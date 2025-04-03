import os
import time
import queue
import threading
from pathlib import Path
import librosa
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datetime import datetime, timezone
from sqlalchemy.exc import IntegrityError
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer
import logging
import colorlog  

from monsoon_biodiversity_common.config import cfg as CFG
from monsoon_biodiversity_common.db_model import init_database, RpiDevices, SpeciesDetections, AudioFiles
from monsoon_biodiversity_common.model import AttModel
from monsoon_biodiversity_common.dataset import TestDataset

# --------------------------
# Colored logging
# --------------------------
logger = colorlog.getLogger()
logger.setLevel(logging.INFO)

# --------------------------
# Formatter for logging
# --------------------------
handler = colorlog.StreamHandler()
formatter = colorlog.ColoredFormatter(
    "%(log_color)s%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    log_colors={
        'DEBUG': 'cyan',
        'INFO': 'green',
        'WARNING': 'yellow',
        'ERROR': 'red',
        'CRITICAL': 'bold_red',
    }
)
handler.setFormatter(formatter)
logger.addHandler(handler)

# file logging
log_directory = "/app/logs"
os.makedirs(log_directory, exist_ok=True)
log_file_path = os.path.join(log_directory, "audio_inference.log")
file_handler = logging.FileHandler(log_file_path)
file_formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# databse
db_path = '/app/app-data/soundscape-model.db'
DATABASE_URL = f'sqlite:///{db_path}'
logger.info(f"Database URL: {DATABASE_URL}")
Session = init_database(DATABASE_URL)

# audio-data
root_directory = '/app/audio-data/'
if not os.path.exists(root_directory) or not os.path.isdir(root_directory):
    raise Exception(f"Audio directory not found: {root_directory}")

contents = os.listdir(root_directory)
folders = [item for item in contents if os.path.isdir(os.path.join(root_directory, item))]
directories_to_monitor = [folder for folder in folders if 'RPiID' in folder]

if directories_to_monitor:
    logger.info("Relevant Folders:")
    for i, folder in enumerate(directories_to_monitor, 1):
        logger.info(f"  {i}. {folder}")
else:
    logger.warning("No relevant folders found in the directory.")

# Setup model
pi_type = 1  
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
weight_path = '/app/weights/soundscape-model.pt'
if not os.path.isfile(weight_path):
    raise Exception(f"Model weights file not found: {weight_path}")

state_dict = torch.load(weight_path, map_location=device, weights_only=False)['state_dict']
model = AttModel(backbone=CFG.backbone, num_class=CFG.num_classes, infer_period=5, cfg=CFG, training=False, device=device)
model.load_state_dict(state_dict)
model = model.to(device)
model.logmelspec_extractor = model.logmelspec_extractor.to(device)
model.eval()

# File queue
file_queue = queue.Queue()

def prediction_for_clip(audio_path):
    prediction_dict = {}
    classification_dict = {}

    try:
        logger.info(f"[LOAD] Loading: {audio_path}")
        clip, sr = librosa.load(audio_path, sr=32000)
    except Exception as e:
        logger.error(f"[ERROR] Loading failed: {e}")
        return {}, {}

    duration = librosa.get_duration(y=clip, sr=32000)
    seconds = list(range(5, int(duration), 5))
    row_ids = [Path(audio_path).stem + f"_{second}" for second in seconds]

    test_df = pd.DataFrame({"row_id": row_ids, "seconds": seconds})
    dataset = TestDataset(df=test_df, clip=clip, cfg=CFG)
    loader = torch.utils.data.DataLoader(dataset, **CFG.loader_params['valid'])

    logger.info(f"[INFER] Processing {len(dataset)} segments")
    for inputs in tqdm(loader, desc=f"Processing segments"):
        row_ids = inputs.pop('row_id')
        with torch.no_grad():
            output = model(inputs)['logit']
            for idx, row_id in enumerate(row_ids):
                logits = output[idx, :].sigmoid().cpu().numpy()
                prediction_dict[row_id] = {CFG.target_columns[i]: logits[i] for i in range(len(CFG.target_columns))}
                classification_dict[row_id] = {
                    'row_id': row_id,
                    'Class': CFG.target_columns[np.argmax(logits)],
                    'Score': np.max(logits)
                }
    return classification_dict

def save_predictions_to_db(audio_path, classification_dict, session):
    path_parts = Path(audio_path).parts
    if len(path_parts) < 3:
        logger.error(f"[ERROR] Invalid path: {audio_path}")
        return

    pi_id = path_parts[-3]
    recording_date_str = path_parts[-2]

    try:
        recording_date = datetime.strptime(recording_date_str, "%Y-%m-%d").date()
    except ValueError:
        logger.error(f"[ERROR] Invalid date format in path: {audio_path}")
        return

    audio_filename = Path(audio_path).name
    unique_file_key = f"{pi_id}_{recording_date}_{audio_filename}"

    device = session.query(RpiDevices).filter_by(pi_id=pi_id).first()
    if not device:
        device = RpiDevices(pi_id=pi_id, pi_type=pi_type)
        session.add(device)
        session.commit()

    audio_file = session.query(AudioFiles).filter_by(file_key=unique_file_key, device_id=device.id, recording_date=recording_date).first()
    if not audio_file:
        audio_file = AudioFiles(device_id=device.id, recording_date=recording_date, file_key=unique_file_key)
        session.add(audio_file)
        session.commit()

    new_detections = []
    for row_id, classification in classification_dict.items():
        exists = session.query(SpeciesDetections).filter_by(audio_file_id=audio_file.id, time_segment_id=row_id).first()
        if not exists:
            new_detections.append(SpeciesDetections(
                audio_file_id=audio_file.id,
                time_segment_id=row_id,
                species_class=classification['Class'],
                confidence_score=classification['Score'],
                created_at=datetime.now(timezone.utc)
            ))

    if new_detections:
        session.add_all(new_detections)
        session.commit()
        logger.info(f"[DB] Added {len(new_detections)} detections")

def delete_file_safely(path):
    try:
        os.remove(path)
        logger.info(f"[CLEANUP] Deleted: {path}")
    except Exception as e:
        logger.warning(f"[CLEANUP] Failed to delete {path}: {e}")

def process_new_audio(audio_path):
    logger.info(f"=== START FILE: {audio_path} ===")

    last_size = -1
    stable_count = 0
    max_wait = 60

    for i in range(max_wait):
        if not os.path.exists(audio_path):
            logger.info(f"[WAIT] File not found yet ({i}s)")
            time.sleep(1)
            continue

        current_size = os.path.getsize(audio_path)
        if current_size == last_size and current_size > 0:
            stable_count += 1
            if stable_count >= 3:
                break
        else:
            stable_count = 0
            last_size = current_size
        time.sleep(1)

    if not os.path.isfile(audio_path) or os.path.getsize(audio_path) == 0:
        logger.error(f"[ERROR] File not stable or empty: {audio_path}")
        return

    session = None
    try:
        session = Session()
        classification_dict = prediction_for_clip(audio_path)
        # if classification_dict:
        #     save_predictions_to_db(audio_path, classification_dict, session)
        #     delete_file_safely(audio_path)
        if classification_dict:
            save_predictions_to_db(audio_path, classification_dict, session)
            # Comment out deletion to retain file for one day
            # delete_file_safely(audio_path)
            logger.info(f"[KEEP] Keeping file for one day: {audio_path}")

        else:
            logger.warning(f"[WARN] No classifications for: {audio_path}")
    except Exception as e:
        logger.error(f"[ERROR] Exception: {str(e)}")
        import traceback
        traceback.print_exc()
    finally:
        if session:
            session.close()

    logger.info(f"=== END FILE: {audio_path} ===")
    logger.info('***' * 10)

def process_queue():
    logger.info("[THREAD] Starting file processor")
    while True:
        try:
            if file_queue.empty():
                time.sleep(2)
                continue
            audio_path = file_queue.get()
            logger.info(f"[QUEUE] Processing: {audio_path}")
            process_new_audio(audio_path)
            file_queue.task_done()
            logger.info(f"[QUEUE] Remaining: {file_queue.qsize()}")
        except Exception as e:
            logger.error(f"[ERROR] Queue processing: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(5)

processing_thread = threading.Thread(target=process_queue, daemon=True)
processing_thread.start()

class AudioFileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory and event.src_path.endswith((".wav", ".ogg", ".mp3")):
            logger.info(f"[NEW FILE] {event.src_path}")
            file_queue.put(event.src_path)

def monitor_directory(directory):
    event_handler = AudioFileHandler()
    observer = Observer()
    for folder in directories_to_monitor:
        observer.schedule(event_handler, path=os.path.join(directory, folder), recursive=True)
    observer.start()
    logger.info(f"[MONITOR] Watching: {directory}")
    try:
        while True:
            logger.debug("[MONITOR] Alive")
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        file_queue.put(None)
    observer.join()

if __name__ == "__main__":
    logger.info(f"[SERVICE] Starting for: {root_directory}")
    monitor_directory(root_directory)
