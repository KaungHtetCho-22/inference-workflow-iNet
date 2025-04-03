import os
import json
import re
import time
import requests
import schedule
import logging
import colorlog
import pandas as pd
import joblib
import sys
from collections import defaultdict
from datetime import datetime

from datetime import datetime, timedelta
from sqlalchemy.orm import sessionmaker
from monsoon_biodiversity_common.db_model import init_database, RpiDevices, SpeciesDetections, AudioFiles

# --------------------------
# Configuration
# --------------------------
OUTPUT_DIR = "json-output"
MODEL_PATH = "/app/weights/xgboost-model.pkl"

TOKEN_URL = os.getenv("TOKEN_URL")
CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
USERNAME = os.getenv("API_USERNAME")
PASSWORD = os.getenv("API_PASSWORD")
API_URL = os.getenv("API_URL")
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///app-data/soundscape-model.db") 
OUTPUT_DIR = ("json-output")

OUTPUT_DIR = ("json-output")
os.makedirs(OUTPUT_DIR, exist_ok=True)  

SELECTED_FEATURES = [
    'hour', 'Abroscopus-superciliaris', 'Acheta-domesticus', 'Alcedo-atthis', 'Alophoixus-pallidus',
    'Anthus-hodgsoni', 'Cacomantis-merulinus', 'Cacomantis-sonneratii', 'Centropus-bengalensis',
    'Centropus-sinensis', 'Ceryle-rudis', 'Chrysocolaptes-guttacristatus', 'Conocephalus-fuscus',
    'Coracias-benghalensis', 'Culicicapa-ceylonensis', 'Cyornis-whitei', 'Dicrurus-leucophaeus',
    'Dicrurus-remifer', 'Erpornis-zantholeuca', 'Eudynamys-scolopaceus', 'Eumodicogryllus-bordigalensis',
    'Galangal-abeculata', 'Gallus-gallus', 'Glaucidium-cuculoides', 'Gryllus-bimaculatus',
    'Harpactes-erythrocephalus', 'Harpactes-oreskios', 'Hierococcyx-sparverioides', 'Hirundo-rustica',
    'Hypothymis-azurea', 'Hypsipetes-leucocephalus', 'Ixos-mcclellandii', 'Merops-leschenaulti',
    'Merops-orientalis', 'Myiomela-leucura', 'Nyctyornis-athertoni', 'Oecanthus-pellucens', 'Parus-minor',
    'Pericrocotus-speciosus', 'Phaenicophaeus-tristis', 'Phaneroptera-falcata', 'Phaneroptera-nana',
    'Phoenicurus-auroreus', 'Phyllergates-cucullatus', 'Phylloscopus-inornatus', 'Phylloscopus-omeiensis',
    'Picumnus-innominatus', 'Psilopogon-asiaticus', 'Psilopogon-haemacephalus', 'Psilopogon-lineatus',
    'Psilopogon-virens', 'Pycnonotus-aurigaster', 'Saxicola-stejnegeri', 'Spilopelia-chinensis',
    'Surniculus-lugubris', 'Turnix-suscitator', 'Turnix-tanki', 'Upupa-epops', 'Urosphena-squameiceps',
    'Yungipicus-canicapillus']

bird_species = set(s for s in SELECTED_FEATURES if s != "hour")
insect_species = set(["Acheta-domesticus", "Conocephalus-fuscus", "Eumodicogryllus-bordigalensis", "Galangal-abeculata",
    "Gryllus-bimaculatus", "Oecanthus-pellucens", "Phaneroptera-falcata", "Phaneroptera-nana",
    "Platypleura-cfcatenata", "Platypleura-plumosa", "Platypleura-sp10", "Platypleura-sp12cfhirtipennis",
    "Platypleura-sp13", "Ruspolia-nitidula"])

# --------------------------
# Logging Setup
# --------------------------
logger = colorlog.getLogger()
logger.setLevel(logging.INFO)
handler = colorlog.StreamHandler()
formatter = colorlog.ColoredFormatter("%(log_color)s%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    log_colors={ 'DEBUG': 'cyan', 'INFO': 'green', 'WARNING': 'yellow', 'ERROR': 'red', 'CRITICAL': 'bold_red'})
handler.setFormatter(formatter)
logger.addHandler(handler)

log_directory = "/app/logs"
os.makedirs(log_directory, exist_ok=True)
file_handler = logging.FileHandler(os.path.join(log_directory, "daily_report.log"))
file_formatter = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

# --------------------------
# Globals
# --------------------------
ACCESS_TOKEN = None
TOKEN_EXPIRATION = None
SessionLocal = init_database(DATABASE_URL)
session = SessionLocal()

# --------------------------
# Score Prediction Logic
# --------------------------
def predict_scores_by_device(target_date):
    model = joblib.load(MODEL_PATH)
    if isinstance(target_date, str):
        target_date = datetime.strptime(target_date, "%Y-%m-%d").date()

    rows = []
    devices = session.query(RpiDevices).join(AudioFiles).filter(AudioFiles.recording_date == target_date).all()
    for device in devices:
        audio_files = session.query(AudioFiles).filter_by(device_id=device.id, recording_date=target_date).all()
        for af in audio_files:
            detections = session.query(SpeciesDetections).filter_by(audio_file_id=af.id).all()
            for d in detections:
                try:
                    second = d.time_segment_id.split("_")[-1]
                    hour = int(int(second) / 3600)
                    rows.append({"device_area": device.pi_id, "hour": hour, "species": d.species_class, "confidence": d.confidence_score})
                except:
                    continue

    df = pd.DataFrame(rows)
    if df.empty:
        return {}

    features_df = df.pivot_table(index=["device_area", "hour"], columns="species", values="confidence", aggfunc="count", fill_value=0).reset_index()
    for col in SELECTED_FEATURES:
        if col not in features_df.columns:
            features_df[col] = 0

    X = features_df[SELECTED_FEATURES]
    preds = model.predict(X)
    features_df['score_prediction'] = preds

    # score_map = {0: 'A', 1: 'B', 2: 'C'}
    # result = features_df.groupby("device_area")['score_prediction'].agg(lambda x: x.value_counts().idxmax()).map(score_map).to_dict()
    
    result = features_df.groupby("device_area")['score_prediction'].agg(lambda x: x.value_counts().idxmax()).to_dict()

    return result

# --------------------------
# API Token
# --------------------------
def get_access_token():
    global ACCESS_TOKEN, TOKEN_EXPIRATION
    data = {
        "grant_type": "password",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
        "username": USERNAME,
        "password": PASSWORD,
        "scope": "profile openid monsoon-server.batchjob"
    }
    max_retries = 3
    for attempt in range(max_retries):
        try:
            logger.info(f"Requesting access token (Attempt {attempt + 1})...")
            response = requests.post(TOKEN_URL, data=data, timeout=20)
            response.raise_for_status()
            token_data = response.json()
            ACCESS_TOKEN = token_data["access_token"]
            TOKEN_EXPIRATION = datetime.now() + timedelta(seconds=token_data.get("expires_in", 3600))
            logger.info(f"Access token obtained, expires at {TOKEN_EXPIRATION}")
            return
        except requests.RequestException as e:
            logger.warning(f"Token request failed on attempt {attempt + 1}: {e}")
            time.sleep(2 ** attempt)
    logger.error("All token request attempts failed.")

def ensure_valid_token():
    if ACCESS_TOKEN is None or TOKEN_EXPIRATION is None or datetime.now() >= TOKEN_EXPIRATION:
        get_access_token()

# --------------------------
# Build JSON Payload
# --------------------------
def get_detections_by_date(target_date):
    result_dict = {}
    score_map = predict_scores_by_device(target_date)

    target_date_obj = datetime.strptime(target_date, "%Y-%m-%d").date()
    devices = session.query(RpiDevices).join(AudioFiles).filter(AudioFiles.recording_date == target_date_obj).all()

    logger.info(f"Found {len(devices)} devices for {target_date_obj}")

    for device in devices:
        device_id = device.pi_id
        audio_files = session.query(AudioFiles).filter_by(device_id=device.id, recording_date=target_date_obj).all()

        logger.info(f"Processing {len(audio_files)} audio files for device {device_id}")

        # Prepare species counts per hour
        species_hourly_counts = defaultdict(lambda: {
            "category": None,
            "hourly_counts": [0] * 24
        })

        for af in audio_files:
            detections = session.query(SpeciesDetections).filter_by(audio_file_id=af.id).all()
            for d in detections:
                sp = d.species_class
                if sp == "nocall":
                    continue  # Skip nocall entirely

                # try:
                #     second = int(d.time_segment_id.split("_")[-1])
                #     hour = int(second / 3600)
                #     hour = max(0, min(hour, 23))  # Ensure hour is 0–23
                # except:
                #     hour = 0


                try:
                    file_key = af.file_key  # e.g. RPiID-0000000090d15aba_2025-04-01_15-10-33_dur=...
                    match = re.search(r"_(\d{4}-\d{2}-\d{2})_(\d{2})-(\d{2})-(\d{2})", file_key)
                    if match:
                        date_str, h, m, s = match.groups()
                        start_seconds = int(h) * 3600 + int(m) * 60 + int(s)
                        relative_second = int(d.time_segment_id.split("_")[-1])
                        absolute_second = start_seconds + relative_second
                        hour = max(0, min(int(absolute_second / 3600), 23))  # Clamp to 0–23
                    else:
                        logger.warning(f"[WARN] Failed to extract start time from file_key: {file_key}")
                        hour = 0
                except Exception as e:
                    logger.warning(f"[WARN] Error parsing hour: {e}")
                    hour = 0


                species_hourly_counts[sp]["category"] = (
                    "bird" if sp in bird_species else
                    "insect" if sp in insect_species else
                    "nocall"
                )
                species_hourly_counts[sp]["hourly_counts"][hour] += 1

        species_data = {
            sp: [data["category"]] + [str(x) for x in data["hourly_counts"]]
            for sp, data in species_hourly_counts.items()
        }

        if species_data:
            result_dict[device_id] = [{
                "date": target_date_obj.strftime("%Y%m%d"),
                "coordinate": [18.8018, 98.9948],
                "score": score_map.get(device_id, "A"),
                "species": species_data
            }]
        else:
            logger.warning(f"No valid detections for {device_id}")

    return result_dict


# --------------------------
# Save and Send
# --------------------------
def save_json_results(data, target_date):
    filename = os.path.join(OUTPUT_DIR, f"predictions_{target_date}.json")
    try:
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)
        logger.info(f"[SAVE] JSON saved to {filename}")
    except Exception as e:
        logger.error(f"[ERROR] Saving JSON: {e}")

def send_predictions(prediction_data):
    ensure_valid_token()
    if not ACCESS_TOKEN:
        logger.warning("[SKIP] No access token, skipping API call.")
        return

    headers = {
        "Authorization": f"Bearer {ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }

    for attempt in range(10):
        try:
            response = requests.post(API_URL, headers=headers, json=prediction_data, timeout=10)
            response.raise_for_status()
            logger.info(f"[API] Prediction sent: {response.json()}")
            return
        except requests.RequestException as e:
            logger.warning(f"[API] Attempt {attempt + 1} failed: {e}")
            time.sleep(2 ** attempt)
    logger.error("[API] All retries failed.")



# --------------------------
# Main Entrypoint
# --------------------------
def daily_task(target_date=None):
    logger.info("=== Starting daily report task ===")
    if target_date is None:
        target_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    logger.info(f"[DATE] Generating report for: {target_date}")
    data = get_detections_by_date(target_date)
    if data:
        save_json_results(data, target_date)
        send_predictions(data)
    else:
        logger.warning("[WARN] No detection data found.")
    logger.info("=== Daily report task complete ===\n")

if __name__ == "__main__":
    target_date_arg = sys.argv[1] if len(sys.argv) > 1 else None
    if target_date_arg:
        daily_task(target_date_arg)
    else:
        schedule.every().day.at("23:59").do(daily_task)
        logger.info("[SCHEDULER] Waiting for 23:59 each day.")
        while True:
            schedule.run_pending()
            time.sleep(60)
