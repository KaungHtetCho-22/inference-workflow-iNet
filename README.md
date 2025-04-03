# inference-workflow-iNet

# Bird Sound Monitoring & Scoring Pipeline

This project automates the process of monitoring bird sounds using a Raspberry Pi, classifying them using machine learning, predicting scores based on classification results, and sending the output to an API in JSON format.

---

![workflow](diagram.png)

### Data Source (Raspberry Pi)
- **Device:** IOT devices (Raspberry Pi + Audiomoth sensors) for audio data collection.
- **Protocol:** FTPS (secure FTP) is used for secure file transfer.
- **Destination:** Files are uploaded to the iNet private cloud:

---

### Audio Collection
- The Raspberry Pi uploads **10-minute audio clips** in `.WAV` format.

---

### Bird Classification Model
- The audio clips are processed by a **Bird Classification Model**.
- This model performs inference/classification to identify bird species. 
- The classification results are stored in a **MySQL database**.

---

### Score Prediction Model
- The **Score Prediction Model** retrieves classification results from MySQL.
- It predicts a **score** based on the results of the bird classification.

---

### API Output
- The predicted scores are formatted as **JSON**.
- The JSON output is sent to an **API endpoint** for further use

---

## Components Summary
| Component                | Description                                  |
|--------------------------|----------------------------------------------|
| Raspberry Pi + Audiomoht + 4G router        | Collects audio data                          |
| FTPS                     | Secure transfer protocol                     |
| iNet Server              | Inferencing machine               |
| Bird Classification Model| Identifies bird sounds from audio            |
| MySQL                    | Stores classification results               |
| Score Prediction Model   | Predicts score based on classification       |
| API                      | Receives JSON payloads from score model      |


---



