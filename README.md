# 🚗 CrashSense: Accident Detection from Dashcam Videos using Deep Learning

This project focuses on detecting road accidents — specifically **Drifting/Skidding** and **Rear-End Collisions** — using real-world dashcam videos. The model is built using a hybrid **CNN + LSTM architecture** and trained on the [HWID12 Highway Incidents Dataset](https://www.kaggle.com/datasets/landrykezebou/hwid12-highway-incidents-detection-dataset).

---

## 📌 Features

- 🎥 Handles real-world dashcam video input (.mp4)
- 🧠 Deep Learning architecture using TimeDistributed CNN + LSTM
- 🚘 Detects:
  - Drifting / Skidding (Class 1)
  - Rear-End Collisions (Class 0)
  - Normal driving behavior (Class 0)
- 📈 Achieved ~79% test accuracy
- ✅ Tested on unseen videos with correct predictions

---

## 🗃️ Dataset

- **Source**: [HWID12 Accident Detection Dataset](https://www.kaggle.com/datasets/landrykezebou/hwid12-highway-incidents-detection-dataset)
- **Video Types Used**:
  - `drifting_or_skidding`
  - `rear_collision`
  - `negative_samples_selected` (normal driving clips)

- **Total Samples Used**:
  - Class 0 (Rear + Normal): 354
  - Class 1 (Drifting): 312

---

## 🧠 Model Architecture

- Input shape: `(10, 64, 64, 3)` → 10-frame video clip
- CNN (via `TimeDistributed`) to extract spatial features per frame
- LSTM to capture temporal patterns across frames
- Dense layers for binary classification

```python
TimeDistributed(Conv2D(16, ...))
TimeDistributed(MaxPooling2D())
...
LSTM(32)
Dense(32, activation='relu')
Dense(1, activation='sigmoid')
