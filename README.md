# 🚀 Application of Machine Learning Methods to Identify and Categorize Radio Pulsar Signal Candidates

This project applies **machine learning techniques** to automatically identify and categorize **radio pulsar signals** from the [HTRU2 dataset](https://archive.ics.uci.edu/dataset/372/htru2).  
It demonstrates a full end-to-end ML pipeline — from data preprocessing and supervised classification to unsupervised clustering and visualization.

---

## 🌌 Project Overview

Pulsars are highly magnetized, rotating neutron stars that emit beams of electromagnetic radiation. Detecting them from telescope data is challenging due to heavy noise and radio frequency interference (RFI).  
This project uses both **supervised** and **unsupervised** ML models to detect pulsars and explore potential subgroups.

---

## ⚙️ Pipeline Structure

| Stage | Description |
|--------|--------------|
| **1. Data Preprocessing** | Loads the HTRU2 dataset, scales features, and splits data into train/test sets. |
| **2. Supervised Learning** | Trains a Random Forest classifier with SMOTE to handle class imbalance. |
| **3. Threshold Optimization** | Tunes decision threshold to maximize recall while reducing false positives. |
| **4. Unsupervised Learning** | Applies K-Means (k=3) and Self-Organizing Map (SOM) to analyze pulsar clusters. |
| **5. Visualization & Evaluation** | Generates ROC, PRC, and PCA-based cluster plots. |

---

## 📊 Key Results

| Metric | Value | Notes |
|--------|--------|-------|
| **Accuracy** | 97.23% | Excellent overall classification performance |
| **ROC AUC** | 0.968 | Strong discrimination between pulsar and non-pulsar signals |
| **Recall (t = 0.5)** | 0.896 | High recall at default threshold |
| **Recall (t = 0.66)** | 0.88 | Balanced recall with fewer false positives |
| **K-Means Silhouette** | 0.388 | Moderate cluster separability |
| **SOM Silhouette** | 0.193 | Weaker micro-cluster distinction |

🧩 These results suggest the dataset’s eight statistical features capture partial structure among pulsars, but deeper separability may require feature extraction from raw time-frequency data.

---

## 🗂️ Repository Structure

pulsar-ml-project/
├── data/
│ └── HTRU_2.csv
├── src/
│ ├── data_preprocess.py
│ ├── train_supervised.py
│ ├── evaluate.py
│ ├── train_unsupervised.py
│ └── utils.py
├── results/
│ ├── models/
│ │ └── rf_best.pkl
│ └── figures/
│ ├── roc.png
│ ├── prc.png
│ ├── kmeans_pos_pca.png
│ └── som_pos_pca.png
├── requirements.txt
├── README.md
└── .gitignore
## 🧩 How to Run

1️⃣ Clone the repository
```bash
git clone https://github.com/sanjanapulla06/pulsar-ml-project.git
cd pulsar-ml-project

2️⃣ Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate  # (on Windows)
pip install -r requirements.txt

3️⃣ Download the dataset

Download the HTRU_2.csv
 file
and place it inside the data/ folder.

4️⃣ Run each stage
python src\data_preprocess.py
python src\train_supervised.py
python src\evaluate.py
python src\train_unsupervised.py

---

## 🧑‍🤝‍🧑 Team Information

**Team 16 – Machine Learning Mini Project**  
**Project Title:** Application of Machine Learning Methods to Identify and Categorize Radio Pulsar Signal Candidates  

| Team Member | SRN | Role |
|--------------|------|------|
| 🧠 **Sanjana Pulla** | PES2UG23CS529 | Model Development, Data Preprocessing, Documentation |
| 🌌 **Sharon A** | PES2UG23CS544 | Model Evaluation, Visualization, Unsupervised Learning |

**Course:** Machine Learning (Mini Project)  
**Department of Computer Science and Engineering**  
**PES University, Electronic City Campus, Bengaluru**  
**Academic Year:** 2025  

---

## 💫 Faculty Guidance
This project was carried out under the guidance of the Kavitha P.
**Department of Computer Science, PES University**.  
We are grateful for the support and mentorship provided by the faculty during the course of this mini project.

---
