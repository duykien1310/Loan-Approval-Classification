<!-- Improved compatibility for back-to-top link -->
<a id="readme-top"></a>

<!-- PROJECT HEADER -->
<div align="center">
  <h3 align="center">Loan Approval Classification</h3>

  <p align="center">
    Build a Machine Learning model to “classify account status with individual subjects wanting to borrow from a bank”.
  </p>
</div>

---

## 📘 Description

This project tackles a **binary classification problem**: determining whether a loan should be **approved (1)** or **rejected (0)** based on multiple financial and personal features.

It uses a **synthetic dataset** (45,000 records, 14 features) generated from financial-risk-inspired data, expanded using **SMOTENC** for balanced categorical and continuous distributions.

The model was implemented entirely from scratch — including:
* Decision tree construction using **Gini Index** and **Entropy**
* Bootstrapped sampling
* Feature subsampling per tree
* Parallelized tree training using joblib.Parallel

---

## ⚙️ Features
* Implemented a **Random Forest** algorithm (no sklearn RandomForestClassifier).
* Supports both **Gini** and **Entropy** split criteria.
* **Parallel training** of trees for faster computation.
* Robust **data preprocessing, encoding**, and **missing value handling**.
* Detailed **model evaluation** with accuracy, precision, recall, and F1 metrics.

---

## Results

| Metric | Weighted Score |
|--------|----------------|
| **Accuracy** | 0.93     |
| **Precision**| 0.93     |
| **Recall**   | 0.93     |
| **F1-Score** | 0.92     |

---

### Classification Report:

| Class | Precision |Recall |F1-Score |Support |
|-------|-----------|-----------|-----------|-----------|
| **0 (Rejected)** | 0.93     |0.98     |0.95     |9108     |
| **1 (Approved)**| 0.93     |0.75     |0.83     |2892     |


#### The model achieves 93% accuracy and demonstrates strong generalization with balanced precision and recall across classes.
---

## ⚙️ Installation
### 🔧 Requirements
* Go 1.21 or higher
* Linux / macOS (for epoll/kqueue support)
* Redis CLI for benchmarking (optional)


### 🧩 Clone the repository:

```bash
git clone https://github.com/duykien1310/Loan-Approval-Classification.git
cd Loan-Approval-Classification
```

### Create a Virtual Environment

```bash
python3 -m venv venv
```

### Activate the Environment

```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install pandas numpy scikit-learn joblib
```

### Run Model

```bash
python3 main.py
```

### The console will display:
* Confusion matrix
* Weighted precision, recall, F1-score
* Classification report
