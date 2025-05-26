
# GAN-based Data Augmentation for Bank Deposit Prediction

##  Project Overview
This project addresses the problem of imbalanced data in predicting bank term deposit subscriptions, using Generative Adversarial Networks (GANs) to generate synthetic samples for the minority class.

##  Tools and Libraries
- Python
- NumPy
- Pandas
- Scikit-learn
- TensorFlow / Keras
- Matplotlib
- Seaborn

##  How to Run
1️ Install the requirements:
```bash
pip install -r requirements.txt
```
or manually:
```bash
pip install numpy pandas scikit-learn tensorflow matplotlib seaborn
```

2️ Run the file:
- If using Python script:
```bash
python GAN_Source_Code.py
```

- If using Jupyter Notebook:
Open `GAN_Notebook.ipynb` and run all the cells.

##  Project Contents
- `GAN_Source_Code.py` → Main project code.
- `GAN_Notebook.ipynb` → Jupyter Notebook version (if provided).
- `README.md` → This file.
- `bank.csv` → Dataset file (if allowed to include).
- `ST Report.docx` → Final project report.

##  Results Description
The project generates synthetic samples using Vanilla GAN and DCGAN, then trains a Random Forest classifier and compares performance.

Evaluation metrics used:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC

##  Project Outputs
- Comparison of model performance before and after augmentation.
- Visualizations of data distribution and improvements.
- Future recommendations for further improvement.
