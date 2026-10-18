# Predicting High-Spending International Visitors in Ghana

Traditional machine learning (Scikit-Learn) and deep learning (TensorFlow / Keras) compared on the Ghana International Travellers' Survey (GITS) 2022–2023. The task is binary classification: predict whether an overnight international visitor is a **high spender** (total spend above the training-set median).

This repository accompanies the Summative Assignment for the Introduction to Machine Learning module.

**Links:** [Written report](#) | [Presentation video](#) | [Google Drive (dataset zip and saved models)](#)

**Author:** [Student name], [Student ID]

---

## Motivation

Destination managers in Africa have limited marketing budgets and rarely know which visitors will spend the most locally. This project asks whether a visitor's origin, purpose of trip, travel arrangements and length of stay can predict high spending. The aim is to support targeting decisions for tourism bodies, including Nile-based attractions such as those around Jinja, Uganda. The model is trained on Ghana data and is not claimed to transfer directly to Uganda.

## Dataset

- **Source:** Ghana Statistical Service, Ghana International Travellers' Survey (GITS) 2022–2023, respondents file (Q1–Q4), Microdata Catalog.
- **Raw rows:** 51,909 respondents, 188 variables. Overnight visitors only are used: **50,415 rows**, which meets the 20,000-row minimum.
- **Encoding:** Latin-1 (the file contains a £ sign that the default UTF-8 reader rejects).
- **Access:** The raw file is not stored in this repository. Download it from the Ghana Statistical Service Microdata Catalog and place it in `data/`, or use the copy in the project Google Drive folder (see link above).

Raw data files are excluded from version control through `.gitignore`.

## Task and target

- **Target:** `high_spender = 1` if `nall` (total visitor spend) is above the median of the training split, otherwise `0`. The training median is 8,702 GHS. The classes are balanced by construction.
- **Split:** 70 / 15 / 15 stratified by continent of residence. Train 35,288, validation 7,564, test 7,563.

## Leakage control

The target is built from spending, so every spending-related column is excluded from the predictors: the expenditure components, the total `nall`, the amount fields `sbq20a`, `sbq22a`, `sbq24a`, `sbq26a`, and the spending band `sbq27a` with its currency field `sbq27b`. The predictor set contains 30 categorical variables and one numeric variable (`sbq15`). The median is computed on the training split only, so no information from validation or test data affects the cut-off.

## Methods

**Traditional models (Scikit-Learn):** logistic regression (default and tuned), random forest (random search), histogram gradient boosting (random search), and linear SVM (grid search). Hyperparameters are tuned with three-fold cross-validation on the training set only.

**Deep learning (TensorFlow / Keras):** four fully connected networks built with the Functional API, with batch normalisation and dropout. Data is read through a `tf.data` pipeline with shuffling, batching, caching and prefetching. Early stopping on validation ROC-AUC (patience 4, maximum 30 epochs), with TensorBoard logs.

| Deep run | Hidden units | Dropout | Learning rate |
|---|---|---|---|
| D1 wide | 256, 128 | 0.3 | 1e-3 |
| D2 narrow | 64, 32 | 0.2 | 1e-3 |
| D3 high lr | 128, 64 | 0.3 | 1e-2 |
| D4 deep, slow | 128, 64, 32 | 0.4 | 3e-4 |

## Results

Test-set performance (held out, used once for final evaluation):

| Model | ROC-AUC | PR-AUC | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| **T4 Gradient Boosting** | **0.8665** | **0.8610** | **0.7857** | 0.7753 | 0.8243 | 0.7991 |
| D1 Deep, wide | 0.8615 | 0.8566 | 0.7775 | 0.7572 | 0.8384 | 0.7957 |
| D4 Deep, slow | 0.8611 | 0.8553 | 0.7768 | 0.7660 | 0.8182 | 0.7912 |
| D3 Deep, high lr | 0.8605 | 0.8544 | 0.7765 | 0.7424 | 0.8696 | 0.8009 |
| D2 Deep, narrow | 0.8588 | 0.8525 | 0.7809 | 0.7755 | 0.8110 | 0.7928 |
| T3 Random Forest | 0.8581 | 0.8539 | 0.7751 | 0.7544 | 0.8376 | 0.7938 |
| T2 Logistic (tuned) | 0.8433 | 0.8309 | 0.7665 | 0.7557 | 0.8102 | 0.7820 |
| T1 Logistic (baseline) | 0.8430 | 0.8305 | 0.7658 | 0.7548 | 0.8102 | 0.7815 |
| T5 Linear SVM | 0.8427 | 0.8308 | 0.7661 | 0.7521 | 0.8169 | 0.7831 |

**Best model:** Gradient Boosting (T4), with a test confusion matrix at threshold 0.5 of 2,719 correct low spenders, 934 false positives, 687 false negatives and 3,223 correct high spenders.

**Key findings**

- Tree-based and deep models outperform linear models by roughly 1.5 to 2.5 ROC-AUC points.
- Length of stay carries most of the predictive signal. Removing the two length-of-stay variables reduces validation ROC-AUC from 0.863 to 0.842.
- Permutation importance ranks `sbq15` first, but retraining without it changes validation ROC-AUC by only 0.002. The variable's meaning is not documented in the codebook used here, so it should be checked before the findings are reused.
- Gradient Boosting overfits (training ROC-AUC well above cross-validation), but its cross-validation score is still rising with more data.

## Repository structure

```
.
├── GITS_high_spender_classification.ipynb   # Full pipeline: EDA, models, evaluation, feature checks
├── requirements.txt                         # Python dependencies
├── README.md
├── .gitignore                               # Excludes raw data and checkpoints
├── data/                                    # Raw GITS file (not committed)
├── models/                                  # Saved models and preprocessor (created on run)
├── figures/                                 # Plots saved by the notebook
└── logs/                                    # TensorBoard logs for deep runs
```

## Installation

Python 3.10 or newer is recommended.

```bash
git clone <repository-url>
cd <repository-folder>
pip install -r requirements.txt
```

## How to run

1. Place the raw file `2022-2023GITS-respondents_Q1-Q4_v1.csv` in `data/`, or adjust `DATA_PATH` in the data-loading cell.
2. Open the notebook in Jupyter or Google Colab and run all cells from top to bottom.
3. The notebook creates `models/`, `figures/` and `logs/` if they do not exist.

**Google Colab:** the submitted notebook mounts Google Drive and reads the file from `My Drive/summative_intro_to_machine_learning/data/`. Change `DATA_PATH` to match your own Drive folder.

**TensorBoard:** `tensorboard --logdir logs`

**Runtime:** the full notebook takes roughly 10 to 15 minutes on a single CPU core, mostly in the random searches and the deep runs.

## Using the saved models

```python
import joblib
import tensorflow as tf

preprocessor = joblib.load("models/preprocessor.joblib")
trad_model = joblib.load("models/best_traditional.joblib")
deep_model = tf.keras.models.load_model("models/best_deep.keras")

X_new = preprocessor.transform(new_visitors_df)   # same predictor columns as training
p_trad = trad_model.predict_proba(X_new)[:, 1]
p_deep = deep_model.predict(X_new).ravel()
```

The notebook also saves the deep model in legacy `.h5` format (`models/best_deep.h5`).

## Reproducibility

- Random seed `42` is set for Python, NumPy and TensorFlow, and passed to all splitters and estimators.
- Package versions are pinned in `requirements.txt`. Results were produced with TensorFlow 2.21.0 and Scikit-Learn 1.8.0 on CPU.
- Small numerical differences can appear across hardware or library builds, particularly in the deep runs.

## Limitations

- The sample covers international travellers leaving Ghana, mostly by air, with West African residents forming a large share. Results may not generalise to other countries or to land-border travellers.
- The currency and units of `nall` are assumed to be Ghana cedis and have not been confirmed against the codebook.
- The total spend does not equal the sum of its listed components for many respondents, so the target variable is not fully validated.
- The median cut-off is a modelling choice, not a fact about tourism.
- Results come from a single split without confidence intervals.
- The deep configuration D4 had not converged after 30 epochs.

## Future work

Bootstrap confidence intervals on the test set, repeated splits with different seeds, training deep models to convergence, probability calibration, a per-night spending target, and collecting comparable survey data in Uganda.

## References

1. Ghana Statistical Service, "Ghana International Travellers' Survey (GITS) 2022–2023, respondents file," Microdata Catalog, 2025.
2. L. Breiman, "Random forests," *Machine Learning*, vol. 45, no. 1, pp. 5–32, 2001.
3. J. H. Friedman, "Greedy function approximation: A gradient boosting machine," *Annals of Statistics*, vol. 29, no. 5, pp. 1189–1232, 2001.
4. F. Pedregosa et al., "Scikit-learn: Machine learning in Python," *JMLR*, vol. 12, pp. 2825–2830, 2011.
5. M. Abadi et al., "TensorFlow: A system for large-scale machine learning," in *Proc. OSDI*, 2016.
6. D. P. Kingma and J. Ba, "Adam: A method for stochastic optimization," in *Proc. ICLR*, 2015.
