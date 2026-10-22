# EduPredict AI: AI/ML-Based Student Performance Prediction & Early Risk Detection

**Final-Year B.Tech Computer Science & Engineering Project**  
*Department of Computer Science & Engineering*

---

## 1. Executive Summary

**EduPredict AI** is a machine learning pipeline engineered to predict secondary school academic achievement and detect students at risk of academic failure. Built on the authentic **UCI Machine Learning Repository Student Performance Dataset** (Cortez & Silva, 2008), EduPredict AI implements an **Early Warning System (EWS)** architecture.

By strictly excluding midterm grades ($G_1$ and $G_2$) alongside final grade targets ($G_3$), the pipeline operates exclusively on pre-enrollment demographics, family socio-economic background, study habits, lifestyle, and behavioral factors. This formulation prevents **data leakage** and produces actionable predictions at the start of an academic semester—empowering academic counselors and faculty to deploy targeted remedial interventions *before* students take their examinations.

```
+-----------------------------------------------------------------------------------------+
|                                    EDUPREDICT AI PIPELINE                               |
+-----------------------------------------------------------------------------------------+
  [ Raw Data Ingestion ]
           |
           v
  [ Schema & Data Audit ]  --> 395 Students, 33 Attributes, 0 Missing, 0 Duplicates
           |
           v
  [ Stratified Train/Test Split ] (80% Train / 20% Test, Stratified on Academic Risk)
           |
           v
  [ Scikit-Learn Pipeline ] (One-Hot Encoding + Standard Scaling, Fit Exclusively on Train)
           |
     +-----+------------------------------+
     |                                    |
     v                                    v
[ Regression: Grade (G3: 0-20) ]   [ Classification: Academic Risk (G3 < 10) ]
  - Linear Regression                - Logistic Regression (Balanced)
  - Random Forest Regressor          - Random Forest Classifier
  - Gradient Boosting Regressor      - Gradient Boosting Classifier
  - XGBoost Regressor                - XGBoost Classifier
     |                                    |
     v                                    v
  [ 5-Fold Cross-Validation Selection ] [ 5-Fold Stratified CV Selection ]
     |                                    |
     +-----------------+------------------+
                       |
                       v
         [ Single Held-Out Test Evaluation ]
                       |
                       v
         [ SHAP Explainability & Risk Factor Engine ]
                       |
                       v
    [ Joblib Serialized Artifacts & Prediction Interface ]
```

---

## 2. Dataset Audit & Choice Justification

### A. Dataset Identification
During workspace inspection, two datasets were identified:
1. `../Project/Students Performance Dataset.csv`: A 5,000-row synthetic Kaggle dataset with generated values.
2. `../student/student-mat.csv` and `student-por.csv`: The peer-reviewed **UCI Student Performance Dataset** collected from two Portuguese secondary schools (Cortez & Silva, 2008).

**Documented Decision:**  
The **UCI Student Performance Dataset (Mathematics Course - `student-mat.csv`)** was selected as the primary benchmark. It represents authentic academic behavior with a 32.9% failure rate, offering a rigorous, real-world evaluation benchmark for academic risk classification and grade regression.

### B. Exploratory Data Summary
* **Records:** 395 students.
* **Attributes:** 33 total columns (17 categorical, 16 numeric).
* **Missing Values:** 0 (100% complete).
* **Duplicate Rows:** 0 (100% unique).
* **Target Variables:**
  * **Continuous Target ($G_3$):** Final academic grade on a 0–20 point scale ($\mu = 10.42, \sigma = 4.58, \text{Median} = 11.0, \text{Min} = 0, \text{Max} = 20$).
  * **Binary Classification Target (`academic_risk`):** Defined as $G_3 < 10.0$ (failing threshold in the Portuguese educational system). Class balance: 67.1% Pass ($N=265$), 32.9% At Risk ($N=130$).

---

## 3. Data Leakage Prevention Architecture

Data leakage occurs when information from outside the training dataset or post-examination states is used to create the model. EduPredict AI enforces three architectural safeguards:

1. **Strict Target Segregation:** The regression target $G_3$ and classification label `academic_risk` are isolated immediately upon ingestion and never appear in feature sets.
2. **Early Prediction Mode (Pre-Exam Interventions):** In standard operational deployment, intermediate grades $G_1$ (Period 1) and $G_2$ (Period 2) correlate strongly with final grade ($r = 0.801$ and $r = 0.905$, respectively). Including them causes the model to merely memorize midterm scores rather than learn early behavioral risk markers. Therefore, $G_1$ and $G_2$ are strictly removed from the early prediction feature space (30 raw features $\rightarrow$ 56 encoded features).
3. **Train-Only Preprocessing Fitting:** The Scikit-Learn `ColumnTransformer` is fitted **strictly on the 80% training partition** ($N=316$). The held-out 20% test partition ($N=79$) is transformed using the training statistics without updating scaler means or encoder dictionaries.
4. **Single Evaluation on Held-Out Test Set:** All hyperparameter benchmarking and candidate model selection occur via 5-fold cross-validation on the training split. Held-out test data is evaluated exactly once.

---

## 4. Measured Experimental Results

All experiments were conducted with a fixed seed (`random_state=42`) using an 80/20 stratified split.

### A. Task 1: Student Grade Prediction (Regression)

Evaluated using Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Coefficient of Determination ($R^2$):

| Model | 5-Fold CV MAE | 5-Fold CV RMSE | 5-Fold CV $R^2$ | Test MAE | Test RMSE | Test $R^2$ | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Linear Regression** | $3.448 \pm 0.158$ | $4.460 \pm 0.275$ | $0.010 \pm 0.159$ | 3.399 | 4.370 | 0.098 | Baseline |
| **Random Forest Regressor** | **$2.916 \pm 0.269$** | **$3.829 \pm 0.328$** | **$0.279 \pm 0.068$** | **2.905** | **3.946** | **0.265** | **Selected Best Model** |
| **Gradient Boosting Regressor** | $3.078 \pm 0.288$ | $4.077 \pm 0.337$ | $0.178 \pm 0.117$ | 2.965 | 3.943 | 0.266 | Candidate |
| **XGBoost Regressor** | $3.040 \pm 0.315$ | $3.986 \pm 0.384$ | $0.215 \pm 0.122$ | 2.990 | 3.924 | 0.273 | Candidate |

* **Selection:** Random Forest Regressor achieved the lowest cross-validation RMSE (3.829) and was selected. On the held-out test set, it achieved an MAE of 2.905 grade points.

---

### B. Task 2: Academic Risk Classification (Binary: $G_3 < 10$)

Evaluated using Accuracy, Precision, Recall, F1-Score, and ROC-AUC on class 1 ("At Risk / Failing"):

| Model | CV F1-Score | CV Recall | CV ROC-AUC | Test Accuracy | Test Precision | Test Recall | Test F1 | Test ROC-AUC | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression (Balanced)** | **0.478** | **0.540** | **0.631** | **69.62%** | **52.94%** | **69.23%** | **0.600** | **0.727** | **Selected Best Model** |
| **Random Forest Classifier** | 0.348 | 0.290 | 0.643 | 70.89% | 57.89% | 42.31% | 0.489 | 0.728 | Candidate |
| **Gradient Boosting Classifier** | 0.434 | 0.346 | 0.657 | 65.82% | 46.15% | 23.08% | 0.308 | 0.651 | Candidate |
| **XGBoost Classifier** | 0.416 | 0.328 | 0.663 | 68.35% | 53.33% | 30.77% | 0.390 | 0.725 | Candidate |

#### Test Set Confusion Matrix (Logistic Regression):
```
                  Predicted Safe (0)   Predicted At Risk (1)
Actual Safe (0)           37                    16             (Specificity = 69.8%)
Actual At Risk (1)         8                    18             (Recall = 69.2%)
```
* **Why Logistic Regression was Selected:** In student welfare and educational counseling, a **False Negative** (failing to identify an at-risk student) carries a high academic cost. Logistic Regression with inverse class-frequency weighting achieved **69.23% recall** (identifying 18 out of 26 failing students), outperforming standard tree models that prioritized majority-class accuracy at the expense of sensitivity.

---

## 5. Model Explainability & Key Drivers

Explainability was implemented using **SHAP (SHapley Additive exPlanations)** and tree feature importances:

1. **Past Academic Failures (`failures`):** The single strongest predictor of academic struggle. Students with $\ge 1$ past failure have significantly elevated risk probabilities.
2. **School Absenteeism (`absences`):** High absences ($> 10$ days) correlate directly with knowledge gaps.
3. **Weekly Study Time (`studytime`):** Students dedicating $\ge 5$ hours/week demonstrate academic resilience.
4. **Parental Education (`Medu`, `Fedu`):** Higher mother's and father's educational levels correlate positively with academic success.
5. **Higher Education Aspirations (`higher`):** Students planning to attend university have significantly higher motivation and lower failure rates.
6. **Alcohol Consumption (`Dalc`, `Walc`):** Elevated workday and weekend alcohol consumption correlates with lower academic engagement.

> **Methodological Note:** Feature importance indicates statistical predictive association within historical data, not confirmed biological or deterministic causation. These metrics are used strictly to prioritize supportive counseling.

---

## 6. Directory Layout

```
AIML PROJECT 2/
|-- data/
|   |-- student-mat.csv               # Primary Mathematics course dataset (UCI)
|   |-- student-por.csv               # Portuguese language course dataset (UCI)
|   |-- student.txt                   # Attribute documentation
|   +-- student-merge.R               # Original merge script
|-- models/
|   |-- full_regression_pipeline.joblib      # End-to-end Pipeline (Preprocessor + Regressor)
|   |-- full_classification_pipeline.joblib  # End-to-end Pipeline (Preprocessor + Classifier)
|   |-- preprocessor.joblib                  # Fitted ColumnTransformer
|   |-- student_grade_regressor.joblib       # Best Regressor (Random Forest)
|   |-- student_risk_classifier.joblib       # Best Classifier (Logistic Regression)
|   +-- model_metadata.json                  # Complete audit trail & hyperparameter specs
|-- outputs/
|   |-- metrics/
|   |   |-- eda_summary.json                 # Dataset statistics and correlation audit
|   |   |-- regression_metrics.json          # CV and test regression results
|   |   +-- classification_metrics.json      # CV and test classification results
|   +-- plots/
|       |-- target_grade_distribution.png    # Histogram & KDE of G3 with threshold
|       |-- missing_and_duplicates.png       # Missing values & duplicate verification
|       |-- correlation_heatmap.png          # Pearson correlation matrix
|       |-- studytime_absences_analysis.png  # Study hours and absences vs. G3
|       |-- academic_risk_distribution.png   # Class balance breakdown
|       |-- socioeconomic_factors.png        # Parental education & aspirations
|       |-- regression_actual_vs_predicted.png # Parity scatter plot with metrics
|       |-- regression_residuals.png         # Residual distributions
|       |-- regression_model_comparison.png  # Comparative bar charts
|       |-- classification_confusion_matrix.png # Test set confusion heatmap
|       |-- classification_roc_curve.png     # ROC curves with AUC annotations
|       |-- classification_model_comparison.png # Precision/Recall/F1 comparison
|       +-- feature_importance_shap.png      # SHAP impact summary
|-- src/
|   |-- __init__.py
|   |-- data_loader.py                # Ingestion, validation, stratified splitting
|   |-- eda.py                        # EDA generation and plot export
|   |-- preprocessing.py              # ColumnTransformer pipeline
|   |-- train_models.py               # Model benchmarking, CV, and persistence
|   |-- evaluate.py                   # Evaluation metrics & diagnostic charts
|   |-- explain.py                    # SHAP attribution & recommendation rules
|   +-- predict.py                    # Reusable inference engine (EduPredictor)
|-- tests/
|   |-- __init__.py
|   |-- test_data_loader.py           # Ingestion and leakage prevention tests
|   |-- test_preprocessing.py         # Pipeline transformation & serialization tests
|   |-- test_prediction.py            # Inference engine, schema & validation tests
|   +-- test_explainability.py        # SHAP and recommendation engine tests
|-- requirements.txt                  # Locked Python dependencies
+-- README.md                         # Engineering documentation
```

---

## 7. Setup & Execution Guide

### Step 1: Environment Setup
Ensure Python 3.10+ (tested on Python 3.14) is installed. Install required libraries:
```powershell
pip install -r requirements.txt
```

### Step 2: Run Exploratory Data Analysis
Generates all exploratory charts and statistical reports:
```powershell
python src/eda.py
```

### Step 3: Run Full Training Pipeline
Executes cross-validation, model benchmarking, diagnostic plot generation, and artifact serialization:
```powershell
python src/train_models.py
```

### Step 4: Run Automated Test Suite
Executes 18 automated unit and integration tests across data loading, leakage prevention, transformations, and inference:
```powershell
python -m pytest tests -v
```

---

## 8. Reusable Prediction Interface (API Reference)

The `EduPredictor` class provides a reusable interface designed for integration into web dashboards (Streamlit, Flask, FastAPI) or batch pipelines.

### Python Code Example:

```python
from src.predict import EduPredictor, get_sample_student

# 1. Initialize predictor (automatically loads saved artifacts)
predictor = EduPredictor(models_dir="models")

# 2. Prepare student data (dictionary with 30 early features)
student_data = {
    "school": "GP", "sex": "F", "age": 17, "address": "U", "famsize": "GT3",
    "Pstatus": "T", "Medu": 2, "Fedu": 2, "Mjob": "services", "Fjob": "other",
    "reason": "course", "guardian": "mother", "traveltime": 1, "studytime": 2,
    "failures": 1, "schoolsup": "no", "famsup": "yes", "paid": "no",
    "activities": "yes", "nursery": "yes", "higher": "yes", "internet": "yes",
    "romantic": "no", "famrel": 4, "freetime": 3, "goout": 3, "Dalc": 1,
    "Walc": 2, "health": 4, "absences": 6
}

# 3. Single student prediction with personalized counseling insights
result = predictor.predict_single(student_data, include_explanation=True)

print(f"Predicted Final Grade: {result['predicted_grade']} / 20.0")
print(f"Academic Risk Status:  {result['risk_status']} ({result['risk_probability']:.1%})")
print(f"Risk Tier:             {result['risk_tier']}")

print("\nActionable Counseling Recommendations:")
for rec in result["explanation"]["actionable_recommendations"]:
    print(f" - {rec}")
```

### Batch Prediction:
```python
import pandas as pd
from src.predict import EduPredictor

predictor = EduPredictor()
students_df = pd.read_csv("data/student-mat.csv", sep=";")

# Returns DataFrame with appended predicted_grade, academic_risk_class, risk_probability, risk_tier
enriched_df = predictor.predict_batch(students_df)
print(enriched_df[["school", "age", "predicted_grade", "risk_probability", "risk_tier"]].head())
```

---

## 9. Academic Project Certification
* **Course:** Final-Year B.Tech Project (CSE)
* **Status:** Complete, Tested, and Verified
* **Unit Tests Passing:** 18 / 18
* **Data Leakage Risk:** 0.0% (Verified by test suite)
