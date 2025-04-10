# Muscle Injury Prediction

This project predicts muscle injuries in athletes using machine learning models. The workflow includes data preprocessing, feature engineering, and training multiple models for different time horizons:
1. **Same-Day Prediction**: Predicts the likelihood of injury on the same day.
2. **One-Day Prediction**: Predicts the likelihood of injury one day ahead.
3. **Weekly Risk Prediction**: Predicts injury risk for the following week.

The models leverage advanced techniques such as **XGBoost**, **Random Forest**, **Logistic Regression**, and **RuleFit**, with hyperparameter optimization using grid search and cross-validation.

---

## Table of Contents
- [Features of the Project](#features-of-the-project)
- [Installation](#installation)
- [Workflow](#workflow)
  - [Data Preprocessing](#data-preprocessing)
  - [Model Training](#model-training)
- [Results](#results)
- [Future Work](#future-work)

---

## Features of the Project

### **Data Preprocessing**
The project involves extensive data preprocessing to ensure high-quality input for model training:
- **Data Cleaning**: Removing players with insufficient information to avoid noise.
- **Imputation**: Handling missing values using simple mean, forward fill, backward fill, or time-based interpolation.
- **Feature Engineering**:
  - Creating new features like absence flags, cumulative training days, and rolling averages (e.g., weekly and monthly averages for training load).
  - Generating binary indicators for recent sleep quality and duration.
  - Tracking injuries and illnesses to create target variables.
- **Master Data Creation**: Integrating multiple data sources (training load, wellness metrics, injuries, illnesses, game performance) into a unified dataset for each player.
- **Visualization**: Exploratory Data Analysis (EDA) to understand feature importance and trends.

### **Model Training**
Three types of models were trained to predict muscle injuries:
1. **Weekly Risk Prediction Model**:
   - Model: LSTM (Long Short-Term Memory).
   - Input: Time-series data of training load, wellness metrics, and game performance.
   - Output: Weekly risk prediction for muscle injuries.

2. **Same-Day Prediction Model**:
   - Models: XGBoost, Random Forest, Logistic Regression, RuleFit.
   - Input: Current day's metrics (training load, wellness, game performance).
   - Output: Binary classification (injury or no injury).

3. **One-Day Prediction Model**:
   - Models: XGBoost, Random Forest, Logistic Regression, RuleFit.
   - Input: Previous day's metrics.
   - Output: Binary classification (injury or no injury).

---

## Installation

To run this project locally:

1. Clone the repository:
   git clone https://github.com/talnazzal/Greening.git
2. Navigate to the project directory:
   cd Greening
3. Install the required Python packages:
   pip install -r requirements.txt

The imlementation code in the directory under name 'Project Code.ipynb'
   ---

## Workflow

### **Data Preprocessing**
The `create_master_dataframe` function processes raw data into a unified master dataset for each player. Key steps include:
1. Integrating multiple data sources (training load, wellness metrics, injuries, illnesses, game performance).
2. Handling missing values using forward fill or time-based interpolation.
3. Creating new features like rolling averages (e.g., weekly training load), absence flags, and cumulative training days.

### **Model Training**
#### Weekly Risk Prediction Model
1. Preprocessed time-series data is fed into an LSTM model.
2. The model predicts injury risk over the next 7 days.

#### Same-Day and One-Day Models
1. Static models like XGBoost, Random Forest, Logistic Regression, and RuleFit are trained using grid search and cross-validation.
2. Optimal thresholds are determined based on evaluation metrics.

---

## Results

### Weekly Risk Prediction Model
Aggregate Performance Across Folds

| Metric        | Mean ± Std Dev |
|---------------|----------------|
| PR-AUC        | 0.038 ± 0.032  |
| ROC-AUC       | 0.744 ± 0.092  |
| Precision     | 0.012 ± 0.014  |
| Recall        | 0.516 ± 0.267  |
| F1-Score      | 0.024 ± 0.026  |
| Optimal F1-Score  | 0.132  |
| Optimal Threshold | 0.651  |



|                | Predicted Class 0 | Predicted Class 1 |
|----------------|-------------------|-------------------|
| **Actual Class 0** | 17466              | 178                |
| **Actual Class 1** | 441                 | 47                |

---

### Same-Day Prediction Model
Confusion Matrix (Optimal Threshold = 0.240):

|                | Predicted Class 0 | Predicted Class 1 |
|----------------|-------------------|-------------------|
| **Actual Class 0** | 4618              | 30                |
| **Actual Class 1** | 5                 | 26                |




---

### One-Day Prediction Model
Confusion Matrix (Optimal Threshold = 0.367):

|                | Predicted Class 0 | Predicted Class 1 |
|----------------|-------------------|-------------------|
| **Actual Class 0** | 4643              | 5                 |
| **Actual Class 1** | 20                | 11                |




---

## Future Work

Potential improvements include:
1. Incorporating additional features like heart rate variability or hydration levels.
2. Experimenting with advanced deep learning architectures such as GRU or Transformers for weekly predictions.
3. Fine-tuning hyperparameters using automated tools like Optuna or Bayesian optimization.









