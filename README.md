# Customer Churn Prediction

This project uses a dataset to predict customer churn using different classficaiton models. The workflow combines exploratory analysis, feature engineering, model training, and performance evaluation inside Jupyter notebooks, with reusable utilities for visualizing confusion matrices and ROC curves.

## Dataset

- Source file: [`dataset.csv`](dataset.csv)
- Target column: `Target_Churn`
- Feature includes demographics (age, gender), financial behavior (income, spend, average transaction size), engagement (purchases, returns, support contacts), sentiment (Satisfaction_Score), and marketing responses.

## Project Structure

- `EDA.ipynb` — exploratory data analysis with visual insights.
- `features.ipynb` — feature engineering and feature selection
- `models.ipynb` — training pipelines for Random Forest, LightGBM, XGBoost classifiers.
- `performance.ipynb` — evaluation plots, metric summaries, and model comparisons.
- `utils.py` — helper functions to plot confusion matrices and metrics calculation.
- `models/` — List of trained models in (`*.pkl`) format.
- `min_max.pkl` — pre-fitted scaler applied during feature preprocessing.
- `model_performance_metrics.csv` — tabular comparison of key metrics for tree-based models.

### 2. Install Dependencies

- Install Using **pip:** 
  ```bash
  pip install --upgrade pip
  pip install -r requirements.txt
  ```

- Install Using **uv:**
  ```bash
  pip install --upgrade pip
  pip install uv 
  uv sync #automatically installs library
  ```

- Install Using **anaconda:**
  
  ```python
  conda create -n churn python=3.10
  conda activate churn
  pip install -r requirements.txt
  ```


## Verdicts on Model Performance

From performance analysis we can see that XGBoost performs well comapared to other models.

- Precision: XGBoost has the highest precision, meaning it makes fewer false positive errors when predicting the positive class.

- Recall: XGBoost's recall is much higher than both RandomForest and LightGBM, meaning it captures a greater proportion of true positive instances.

- F1-Score: XGBoost maintains a good balance between precision and recall, reflected in its higher F1-score.

- ROC-AUC: XGBoost demonstrates a stronger ability to separate positive and negative classes, as indicated by its higher ROC-AUC score comapred to others.


## Reasons Why XGBoost Performs Well

- Uses decesion trees as weak learners and corrects utilizing previous models erros
- Uses L1 and L2 regularization to prevent overfitting
- User gradient boosting algorithm, sequentially improving models performance


