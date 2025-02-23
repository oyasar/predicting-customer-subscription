---
title: "Model Training"
author: "Ozge Yasar"
date: "2025-02-22"
format:
  html:
    self-contained: true
    toc: true            # Enable Table of Contents
    toc-depth: 3         # Set TOC depth (adjust as needed)
    number-sections: true # Automatically number sections
    code-fold: true      # Allow folding (collapsing) of code blocks
    code-tools: true     # Enable code tools (like copy buttons)
execute:
  echo: true              # Show code by default
  warning: false          # Hide warnings (optional)
  error: false            # Hide errors (optional)
jupyter: python3          # Specify the Jupyter kernel
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.16.7
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

```python
import pandas as pd
from predicting_customer_subscription.train import Model
from predicting_customer_subscription.preprocessing import Preprocess
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "png"
import shap
```

```python
# Load the data
df= pd.read_excel('../data/train_file.xlsx')
```

* For the modelling task, I'll be experimenting with 2 algorithms. XGBoost and CatBoost. I'll train both models and compare their performance.
Because Catboost has a built-in categorical feature handling, I'll use one hot encoder for XGBoost model training only. Due to this reason, I'll split the data into train and test sets separately for each experiment.
* I first preprocess the data by encoding the categorical features and splitting the data into train and test sets as can be seen in preprocessing.py module.
* As can be seen in train.py module, Model method is used to train the models. It takes the model type as an argument and trains the model with 5-fold cross validation and hyperparameter tuning. It also evaluates the model on the test set and generates the evaluation metrics such as PRCurve, Feature importance, etc. I've also looked at the shap feature importance scores for both models.



# 1. XGBoost Model Training

```python
df_train = df.copy()
pp = Preprocess(data=df_train, target='y')
```

```python
# encoding is applied for xgboost model training
X_train_x, X_test_x, y_train_x, y_test_x = pp.split_data(encode=True)
```

```python
# Train and evaluate XGBoost model - 5 fold cross validation
xgb_model = Model(model_type='xgboost', n_splits=5)
xgb_model.train(X_train_x, y_train_x)
```

```python
xgb_model.evaluate(X_test_x, y_test_x)
```

```python
# feature importance
xgb_importance = xgb_model.feature_importance().sort_values(by='Importance', ascending=False)
fig_x = px.bar(xgb_importance.head(15).sort_values(by='Importance', ascending=True),
               y='Feature', x='Importance', title='Feature Importance', orientation='h')
fig_x.show()
```

```python
# Shap Scores
shap.initjs()
explainer_x = shap.Explainer(xgb_model.model, X_test_x)
shap_values_x = explainer_x(X_test_x)

```

```python
shap.plots.beeswarm(shap_values_x, max_display=15)
```

# 2. CatBoost Model Training

```python
X_train_c, X_test_c, y_train_c, y_test_c = pp.split_data(encode=False)
```

```python
# Train and evaluate CatBoost model
cb_model = Model(model_type='catboost', n_splits=5, cat_features=cat_cols)
cat_var = X_train_c.select_dtypes(include=['object']).columns.tolist()
cb_model.train(X_train_c, y_train_c, cat_features=cat_var)
```

```python
cb_model.evaluate(X_test_c, y_test_c)
```

```python
cb_importance = cb_model.feature_importance().sort_values(by='Importance', ascending=False)
fig_c = px.bar(cb_importance.head(15).sort_values(by='Importance', ascending=True), y='Feature', x='Importance', title='Feature Importance', orientation='h')
```

```python
fig_c.show()
```

```python
# shap scores
explainer_c = shap.Explainer(cb_model.model, X_test_x)
shap_values_c = explainer_c(X_test_x)
```

```python
shap.plots.beeswarm(shap_values_c, max_display=15)
```

# 3. Evaluation of the models


* In order to evaluate the models I've generated both ROC curves and Precision Recall curves (with auc scores). However, Because the dataset is imbalanced, I've used the Precision Recall curve as the main evaluation metric.
* As can be seen in the plots above, they perform similarly in terms of Precision Recall curves. CatBoost model has a slightly better performance in terms of precision and recall scores (0.1 increase in PR auc).
* There's a spike in the precision recall curve of XGB model indicating lower number of cases predicted as positive at high thresholds. This might be due to the fact that the dataset is imbalanced. Other than that the shape of the PR curves are very similar.
* In terms of the feature importance, both models have similar top features. However, the top important features are slightly different. For example, the top feature for XGBoost model is 'potucome_success' whereas it is 'duration' for CatBoost model. Given the EDA, I think both make sense and not unexpected. However, in the current method, catboost doesn't reveal which value of the categorical features are important for the prediction. This is a limitation of my implementation currently.
* For the shap scores, the top features contributing to the model is the same (duration). However, the remaining seems to be different. Catboost model seems to be relying most on the continuous columns rather than the categorical ones.
* On the other hand, XGB shap scores are more balanced between the categorical and continuous features. This might be due to the one hot encoding of the categorical features.
* Training complexity and time is higher for CatBoost model compared to XGBoost model.
* In conclusion, I've chosen to use XGBoost model for the prediction task due to the performance being as high as Catboost model on the holdout set, better explainability and reduced complexity even though it still has some pre-processing complexity with data encoding.
