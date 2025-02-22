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

For the modelling task, I'll be experimenting with 2 algorithms. XGBoost and CatBoost. I'll train both models and compare their performance.
Because catboost has a built-in categorical feature handling, I'll use onehot encoder for XGBoost model training only. Due to this reason, I'll split the data into train and test sets seperatelty for each experiment.


# 1. XGBoost Model Training

```python
df_train = df.copy()
pp = Preprocess(data=df_train, target='y')
```

```python
X_train_x, X_test_x, y_train_x, y_test_x = pp.split_data(encode=True)
```

```python
# Train and evaluate XGBoost model
xgb_model = Model(model_type='xgboost', n_splits=5)
xgb_model.train(X_train_x, y_train_x)
```

```python
xgb_model.evaluate(X_test_x, y_test_x)
```

```python
# feature importance

xgb_importance = xgb_model.feature_importance().sort_values(by='Importance', ascending=False)
fig_x = px.bar(xgb_importance.head(15).sort_values(by='Importance', ascending=True), y='Feature', x='Importance', title='Feature Importance', orientation='h')
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
cb_model = Model(model_type='catboost', n_splits=5)
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

```python
# from sklearn.metrics import precision_recall_curve
# y_pred = xgb_model.model.predict(X_test_x)
# y_proba = xgb_model.model.predict_proba(X_test_x)[:, 1]
# precision, recall, thresholds = precision_recall_curve(y_pred, y_proba)
```

```python
# import matplotlib.pyplot as plt
# thresholds = list(thresholds)
# # thresholds.append(1.0)
#
# # Plot the PR curve
# plt.figure(figsize=(10, 6))
# plt.plot(thresholds, precision, label='Precision', marker='o')
# plt.plot(thresholds, recall, label='Recall', marker='o')
# plt.xlabel('Threshold')
# plt.ylabel('Score')
# plt.title('Precision-Recall Curve with Thresholds')
# plt.legend()
# plt.grid()
# plt.show()
```

```python

```
