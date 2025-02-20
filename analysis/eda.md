---
title: "Exploratory Data Analysis"
author: "Ozge Yasar"
date: "2025-02-20"
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
---

```python
from predicting_customer_subscription.eda_utils import *
import plotly.io as pio
pio.renderers.default = "png"
```

## 1. Dataset

Based on the results below, here's the summary of the dataset:
* The dataset is consist of 15 columns. 4 of these include numerical values, 10 of them are categorical variables, and the colum 'y' is the binary outcome variable.
* The columns don't have any missing values
* The positive output ratio is around 1 to 9 (11% to 89%) which indicates an imbalance (but not too bad for now).



```python
df = pd.read_excel('../data/train_file.xlsx')
```


```python
data_summary(df, outcome_col= 'y', head_rows=5)
```

## 2. Correlations

Correlations among the continous variables are quite weak as can be seen from the matrix below.


```python
plot_correlation_matrix(df, num_columns=['age', 'duration', 'campaign', 'previous'], method= 'pearson')
```


```python
l_num_cols=['age', 'duration', 'campaign', 'previous']
l_cat_cols=['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
```

## 3. Distributions - Continous Variables

* Skewed distributions for all variables with outliers especially for duration variable (average is 258)
* Relationships of the continous variables with the outcome shows some interesting insights.
* Based on the histograms below, the marketing of the product seems to be more successful among the middle age group (30-40), however, it doesn't indicate a direct relationship given the number of customers contacted within this age group
* Duration (call duration) seems to have a slightly positive relationship with the outcome.
* Campaing (number of contact) distrtibution indicates that the higher number of contacts are not likely to be successful for the marketing of the product
*  similar to this finding - there were more successful contacts in the not-previously-contacted bucket.


```python
plot_distributions(df, cols=l_num_cols, outcome='y')
```

## 4. Distributions - Categorical Variables

Majority of the target customers
* are blue-collar worker,s admins, and technicians (this group also include the highest number of success).
* are married
* have a university degree (majority success)
* doesn't have a credit in default (majority success which seems sensible)
* has a housing loan (although not a significant difference with the no-housing loan group). Given the success of the product marketing, having a house loan doesn't seem to have a big impact.
* doesn't have a personal loan. Personal loans seem to have sone impact.
And most (last) contacts were made in May.
Previous outcome column indicates that most of the customers doesn't have a previous outcome history which may indicate that they are the first time contact - or this information cannot be gathered. Again, there's higher success in this group.


```python
plot_distributions(df, cols=l_cat_cols, outcome='y')
```


```python

```


```python

```


```python

```
