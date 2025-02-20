```python
import pandas as pd
import plotly.express as px
```


```python
df = pd.read_excel('../data/train_file.xlsx')
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>job</th>
      <th>marital</th>
      <th>education</th>
      <th>default</th>
      <th>housing</th>
      <th>loan</th>
      <th>contact</th>
      <th>month</th>
      <th>day_of_week</th>
      <th>duration</th>
      <th>campaign</th>
      <th>previous</th>
      <th>poutcome</th>
      <th>y</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>49</td>
      <td>blue-collar</td>
      <td>married</td>
      <td>basic.9y</td>
      <td>unknown</td>
      <td>no</td>
      <td>no</td>
      <td>cellular</td>
      <td>nov</td>
      <td>wed</td>
      <td>227</td>
      <td>4</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>no</td>
    </tr>
    <tr>
      <th>1</th>
      <td>37</td>
      <td>entrepreneur</td>
      <td>married</td>
      <td>university.degree</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>telephone</td>
      <td>nov</td>
      <td>wed</td>
      <td>202</td>
      <td>2</td>
      <td>1</td>
      <td>failure</td>
      <td>no</td>
    </tr>
    <tr>
      <th>2</th>
      <td>78</td>
      <td>retired</td>
      <td>married</td>
      <td>basic.4y</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>cellular</td>
      <td>jul</td>
      <td>mon</td>
      <td>1148</td>
      <td>1</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>36</td>
      <td>admin.</td>
      <td>married</td>
      <td>university.degree</td>
      <td>no</td>
      <td>yes</td>
      <td>no</td>
      <td>telephone</td>
      <td>may</td>
      <td>mon</td>
      <td>120</td>
      <td>2</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>no</td>
    </tr>
    <tr>
      <th>4</th>
      <td>59</td>
      <td>retired</td>
      <td>divorced</td>
      <td>university.degree</td>
      <td>no</td>
      <td>no</td>
      <td>no</td>
      <td>cellular</td>
      <td>jun</td>
      <td>tue</td>
      <td>368</td>
      <td>2</td>
      <td>0</td>
      <td>nonexistent</td>
      <td>no</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (32910, 15)




```python
df.columns
```




    Index(['age', 'job', 'marital', 'education', 'default', 'housing', 'loan',
           'contact', 'month', 'day_of_week', 'duration', 'campaign', 'previous',
           'poutcome', 'y'],
          dtype='object')




```python
df.y.value_counts()
```




    y
    no     29203
    yes     3707
    Name: count, dtype: int64




```python
3707/29203
```




    0.12693901311509093




```python
df.isnull().sum()
```




    age            0
    job            0
    marital        0
    education      0
    default        0
    housing        0
    loan           0
    contact        0
    month          0
    day_of_week    0
    duration       0
    campaign       0
    previous       0
    poutcome       0
    y              0
    dtype: int64




```python
df.dtypes
```




    age             int64
    job            object
    marital        object
    education      object
    default        object
    housing        object
    loan           object
    contact        object
    month          object
    day_of_week    object
    duration        int64
    campaign        int64
    previous        int64
    poutcome       object
    y              object
    dtype: object




```python
df.job.unique()
```




    array(['blue-collar', 'entrepreneur', 'retired', 'admin.', 'student',
           'services', 'technician', 'self-employed', 'management',
           'unemployed', 'unknown', 'housemaid'], dtype=object)




```python
df.default.unique()
```




    array(['unknown', 'no', 'yes'], dtype=object)




```python
df.duration.max(), df.duration.min()
```




    (4918, 0)




```python
# TODO:
# requirements versions
# convert outcomes to 0,1, NA
# relationships between categorical variables and the outcome
# relationships between continous variables and the outcome
# bin age and others
# push to github
```


```python

```


```python

```
