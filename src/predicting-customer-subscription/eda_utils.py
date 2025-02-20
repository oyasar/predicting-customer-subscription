import pandas as pd
import plotly.express as px
from utils import logger

log = logger()


def data_summary(df: pd.DataFrame, head_rows: 5):
    """
    Print basic summary information of the DataFrame.

    Args:
        df: DataFrame to summarize.
        head_rows: Number of rows to display from head.
    """
    log.info("Data Shape: %s", df.shape)
    print("Head:")
    print(df.head(head_rows))
    print("\nData Types:")
    print(df.dtypes)
    print("\nMissing Values:")
    print(df.isnull().sum())


def plot_correlation_matrix(df: pd.DataFrame, num_columns, method: 'pearson'):
    """
    Plot the correlation matrix of numerical features using Plotly.

    Args:
        df: DataFrame containing the features.
        num_columns: List of numerical columns to include
        in the correlation matrix.
        method: Correlation method - 'pearson', 'spearman', etc.
    """
    corr = df[num_columns].corr(method=method)
    fig = px.imshow(corr,
                    text_auto=True,
                    title=f"Correlation Matrix ({method.title()})")
    fig.show()


def plot_missing_values(df: pd.DataFrame):
    """
    Plot a bar chart of missing values count for each column.
    check if there are missing values in columns first

    Args:
        df: DataFrame to be analyzed.
    """
    missing_values = df.isnull().sum()
    missing_values = missing_values[missing_values > 0]
    if missing_values.empty:
        log.info("No missing values found.")
    else:
        fig = px.bar(missing_values,
                     title="Missing Values Count",
                     labels={'index': 'Column', 'value': 'Count'})
        fig.show()

def plot_distributions(df: pd.DataFrame, cols: list, col_type: str):
    """
    Plot the distribution of numerical features using Plotly.
    Depending on the column type plot histograms or bar charts.

    Args:
        df: DataFrame containing the features.
    """
    if col_type == 'numerical':
        for col in cols:
            fig = px.histogram(df, x=col, title=f"{col} Distribution")
            fig.show()
    elif col_type == 'categorical':
        for col in cols:
            fig = px.bar(df[col].value_counts().reset_index(),
                         x='index', y=col, title=f"{col} Distribution")
            fig.show()
    else:
        log.error("Invalid column type."
                  " Choose 'numerical' or 'categorical'.")