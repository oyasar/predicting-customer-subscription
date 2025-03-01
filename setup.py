from setuptools import setup, find_packages

setup(
    name="predicting_customer_subscription",
    version="0.1.0",
    description="A package for predicting customer subscription",
    package_dir={'': 'src'},
    packages=find_packages(where='src'),
    install_requires=[
        "pandas==2.2.3",
        "numpy==1.26.4",
        "scikit-learn==1.6.1",
        "matplotlib==3.10.0",
        "seaborn==0.13.2",
        "mlflow==2.20.2",
        "fastapi==0.115.8",
        "uvicorn==0.34.0",
        "pytest==8.3.4",
        "black==25.1.0",
        "flake8==7.1.2",
        "mypy==1.15.0",
        "sphinx==8.1.3",
        "plotly==6.0.0",
        "xgboost==2.1.4",
        "lightgbm==4.6.0",
        "catboost==1.2.7",
        "jupyter==1.1.1",
        "notebook==7.3.2",
        "openpyxl==3.1.5",
        "kaleido==0.2.1",
        "shap==0.46.0",
        "jupytext==1.16.7"
    ],
)
