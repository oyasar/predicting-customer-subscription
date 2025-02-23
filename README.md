# Predicting Customer Subscription

This is the python package of the project. It’s developed using **Python version 3.10**. The project structure is mainly created with a cookie cutter template.

## Installation:

It’s recommended to create a venv after cloning the repo. Assuming you are in the project directory, run below:

```sh
python -m venv venv
source venv/bin/activate
```

Then install the requirements:

```sh
pip install -r requirements.txt
```

## Project:

### Analysis: 

* This folder include the notebooks used for the EDA and model training in html and .md formats. 
* html versions include all outputs (plots, printed outputs, etc. - these are generated with quarto that’s installed in the local environment) and the codes whereas markdown versions only include the code. 
* In order to view the HTMLs please download it from Github GUI (if not cloning the repo). 
* If you would like to run the notebooks and reproduce the results, you will need to convert the md files to python notebooks with the below command after project setup.

```sh
jupytext --to notebook analysis/eda.md
```

### Data:

Git ignores the files in this folder. Please add the train and test excel files provided to me for this task in order to run the project.

### Models:

Were the serialised models stored in pickle format. These are not pushed to the remote repository due to the size of the files. 

### Output:

The results of the scoring is saved in csv format

### Project root - src/predicting_customer_subscription

Source code used in notebooks and modules are stored in this folder. File names are self-explanatory.

## Scoring the model:
After the model training is done, selected model is saved in models folder. 

I've created a simple Flask App with a simple UI to score the model (see predict.py). 
Then I created a docker image to run the app. See docker file for the configuration.

In order to build the image requirements.txt should be updated to remove below line which installs the package in development mode:

```sh
-e .
```
Building the image:

```sh
docker build -t predicting_customer_subscription . 
```

To run the image and save the output to the project folder:

```sh
docker run --name pcs -p 8000:8000 -v local_project_path/predicting-customer-subscription/output:/app/output -it predicting_customer_subscription
```