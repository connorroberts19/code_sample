# Titanic Survival Prediction Project

This project applies machine learning techniques to predict the survival of passengers aboard the Titanic. It leverages Python and several libraries, including pandas, scikit-learn, and matplotlib, to preprocess data, compare models, and evaluate their performance. The goal is to split the project into succinct pieces that can be easily run and evaluated.

## Project Structure

The project is organized into several key components:

- `feature_engineering.py`: Implements functions to engineer new features from the Titanic dataset to improve model predictions.
- `data_preprocessor.py`: Contains the `DataPreprocessor` class for cleaning and preprocessing the dataset before model training.
- `model_comparer.py`: Features the `ModelComparer` class to compare different machine learning models and select the best performer based on accuracy.
- `model_metrics.py`: Includes the `ModelMetrics` class for evaluating the performance of the best model using various metrics and visualizations.

## Installation

   ```bash
   git clone https://github.com/connorroberts19/code_sample.git
   ```

## Python Packages
Packages are located in the `requirements.txt` file

- `matplotlib==3.8.3`,
- `numpy==1.26.4`,
- `pandas==2.2.1`,
- `scikit-learn==1.4.1.post1`,
- `seaborn==0.13.2`,
- `xgboost==2.0.3`,
- `ipykernel==6.29.3`

## Unit Tests
Use the below code in the code_sample folder to run all the relevant unit tests
```bash 
python -m unittest discover
``` 

## Usage
The classes are utilized in the `TitanicNotebook.ipynb` jupyter notebook. After the relevant packages are installed, run all the cells in the notebook. Each cell uses a different class to perform a variety of operations and it goes through the workflow of a standard data science project.
