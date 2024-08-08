**House-Prices---Advanced-Regression-Techniques**
**House Prices Prediction in Iowa**

This project is a submission for the Kaggle competition "House Prices: Advanced Regression Techniques." The goal is to predict the final price of each home in Ames, Iowa.

**Project Overview**

In this project, we explore and model the dataset to predict house prices. The dataset contains various features related to housing attributes such as the number of rooms, the year built, and the overall quality of the house.

**Dataset**

The dataset used in this project is provided by the competition and can be found [here](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data). It includes:

- `train.csv`: The training set
- `test.csv`: The test set
- `data_description.txt`: Full description of each attribute
- `sample_submission.csv`: A sample submission file in the correct format

**Project Structure**

The project is organized as follows:

- `House_Prices_Prediction_in_Iowa.ipynb`: Jupyter notebook containing the code for data preprocessing, feature engineering, model training, and evaluation.
- `data/`: Folder containing the dataset files (train, test, and data description).
- `models/`: Folder to save trained models.
- `output/`: Folder to save the submission files.

**Installation**

To run this project, you need to have Python installed along with the following libraries:

- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- xgboost
- lightgbm
- catboost

You can install these libraries using pip:

```bash
pip install pandas numpy scikit-learn matplotlib seaborn xgboost lightgbm catboost
```

**Usage**

1. Clone the repository:
    ```bash
    git clone https://github.com/Yeshh45/house-prices-prediction.git
    cd house-prices-prediction
    ```

2. Download the dataset from Kaggle and place it in the `data/` folder.

3. Run the Jupyter notebook:
    ```bash
    jupyter notebook House_Prices_Prediction_in_Iowa.ipynb
    ```

**Data Preprocessing and Feature Engineering**

The notebook includes the following steps:

1. **Data Cleaning**: Handling missing values, outliers, and data types.
2. **Exploratory Data Analysis**: Visualizing and understanding the distribution and relationships of features.
3. **Feature Engineering**: Creating new features, encoding categorical variables, and scaling numerical features.

**Model Training and Evaluation**

Several regression models were trained and evaluated, including:

- Linear Regression
- Ridge Regression
- Lasso Regression
- Elastic Net
- Decision Tree Regressor
- Random Forest Regressor
- Gradient Boosting Regressor
- XGBoost Regressor
- LightGBM Regressor
- CatBoost Regressor

The models were evaluated using cross-validation and the best model was selected based on the Root Mean Squared Error (RMSE) metric.

**Results**

The final model achieved a siginificant RMSE on the test set, ranking in the top 15% of the competition.

