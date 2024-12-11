# Housing Price Prediction

This project aims to predict housing prices using machine learning techniques. The dataset consists of various features such as the number of bedrooms, bathrooms, living area size, and location, which are used to predict the price of a house. The project utilizes a **Linear Regression** model to perform the predictions.

## Dataset

The dataset used in this project is the **Housing Data** available on [Kaggle](https://www.kaggle.com/datasets/shree1992/housedata). It contains information about house sales in Seattle and includes the following columns:

- **date**: The date the house was sold
- **price**: The price of the house
- **bedrooms**: The number of bedrooms in the house
- **bathrooms**: The number of bathrooms in the house
- **sqft_living**: The size of the living area in square feet
- **sqft_lot**: The size of the lot in square feet
- **floors**: The number of floors in the house
- **waterfront**: Whether the house has a waterfront view (1 if yes, 0 if no)
- **view**: The view quality (from 0 to 4)
- **condition**: The condition of the house (from 1 to 5)
- **sqft_above**: The size of the above-ground living area in square feet
- **sqft_basement**: The size of the basement in square feet
- **yr_built**: The year the house was built
- **yr_renovated**: The year the house was renovated
- **street**: The street address
- **city**: The city where the house is located
- **statezip**: The state and zip code of the house location
- **country**: The country where the house is located

## Approach

The approach taken for this project follows the typical machine learning pipeline, consisting of data loading, preprocessing, model building, and evaluation. Below is the breakdown of the approach:

### 1. **Data Loading and Exploration**
   - The dataset is loaded using **pandas** and the first 10,000 rows are taken for analysis.
   - The columns and null values are checked to ensure that all necessary data is available for processing.
   
### 2. **Data Preprocessing**
   - **Drop Unnecessary Columns**: The 'date' column is removed from the dataset since it's not relevant for predicting the housing price.
   - **Feature Engineering**: The categorical variables (such as the 'street', 'city', 'statezip', and 'country' columns) are converted into dummy variables using `pd.get_dummies()`. This step helps to convert the categorical variables into a format that the machine learning model can use.
   - **Handling Missing Values**: Any missing values are handled by dropping rows with NaN values.
   - **Outlier Removal**: Outliers are detected using the Interquartile Range (IQR) method. Any rows that contain outliers in numerical columns are removed. This ensures that the model doesn't get skewed by extreme values.

### 3. **Model Training**
   - The data is split into training and testing sets using an 80-20 ratio.
   - A **Linear Regression** model is trained on the training data to learn the relationship between the features and the target variable (house price).

### 4. **Model Evaluation**
   - The performance of the model is evaluated using three main metrics:
     - **Mean Absolute Error (MAE)**: Measures the average magnitude of the errors in the predictions, without considering their direction.
     - **Mean Squared Error (MSE)**: Measures the average squared difference between predicted and actual values. This metric gives more weight to larger errors.
     - **R² Score**: Indicates how well the model fits the data. The closer the R² score is to 1, the better the model.
   - A plot is generated to visually compare the actual vs. predicted prices.

### 5. **Results**
   - The model's evaluation metrics are displayed, and the results help in understanding the model's performance. The scatter plot gives an intuitive understanding of how well the model predictions align with the actual house prices.

### 6. **Conclusion**
   - The project demonstrates how to use machine learning techniques, particularly **Linear Regression**, to predict housing prices based on multiple features. The model achieves reasonable performance, with an R² score of **[your R² score here]**.

## Project Structure

The project contains the following files:

- **housing_price_prediction.py**: The Python script that loads the dataset, processes the data, handles missing values and outliers, applies feature engineering, and builds the linear regression model to predict house prices.
- **data.csv**: The dataset file used for model training and testing.
- **README.md**: This file.

## Requirements

To run this project, you need to install the following Python packages:

- pandas
- numpy
- matplotlib
- scikit-learn

You can install them using pip:

```bash
pip install pandas numpy matplotlib scikit-learn
