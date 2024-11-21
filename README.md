# Machine Learning Project for User Classification

This project aims to classify users based on their interaction data using various machine learning models. The project involves data preprocessing, feature engineering, model training, and evaluation.

## Table of Contents

- [Project Overview](#project-overview)
- [Data Preprocessing](#data-preprocessing)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Results](#results)
- [Conclusion](#conclusion)
- [Dependencies](#dependencies)

## Project Overview

The goal of this project is to classify users into different categories based on their interaction data. The dataset used for this project contains information about users' activities on a platform, such as the number of days on the platform, minutes watched, courses started, practice exams started and passed, and minutes spent on exams.

## Data Preprocessing

Data preprocessing is a crucial step in any machine learning project. In this project, the following preprocessing steps were performed:

1. **Importing the Data**: The data was imported from a CSV file named `ml_datasource.csv`.
2. **Removing Outliers**: Outliers were removed based on specific thresholds for `minutes_watched`, `courses_started`, `practice_exams_started`, and `minutes_spent_on_exams`.
3. **Checking for Multicollinearity**: Variance Inflation Factor (VIF) was calculated to check for multicollinearity among the features. Features with high VIF values were dropped to prevent multicollinearity.
4. **Dealing with NaN Values**: Missing values in the `student_country` column were replaced with the string 'NAM'.
5. **Encoding Categorical Variables**: The `student_country` column was encoded using an ordinal encoder.

## Model Training and Evaluation

Several machine learning models were trained and evaluated on the preprocessed data:

1. **Logistic Regression**: A logistic regression model was trained using the `sm.Logit` function from the `statsmodels` library.
2. **K-Nearest Neighbors (KNN)**: A KNN model was trained using the `KNeighborsClassifier` from the `sklearn` library. Grid search was used to find the optimal hyperparameters.
3. **Support Vector Machines (SVM)**: An SVM model was trained using the `SVC` from the `sklearn` library.
4. **Decision Trees**: A decision tree model


### K-Nearest Neighbors (KNN)
```
# Define the parameter grid for the grid search
parameters_knn = {'n_neighbors': range(1, 51), 'weights': ['uniform', 'distance']}

# Initialize a GridSearchCV object with KNeighborsClassifier estimator
grid_search_knn = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=parameters_knn, scoring='accuracy')

# Fit the grid search object on the training data
grid_search_knn.fit(x_train_array, y_train_array)

# Store the best estimator (model with optimal parameters) in knn_clf
knn_clf = grid_search_knn.best_estimator_

# Predict the target variable for the test dataset using the optimal model
y_test_pred_knn = knn_clf.predict(x_test_array)
```

### Support Vector Machines (SVM)
```
# Create an instance of the MinMaxScaler
scaler = MinMaxScaler()

# Scale the training and testing data
x_train_scaled = scaler.fit_transform(x_train_array)
x_test_scaled = scaler.transform(x_test_array)

# Initialize and train the SVM model
svm_clf = SVC()
svm_clf.fit(x_train_scaled, y_train_array)

# Predict the target variable for the test dataset
y_test_pred_svm = svm_clf.predict(x_test_scaled)
```

### Decision Trees
```
# Initialize and train the decision tree model
dt_clf = DecisionTreeClassifier()
dt_clf.fit(x_train_array, y_train_array)

# Predict the target variable for the test dataset
y_test_pred_dt = dt_clf.predict(x_test_array)
```

### Random Forest
```
# Initialize and train the random forest model
rf_clf = RandomForestClassifier()
rf_clf.fit(x_train_array, y_train_array)

# Predict the target variable for the test dataset
y_test_pred_rf = rf_clf.predict(x_test_array)
```

### Results
The performance of each model was evaluated using accuracy, precision, recall, and F1-score. Confusion matrices were also generated to visualize the performance of each model.

### Dependencies
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- statsmodels
