import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

df = pd.read_csv("Project/Dataset/calories.csv")
# print(df.head(5))
# print(df.dtypes)
# print(df.corr(numeric_only=True))

df = df.drop("User_ID", axis=1)         #dropping user_id because its irrelevant
# print(df.head(5))

df["Gender"] = df["Gender"].map({       # Gender from string to int
    "male": 1,
    "female": 0
})

X = df[
    [
        "Gender",
        "Age",
        "Height",
        "Weight",
        "Duration",
        "Heart_Rate",
        "Body_Temp"
    ]
]

y = df["Calories"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
#-------------------------------RANDOM FOREST REGRESSOR--------------------------#
model_rf = RandomForestRegressor(random_state=42)
model_rf.fit(X_train, y_train)



# print("Predicted Calories:", prediction)

# print("--------------------Random Forest Regression--------------------")
# r2_rf = r2_score(y_test, prediction)
# print("R2 Score:", r2_rf)
# mae_rf = mean_absolute_error(y_test, prediction)
# print("Mean Absolute Error:", mae_rf)
# rmse_rf = root_mean_squared_error(y_test, prediction)
# print("Root Mean Squared Error :", rmse_rf)

# plt.figure(figsize=(8,5))

# plt.scatter(y_test, prediction)

# plt.plot(
#     [y_test.min(), y_test.max()],
#     [y_test.min(), y_test.max()]
# )

# plt.xlabel("Actual Calories")
# plt.ylabel("Predicted Calories")
# plt.title("Actual vs Predicted Calories (Random Forest)")

# plt.show()

#-----------------------------LINEAR REGRESSOR----------------------------------#.

# lr_model = LinearRegression()
# lr_model.fit(X_train, y_train)
# lr_predictions = lr_model.predict(X_test)

# print("\n-----------------Linear Regression-----------------")
# r2_lr = r2_score(y_test, lr_predictions)
# print("R2 Score:", r2_lr)
# mae_lr = mean_absolute_error(y_test, lr_predictions)
# print("Mean Absolute Error:", mae_lr)
# rmse_lr = root_mean_squared_error(y_test, lr_predictions)
# print("Root Mean Squared Error :", rmse_lr)

# #----------------------------DECISION TREE REGRESSOR---------------------------#


# dt_model = DecisionTreeRegressor(random_state=42)
# dt_model.fit(X_train, y_train)
# dt_predictions = dt_model.predict(X_test)
# print("\n-----------------Decision Tree-----------------")
# r2_dt = r2_score(y_test, dt_predictions)
# print("R2 Score:", r2_dt)
# mae_dt = mean_absolute_error(y_test, dt_predictions)
# print("Mean Absolute Error:", mae_dt)
# rmse_dt = root_mean_squared_error(y_test, dt_predictions)
# print("Root Mean Squared Error :", rmse_dt)

# models = ["Random Forest", "Linear Regression", "Decision Tree"]

# #COMPARISON OF R2 SCORES
# r2_scores = [r2_rf,r2_lr,r2_dt]

# plt.figure(figsize=(8,5))
# plt.bar(models, r2_scores)

# plt.title("Comparison of R² Scores")
# plt.xlabel("Models")
# plt.ylabel("R² Score")

# plt.ylim(0.9, 1.0)

# plt.show()
# #COMPARISON OF MAE SCORES
# mae_scores = [mae_rf,mae_lr,mae_dt]

# plt.figure(figsize=(8,5))
# plt.bar(models, mae_scores)

# plt.title("Mean Absolute Error Comparison")
# plt.xlabel("Models")
# plt.ylabel("MAE")

# plt.show()

import joblib

joblib.dump(model_rf, "calorie_model.pkl")