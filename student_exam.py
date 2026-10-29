import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error , mean_squared_error , r2_score

df=pd.read_csv("student_exam_scores.csv")
# print(df.head())
# print(df.shape)
# print(df.columns)

print(df.info())

print(df.describe())

print(df.isnull().sum())

# target(y) and data(x)
x=df[["hours_studied","sleep_hours","attendance_percent","previous_scores"]]

y=df["exam_score"]

#split test & train data
x_test , x_train , y_test , y_train=train_test_split(x,y, test_size=0.2 , random_state=42)


#model
model=LinearRegression()

model.fit(x_train ,y_train)

print("Model training completed!")
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)

#predict exam score
y_pred=model.predict(x_test)

#evaluate the model 
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("mean_absolute_error: " ,round(mae, 2))
print("mean_squared_error: " , round(rmse, 2))
print("R2_score: " , round(r2, 3))


#plots
#1.actual score vs predicted score
plt.figure(figsize=(7, 5))
plt.scatter(y_test, y_pred)
low = min(y_test.min(), y_pred.min())
high = max(y_test.max(), y_pred.max())
plt.plot([low, high], [low, high], "r--", label="Perfect Prediction")
plt.xlabel("Actual Exam Scores")
plt.ylabel("Predicted Exam Scores")
plt.title("Actual vs Predicted Scores")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

#2.study hours vs exam score
plt.figure(figsize=(7, 5))
plt.scatter(df["hours_studied"], df["exam_score"])
plt.xlabel("Hours Studied")
plt.ylabel("Exam Score")
plt.title("Study Hours vs Exam Scores")
plt.grid(True)
plt.tight_layout()
plt.show()


# 3. Prediction errors
errors = y_test - y_pred
plt.figure(figsize=(7, 5))
plt.scatter(y_pred, errors)
plt.axhline(y=0, linestyle="--")
plt.xlabel("Predicted Exam Scores")
plt.ylabel("Prediction Error")
plt.title("Prediction Errors")
plt.grid(True)
plt.tight_layout()
plt.show()



















