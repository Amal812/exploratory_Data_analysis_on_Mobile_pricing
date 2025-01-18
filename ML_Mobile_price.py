# Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBRegressor
from sklearn.metrics import r2_score
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

#  Load the Dataset
data_set = pd.read_csv(r"C:\SMEC\Data_Science\Project\DATASET\Mobile-Price-Prediction-cleaned_data.csv")
print(data_set.columns)
data_set.dropna(inplace=True)
print(data_set)

data_set=data_set[data_set["Price"]<60000]


sns.boxplot(data=data_set)
plt.show()

correlation=data_set.corr(numeric_only=True)
sns.heatmap(correlation,annot=True)
plt.show()
df1=data_set.select_dtypes(exclude=['object'])
for coloumn in df1:
    plt.figure(figsize=(17,1))
    sns.boxplot(data=df1,x=coloumn)
print(df1)
plt.show()

# # removed coloum

x=data_set.drop(columns=["Mobile_Size","Primary_Cam","Selfi_Cam","Price"],axis=1)
y=data_set['Price']
print(x)
print(y)

sns.scatterplot(x="Battery_Power",y="Price",data=data_set)
plt.show()

# Splitting the dataset into training and test set.

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.2,random_state=0)

#Fitting the MLR model to the training set:

regressor= LinearRegression()
regressor.fit(x_train, y_train)

#Predicting the Test set result;
y_pred= regressor.predict(x_test)

#To compare the actual output values for X_test with the predicted value
df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df.to_string())
print("Mean")
print(data_set.describe())
print("-------------------------------------")

#Evaluating the Algorithm
from sklearn import metrics
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test,y_pred))
print('Mean Squared Error:', metrics.mean_squared_error(y_test,y_pred))
print('Root Mean Squared Error:',np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# predicting the accuracy score

score=r2_score(y_test,y_pred)
print("r2 socre is ",score*100,"%")

# boosting the r2 score

model=LinearRegression()
model.fit(x_train,y_train)

y_pred=model.predict(x_test)
print(y_pred)
accuracy1=r2_score(y_pred,y_test)
print(accuracy1)

model2=XGBRegressor()
model2.fit(x_train,y_train)
y_pred=model2.predict(x_test)
accuracy2=r2_score(y_test,y_pred)
print(accuracy2)

# Model 2 outperformed the other models, achieving the highest accuracy and providing the most reliable predictions.

