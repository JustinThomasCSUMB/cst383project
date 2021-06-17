import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, export_graphviz

url = "https://raw.githubusercontent.com/JustinThomasCSUMB/cst383project/main/healthcare-dataset-stroke-data.csv"
df = pd.read_csv(url)

shape = df.shape
print("Data Shape:", shape)
print('\nInfo:', df.info())
print('\nDescribe:',df.describe())

print("Percent of values null per column")
print(df.isnull().sum() / len(df) * 100) # % of null values

print("\nTotal null values")
print(df.isnull().sum()) #
print(df.head(10))

possible_columns = ['gender','age','hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'smoking_status']
print("Useful Predictors:\n", possible_columns)

df = df.dropna() #drop all na/null values
df = df.drop(columns=['ever_married','work_type','Residence_type','id']) # remove useless columns/ not needed
print("Data with removed columns:\n", df.head(5))

fig, ax = plt.subplots(1, len(possible_columns), figsize=[30,5])
for i,p in enumerate(possible_columns):
    sns.histplot(data=df, x=p,ax=ax[i])
    
plt.figure()
gender = df['gender'].value_counts()
sns.barplot(x=gender.index, y=gender)
plt.title("Gender of Participants")
plt.xlabel('Gender')
plt.ylabel('Total Participants')

plt.figure()
heartDisease = df['heart_disease'].value_counts()
sns.barplot(x=heartDisease.index, y=heartDisease)
plt.title("With vs Without Heart Disease")
plt.xlabel('Has Heart Disease')
plt.ylabel('Total Participants')

plt.figure()
smoker = df['smoking_status'].value_counts()
sns.barplot(x=smoker.index, y=smoker)
plt.title("Smoking Status")
plt.xlabel('Status')
plt.ylabel('Total Participants')

plt.figure()
ageBmi = df[['age','bmi']]
sns.scatterplot(x='age', y='bmi', data=ageBmi)
plt.title("Age vs Bmi")
plt.xlabel('Age')
plt.ylabel('Bmi (Body Mass Index') 

plt.figure()
scatNumDf = df[['age','bmi','avg_glucose_level']]
sns.pairplot(scatNumDf)

plt.figure()
sns.violinplot(x='gender', y='age', data=df)
plt.title('Gender and Age Comparison')

plt.figure()
sns.violinplot(x='smoking_status', y='age', data=df)
plt.title('Smoking and Age Comparison')

plt.figure()
sns.scatterplot(x='bmi', y='avg_glucose_level', hue='smoking_status', data=df)
plt.title('Ave. Glucose Level vs Age')

# more clean up
# only compare male and female
df = df[df.gender != 'Other']
#replace smoking strings with numerical representation, replace gender with binary representation
df = df.replace({'never smoked':0,'formerly smoked':1,'smokes':2,'Unknown':3,'Male':1,'Female':0})
dfdum = pd.get_dummies(df, drop_first=True)

print(dfdum.info())
print("Dummy shape: ",dfdum.shape)

# mse and rmse helper
def rmse(predicted, actual):
    return np.sqrt(((predicted - actual)**2).mean())

def mse(predicted, actual):
    return np.square(actual - predicted).mean()

# training data linear regression
# column names changed due to dummy data
predictions = ['gender','age','hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'smoking_status']
target = ['stroke']
x = dfdum[predictions].values
y = dfdum[target].values

# create training model that predics stroke occurance based on
# age, hypertension, smoking_habits
dfPred1Cols = ['age', 'hypertension', 'smoking_status']
dfPred1 = dfdum[(df.age >= 40) & (df.hypertension == 1) & (df.smoking_status == 1) | (df.smoking_status == 2)]
x1 = dfPred1[dfPred1Cols].values
y1 = dfPred1[target].values

x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size=0.30, random_state=0)
reg = LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)
reg.fit(x_train,y_train)
pred1 = reg.predict(x1)

plt.figure()
sns.scatterplot(y1.ravel(), pred1.ravel())
plt.plot([0,1],[0,1], color='grey', linestyle='dashed')





















