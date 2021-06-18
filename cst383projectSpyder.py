import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn import svm
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures, MinMaxScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, export_graphviz
from sklearn.metrics import roc_curve, RocCurveDisplay, confusion_matrix, plot_confusion_matrix, roc_auc_score, average_precision_score
from sklearn.metrics import ConfusionMatrixDisplay, precision_recall_curve, PrecisionRecallDisplay, accuracy_score

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

# clean up
# only compare male and female
df = df[df.gender != 'Other']
#replace smoking strings with numerical representation, replace gender with binary representation
df = df.replace({'never smoked':0,'formerly smoked':1,'smokes':2,'Unknown':3,'Male':1,'Female':0})
dfdum = pd.get_dummies(df, drop_first=True)

#normalize data
scaler = MinMaxScaler()
ageScaled = scaler.fit_transform(df[['age']])
glucScaled = scaler.fit_transform(df[['avg_glucose_level']])
bmiScaled = scaler.fit_transform(df[['bmi']])
dfnorm = df.copy()
dfnorm['age'] = ageScaled
dfnorm['bmi'] = bmiScaled
dfnorm['avg_glucose_level'] = glucScaled

print(dfdum.info())
print("Dummy shape: ",dfdum.shape)

# mse and rmse helper
def rmse(predicted, actual):
    return np.sqrt(((predicted - actual)**2).mean())

def mse(predicted, actual):
    return np.square(actual - predicted).mean()

# training data predictions
predictions = ['gender','age','hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'smoking_status']
target = ['stroke']
xdum = dfdum[predictions].values
ydum = dfdum[target].values
    
target = ['stroke']
y = dfnorm[target].values

#prediction 1 all columns
pred1Cols = ['gender','age','hypertension', 'heart_disease', 'avg_glucose_level', 'bmi', 'smoking_status']
x1 = dfnorm[pred1Cols].values

x1_train, x1_test, y1_train, y1_test = train_test_split(x1, y, test_size=0.30, random_state=0)

clf = make_pipeline(StandardScaler(), LogisticRegression(random_state=0))
clf.fit(x1_test,y1_test)

# RocCurveDsiplay
plt.figure()
y1_score = clf.decision_function(x1_test)
fpr, tpr, _ = roc_curve(y1_test, y1_score, pos_label=clf.classes_[1])
rocauc1 = roc_auc_score(y1_test, y1_score)
roc_display = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=rocauc1, estimator_name='AUC Score').plot()
plt.title('ROC Curve Test Data - All Columns')

# precision recall display
plt.figure()
prec, recall, _ = precision_recall_curve(y1_test, y1_score, pos_label=clf.classes_[1])
aps1 = average_precision_score(y1_test, y1_score)
pr_display = PrecisionRecallDisplay(precision=prec, recall=recall, average_precision=aps1, estimator_name='Ave. Precision').plot()
plt.title('Precision Recall Test Data - All Columns')

# confusion matrix
plt.figure()
y1pred = clf.predict(x1_test)
cm = confusion_matrix(y1_test,y1pred)
cm_display=ConfusionMatrixDisplay(cm, display_labels=['']).plot()
plt.title('Conf. Matrix Test Data - All Columns')

# results
# accuracy
accuracy1 = accuracy_score(y1_test, y1pred)
mse1 = mse(y1pred, y1_test)
rmse1 = rmse(y1pred, y1_test)
r21 = r2_score(y1_test, y1pred)

print('Prediction 1 contains data from all columns in the dataset.')
print('Accuracy for Prediction 1: {:.2f}'.format(accuracy1))
print('MSE for Prediction 1: {:.2f}'.format(mse1))
print('RMSE for Prediction 1: {:.2f}'.format(rmse1))
print('R Squared for Prediction 1: {:.2f}'.format(r21))

# ----------------------------------------------------------

#prediction 2 age, hypertension, heart_disease, and smoking_status
pred2Cols = ['age','hypertension','heart_disease','smoking_status']
x2 = dfnorm[pred2Cols].values

x2_train, x2_test, y2_train, y2_test = train_test_split(x2, y, test_size=0.30, random_state=0)

clf2 = make_pipeline(StandardScaler(), LogisticRegression(random_state=0))
clf2.fit(x2_test,y2_test)

# RocCurveDsiplay
plt.figure()
y2_score = clf2.decision_function(x2_test)
fpr2, tpr2, _ = roc_curve(y2_test, y2_score, pos_label=clf.classes_[1])
rocauc2 = roc_auc_score(y2_test, y2_score)
roc_display = RocCurveDisplay(fpr=fpr2, tpr=tpr2, roc_auc=rocauc2, estimator_name='AUC Score').plot()
plt.title('ROC Curve Test Data - All Columns')

# precision recall display
plt.figure()
prec2, recall2, _ = precision_recall_curve(y2_test, y2_score, pos_label=clf.classes_[1])
aps2 = average_precision_score(y2_test, y2_score)
pr_display2 = PrecisionRecallDisplay(precision=prec2, recall=recall2, average_precision=aps2, estimator_name='Ave. Precision').plot()
plt.title('Precision Recall Test Data - All Columns')

# confusion matrix
plt.figure()
y2pred = clf2.predict(x2_test)
cm2 = confusion_matrix(y2_test,y1pred)
cm_display2 = ConfusionMatrixDisplay(cm2, display_labels=['']).plot()
plt.title('Conf. Matrix Test Data - All Columns')

# results
# accuracy
accuracy2 = accuracy_score(y2_test, y2pred)
mse2 = mse(y2pred, y2_test)
rmse2 = rmse(y2pred, y2_test)
r22 = r2_score(y2_test, y2pred)

print('Prediction 2 contains data from age, hypertension, heart_disease, and smoking_status columns in the dataset.')
print('Accuracy for Prediction 2: {:.2f}'.format(accuracy2))
print('MSE for Prediction 2: {:2f}'.format(mse2))
print('RMSE for Prediction 2: {:2f}'.format(rmse2))
print('R Squared for Prediction 2: {:.2f}'.format(r22))

# ----------------------------------------------------------

#prediction 3 bmi, avg_glucose_level, gender
pred3Cols = ['bmi','avg_glucose_level','gender']
x3 = dfnorm[pred3Cols].values

x3_train, x3_test, y3_train, y3_test = train_test_split(x3, y, test_size=0.30, random_state=0)

clf3 = make_pipeline(StandardScaler(), LogisticRegression(random_state=0))
clf3.fit(x3_test,y3_test)

# RocCurveDsiplay
plt.figure()
y3_score = clf3.decision_function(x3_test)
fpr3, tpr3, _ = roc_curve(y3_test, y3_score, pos_label=clf.classes_[1])
rocauc3 = roc_auc_score(y1_test, y1_score)
roc_display = RocCurveDisplay(fpr=fpr3, tpr=tpr3, roc_auc=rocauc3, estimator_name='AUC Score').plot()
plt.title('ROC Curve Test Data - All Columns')

# precision recall display
plt.figure()
prec3, recall3, _ = precision_recall_curve(y3_test, y3_score, pos_label=clf.classes_[1])
aps3 = average_precision_score(y3_test, y3_score)
pr_display3 = PrecisionRecallDisplay(precision=prec3, recall=recall3, average_precision=aps3, estimator_name='Ave. Precision').plot()
plt.title('Precision Recall Test Data - All Columns')

# confusion matrix
plt.figure()
y3pred = clf3.predict(x3_test)
cm3 = confusion_matrix(y3_test,y3pred)
cm_display3 = ConfusionMatrixDisplay(cm3, display_labels=['']).plot()
plt.title('Conf. Matrix Test Data - All Columns')

# results
# accuracy
accuracy3 = accuracy_score(y3_test, y3pred)
mse3 = mse(y3pred, y3_test)
rmse3 = rmse(y3pred, y3_test)
r23 = r2_score(y3_test, y3pred)

print('Prediction 2 contains data from age, hypertension, heart_disease, and smoking_status columns in the dataset.')
print('Accuracy for Prediction 2: {:.2f}'.format(accuracy3))
print('MSE for Prediction 2: {:2f}'.format(mse3))
print('RMSE for Prediction 2: {:2f}'.format(rmse3))
print('R Squared for Prediction 2: {:.2f}'.format(r23))

# ----------------------------------------------------------












