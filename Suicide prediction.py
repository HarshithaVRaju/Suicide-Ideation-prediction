# -*- coding: utf-8 -*-
"""
Created on Mon Mar  6 11:42:14 2023

@author: Harshitha and Aarushi
"""

import os
os.getcwd()
import pandas as pd
os.chdir("F:/python datasets")
mydata=pd.read_csv("suicide survey.csv")
print(mydata.head())
print(mydata.isnull().sum())
print(mydata.dtypes)
print(mydata.describe(include=["object"]))
mydata["Age"].value_counts()
import seaborn as sns 
import matplotlib.pyplot as plt

# we filter the age between the group of 18 to 80
mydata= mydata[(mydata['Age'] >= 18) & (mydata['Age'] <= 80)]
print(mydata)
mydata["Age"].value_counts()
plt.figure(figsize=(12,8))
sns.distplot(mydata["Age"], bins=20)
plt.title("Distribution and density by Age")
plt.xlabel("Age")

g = sns.FacetGrid(mydata, col='suicide ideation', size=5)
g = g.map(sns.distplot, "Age")

#Ranges of Age
mydata['age_range'] = pd.cut(mydata['Age'], [18,30,50,65,100], labels=["18-30", "30-50", "50-65","65-80"], include_lowest=True)

#Features Scaling We're going to scale age, because is extremely different from the other ones.

# Scaling Age
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
mydata['Age'] = scaler.fit_transform(mydata[['Age']])
#mydata['no_of_friends']=scaler.fit_transform(mydata[['no_of_friends']])
mydata.head()

#to group gender into male,female and trans
mydata["Gender"].value_counts()
male_list = ["male", "m", "male-ish", "maile", "mal", "male (cis)", "make", "male ", "man","msle", "mail", "malr","cis man", "Cis Male", "cis male"]
trans_list = ["trans-female", "something kinda male?", "queer/she/they", "non-binary","nah", "all", "enby", "fluid", "genderqueer", "androgyne", "agender", "male leaning androgynous", "guy (-ish) ^_^", "trans woman", "neuter", "female (trans)", "queer", "ostensibly male, unsure what that really means"]           
female_list = ["cis female", "f", "female", "woman",  "femake", "female ","cis-female/femme", "female (cis)", "femail"]

for (row, col) in mydata.iterrows():

    if str.lower(col.Gender) in male_list:
        mydata['Gender'].replace(to_replace=col.Gender, value='male', inplace=True)

    if str.lower(col.Gender) in female_list:
        mydata['Gender'].replace(to_replace=col.Gender, value='female', inplace=True)

    if str.lower(col.Gender) in trans_list:
        mydata['Gender'].replace(to_replace=col.Gender, value='trans', inplace=True)

char_list = ['A little about you', 'p']
mydata = mydata[~mydata['Gender'].isin(char_list)]
print(mydata['Gender'].unique())
print(mydata)
mydata["Gender"].value_counts()

#to fill the missing values in the column self employed

mydata["self_employed"].value_counts()
mydata.self_employed.fillna("No",inplace=True)
print(mydata)
print(mydata.isnull().sum())

mydata["state"].value_counts()
mydata["Country"].value_counts()

#to fill the missing values in work interference
mydata["work_interfere"].value_counts()
mydata.work_interfere.fillna("Unknown",inplace=True)
print(mydata)

#to fill the missing values in comments column
mydata.comments.fillna("Not defined",inplace=True)
print(mydata)

#to 
mydata['no_of_friends'] = pd.to_numeric(mydata['no_of_friends'], errors='coerce')
# Fill in any NaN values with 0
mydata['no_of_friends'].fillna(0, inplace=True)
# Convert the column to integers
mydata['no_of_friends'] = mydata['no_of_friends'].astype(int)
mean_value = mydata['no_of_friends'].mean()
int_mean = int(mean_value)
mydata['no_of_friends'] = mydata['no_of_friends'].replace(0, int_mean)
mydata['no_of_friends']=scaler.fit_transform(mydata[['no_of_friends']])
mydata.head()


#dropping states with missing values
mydata.drop(['state','Country','Timestamp'], axis= 1, inplace=True)
mydata.drop(['comments'],axis=1,inplace=True)
print(mydata)

from sklearn.preprocessing import LabelEncoder, MinMaxScaler
#checking correlation and visualization
from scipy.stats import chi2_contingency
from sklearn import preprocessing

labelDict = {}
for feature in mydata:
    le = preprocessing.LabelEncoder()
    le.fit(mydata[feature])
    le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    mydata[feature] = le.transform(mydata[feature])
    # Get labels
    labelKey = 'label_' + feature
    labelValue = [*le_name_mapping]
    labelDict[labelKey] =labelValue
   
for key, value in labelDict.items():    
    print(key, value)

sns.heatmap(mydata.corr(),annot=True)
plt.show()
corrmat = mydata.corr()
f, ax = plt.subplots(figsize=(20, 9))
sns.heatmap(corrmat, vmax=1.0, square=True);
plt.show()

import numpy as np      
k = 10 #number of variables for heatmap
cols = corrmat.nlargest(k, 'suicide ideation')['suicide ideation'].index
cm = np.corrcoef(mydata[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
     
o = labelDict['label_age_range']
g = sns.factorplot(x="age_range", y="suicide ideation", hue="Gender", data=mydata, kind="bar",  ci=None, size=5, aspect=2, legend_out = True)
g.set_xticklabels(o)
plt.title('Probability of suicide ideation')
plt.ylabel('Probability')
plt.xlabel('Age')
# replace legend labels
new_labels = labelDict['label_Gender']
for t, l in zip(g._legend.texts, new_labels): t.set_text(l)
# Positioning the legend
g.fig.subplots_adjust(top=0.9,right=0.8)
plt.show()

o = labelDict['label_family_history']
g = sns.factorplot(x="family_history", y="suicide ideation", hue="Gender", data=mydata, kind="bar", ci=None, size=5, aspect=2, legend_out = True)
g.set_xticklabels(o)
plt.title('Probability of suicide ideation')
plt.ylabel('Probability')
plt.xlabel('Family History')
# replace legend labels
new_labels = labelDict['label_Gender']
for t, l in zip(g._legend.texts, new_labels): t.set_text(l)
# Positioning the legend
g.fig.subplots_adjust(top=0.9,right=0.8)
plt.show()

o = labelDict['label_care_options']
g = sns.factorplot(x="care_options", y="suicide ideation", hue="Gender", data=mydata, kind="bar", ci=None, size=5, aspect=2, legend_out = True)
g.set_xticklabels(o)
plt.title('Probability of suicide ideation')
plt.ylabel('Probability')
plt.xlabel('care options')
# replace legend labels
new_labels = labelDict['label_Gender']
for t, l in zip(g._legend.texts, new_labels): t.set_text(l)
# Positioning the legend
g.fig.subplots_adjust(top=0.9,right=0.8)
plt.show()

o = labelDict['label_benefits']
g = sns.factorplot(x="benefits", y="suicide ideation", hue="Gender", data=mydata, kind="bar", ci=None, size=5, aspect=2, legend_out = True)
g.set_xticklabels(o)
plt.title('Probability of suicide ideation')
plt.ylabel('Probability')
plt.xlabel('Benefits')
# replace legend labels
new_labels = labelDict['label_Gender']
for t, l in zip(g._legend.texts, new_labels): t.set_text(l)
# Positioning the legend
g.fig.subplots_adjust(top=0.9,right=0.8)
plt.show()

o = labelDict['label_work_interfere']
g = sns.factorplot(x="work_interfere", y="suicide ideation", hue="Gender", data=mydata, kind="bar", ci=None, size=5, aspect=2, legend_out = True)
g.set_xticklabels(o)
plt.title('Probability of suicide ideation')
plt.ylabel('Probability')
plt.xlabel('work_interfere')
# replace legend labels
new_labels = labelDict['label_Gender']
for t, l in zip(g._legend.texts, new_labels): t.set_text(l)
# Positioning the legend
g.fig.subplots_adjust(top=0.9,right=0.8)
plt.show()
print(mydata)


#Since p value  greater than 0.05, we would fail to reject the null hypothesis
CR1=pd.crosstab(index=mydata['Gender'],columns=mydata['suicide ideation'])
print(CR1)
sns.countplot(x='Gender',hue='suicide ideation',data=mydata)
plt.xlabel('Gender')
plt.ylabel('suicide ideation')
plt.show()
result1= chi2_contingency(CR1)
print(result1[1])


CR3=pd.crosstab(index=mydata['self_employed'],columns=mydata['suicide ideation'])
print(CR3)
sns.countplot(x='self_employed',hue='suicide ideation',data=mydata)
plt.xlabel('self_employed')
plt.ylabel('suicide ideation')
plt.show()
result3= chi2_contingency(CR3)
print(result3[1])

CR4=pd.crosstab(index=mydata['family_history'],columns=mydata['suicide ideation'])
print(CR4)
sns.countplot(x='family_history',hue='suicide ideation',data=mydata)
plt.xlabel('family_history')
plt.ylabel('suicide ideation')
plt.show()
result4= chi2_contingency(CR4)
print(result4[1])

CR5=pd.crosstab(index=mydata['treatment'],columns=mydata['suicide ideation'])
print(CR5)
sns.countplot(x='treatment',hue='suicide ideation',data=mydata)
plt.xlabel('treatment')
plt.ylabel('suicide ideation')
plt.show()
result5= chi2_contingency(CR5)
print(result5[1])

CR6=pd.crosstab(index=mydata['work_interfere'],columns=mydata['suicide ideation'])
print(CR6)
sns.countplot(x='work_interfere',hue='suicide ideation',data=mydata)
plt.xlabel('work_interfere')
plt.ylabel('suicide ideation')
plt.show()
result6= chi2_contingency(CR6)
print(result6[1])

CR7=pd.crosstab(index=mydata['abuse'],columns=mydata['suicide ideation'])
print(CR7)
sns.countplot(x='abuse',hue='suicide ideation',data=mydata)
plt.xlabel('abuse')
plt.ylabel('suicide ideation')
plt.show()
result7= chi2_contingency(CR7)
print(result7[1])

CR8=pd.crosstab(index=mydata['tech_company'],columns=mydata['suicide ideation'])
print(CR8)
sns.countplot(x='tech_company',hue='suicide ideation',data=mydata)
plt.xlabel('tech_company')
plt.ylabel('suicide ideation')
plt.show()
result8= chi2_contingency(CR8)
print(result8[1])

CR9=pd.crosstab(index=mydata['benefits'],columns=mydata['suicide ideation'])
print(CR9)
sns.countplot(x='benefits',hue='suicide ideation',data=mydata)
plt.xlabel('benefits')
plt.ylabel('suicide ideation')
plt.show()
result9= chi2_contingency(CR9)
print(result9[1])

CR10=pd.crosstab(index=mydata['care_options'],columns=mydata['suicide ideation'])
print(CR10)
sns.countplot(x='care_options',hue='suicide ideation',data=mydata)
plt.xlabel('care_options')
plt.ylabel('suicide ideation')
plt.show()
result10= chi2_contingency(CR10)
print(result10[1])

CR11=pd.crosstab(index=mydata['wellness_program'],columns=mydata['suicide ideation'])
print(CR11)
sns.countplot(x='wellness_program',hue='suicide ideation',data=mydata)
plt.xlabel('wellness_program')
plt.ylabel('suicide ideation')
plt.show()
result11= chi2_contingency(CR11)
print(result11[1])

CR12=pd.crosstab(index=mydata['seek_help'],columns=mydata['suicide ideation'])
print(CR12)
sns.countplot(x='seek_help',hue='suicide ideation',data=mydata)
plt.xlabel('seek_help')
plt.ylabel('suicide ideation')
plt.show()
result12= chi2_contingency(CR12)
print(result12[1])

CR13=pd.crosstab(index=mydata['anonymity'],columns=mydata['suicide ideation'])
print(CR13)
sns.countplot(x='anonymity',hue='suicide ideation',data=mydata)
plt.xlabel('anonymity')
plt.ylabel('suicide ideation')
plt.show()
result13= chi2_contingency(CR13)
print(result13[1])

CR14=pd.crosstab(index=mydata['leave'],columns=mydata['suicide ideation'])
print(CR14)
sns.countplot(x='leave',hue='suicide ideation',data=mydata)
plt.xlabel('leave')
plt.ylabel('suicide ideation')
plt.show()
result14= chi2_contingency(CR14)
print(result14[1])

CR15=pd.crosstab(index=mydata['mental_health_consequence'],columns=mydata['suicide ideation'])
print(CR15)
sns.countplot(x='mental_health_consequence',hue='suicide ideation',data=mydata)
plt.xlabel('mental_health_consequence')
plt.ylabel('suicide ideation')
plt.show()
result15= chi2_contingency(CR15)
print(result15[1])

CR16=pd.crosstab(index=mydata['phys_health_consequence'],columns=mydata['suicide ideation'])
print(CR16)
sns.countplot(x='phys_health_consequence',hue='suicide ideation',data=mydata)
plt.xlabel('phys_health_consequence')
plt.ylabel('suicide ideation')
plt.show()
result16= chi2_contingency(CR16)
print(result16[1])

CR17=pd.crosstab(index=mydata['coworkers'],columns=mydata['suicide ideation'])
print(CR17)
sns.countplot(x='coworkers',hue='suicide ideation',data=mydata)
plt.xlabel('coworkers')
plt.ylabel('suicide ideation')
plt.show()
result17= chi2_contingency(CR17)
print(result17[1])

CR18=pd.crosstab(index=mydata['supervisor'],columns=mydata['suicide ideation'])
print(CR18)
sns.countplot(x='supervisor',hue='suicide ideation',data=mydata)
plt.xlabel('supervisor')
plt.ylabel('suicide ideation')
plt.show()
result18= chi2_contingency(CR18)
print(result18[1])

CR19=pd.crosstab(index=mydata['mental_health_interview'],columns=mydata['suicide ideation'])
print(CR19)
sns.countplot(x='mental_health_interview',hue='suicide ideation',data=mydata)
plt.xlabel('mental_health_interview')
plt.ylabel('suicide ideation')
plt.show()
result19= chi2_contingency(CR19)
print(result19[1])

CR20=pd.crosstab(index=mydata['phys_health_interview'],columns=mydata['suicide ideation'])
print(CR20)
sns.countplot(x='phys_health_interview',hue='suicide ideation',data=mydata)
plt.xlabel('phys_health_interview')
plt.ylabel('suicide ideation')
plt.show()
result20= chi2_contingency(CR20)
print(result20[1])

CR21=pd.crosstab(index=mydata['Alcohol consumption'],columns=mydata['suicide ideation'])
print(CR21)
sns.countplot(x='Alcohol consumption',hue='suicide ideation',data=mydata)
plt.xlabel('Alcohol consumption')
plt.ylabel('suicide ideation')
plt.show()
result21= chi2_contingency(CR21)
print(result21[1])

CR22=pd.crosstab(index=mydata['smoking'],columns=mydata['suicide ideation'])
print(CR22)
sns.countplot(x='smoking',hue='suicide ideation',data=mydata)
plt.xlabel('smoking')
plt.ylabel('suicide ideation')
plt.show()
result22= chi2_contingency(CR22)
print(result22[1])



newdata=mydata.drop(["ID","Gender","family_history","work_interfere","tech_company","wellness_program","care_options"
                     ,"anonymity","phys_health_consequence","coworkers","supervisor","mental_health_interview"
                     ,"phys_health_interview"],axis=1)
# print the updated dataframe
print(newdata)

#fitting the model
y=newdata[["suicide ideation"]]
x=newdata.drop(["suicide ideation"],axis=1)
print(x)

from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.20,random_state=17)


from xgboost import XGBClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score,roc_auc_score, roc_curve
methodDict={}
def XGboost():
    model = XGBClassifier()
    model.fit(xtrain, ytrain)
    predicted_value = model.predict(xtest)
    cm = confusion_matrix(ytest, predicted_value)
    report = classification_report(ytest, predicted_value)
    accuracy = accuracy_score(ytest, predicted_value)
    precision = precision_score(ytest, predicted_value)
    recall = recall_score(ytest, predicted_value)
    f1 = f1_score(ytest, predicted_value)
    
    methodDict['XGBoost Algo'] = accuracy * 100
    return accuracy * 100, report, cm, precision, recall, f1

accuracy, report, matrix, precision, recall, f1 = XGboost()
print("Confusion Matrix:\n", matrix)
print("Classification Report:\n", report)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


#LOGISTIC REGRESSION
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, precision_score, recall_score, f1_score, roc_auc_score, roc_curve

def LOGISTIC():
    classifier = LogisticRegression()
    classifier.fit(xtrain, ytrain)
    y_pred = classifier.predict(xtest) 
    accuracy = accuracy_score(ytest, y_pred)
    
    y_pred_prob = classifier.predict_proba(xtest)[:, 1]
    auc = roc_auc_score(ytest, y_pred_prob)
    
    fpr, tpr, thresholds = roc_curve(ytest, y_pred_prob)

    methodDict['Logistic Regression'] = accuracy * 100
    return accuracy * 100,auc,fpr,tpr

accuracy1,auc1,fpr1,tpr1= LOGISTIC()

print("Accuracy:", accuracy1)
print("AUC:", auc1)
plt.figure()
plt.plot(fpr1, tpr1, label='Logistic Regression (AUC = {:.2f})'.format(auc1))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()
   
def Knn():
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=8)
    knn.fit(xtrain, ytrain)
    pred = knn.predict(xtest)
    from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
    
    accuracy=accuracy_score(ytest, pred)
   
    # Calculate AUC
    y_pred_prob = knn.predict_proba(xtest)[:, 1]
    auc = roc_auc_score(ytest, y_pred_prob)
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(ytest, y_pred_prob)
    methodDict['K-Neighbors'] = accuracy * 100
    return accuracy *100,auc,fpr,tpr
accuracy3,auc3,fpr3,tpr3=Knn()

print("Accuracy:",accuracy3)
print("AUC:", auc3)

# Plot ROC curve
plt.figure()
plt.plot(fpr3, tpr3, label='K-Neighbors (AUC = {:.2f})'.format(auc3))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

def SVM():
    from sklearn.svm import SVC
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
    
    classifier = SVC(kernel='linear')
    classifier.fit(xtrain, ytrain)
    y_pred = classifier.predict(xtest) 
    accuracy = accuracy_score(ytest, y_pred)
    
    # Calculate AUC
    y_pred_prob = classifier.decision_function(xtest)
    auc = roc_auc_score(ytest, y_pred_prob)
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(ytest, y_pred_prob)
    methodDict['SVM'] = accuracy * 100
    return accuracy*100,auc,fpr,tpr
    
accuracy4,auc4,fpr4,tpr4 = SVM()

print("Accuracy:", accuracy4)
print("AUC:", auc4)

# Plot ROC curve
plt.figure()
plt.plot(fpr4, tpr4, label='SVM (AUC = {:.2f})'.format(auc4))
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()

def Hybrid():
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.ensemble import VotingClassifier
    logreg = LogisticRegression()
    knn = KNeighborsClassifier(n_neighbors=5)
    ensemble = VotingClassifier(estimators=[('lr', logreg), ('knn', knn)], voting='hard')
    ensemble.fit(xtrain, ytrain)
# Make predictions on the test data
    pred = ensemble.predict(xtest)
# Evaluate the performance of the ensemble model
    from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
    cm = confusion_matrix(ytest,pred)
    cm
    accuracy = accuracy_score(ytest,pred)
    report=classification_report(ytest,pred)
    methodDict['Hybrid'] = accuracy * 100
    return accuracy *100,report,cm
accuracy5=Hybrid()

print("Accuracy of the hybrid algorithm:", accuracy5)


        

   
def plotSuccess():
    s = pd.Series(methodDict)
    s = s.sort_values(ascending=False)
    plt.figure(figsize=(12,8))
    ax = s.plot(kind='bar') 
    for p in ax.patches:
        ax.annotate(str(round(p.get_height(),2)), (p.get_x() * 1.005, p.get_height() * 1.005))
    plt.ylim([70.0, 90.0])
    plt.xlabel('Method')
    plt.ylabel('Percentage')
    plt.title('Success of methods')
    plt.show()
plotSuccess()
plt.show()       

