# -*- coding: utf-8 -*-


import pandas as pd
from sklearn.decomposition import KernelPCA
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import numpy as np
from sklearn.model_selection import train_test_split
import sklearn.metrics as sm
from sklearn.svm import SVC
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import SelectKBest, chi2
from sklearn import preprocessing
from mpl_toolkits.mplot3d import Axes3D
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import classification_report
import seaborn as sns





#Importing Dataset
#note: linear dataset takes long time for loading therefore wait for data to fully load
Gene_Data = pd.read_csv('/content/data_linear_CNA.csv')
Patient_Data = pd.read_csv('/content/data_clinical_sample.csv')

Gene_Data

Patient_Data

#Extracting Y Value
Y = Patient_Data[['IHC_HER2', 'SAMPLE_ID']]
S = 'IHC_HER2'
Y

#Drop ID Columns
Gene_Data = Gene_Data.drop('Hugo_Symbol',axis=1)
Gene_Data = Gene_Data.drop('Entrez_Gene_Id',axis=1)
Gene_Data

#Transposing Data
X_old = Gene_Data.transpose()
X_old

#Normalizing Data to get rid of negative values
mm_scaler = preprocessing.MinMaxScaler()
X_old = pd.DataFrame(mm_scaler.fit_transform(X_old))
Y = Y.set_index('SAMPLE_ID')
X = pd.concat([X_old, pd.DataFrame(Y.values)], axis=1, ignore_index=True)
X = X.rename(columns={22247: "IHC_HER2"})
X = X.drop(X.index[-1])
X = X.drop(X.index[-1])
X

#Bar Chart to Display Classes
Y.IHC_HER2.value_counts().plot(kind='bar', title='Before Over Sampling');

#Replacing Classnames with integers
ClassList = []
for item in X[S]:
    if item == 'Negative':
        ClassList.append(0)
    elif item == 'Equivocal':
        ClassList.append(1)
    elif item == 'Positive':
        ClassList.append(2)
    else:
        ClassList.append(-1)

#Adding Y to dataset
X[S] = ClassList

#Seperating Classes
D0 = X.query('IHC_HER2 == 0')
D1 = X.query('IHC_HER2 == 1')
D2 = X.query('IHC_HER2 == 2')

#Appending Selected Classes
DF = D0.append(D1).append(D2)
DF

#Bar Chart to Display Selected Classes
DF.IHC_HER2.value_counts().plot(kind='bar', title='Selected');

X_o = DF.iloc[:, 0:22247]
Y_o = DF['IHC_HER2']

#OverSampling unbalanced classes
smo = SMOTE(random_state=42,k_neighbors=10, sampling_strategy='not majority')
X_s, Y_s = smo.fit_resample(X_o, Y_o)

#Making Dataframes to make Bar Chart
X_d = pd.DataFrame(X_s)
Y_d = pd.DataFrame(Y_s)
X_d['IHC_HER2'] = Y_d

#Bar Chart to Display Selected Classes after Oversampling
X_d.IHC_HER2.value_counts().plot(kind='bar', title='After Over Sampling')

#Selecting K-Best Features
X_new = SelectKBest(chi2, k=18000).fit_transform(X_s,Y_s)

#Splitting Data to Training and Testing Data
Points_train, Points_test, Label_train, Label_test = train_test_split(X_new, Y_s,train_size=0.3,shuffle=True)

#Applying Support Vector Machine Classifier with linear kernel
svm = SVC(gamma= 0.1, kernel='linear')
model = svm.fit(Points_train, Label_train)
svm_pred = svm.predict(Points_test)
print('SVM linear score:',svm.score(Points_train,Label_train))

# making function to create ROC curve
from yellowbrick.classifier import ROCAUC
def plot_roc_curve(model,xtrain,ytrain,xtest,ytest):
  visualizer = ROCAUC(model, encoder={0:'Negative',
                                      1:'Equivocal',
                                      2:'Positive'})
  visualizer.fit(xtrain,ytrain)
  visualizer.score(xtest,ytest)
  visualizer.show()

  return visualizer

#Confusion matrix SVM Classifier with linear kernel
c_mat= confusion_matrix(Label_test,svm_pred,labels = svm.classes_)
c_mat

#Checking SVM accuracy (linear kernel)
print("SVM accuracy",str(round(sm.accuracy_score(Label_test, svm_pred)*100,1)))

# applying cross validation on training data
from sklearn.model_selection import KFold, cross_val_score
scores = cross_val_score(model, Points_train, Label_train, cv = KFold(n_splits = 5))

scores

print(np.mean(scores))

# applying cross validation on testing data
from sklearn.model_selection import KFold, cross_val_score
scorespred = cross_val_score(svm, Points_test, Label_test)

scorespred

print(np.mean(scorespred))

print(classification_report(Label_test,svm_pred))

plot_roc_curve(svm,Points_train,Label_train,Points_test,Label_test)

#Applying Support Vector Machine Classifier with rbf kernel
svmrbf = SVC(gamma= 0.1, kernel='rbf')
model = svmrbf.fit(Points_train, Label_train)
svmrbf_pred = svmrbf.predict(Points_test)
print('SVM rbf score:',svmrbf.score(Points_train,Label_train))

#Confusion matrix Support Vector Machine Classifier with rbf kernel
c_mat= confusion_matrix(Label_test,svmrbf_pred)
c_mat

#Checking SVM accuracy (rbf kernel)
print("SVM accuracy",str(round(sm.accuracy_score(Label_test, svmrbf_pred)*100,1)))

# applying cross validation on training data
from sklearn.model_selection import KFold, cross_val_score
scores = cross_val_score(model, Points_train, Label_train, cv = KFold(n_splits = 5))

scores

print(np.mean(scores))

# applying cross validation on testing data
from sklearn.model_selection import KFold, cross_val_score
scorespred = cross_val_score(svmrbf, Points_test, Label_test)

scorespred

print(np.mean(scorespred))

print(classification_report(Label_test,svmrbf_pred))

plot_roc_curve(svmrbf,Points_train,Label_train,Points_test,Label_test)

#############################################################################################

#Random Forest
rf = RandomForestClassifier(n_estimators = 1000, random_state = 42)
model = rf.fit(Points_train, Label_train);
rf_pred = rf.predict(Points_test)
print("Random Forest Score:",metrics.accuracy_score(Label_test, rf_pred))

plot_roc_curve(rf,Points_train,Label_train,Points_test,Label_test)

# applying cross validation on training data
from sklearn.model_selection import KFold, cross_val_score
scores = cross_val_score(model, Points_train, Label_train, cv = KFold(n_splits = 5))

scores

print(np.mean(scores))

# applying cross validation on testing data
from sklearn.model_selection import KFold, cross_val_score
scorespred = cross_val_score(rf, Points_test, Label_test)

scorespred

print(np.mean(scorespred))

c_mat= confusion_matrix(Label_test,rf_pred)
c_mat

print(classification_report(Label_test,rf_pred))

###########################################################

#KNN
knn = KNeighborsClassifier(n_neighbors=5)
model = knn.fit(Points_train, Label_train)
knn_pred = knn.predict(Points_test)
print("KNN Score:",knn.score(Points_test, Label_test))

# applying cross validation on training data
from sklearn.model_selection import KFold, cross_val_score
scores = cross_val_score(model, Points_train, Label_train, cv = KFold(n_splits = 5))

scores

print(np.mean(scores))

# applying cross validation on testing data
from sklearn.model_selection import KFold, cross_val_score
scorespred = cross_val_score(knn, Points_test, Label_test)

scorespred

print(np.mean(scorespred))

c_mat= confusion_matrix(Label_test,knn_pred)
c_mat

print(classification_report(Label_test,knn_pred))

plot_roc_curve(knn,Points_train,Label_train,Points_test,Label_test)

#################################################################

#Reducing dimensions for data visualization
kpca = KernelPCA(n_components=3, kernel='rbf',gamma=0.1)
X_pca = kpca.fit_transform(X_new)

#Plotting Graph
fig = plt.figure(figsize=(15, 15))
ax = fig.add_subplot(111, projection='3d')
x = X_pca[:,0]
y = X_pca[:,1]
z = X_pca[:,2]
ax.scatter(x, y, z, c=Y_s,s=100, marker='o')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
plt.show()