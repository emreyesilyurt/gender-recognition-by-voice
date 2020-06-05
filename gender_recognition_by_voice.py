# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score
from sklearn import metrics
import sys
if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")
%matplotlib inline
    
# %%
data = pd.read_csv('voice.csv')

# %%

print(data.isnull().sum())

# %%
print(data.dtypes)

# %%
print(len(data.columns))

# %%

sns.countplot(x= 'label', data = data)
plt.show() 

# %%

print ("Rows     : " ,data.shape[0])
print ("Columns  : " ,data.shape[1])
print ("\nFeatures : \n" ,data.columns.tolist())
print ("\nMissing values :  ", data.isnull().sum().values.sum())
print ("\nUnique values :  \n",data.nunique())


print(data['label'].unique())
print(data['meanfun'].unique())
print(data['IQR'].unique())


# %%

X = data.drop(['label'], axis = 1).values
Y = data.iloc[:,-1:].values


# %%

from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test = train_test_split(X, Y,test_size=0.2, random_state=0)

# %%

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(x_train)
X_test = sc.fit_transform(x_test)

# %%

"""
from sklearn.decomposition import PCA

pca = PCA(n_components = 9)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
"""

# %%


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis(n_components = 2)
X_train_lda = lda.fit_transform(X_train, y_train)
X_test_lda = lda.transform(X_test)


# %%

from sklearn.svm import SVC
model = SVC(kernel = 'linear') 
model.fit(X_train_lda, y_train)
y_pred = model.predict(X_test_lda)



# %%


cm = confusion_matrix(y_test, y_pred)
print('Accuracy: ', metrics.accuracy_score(y_test, y_pred), '\n')
print("SVC Confusion Matrix\n", cm)
sns.heatmap(cm/np.sum(cm), annot=True, fmt='.2%', cmap='Blues')


# %%

crossVal= cross_val_score(estimator = model, X = X_train, y = y_train, cv = 2)
print('SVC Accuracy: ', crossVal.mean())
print('SVC Std: ', crossVal.std())


# %%