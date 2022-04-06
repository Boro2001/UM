# %%
from sklearn import datasets
from sklearn.metrics import accuracy_score

# %%
data_breast_cancer_X, data_breast_cancer_y = datasets.load_breast_cancer(return_X_y= True, as_frame=True)
data_breast_cancer_y

# %%
data_breast_cancer_X = data_breast_cancer_X[(["mean area", "mean smoothness"])]
data_breast_cancer_X

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data_breast_cancer_X, data_breast_cancer_y, test_size=0.2)


# %%
import numpy as np
from sklearn import datasets
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
import pandas as pd

# %%
svm_clf = Pipeline([("linear_svc", LinearSVC(C=1,loss="hinge",random_state=42)),])

# %%
svm_clf.fit(X_train, y_train)

# %%
acc = []
y_train_predict = svm_clf.predict(X_train)
y_test_predict = svm_clf.predict(X_test)
accuracy0 = np.array([accuracy_score(y_train, y_train_predict, normalize=True, sample_weight=None), accuracy_score(y_test, y_test_predict, normalize=True, sample_weight=None)])
accuracy0

# %%
svm_clf_scaled = Pipeline([("scaler", StandardScaler()),("linear_svc", LinearSVC(C=1,loss="hinge",random_state=42)),])
svm_clf_scaled.fit(X_train, y_train)

# %%
y_train_predict = svm_clf_scaled.predict(X_train)
y_test_predict = svm_clf_scaled.predict(X_test)
accuracy1 = np.array([accuracy_score(y_train, y_train_predict, normalize=True, sample_weight=None), accuracy_score(y_test, y_test_predict, normalize=True, sample_weight=None)])
accuracy1

# %%
acc = [accuracy0[0], accuracy1[0], accuracy0[1], accuracy1[1]]
acc 

# %%
import pickle
with open('bc_acc.pkl', 'wb') as f_cva:
    pickle.dump(acc, f_cva)

# %%


# %%
data_iris = datasets.load_iris()
data_iris_X, data_iris_y = datasets.load_iris(return_X_y=True, as_frame=True)
data_iris_X = data_iris_X[(["sepal length (cm)", "sepal width (cm)"])]
data_iris_y

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data_iris_X, data_iris_y, test_size=0.2)


# %%
svm_clf = Pipeline([("linear_svc", LinearSVC(C=1,loss="hinge",random_state=42)),])

# %%
svm_clf.fit(X_train, y_train)

# %%
acc = []
y_train_predict = svm_clf.predict(X_train)
y_test_predict = svm_clf.predict(X_test)
accuracy0 = np.array([accuracy_score(y_train, y_train_predict, normalize=True, sample_weight=None), accuracy_score(y_test, y_test_predict, normalize=True, sample_weight=None)])
accuracy0

# %%
svm_clf_scaled = Pipeline([("scaler", StandardScaler()),("linear_svc", LinearSVC(C=1,loss="hinge",random_state=42)),])
svm_clf_scaled.fit(X_train, y_train)

# %%
y_train_predict = svm_clf_scaled.predict(X_train)
y_test_predict = svm_clf_scaled.predict(X_test)
accuracy1 = np.array([accuracy_score(y_train, y_train_predict, normalize=True, sample_weight=None), accuracy_score(y_test, y_test_predict, normalize=True, sample_weight=None)])
accuracy1

# %%
acc = [accuracy0[0], accuracy1[0], accuracy0[1], accuracy1[1]]
acc 

# %%
import pickle
with open('iris_acc.pkl', 'wb') as f_cva:
    pickle.dump(acc, f_cva)

# %%
## code do review 


