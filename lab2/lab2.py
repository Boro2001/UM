import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
mnist.keys()
X, y = mnist["data"], mnist["target"].astype(np.uint8)
X_train, X_test = X[:56000], X[56000:]
y_train, y_test = y[:56000], y[56000:]
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)
y_train_0 = (y_train == 0)
y_test_0 = (y_test ==0)
from sklearn.linear_model import SGDClassifier
sgd_clf = SGDClassifier(random_state=30)
sgd_clf.fit(X_train, y_train_0)
predict_y_train_0 = sgd_clf.predict(X_train)
predict_y_test_0 = sgd_clf.predict(X_test)
from sklearn.metrics import accuracy_score
accuracy = np.array([accuracy_score(y_train_0, predict_y_train_0, normalize=True, sample_weight=None), accuracy_score(y_test_0, predict_y_test_0, normalize=True, sample_weight=None)])
accuracy = accuracy.tolist()    
print(accuracy)
import pickle
with open('sgd_acc.pkl', 'wb') as f_acc:
    pickle.dump(accuracy, f_acc)
from sklearn.model_selection import cross_val_score
cva = cross_val_score(sgd_clf, X_train, y_train_0, cv=3, scoring="accuracy")
cva = cva.tolist()
import pickle
with  open('sgd_cva.pkl', 'wb') as f_cva:
    pickle.dump(cva, f_cva)
sgd_m_clf = SGDClassifier(random_state=30,n_jobs=-1)
sgd_m_clf.fit(X_train, y_train)

from sklearn.model_selection import cross_val_predict
y_pred = cross_val_predict(sgd_m_clf, X, y, cv=3, n_jobs=-1)
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y, y_pred)
confusion_matrix.tolist()
with open('sgd_cmx.pkl', 'wb') as f_conf:
    pickle.dump(confusion_matrix.tolist(), f_conf)