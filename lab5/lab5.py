# %%
from sklearn import datasets
import pandas as pd
data_breast_cancer = datasets.load_breast_cancer(as_frame=True)
#print(data_breast_cancer['DESCR'])

# %%
data_breast_cancer_X, data_breast_cancer_y = datasets.load_breast_cancer(return_X_y= True, as_frame=True)
data_breast_cancer_X = data_breast_cancer_X[['mean texture','mean symmetry']]
data_breast_cancer_X

# %% [markdown]
# 

# %%
import numpy as np
import pandas as pd
size = 300

# %%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data_breast_cancer_X, data_breast_cancer_y, test_size=0.2)

# %%
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
depth = 8
tree_clf = DecisionTreeClassifier(max_depth=depth,random_state=42)
tree_clf.fit(data_breast_cancer_X, data_breast_cancer_y)
f = "bc"
export_graphviz(tree_clf,out_file=f, rounded=True, filled=True)

# %%
import graphviz
graph = graphviz.Source.from_file(f)

# %%
print(graphviz.render('dot', 'png', f))

# %%
# współczynniki f1 
import sklearn.metrics

y_train_predict = tree_clf.predict(X_train)
y_test_predict = tree_clf.predict(X_test)

f1_train = sklearn.metrics.f1_score(y_train_predict, y_train)
f1_test = sklearn.metrics.f1_score(y_test_predict, y_test)

acc_train = sklearn.metrics.accuracy_score(y_train_predict, y_train)
acc_test = sklearn.metrics.accuracy_score(y_test_predict, y_test)

f1acc_tree  = [depth, f1_train, f1_test, acc_train, acc_test]
f1acc_tree

# %%
import pickle
with open('f1acc_tree.pkl', 'wb') as f_cva:
    pickle.dump(f1acc_tree, f_cva)

# %%
import numpy as np
import pandas as pd
size = 300
X = np.random.rand(size)*5-2.5
w4, w3, w2, w1, w0 = 1, 2, 1, -4, 2
y = w4*(X**4) + w3*(X**3) + w2*(X**2) + w1*X + w0 + np.random.randn(size)*8-4
df = pd.DataFrame({'x': X, 'y': y})
df.plot.scatter(x='x',y='y')

# %%
X_train, X_test = X[:240], X[240:]
y_train, y_test = y[:240], y[240:]
X_train = X_train.reshape(-1,1)
y_train = y_train.reshape(-1,1)
X_test = X_test.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# %%
from sklearn.tree import DecisionTreeRegressor
depth = 4
tree_reg = DecisionTreeRegressor(max_depth=depth)
tree_reg.fit(X_train, y_train)

# %%
from matplotlib import pyplot as plt
import matplotlib

plt.plot(X_train, tree_reg.predict(X_train), "r.")
plt.plot(X_train, y_train, ".")

# %%
from sklearn.metrics import mean_squared_error

y_train_predict = tree_reg.predict(X_train)
y_test_predict = tree_reg.predict(X_test)

mse_train = mean_squared_error(y_train, y_train_predict)
mse_test = mean_squared_error(y_test, y_test_predict)

# %%
mse_test

# %%
mse_train

# %%
f = "reg"
export_graphviz(tree_reg,out_file=f, rounded=True, filled=True)

# %%
import graphviz
graph = graphviz.Source.from_file(f)
print(graphviz.render('dot', 'png', f))

# %%
mse_tree = [depth, mse_train, mse_test]
import pickle
with open('mse_tree.pkl', 'wb') as f_cva:
    pickle.dump(mse_tree, f_cva)

mse_tree


