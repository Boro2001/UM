# %%
import pandas as pd
import numpy as np
import matplotlib 
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
#1, 2 

# %%
import numpy as np
size = 300
X = np.random.rand(size)*5-2.5
w4, w3, w2, w1, w0 = 1, 2, 1, -4, 2
y = w4*(X**4) + w3*(X**3) + w2*(X**2) + w1*X + w0  + np.random.randn(size)*8-4
df = pd.DataFrame({'x': X, 'y': y})
df.to_csv('dane_do_regresji.csv',index=None)
df.plot.scatter(x='x',y='y')

# %%
X_train, X_test = X[:240], X[240:]
y_train, y_test = y[:240], y[240:]
X_train = X_train.reshape(-1,1)
y_train = y_train.reshape(-1,1)
X_test = X_test.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)

# %%
test_errors = []
train_errors = []

# %%
X_new = np.array([[0], [2]])

# %%
#regresja linowa
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_train,y_train)
print(lin_reg.intercept_, lin_reg.coef_, "\n",lin_reg.predict(X_new))

# %%
#mse lin_reg
y_train_predict = lin_reg.predict(X_train)
y_test_predict = lin_reg.predict(X_test)

train_errors.append(mean_squared_error(y_train, y_train_predict))
test_errors.append(mean_squared_error(y_test, y_test_predict))

# %%
from matplotlib import pyplot as plt
import matplotlib
plt.plot(X_train, lin_reg.predict(X_train), "r-")
plt.plot(X_train, y_train, ".")

# %%
#regresja kNN k = 3
import sklearn.neighbors 
knn_3_reg = sklearn.neighbors.KNeighborsRegressor(n_neighbors=3)
knn_3_reg.fit(X_train, y_train)

# %%
#mse knn3
y_train_predict = knn_3_reg.predict(X_train)
y_test_predict = knn_3_reg.predict(X_test)
train_errors.append(mean_squared_error(y_train, y_train_predict))
test_errors.append(mean_squared_error(y_test, y_test_predict))
print(test_errors)
print(train_errors)

# %%
from matplotlib import pyplot as plt
import matplotlib
plt.plot(X_train, knn_3_reg.predict(X_train), "r.")
plt.plot(X_train, y_train, ".")

# %%
#regresja kNN k = 5
import sklearn.neighbors 
knn_5_reg = sklearn.neighbors.KNeighborsRegressor(n_neighbors=5)
knn_5_reg.fit(X_train, y_train)

#print(knn_reg.predict(X_train))

# %%
#mse knn5
y_train_predict = knn_5_reg.predict(X_train)
y_test_predict = knn_5_reg.predict(X_test)
train_errors.append(mean_squared_error(y_train, y_train_predict))
test_errors.append(mean_squared_error(y_test, y_test_predict))
print(test_errors)
print(train_errors)

# %%
from matplotlib import pyplot as plt
import matplotlib
plt.plot(X_train, knn_5_reg.predict(X_train), "r.")
plt.plot(X_train, y_train, ".")
# the red dots are actual values, blue are predicted

# %%
# regresja wielomianowa st = 2
from sklearn.preprocessing import PolynomialFeatures
poly_features_2 = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly_features_2.fit_transform(X_train)
poly_2_reg = LinearRegression()
poly_2_reg.fit(X_poly, y_train)
print(X[0], X_poly[0])
print(poly_2_reg.intercept_, poly_2_reg.coef_)
print(poly_2_reg.predict(poly_features_2.fit_transform([[0],[2]])))
print(poly_2_reg.coef_[0][1] * 2**2 + poly_2_reg.coef_[0][0] * 2+ poly_2_reg.intercept_[0])

# %%
#mse poly2
X_poly_test = poly_features_2.fit_transform(X_test)
y_train_predict = poly_2_reg.predict(X_poly)
y_test_predict = poly_2_reg.predict(X_poly_test)
train_errors.append(mean_squared_error(y_train, y_train_predict))
test_errors.append(mean_squared_error(y_test, y_test_predict))
print(test_errors)
print(train_errors)

# %%
from matplotlib import pyplot as plt
import matplotlib
plt.plot(X_train, poly_2_reg.predict(X_poly), "r.")
plt.plot(X_train, y_train, ".")
# the red dots are actual values, blue are predicted

# %%
from sklearn.preprocessing import PolynomialFeatures
poly_features_3 = PolynomialFeatures(degree=3, include_bias=False)
X_poly = poly_features_3.fit_transform(X_train)
poly_3_reg = LinearRegression()
poly_3_reg.fit(X_poly, y_train)
print(X[0], X_poly[0])
print(poly_3_reg.intercept_, poly_3_reg.coef_)
print(poly_3_reg.predict(poly_features_3.fit_transform([[0],[2]])))
print(poly_3_reg.coef_[0][1] * 2**2 + poly_3_reg.coef_[0][0] * 2+ poly_3_reg.intercept_[0])

# %%
#mse poly3
X_poly_test = poly_features_3.fit_transform(X_test)
y_train_predict = poly_3_reg.predict(X_poly)
y_test_predict = poly_3_reg.predict(X_poly_test)
train_errors.append(mean_squared_error(y_train, y_train_predict))
test_errors.append(mean_squared_error(y_test, y_test_predict))
print(test_errors)
print(train_errors)

# %%
from matplotlib import pyplot as plt
import matplotlib
plt.plot(X_train, poly_3_reg.predict(X_poly), "r.")
plt.plot(X_train, y_train, ".")
# the red dots are actual values, blue are predicted

# %%
from sklearn.preprocessing import PolynomialFeatures
poly_features_4 = PolynomialFeatures(degree=4, include_bias=False)
X_poly = poly_features_4.fit_transform(X_train)
poly_4_reg = LinearRegression()
poly_4_reg.fit(X_poly, y_train)
print(X[0], X_poly[0])
print(poly_4_reg.intercept_, poly_4_reg.coef_)
print(poly_4_reg.predict(poly_features_4.fit_transform([[0],[2]])))
print(poly_4_reg.coef_[0][1] * 2**2 + poly_4_reg.coef_[0][0] * 2+ poly_4_reg.intercept_[0])

# %%
#mse poly4
X_poly_test = poly_features_4.fit_transform(X_test)
y_train_predict = poly_4_reg.predict(X_poly)
y_test_predict = poly_4_reg.predict(X_poly_test)
train_errors.append(mean_squared_error(y_train, y_train_predict))
test_errors.append(mean_squared_error(y_test, y_test_predict))
print(test_errors)
print(train_errors)

# %%
from matplotlib import pyplot as plt
import matplotlib
plt.plot(X_train, poly_4_reg.predict(X_poly), "r.")
plt.plot(X_train, y_train, ".")
# the red dots are actual values, blue are predicted

# %%
from sklearn.preprocessing import PolynomialFeatures
poly_features_5 = PolynomialFeatures(degree=5, include_bias=False)
X_poly = poly_features_5.fit_transform(X_train)
poly_5_reg = LinearRegression()
poly_5_reg.fit(X_poly, y_train)
print(X[0], X_poly[0])
print(poly_5_reg.intercept_, poly_5_reg.coef_)
print(poly_5_reg.predict(poly_features_5.fit_transform([[0],[2]])))
print(poly_5_reg.coef_[0][1] * 2**2 + poly_5_reg.coef_[0][0] * 2+ poly_5_reg.intercept_[0])

# %%
#mse poly5
X_poly_test = poly_features_5.fit_transform(X_test)
y_train_predict = poly_5_reg.predict(X_poly)
y_test_predict = poly_5_reg.predict(X_poly_test)
train_errors.append(mean_squared_error(y_train, y_train_predict))
test_errors.append(mean_squared_error(y_test, y_test_predict))
print(test_errors)
print(train_errors)

# %%
from matplotlib import pyplot as plt
import matplotlib
plt.plot(X_train, poly_5_reg.predict(X_poly), "r.")
plt.plot(X_train, y_train, ".")
# the red dots are actual values, blue are predicted

# %%
mse = pd.DataFrame({"train_mse": train_errors, "test_mse": test_errors})
import pickle
with open('mse.pkl', 'wb') as f_acc:
    pickle.dump(mse, f_acc)

# %%
reg = [(lin_reg, None), (knn_3_reg, None), (knn_5_reg, None), (poly_2_reg,poly_features_2), (poly_3_reg, poly_features_3), (poly_4_reg, poly_features_4),(poly_5_reg, poly_features_5)]
with open('reg.pkl', 'wb') as f_acc:
    pickle.dump(mse, f_acc)

# %%
mse = mse.rename(index={0 : "lin_reg", 1 : "knn_3_reg", 2 : "knn_5_reg", 3 : "poly_2_reg", 4 : "poly_3_reg", 5 : "poly_4_reg", 6 : "poly_5_reg"})
mse

# %%



