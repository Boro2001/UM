import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import os
import tarfile
import urllib.request
import gzip
import numpy as np
import shutil
import pickle
#zbior danych
urllib.request.urlretrieve("https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.tgz", "housing.tgz")
file = tarfile.open('housing.tgz')
file.extractall()
file.close()
with open('housing.csv', 'rb') as f_in, gzip.open('housing.csv.gz', 'wb') as f_out:
    shutil.copyfileobj(f_in, f_out)
#Informacje o zbiorze danych
df = pd.read_csv('housing.csv.gz')
df.head()
df.info()
df['ocean_proximity'].dtypes
df['ocean_proximity'].describe()
df['ocean_proximity'].value_counts()
#Wizualizacja
df.hist(bins=50, figsize=(20,15))
plt.savefig('obraz1.png')
df.plot(kind="scatter", x="longitude", y="latitude",
        alpha=0.1, figsize=(7,4))
plt.savefig('obraz2.png')
df.plot(kind="scatter", x="longitude", y="latitude",
        alpha=0.4, figsize=(7,3), colorbar=True,
        s=df["population"]/100, label="population", 
        c="median_house_value", cmap=plt.get_cmap("jet"))
plt.savefig('obraz3.png')
#Analiza
korelacja = df.corr()["median_house_value"].sort_values(ascending=False)
korelacja.reset_index()\
.rename(columns=({"index":"atrybut", "median_house_value":"wspolczynnik_korelacji"})).\
to_csv("korelacja.csv")
import seaborn as sns
sns.pairplot(df)
#Przygotowanie do uczenia
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(df,
                                       test_size=0.2,
                                       random_state=42)
len(train_set),len(test_set)
pickle.dump(train_set, open('train_set.pkl', 'wb'))
pickle.dump(test_set, open('test_set.pkl', 'wb'))
