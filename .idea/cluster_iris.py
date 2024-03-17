import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()
iris_df = pd.DataFrame(iris.data, columns = iris.feature_names)
print(iris_df.head())
print(iris_df.shape)
print(iris_df.isnull().sum())
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

iris_scaled = scaler.fit_transform(iris_df)

iris_df_scaled = pd.DataFrame(iris_scaled, columns = iris.feature_names)

iris_df_scaled.round(2).head()
X = iris_df_scaled
print(X.head())
from sklearn.cluster import KMeans

wcss = []


for i in range(1, 11):


    kmeans = KMeans(n_clusters = i, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 42)

    kmeans.fit(X)

    wcss.append(kmeans.inertia_)


plt.figure(figsize = (10,6))

plt.plot(range(1, 11), wcss)

plt.title('Выбор количества кластеров методом локтя')
plt.xlabel('Количество кластеров')
plt.ylabel('WCSS')
kmeans = KMeans(n_clusters = 3, init = 'k-means++', max_iter = 300, n_init = 10, random_state = 42)

y_pred = kmeans.fit_predict(X)

kmeans.inertia_
print(iris.target)
print(y_pred)
for i in range(len(y_pred)):
  # если было 1, заменим на 0
  if y_pred[i] == 1:
    y_pred[i] = 0
  # если было 0, будет 1
  elif y_pred[i] == 0:
    y_pred[i] = 2
  # если было 2, будет 1
  elif y_pred[i] == 2:
    y_pred[i] = 1
print(y_pred)
d = {'Target' : iris.target, 'Prediction' : y_pred}

result = pd.DataFrame(d, columns = ['Target', 'Prediction'])
print(result.head(2))
comparison = np.where(result['Target'] == result['Prediction'], True, False)

print(type(comparison))
print(comparison[:5])
result['Comparison'] = comparison
print(result.head())
print(result['Comparison'].value_counts(normalize = True).round(2))
plt.figure(figsize = (10,6))

plt.scatter(X.iloc[:,0], X.iloc[:,1], c = iris.target, cmap = 'Paired')
plt.figure(figsize = (10,6))


plt.scatter(X.iloc[:,0], X.iloc[:,1], c = y_pred, cmap='Paired')


plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = 150, c = 'red', marker = '^', label = 'Centroids')


plt.legend(loc = 'upper right')
plt.show()