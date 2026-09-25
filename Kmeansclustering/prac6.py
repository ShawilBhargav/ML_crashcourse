import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

customers = pd.read_csv('CC GENERAL.csv')
print(customers.head())
print(customers.info())
print(customers.shape)
print(customers.isnull().sum())

customers = customers.drop(columns=['CUST_ID'])
customers['CREDIT_LIMIT'] = customers['CREDIT_LIMIT'].fillna(customers['CREDIT_LIMIT'].median())
customers['MINIMUM_PAYMENTS'] = customers['MINIMUM_PAYMENTS'].fillna(customers['MINIMUM_PAYMENTS'].median())

x = customers[['BALANCE', 'PURCHASES', 'CASH_ADVANCE', 'CREDIT_LIMIT', 'PAYMENTS', 'MINIMUM_PAYMENTS', 'PRC_FULL_PAYMENT']]

scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Elbow method
k_values = range(1,11)
inertia = []
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit_predict(x_scaled)
    inertia.append(kmeans.inertia_)

plt.plot(k_values, inertia)
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

# Silhouette score (helps to select k, as elbow method can be less differentiable in large dataset)
for k in [3,4,5,6]:
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(x_scaled)
    score = silhouette_score(x_scaled, labels)
    print(f"K={k} Silhouette score: {score:.2f}")

kmeans = KMeans(n_clusters=3, random_state=42)
customers['Cluster'] = kmeans.fit_predict(x_scaled)

plt.scatter(customers['BALANCE'], customers['PURCHASES'], c=customers['Cluster'])
plt.show()

pd.set_option('display.max_columns', None)
print(customers.groupby('Cluster').mean())