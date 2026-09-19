import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('netflix_titles.csv')
print(df.head())
# print(df.info())
# print(df.describe())
# print(df.shape)

# CLEANING
df['duration'] = (df['duration'].str.split(' ').str[0]).astype('Int64')
# print(df['duration'])
df.loc[df['type'] == 'Movie', 'duration_min'] = df['duration']
df.loc[df['type'] == 'TV Show', 'duration_season'] = df['duration']

df = df.dropna(subset=['date_added', 'rating', 'duration'])
df['director'] = df['director'].fillna('Unknown')
df['cast'] = df['cast'].fillna('Not Specified')
df['country'] = df['country'].fillna('Not Specified')
# print(df.isnull().sum())

# DUPLICATES
# print(df.duplicated().sum())

# PLOTTING
sns.countplot(df['type'])
plt.show()
sns.countplot(df['rating'])
plt.show()
sns.histplot(df['release_year'])
plt.show()
pivot = pd.pivot_table(df, values='title',index='release_year', columns='type',aggfunc='count')
sns.lineplot(data=pivot)
plt.show()

# ANALYSIS
# print(df['country'].value_counts())
df['country'] = (df['country'].str.split(',')).explode('country')
print(df['country'].value_counts())

exploded_listed = (df['listed_in'].str.split(',')).explode()
print(exploded_listed.value_counts())

pivot = pd.pivot_table(df, values='title', index='rating', columns='type', aggfunc='count')
print(pivot)