import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Loading the dataset
url = "C:/Users/Tarush Tarang/Desktop/projects/sports analytics/nba_2022_2023.csv"
data = pd.read_csv(url)

# Display the first few rows of the dataset
print(data.head())

# Basic data cleaning
data.dropna(inplace=True)  # Remove missing values
data = data[data['MinutesPlayed'] > 0]  # Filter out players with zero minutes played

# Selecting relevant columns
select_columns = ['Player', 'Team', 'GamesPlayed', 'MinutesPlayed', 'Points', 'Assists', 'Rebounds', 'Steals', 'Blocks']
data = data[select_columns]

print(data.info())

# Summary statistics
print(data.describe())

# Top 10 players by points
top_scorers = data.sort_values(by='Points', ascending=False).head(10)
print(top_scorers[['Player', 'Points']])

# Exclude non-numeric columns
numeric_data = data.select_dtypes(include=[np.number])

# Correlation matrix
correlation_matrix = numeric_data.corr()
print(correlation_matrix)

# Top 10 players graph by points
plt.figure(figsize=(10, 6))
plt.barh(top_scorers['Player'], top_scorers['Points'], color='skyblue')
plt.xlabel('Points')
plt.title('Top 10 NBA Players by Points')
plt.gca().invert_yaxis()
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Player Performance Metrics')
plt.show()

# Clustering players
features = data[['Points', 'Assists', 'Rebounds', 'Steals', 'Blocks']]
kmeans = KMeans(n_clusters=3, n_init=10)
data['Cluster'] = kmeans.fit_predict(features)

# Visualize clusters
plt.figure(figsize=(10, 6))
sns.scatterplot(data=data, x='Points', y='Assists', hue='Cluster', palette='viridis')
plt.title('Player Clusters Based on Performance Metrics')
plt.show()