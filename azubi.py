# Importing required libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Task 1: Load and Explore the Dataset
# Load the Iris dataset from sklearn
from sklearn.datasets import load_iris

# Load the data into a pandas DataFrame
iris_data = load_iris()
df = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)

# Add the target column for species
df['species'] = iris_data.target
df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})

# Display the first few rows of the dataset
print(df.head())

# Explore the structure of the dataset
print("\nData Types:")
print(df.dtypes)

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Task 2: Basic Data Analysis
# Compute basic statistics for numerical columns
print("\nBasic Statistics:")
print(df.describe())

# Group by species and compute the mean for numerical columns
grouped_data = df.groupby('species').mean()
print("\nMean by Species:")
print(grouped_data)

# Task 3: Data Visualization
# Set the style of the plots using seaborn
sns.set(style="whitegrid")

# Line chart: Trend of sepal length for each species
plt.figure(figsize=(10, 6))
sns.lineplot(x="species", y="sepal length (cm)", data=df, marker="o")
plt.title('Sepal Length Trend by Species')
plt.xlabel('Species')
plt.ylabel('Sepal Length (cm)')
plt.show()

# Bar chart: Average petal length per species
plt.figure(figsize=(10, 6))
sns.barplot(x="species", y="petal length (cm)", data=df)
plt.title('Average Petal Length by Species')
plt.xlabel('Species')
plt.ylabel('Petal Length (cm)')
plt.show()

# Histogram: Distribution of sepal width
plt.figure(figsize=(10, 6))
sns.histplot(df['sepal width (cm)'], bins=20, kde=True)
plt.title('Distribution of Sepal Width')
plt.xlabel('Sepal Width (cm)')
plt.ylabel('Frequency')
plt.show()

# Scatter plot: Sepal length vs Petal length
plt.figure(figsize=(10, 6))
sns.scatterplot(x="sepal length (cm)", y="petal length (cm)", hue="species", data=df)
plt.title('Sepal Length vs Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend(title='Species')
plt.show()

# Additional Findings or Observations
# Here you can include any patterns or observations from the data analysis and visualizations
print("\nObservations:")
print("1. There is a significant difference in petal length between species.")
print("2. Setosa has the smallest average sepal and petal length.")
print("3. The distribution of sepal width is relatively uniform.")
