import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = sns.load_dataset('iris')

print("Dataset Shape:", df.shape)

print("Columns:", df.columns.tolist())

df.head()

print(df.info())

df.describe()

plt.figure(figsize=(8, 5))
sns.scatterplot(data=df, x='sepal_length', y='sepal_width', hue='species')
plt.title('Sepal Length vs Sepal Width')
plt.show()

df.hist(figsize=(10, 8), bins=20, edgecolor='black')
plt.suptitle('Feature Distributions', y=0.95)
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='species', y='petal_length')
plt.title('Petal Length Distribution by Species')
plt.show()
