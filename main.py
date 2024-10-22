# Import necessary libraries
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Load the Titanic dataset from Seaborn
titanic = sns.load_dataset('titanic')

# Display the first few rows of the dataset
print(titanic.head())

# Check for missing values
print(titanic.isnull().sum())

# Handle missing values
# For 'age', fill missing values with the median
titanic['age'].fillna(titanic['age'].median(), inplace=True)

# For 'embark_town', fill missing values with the most frequent value (mode)
titanic['embark_town'].fillna(titanic['embark_town'].mode()[0], inplace=True)

# For 'deck', add 'Unknown' category and then fill missing values with 'Unknown'
titanic['deck'] = titanic['deck'].cat.add_categories('Unknown')
titanic['deck'].fillna('Unknown', inplace=True)

# After cleaning, check again for missing values
print(titanic.isnull().sum())

# Basic descriptive statistics
print(titanic.describe())

# Survival rate by gender
survival_by_gender = titanic.groupby('sex')['survived'].mean()
print(survival_by_gender)

# Plot survival rate by gender
sns.barplot(x='sex', y='survived', data=titanic)
plt.title('Survival Rate by Gender')
plt.show()

# Survival rate by passenger class
survival_by_class = titanic.groupby('class')['survived'].mean()
print(survival_by_class)

# Plot survival rate by class
sns.barplot(x='class', y='survived', data=titanic)
plt.title('Survival Rate by Passenger Class')
plt.show()

# Age distribution of passengers
plt.hist(titanic['age'], bins=30, color='skyblue')
plt.title('Age Distribution of Titanic Passengers')
plt.xlabel('Age')
plt.ylabel('Number of Passengers')
plt.show()

# Survival based on age
plt.figure(figsize=(10,6))
sns.histplot(titanic[titanic['survived'] == 1]['age'], bins=30, color='green', label='Survived', kde=False)
sns.histplot(titanic[titanic['survived'] == 0]['age'], bins=30, color='red', label='Did not Survive', kde=False)
plt.legend()
plt.title('Survival based on Age')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()

# Exclude non-numeric columns for the correlation matrix
numeric_columns = titanic.select_dtypes(include=[np.number])

# Correlation between numeric features
plt.figure(figsize=(10,6))
sns.heatmap(numeric_columns.corr(), annot=True, cmap='coolwarm')
plt.title('Feature Correlation Heatmap')
plt.show()

# Save cleaned dataset (Optional)
titanic.to_csv('cleaned_titanic.csv', index=False)
