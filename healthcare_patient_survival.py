import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
df = pd.read_csv("healthcare_patient_survival.csv").head(30)
print(df)
# Introduce some null values for demonstration
df.loc[np.random.choice(df.index, size=20, replace=False), 'Age'] = np.nan

df.loc[np.random.choice(df.index, size=15, replace=False), 'Hospital_Stay_Days'] = np.nan

print(df)

# Check for null values
print("Missing Values:\n", df.isnull().sum())


# Handling missing values
# Fill missing values with mean/median/mode
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Hospital_Stay_Days'].fillna(df['Hospital_Stay_Days'].mean(), inplace=True)

print(df)

# Drop rows with missing values
df.dropna(inplace=True)
print(df)
# Convert categorical variables to numerical
categorical_cols = ['Gender', 'Medical_Condition', 'Treatment_Type']
df1 = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
print(df1)

# Display basic info
print(df.info())
print(df.describe())

# Exploratory Data Analysis
sns.histplot(df['Age'], bins=20, kde=True)
plt.title("Age Distribution")
plt.show()

sns.boxplot(x='Survival_Status', y='Age', data=df)
plt.title("Age vs Survival Status")
plt.show()

# Additional visualizations
sns.violinplot(x='Survival_Status', y='Age', data=df)
plt.title("Age Distribution by Survival Status")
plt.show()
