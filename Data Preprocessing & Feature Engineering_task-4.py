print("Data Preprocessing & Feature Engineering")
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Sample dataset
data = {
    'Age': [25, np.nan, 30, 35, np.nan, 40, 45],
    'Salary': [50000, 54000, np.nan, 60000, 62000, 65000, np.nan],
    'City': ['Delhi', 'Mumbai', 'Kolkata', 'Delhi', 'Kolkata', 'Mumbai', 'Delhi'],
    'Education': ['Graduate', 'Postgraduate', 'Graduate', 'Doctorate', 'Graduate', 'Postgraduate', np.nan]
}

df = pd.DataFrame(data)
print("Original Dataset:\n", df)
# Impute missing numerical values with Mean
df['Age'].fillna(df['Age'].mean(), inplace=True)
df['Salary'].fillna(df['Salary'].median(), inplace=True)

# Impute missing categorical values with Mode
df['Education'].fillna(df['Education'].mode()[0], inplace=True)

print("\nDataset after Handling Missing Data:\n", df)
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
print("use of encoders")
# One-Hot Encoding for 'City'
df = pd.get_dummies(df, columns=['City'])


# Ordinal Encoding for 'Education'
education_order = [['Graduate', 'Postgraduate', 'Doctorate']]
ordinal_encoder = OrdinalEncoder(categories=education_order)
df['Education'] = ordinal_encoder.fit_transform(df[['Education']])

print("\nFinal Encoded Dataset:\n", df)
# Sample data
data = {'Feature1': [10, 20, 30, 40, 50], 'Feature2': [5, 15, 25, 35, 45]}
df = pd.DataFrame(data)
print("The data after Standard scalar")
# Apply StandardScaler
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# Convert back to DataFrame
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
print(df_scaled)
print("Use of MinMaxScaler")
from sklearn.preprocessing import MinMaxScaler

# Apply MinMaxScaler
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df)

# Convert back to DataFrame
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
print(df_scaled)
print("use of robustScaler")
from sklearn.preprocessing import RobustScaler

# Apply RobustScaler
scaler = RobustScaler()
df_scaled = scaler.fit_transform(df)

# Convert back to DataFrame
df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
print(df_scaled)

print("outlier Detection and Handling")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

# Generate sample data with outliers
np.random.seed(42)
data = {
    'Feature1': np.append(np.random.normal(50, 10, 50), [120, 130, 140]),  # Adding outliers
}

df = pd.DataFrame(data)

# Visualizing with a boxplot before outlier removal
plt.figure(figsize=(8, 5))
sns.boxplot(y=df['Feature1'], color='lightblue')
plt.title("Boxplot Before Removing Outliers")
plt.show()
# Compute Z-scores
df['Z-Score'] = zscore(df['Feature1'])

# Filter out outliers using Z-score (|Z| > 3)
df_zscore_cleaned = df[abs(df['Z-Score']) < 3]

# Plot after removing outliers
plt.figure(figsize=(8, 5))
sns.boxplot(y=df_zscore_cleaned['Feature1'], color='lightgreen')
plt.title("Boxplot After Removing Outliers (Z-Score Method)")
plt.show()
# Compute Q1, Q3, and IQR
Q1 = df['Feature1'].quantile(0.25)
Q3 = df['Feature1'].quantile(0.75)
IQR = Q3 - Q1

# Define bounds
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

# Filter out outliers
df_iqr_cleaned = df[(df['Feature1'] >= lower_bound) & (df['Feature1'] <= upper_bound)]

# Plot after removing outliers
plt.figure(figsize=(8, 5))
sns.boxplot(y=df_iqr_cleaned['Feature1'], color='lightcoral')
plt.title("Boxplot After Removing Outliers (IQR Method)")
plt.show()
plt.figure(figsize=(10, 5))

# Scatter plot before removing outliers
plt.subplot(1, 2, 1)
sns.scatterplot(data=df, x=np.arange(len(df)), y="Feature1", color='red')
plt.title("Scatter Plot Before Removing Outliers")

# Scatter plot after removing outliers using IQR
plt.subplot(1, 2, 2)
sns.scatterplot(data=df_iqr_cleaned, x=np.arange(len(df_iqr_cleaned)), y="Feature1", color='green')
plt.title("Scatter Plot After Removing Outliers (IQR)")

plt.tight_layout()
plt.show()

print("Dimensionality Reduction (PCA, t-SNE, LDA for feature selection)")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_iris

# Load dataset
iris = load_iris()
X = iris.data
y = iris.target

# Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=2)  # Reduce to 2 dimensions
X_pca = pca.fit_transform(X_scaled)

# Plot PCA result
plt.figure(figsize=(8, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', alpha=0.7)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA: Iris Dataset')
plt.colorbar(label='Target Classes')
plt.show()


from sklearn.manifold import TSNE

# Apply t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# Plot t-SNE result
plt.figure(figsize=(8, 5))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='plasma', edgecolor='k', alpha=0.7)
plt.xlabel('t-SNE Component 1')
plt.ylabel('t-SNE Component 2')
plt.title('t-SNE: Iris Dataset')
plt.colorbar(label='Target Classes')
plt.show()
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# Apply LDA
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X_scaled, y)

# Plot LDA result
plt.figure(figsize=(8, 5))
plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y, cmap='coolwarm', edgecolor='k', alpha=0.7)
plt.xlabel('LDA Component 1')
plt.ylabel('LDA Component 2')
plt.title('LDA: Iris Dataset')
plt.colorbar(label='Target Classes')
plt.show()



