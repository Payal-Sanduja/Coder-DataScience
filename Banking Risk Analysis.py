import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt



df = pd.read_csv("banking_risk_analysis.csv")

print(df)
#using the describe function
print("CHECK THE STATISTICS VALUE USING DESCRIBE FUNCTION")

print(df.describe())

#Checking whether the null is present or not
print(df.isnull().sum())

#Plot the bargraph between the paid and defaulted status using the count plot
sns.countplot(x='Current Loan Status', data=df,color="orange")
plt.title("Loan Status Distribution")
plt.show()

#Plot the histogram to check the frequency among the income  range
plt.hist(df['Income'],color="red",edgecolor='brown',linewidth="5")
plt.title("Income Distribution")
plt.xlabel("Income")
plt.ylabel("Frequency")
plt.show()

#create the boxplot 
sns.boxplot(df['Credit Score'],color="green")
plt.title("Credit Score Box Plot")
plt.show()


#A scatter plot to check the relationship between income and loan amount.
colors=["red"]
plt.scatter(x='Income', y='Loan Amount', data=df,c=colors)
plt.xlabel("Income")
plt.ylabel("Loan Amount")
plt.title("income v/s loan amount")
plt.show()            

#Check how  Age, Income, Credit Score, and Loan Amount relate to each other

sns.pairplot(df[['Age', 'Income', 'Credit Score', 'Loan Amount']])
plt.show()


#check the Branch wise Status
sns.countplot(y='Bank Branch', hue='Current Loan Status', data=df)
plt.title("Loan Status by Bank Branch")
plt.show()
