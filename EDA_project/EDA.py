import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns 
import os 

print("Understanding the dataset")

file_name='sales_data.csv'
if not os.path.exists(file_name):
    print(f"Error: {file_name} is not found")
    exit()

# Load the dataset
df=pd.read_csv("sales_data.csv")
print("Successfully loaded!")
print(f"Shape of the dataset: Rows:{df.shape[0]} , Columns:{df.shape[1]}")

print(f"\n First 5 rows of the data:\n {df.head()}")
print(f"\n Last 5 rows of the data:\n {df.tail()}")
print(f"\n Data description: \n{df.describe()}")

print("\n Handling the missing values:")
print(df.isnull().sum())

# With using Median
median_age = df['Age'].median()
df['Age'] = df['Age'].fillna(median_age)
print(median_age)

median_Spending = df['Spending'].median()
df['Spending'] = df['Spending'].fillna(median_Spending)
print(median_Spending)

# Using Mean

mean_Spending = df['Spending'].mean()
df['Spending'] = df['Spending'].fillna(mean_Spending)
print(mean_Spending)

mean_age = df['Age'].mean()
df['Age'] = df['Age'].fillna(mean_age)
print(mean_age)

# Distribution Analysis

plt.figure(figsize=(7,4))
df['Spending'].hist(bins=10,color='skyblue',edgecolor='black')
plt.title('Distribution of Spending')
plt.xlabel('Speding Amount')
plt.ylabel('Number of Customers')
plt.show()

# Correlation matrix

correlation = df.corr(numeric_only=True)
print(correlation)

print("Plotting Correlation Heatmap")
plt.figure(figsize=(7,4))
sns.heatmap(correlation,annot=True,cmap='coolwarm',fmt=".2f")
plt.title("Correlation Heatmap")
plt.show()

#Outlier detection
plt.figure(figsize=(7,4))
sns.boxplot(x=df['Age'],color='lightgreen')
plt.title("Boxplot of Customer age")
plt.xlabel("Age")
plt.show()

print("Find the Outliers of Age:")
outliers=df[df['Age']>100]
print("Found Outliers")
print(outliers)

#Detect Outliers using IQR
df[df['Age']>100]
Q1 = df['Age'].quantile(0.25)
Q3 = df['Age'].quantile(0.75)

IQR = Q3 - Q1

lower = Q1 - 1.5*IQR
upper = Q3 + 1.5*IQR

outliers = df[(df['Age']<lower) | (df['Age']>upper)]

print(outliers)

#Duplicate Detection
print("Duplicate Rows:")
print(df.duplicated().sum())

df.drop_duplicates(inplace=True)


#Unique Values
print(df['City'].unique())

print(df['City'].value_counts())


#Top Spending Customers
top = df.sort_values(by="Spending", ascending=False)
print(top.head(10))

#Bottom Spending Customers
low = df.sort_values(by="Spending")
print(low.head(10))


#Summary Report
print("========== SUMMARY ==========")

print("Total Customers :",len(df))

print("Average Age :",df['Age'].mean())

print("Average Spending :",df['Spending'].mean())

print("Highest Spending :",df['Spending'].max())

print("Lowest Spending :",df['Spending'].min())

print("Most Common City :",
      df['City'].mode()[0])