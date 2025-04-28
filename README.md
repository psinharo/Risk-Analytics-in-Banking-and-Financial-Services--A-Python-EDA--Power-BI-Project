# Risk Analytics in Banking and Financial Services- A Python EDA and Power-BI-Project

 # Re-import necessary libraries since the execution state was reset
 # END GOAL: RISK ANALYSIS and SHOW the IMP KPIS
 import pandas as pd
 import matplotlib.pyplot as plt
 import seaborn as sns
 import numpy as np
 #banking EDA
 file_path = r"C:\Users\priya\Downloads\Banking.csv"
 df = pd.read_csv(file_path)
 print(df.head())
 df.info()
 #Generating descriptive statistics for the dataframe
 print(df.describe())
 #Estimated Income(converting a numerical coloumn into categorical for 
segmentation )
 bins=[0,100000,300000,float('inf')]
 labels=['low','med','high']
 df['Income Band']=pd.cut(df['Estimated 
Income'],bins=bins,labels=labels,right=False)
 df['Income Band'].value_counts().sort_index().plot(kind='bar', color='skyblue')
 # Show the plot
 plt.title('Estimated Income Bands')
 plt.xlabel('Income Band')
 plt.ylabel('Count')
 plt.show()
 #BI-VARIATE ANALYSIS(w.r.t gender,Nationality)
 #Examine  the of unique categories in categorical coloumns(non numerical cols)
 categorical_cols=df[["BRId","GenderId","IAId","Amount of Credit Cards","Risk 
Weighting","Nationality","Occupation","Fee Structure","Loyalty 
Classification","Properties Owned","Income Band"]].columns
 for col in categorical_cols:
 continue
 # if col in ["Client ID","Name","Joined Bank"]:
 #   print(f"\nValue Counts for '{col}':")
 print(df[col].value_counts())
 for predictor in categorical_cols:
 plt.figure(figsize=(8, 5))
 sns.countplot(data=df, x=predictor, hue='Nationality')# here hue is for
 bi-variate analysis ,gender/nationality  is used and hue is like 
groupping the data w.r.t gender/nationality
 plt.title(f"Count of {predictor} by Nationality")
 plt.xticks(rotation=45)
 plt.tight_layout()
 plt.show()
 #Histogram of value counts for different occupation
 plt.figure(figsize=(10, 6))
 sns.countplot(data=df, x='Occupation', order=df['Occupation'].value_counts
 ().index, color='skyblue')
 plt.title('Histogram of Occupation Count')
 plt.xlabel('Occupation')
 plt.ylabel('Count')
 plt.xticks(rotation=45)
 plt.tight_layout()
 plt.show()
 #Numerical Analysis
 # Numerical analysis and exploration
 numerical_cols = ['Fee Structure','Age', 'Estimated Income', 'Superannuation 
Savings', 'Credit Card Balance', 'Bank Loans', 'Bank Deposits', 'Checking 
Accounts', 'Saving Accounts', 'Foreign Currency Account', 'Business Lending']
 # Univariate analysis and visualization
 plt.figure(figsize=(15, 10))
 for i, col in enumerate(numerical_cols):
 plt.subplot(4, 3, i + 1)
 sns.histplot(df[col], kde=True)
 plt.title(col)
 plt.tight_layout()
 plt.show()
 #Heatmaps
 # Select numerical columns for correlation analysis
 numerical_cols = ['Age', 'Estimated Income', 'Superannuation Savings', 'Credit 
Card Balance',
 'Bank Loans', 'Bank Deposits', 'Checking Accounts', 'Saving 
Accounts', 'Foreign Currency Account', 'Business Lending', 'Properties 
Owned']
 # Calculate the correlation matrix
 correlation_matrix = df[numerical_cols].corr()
 # Create a heatmap of the correlation matrix
 plt.figure(figsize=(12, 10))
 sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
 plt.title('Correlation Matrix of Numerical Features')
 plt.show()
 #Further EDA on different pairs
 pairs_to_plot = [
 ('Bank Deposits', 'Saving Accounts'),
 ('Checking Accounts', 'Saving Accounts'),
 ('Checking Accounts', 'Foreign Currency Account'),
 ('Age', 'Superannuation Savings'),
 ('Estimated Income', 'Checking Accounts'),
 ('Bank Loans', 'Credit Card Balance'),
 ('Business Lending', 'Bank Loans'),
 ]
 for x_col, y_col in pairs_to_plot:
 plt.figure(figsize=(8, 6))
 sns.regplot(
 data=df,
 x=x_col,
 y=y_col,
 scatter_kws={'alpha': 0.4},     
line_kws={'color': 'red'}       
)
 # semi-transparent points
 # best-fit line color
 plt.title(f'Relationship between {x_col} and {y_col}', fontsize=14)
 plt.xlabel(x_col, fontsize=12)
 plt.ylabel(y_col, fontsize=12)
 plt.tight_layout()
 plt.show()
 #Risk Weighting Distribution & Profile
 #Goal: Understand how customers are distributed across different risk levels.
 # Risk Weighting counts
 plt.figure(figsize=(8, 5))
 sns.countplot(data=df, x='Risk Weighting', palette='viridis')
 plt.title('Distribution of Risk Weighting')
 plt.xlabel('Risk Weighting Level')
 plt.ylabel('Customer Count')
 plt.show()
 # Boxplots of Financial Metrics by Risk Category
 #Goal: Explore how various financial indicators differ across risk groups.
 financial_vars = ['Estimated Income', 'Superannuation Savings', 'Credit Card 
Balance', 'Bank Loans', 'Bank Deposits']
 for var in financial_vars:
 plt.figure(figsize=(8, 5))
 sns.boxplot(data=df, x='Risk Weighting', y=var)
 plt.title(f'{var} by Risk Weighting')
 plt.xlabel('Risk Weighting')
 plt.ylabel(var)
 plt.show()
 #Time-Based Insight – Join Date Analysis
 #Goal: See how customer tenure relates to risk and financial activity.
 # Convert 'Joined Bank' to datetime
 df['Joined Bank'] = pd.to_datetime(df['Joined Bank'], dayfirst=True)
 # Extract year of joining
 df['Join Year'] = df['Joined Bank'].dt.year
 plt.figure(figsize=(10, 5))
 sns.countplot(data=df, x='Join Year', palette='mako')
 plt.title('Customer Join Year Distribution')
 plt.xticks(rotation=45)
 plt.show()
 # Credit Card vs Loan Risk Interaction
 #Goal: See if people with more credit cards and higher balances tend to have 
higher risk.
 plt.figure(figsize=(8, 6))
 sns.scatterplot(data=df, x='Amount of Credit Cards', y='Credit Card Balance', 
hue='Risk Weighting', palette='coolwarm')
 plt.title('Credit Cards vs Balance Colored by Risk')
 plt.show()
 #Income Band vs Risk Profile
 #Goal: You've already created Income Band, now map it against risk.
 plt.figure(figsize=(8, 5))
 sns.countplot(data=df, x='Income Band', hue='Risk Weighting', 
palette='Spectral')
 plt.title('Risk Weighting by Income Band')
 plt.show()
 # Properties Owned vs Risk
 #Goal: Investigate if property ownership reduces risk.
 sns.barplot(data=df, x='Properties Owned', y='Risk Weighting', ci=None, 
palette='Set2')
 plt.title('Average Risk Weighting by Number of Properties Owned')
 plt.show()
 #Insights
 #Banking Deposits has a very high co relation with Savings Accounts, Checking 
Accounts and Foreign Currency Account indicating that 
#customers who maintain high balance in one account type often hold substantial
 amount/funds across other accounts as well.
 #Credit usage patters vary independently so far.
 #Clustering Analysis using Kmeans Machine Learning
 from sklearn.preprocessing import StandardScaler
 from sklearn.cluster import KMeans
 from sklearn.decomposition import PCA
 # Select numerical features for clustering
 features = [
 'Estimated Income', 'Superannuation Savings', 'Credit Card Balance',
 'Bank Loans', 'Bank Deposits', 'Checking Accounts', 'Saving Accounts',
 'Foreign Currency Account', 'Business Lending', 'Properties Owned'
 ]
 # Standardize the data
 scaler = StandardScaler()
 X_scaled = scaler.fit_transform(df[features])
 # KMeans clustering
 kmeans = KMeans(n_clusters=4, random_state=42)
 df['Cluster'] = kmeans.fit_predict(X_scaled)
 # PCA for visualization
 pca = PCA(n_components=2)
 pca_components = pca.fit_transform(X_scaled)
 df['PCA1'] = pca_components[:, 0]
 df['PCA2'] = pca_components[:, 1]
 # Visualize the clusters
 plt.figure(figsize=(10, 6))
 sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='Cluster', palette='Set2', 
alpha=0.7)
 plt.title('Customer Segments Based on Financial Behavior')
 plt.xlabel('PCA Component 1')
 plt.ylabel('PCA Component 2')
 plt.legend(title='Cluster')
 plt.tight_layout()
 plt.show()
 # Optional: Save clustered data to CSV for Power BI
 df.to_csv("C:/Users/priya/Downloads/Banking_Clustered.csv", index=False)
 # Visual Insight:
 #The PCA plot shows 4 distinct customer segments based on their financial 
behaviors and assets. 
#The clusters are separated clearly, indicating that the customer data has 
meaningful groupings.
 #Cluster Profiles (Summarized):
 #Cluster
 Key Traits
 #0 High income, mid-range deposits/loans, moderate credit use. Risk-balanced 
segment.
 #1 Low income, lowest balances and savings. But owns most properties – maybe 
retired or land-rich?
 #2 High-value customers: High deposits, loans, business lending. Big bank 
players.
 #3 Lowest income, low balances, fewest properties – potentially high-risk or 
early-stage earners.
 # Ideas to Act On:
 #Target Cluster 2 with premium services or financial advice (they’re power 
users).
 #Monitor Cluster 3 closely for risk. They could benefit from financial support 
or education.
 #Cluster 1 may appear low-income but stable due to their property 
ownership.
 #Segmented strategies can be designed for marketing, credit decisions, or advisory services.
