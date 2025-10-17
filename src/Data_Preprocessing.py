import numpy as np
import pandas as pd
df = pd.read_csv(r'C:\Users\User\ML_Projects\data\Titanic-Dataset.csv')
print(df.head())
print(df.columns)
# Dropping unnecessary columns
df = df.drop(columns=['PassengerId','Cabin','Name'],errors='ignore')

# Identifying Categorical and numerical columns
categorical_cols = df.select_dtypes(exclude=np.number).columns
numerical_cols = df.select_dtypes(include=np.number).columns
print(f'Categorical columns in the DataFrame are {categorical_cols.tolist()}')
print(f'Numerical columns in the DataFrame are {list(numerical_cols)}')

# Data Cleaning, and imputing missing values
for col in df.columns:
    if df[col].isna().sum() > 0:
        if col in numerical_cols:
            df[col] = df[col].fillna(df[col].median())
        elif col in categorical_cols:
            df[col] = df[col].fillna(df[col].mode()[0])
    else:
        pass

df.loc[df['Fare'] == 0,'Fare'] = df.groupby('Pclass')['Fare'].transform('median')

# Handling outliers

# Feature engineering 
df['Age_groups'] = pd.cut(x=df['Age'],bins=4,labels=['Child','Teens','Adults','Seniors'])
df['Family_size'] = df[['Parch','SibSp']].sum(axis=1) + 1
df['Alone'] = (df['Family_size'] == 1).astype('int64')
