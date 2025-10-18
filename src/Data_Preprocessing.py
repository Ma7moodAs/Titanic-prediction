import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer

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

# Feature engineering 
df['Family_size'] = df[['Parch','SibSp']].sum(axis=1) + 1
df['Alone'] = (df['Family_size'] == 1).astype('int64')

# Encoding Categorical variables 
df['Sex'] = df['Sex'].map({'male':1,'female':0})
onehotencoder = OneHotEncoder(drop='first',sparse_output=False)
Embarked_encoded = onehotencoder.fit_transform(df[['Embarked']])
encoded_cols = onehotencoder.get_feature_names_out(['Embarked'])
embarked_df = pd.DataFrame(Embarked_encoded,columns=encoded_cols)
df = pd.concat([df,embarked_df],axis=1)
df.drop(columns=['Embarked'],inplace=True,errors='ignore')

# Scaling numerical features (Age,Fare)
scaler = StandardScaler()
df[['Age','Fare']] = scaler.fit_transform(df[['Age','Fare']])

# Dropping unnecessary columns
df = df.drop(columns=['PassengerId','Name','Age_groups','Ticket'],errors='ignore')

# Save the cleaend df
df.to_csv(r'C:\Users\User\ML_Projects\data\cleaned_df.csv',index=False)
print('Titanic data cleaned and save successfully')

