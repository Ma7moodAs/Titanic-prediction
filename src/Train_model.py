import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import joblib

# Loading the cleaned data frame
df = pd.read_csv(r'C:\Users\User\ML_Projects\data\cleaned_df.csv') 
# Separating features and target
X = df.drop(columns='Survived',errors='ignore')
y = df['Survived']

# Splitting into train and test data frames
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
print(f'Training Data shape is: {X_train.shape}')
print(f'Testing Data shape is: {X_test.shape}')
print(y_train.shape)
print(y_test.shape)

# Model training:

# Logistic regression
LR_model = LogisticRegression(max_iter=1000,random_state=42)
LR_model.fit(X_train,y_train)
joblib.dump(LR_model,r'C:\Users\User\ML_Projects\models\Logistic_regression.pkl')
print('Logistic regression model has been trained and saved successfully')

# Random forest
RF_model = RandomForestClassifier(random_state=42)
RF_model.fit(X_train,y_train)
joblib.dump(RF_model,r'C:\Users\User\ML_Projects\models\Random_Forest.pkl')
print('Random Forest model has been trained and saved successfully')
