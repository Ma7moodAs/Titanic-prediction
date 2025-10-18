import joblib
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from Train_model import X_test,y_test
print('Data has been imported successfully')

model_1 = joblib.load(r'C:\Users\User\ML_Projects\models\Logistic_regression.pkl')
y_prediction = model_1.predict(X_test)
print('Logistic regression Model Evaluation results:\n')
print('Accuracy:',accuracy_score(y_test,y_prediction))
print('\nConfusion Matrix:\n',confusion_matrix(y_test,y_prediction))
print('\nClassification Report:\n',classification_report(y_test,y_prediction))

model_2 = joblib.load(r'C:\Users\User\ML_Projects\models\Random_Forest.pkl')
y_pred = model_2.predict(X_test)
print('Random Forest Model Evaluation results:\n')
print('Accuracy:',accuracy_score(y_test,y_pred))
print('\nConfusion Matrix:\n',confusion_matrix(y_test,y_pred))
print('\nClassification Report:\n',classification_report(y_test,y_pred))