
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

# 1. Load the dataset
df = pd.read_csv('parkinsons.csv')

# 2. Select features
X = df[['MDVP:Fo(Hz)', 'PPE']]
y = df['status']

# 3. Scale the data
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# 4. Split the data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_split=0.2)

# 5. Choose a model
model = LogisticRegression()
model.fit(X_train, y_train)

# 6. Test the accuracy
y_pred = model.predict(X_test)
print(accuracy_score(y_test, y_pred))

# 7. Save the model
joblib.dump(model, 'my_model.joblib')
