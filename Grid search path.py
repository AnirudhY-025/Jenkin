from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
import pandas as pd

# Load your dataset (update the file path)
file_path = r'/content/SILKYSKY_DATA_CW2.csv'
df = pd.read_csv(file_path, encoding='ISO-8859-1')

# 1. Encode categorical columns
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['Satisfied'] = label_encoder.fit_transform(df['Satisfied'])
df['Age Band'] = label_encoder.fit_transform(df['Age Band'])
df['Type of Travel'] = label_encoder.fit_transform(df['Type of Travel'])
df['Class'] = label_encoder.fit_transform(df['Class'])
df['Destination'] = label_encoder.fit_transform(df['Destination'])
df['Continent'] = label_encoder.fit_transform(df['Continent'])

# 2. Handle missing values for 'Arrival Delay in Minutes'
imputer = SimpleImputer(strategy='mean')
df['Arrival Delay in Minutes'] = imputer.fit_transform(df[['Arrival Delay in Minutes']])

# 3. Split dataset into features (X) and target (y)
X = df.drop(columns=['Satisfied', 'Ref', 'id'])  # Drop the target and irrelevant columns
y = df['Satisfied']

# 4. Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 5. Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 6. Define hyperparameters to tune for XGBoost
param_grid = {
'learning_rate': [0.01, 0.05, 0.1, 0.2],  # Additional intermediate learning rates
    'max_depth': [4, 6, 8],  # Increased depth for better capturing of complexities
    'n_estimators': [150, 250, 350],  # Higher range of estimators
    'subsample': [0.75, 0.85, 1.0],  # Expanded range to prevent overfitting
    'colsample_bytree': [0.75, 0.85, 1.0],  # Similar range expansion for regularization
    'gamma': [0, 0.1, 0.2, 0.3]  # Refined range for flexibility in penalizing leaves
}

# Re-define the GridSearchCV object
grid_search_xgb = GridSearchCV(xgb, param_grid, cv=5, scoring='accuracy', n_jobs=-1)

# Fit the grid search to the training data
grid_search_xgb.fit(X_train_scaled, y_train)

# Get the best model and evaluate
best_xgb = grid_search_xgb.best_estimator_
y_pred_best = best_xgb.predict(X_test_scaled)
best_accuracy = accuracy_score(y_test, y_pred_best)

print("Best Parameters:", grid_search_xgb.best_params_)
print("Best Accuracy:", best_accuracy)