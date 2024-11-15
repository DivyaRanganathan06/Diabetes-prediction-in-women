#predict the diabetes prossibility in women based on the factors such as BMI, glocose, insulin during pregnancy
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from xgboost import XGBClassifier
import plotly.express as px

# Load the dataset
diabetes_df = pd.read_csv(r'C:\Users\Divya\Downloads\diabetes.csv')

# Show the basic info and description of the dataset
print(diabetes_df.info())
print(diabetes_df.describe())

# Distribution of Glucose vs. Outcome
fig = px.histogram(diabetes_df, x='Glucose', marginal='box', color='Outcome', nbins=47,
                   color_discrete_sequence=['red', 'grey'], title='Distribution of Glucose')
fig.update_layout(bargap=0.1)
fig.show()

# Distribution of BMI vs. Outcome
fig = px.histogram(diabetes_df, x='BMI', marginal='box', color='Outcome', nbins=47,
                   color_discrete_sequence=['red', 'grey'], title='Distribution of BMI')
fig.update_layout(bargap=0.1)
fig.show()

# Count of outcomes (0 = non-diabetic, 1 = diabetic)
print(diabetes_df.Outcome.value_counts())

# Features and target variable
X = diabetes_df.drop('Outcome', axis=1)
y = diabetes_df['Outcome']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build and train the XGBoost classifier model
model = XGBClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Model prediction and accuracy
y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Function to get user input for predictions
def get_user_input():
    features = {}
    for column in X.columns:
        while True:
            try:
                value = float(input(f"Enter {column}: "))
                features[column] = value
                break
            except ValueError:
                print("Please enter a valid number.")
    return pd.DataFrame([features])

# Function to make predictions based on user input
def predict(input_data):
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    probability = model.predict_proba(input_scaled)[0][1]
    return prediction[0], probability

# Main loop to make multiple predictions
while True:
    print("\nEnter patient data:")
    user_input = get_user_input()
    prediction, probability = predict(user_input)
    
    print(f"\nPrediction: {'Diabetic' if prediction == 1 else 'Non-Diabetic'}")
    print(f"Probability of being Diabetic: {probability * 100:.2f}%")
    
    another = input("\nMake another prediction? (y/n): ")
    if another.lower() != 'y':
        break

print("Thank you for using the Model!")
