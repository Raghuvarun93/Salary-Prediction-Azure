import argparse
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import joblib

parser = argparse.ArgumentParser()
parser.add_argument("--data", type=str)
args = parser.parse_args()

# Load data
data = pd.read_csv(args.data)

X = data[["YearsExperience"]]
y = data["Salary"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
print("R2 Score:", r2_score(y_test, predictions))

# Save model
joblib.dump(model, "salary_model.pkl")
