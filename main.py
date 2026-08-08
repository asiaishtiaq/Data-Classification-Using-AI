"""
AI Project 2: Data Classification Using AI
Iris dataset + StandardScaler + KNN(k=5)
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score

# 1. Load dataset
df = pd.read_csv("data/iris.csv")

features = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
X = df[features]
y = df["species"]

# 2. Shuffle + split into training and testing sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y,
    shuffle=True
)

# 3. Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Create and train KNN model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train_scaled, y_train)

# 5. Predict
predictions = model.predict(X_test_scaled)

# 6. Evaluate
accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(y_test, predictions, average="weighted")
cm = confusion_matrix(y_test, predictions)

print("AI Project 2 - Data Classification Using AI")
print("--------------------------------------------")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples : {len(X_test)}")
print(f"Accuracy        : {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Weighted F1     : {f1:.4f}")
print("\nConfusion Matrix:")
print(cm)
print("\nClassification Report:")
print(classification_report(y_test, predictions))
