import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import mlflow
import mlflow.sklearn
import kagglehub

#download dataset
print(" Downloading Iris dataset...")
path = kagglehub.dataset_download("uciml/iris")

print(" Dataset downloaded at:", path)

csv_path = os.path.join(path, "Iris.csv")
print(" Reading CSV file from:", csv_path)

df = pd.read_csv(csv_path)

# Preprocess the dataset
print(" Preprocessing data now :")
df = df.dropna()
X = df[["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"]]
y = df["Id"]  # Using 'Id' column as regression target for demonstration

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step up mlflow exper
mlflow.set_experiment("Iris Linear Regression")

# Train and log the model
with mlflow.start_run():
    print(" Training Linear Regression model...")
    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    #  Evaluate the model
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"MSE: {mse:.4f}, R²: {r2:.4f}")

    #  Log parameters, metrics, model
    mlflow.log_param("model_type", "LinearRegression")
    mlflow.log_metric("mse", mse)
    mlflow.log_metric("r2_score", r2)
    mlflow.sklearn.log_model(model, "model")

    #  Save and log plot
    print("Generating Actual vs Predicted plot...")
    fig, ax = plt.subplots()
    ax.scatter(y_test, predictions, alpha=0.7)
    ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")

    plot_path = "actual_vs_predicted.png"
    fig.savefig(plot_path)
    mlflow.log_artifact(plot_path)
    plt.close(fig)

    print(" Run logged to MLflow with Run ID:", mlflow.active_run().info.run_id)
