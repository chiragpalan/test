import os
import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import requests
import numpy as np

# Set paths to the database and output directories
DATA_DB = 'joined_data.db'  # Local file name after downloading
MODELS_DIR = 'models'  # Folder to store trained models
PREDICTIONS_DB = 'data/predictions.db'  # SQLite DB for predictions

# Ensure models and data directories exist
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs('data', exist_ok=True)

def download_database():
    """Download the database from GitHub."""
    url = 'https://raw.githubusercontent.com/chiragpalan/final_project/main/database/joined_data.db'
    response = requests.get(url)
    if response.status_code == 200:
        with open(DATA_DB, 'wb') as f:
            f.write(response.content)
        print("Database downloaded successfully.")
    else:
        raise Exception("Failed to download the database.")

def get_table_names(db_path):
    """Fetch all table names from the SQLite database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    return tables

def load_data_from_table(db_path, table_name):
    """Load data from a specific table."""
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df

def train_model(X_train, y_train, model_type):
    """Train a regression model based on the model type."""
    if model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=800, n_jobs=-1)
    elif model_type == 'gradient_boosting':
        model = GradientBoostingRegressor(n_estimators=800, learning_rate=0.01)
    elif model_type == 'xgboost':
        model = XGBRegressor(n_estimators=800, learning_rate=0.01, n_jobs=-1)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    model.fit(X_train, y_train)
    return model

def save_model(model, table_name, model_type):
    """Save the trained model to the models directory."""
    filename = f"{MODELS_DIR}/{table_name}_{model_type}.joblib"
    joblib.dump(model, filename)
    print(f"Saved model: {filename}")

def get_percentiles_for_data_points(model, X):
    """Compute 5th and 95th percentiles for each data point from the ensemble model."""
    predictions = np.array([tree.predict(X) for tree in model.estimators_]).T  # Shape: (n_samples, n_learners)
    p5 = np.percentile(predictions, 5, axis=1)  # 5th percentile for each data point
    p95 = np.percentile(predictions, 95, axis=1)  # 95th percentile for each data point
    return p5, p95

def save_predictions_to_db(table_name, predictions_df):
    """Save predictions and actual values to a new table in the predictions database."""
    conn = sqlite3.connect(PREDICTIONS_DB)
    predictions_df.to_sql(f'predictions_{table_name}', conn, if_exists='replace', index=False)
    conn.close()
    print(f"Saved predictions for {table_name} to {PREDICTIONS_DB}")

def main():
    # Download the database
    download_database()

    # Check if the database exists
    if not os.path.exists(DATA_DB):
        raise FileNotFoundError(f"Database not found at {DATA_DB}")

    # Get all table names from the database
    tables = get_table_names(DATA_DB)

    # Train models and generate predictions for each table
    for table in tables:
        print(f"Processing table: {table}")
        df = load_data_from_table(DATA_DB, table)

        # Drop rows with missing values
        df = df.dropna()

        # Ensure 'date' is a column (if it's set as index, reset it)
        if 'date' in df.index.names:
            df.reset_index(inplace=True)

        # Split into features and target
        X = df.drop(columns=['Date', 'target_n7d'])  # Replace 'target_n7d' with your target column if needed
        y = df['target_n7d']
        dates = df['Date']  # Store the dates for predictions

        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=True
        )

        # Scale the training data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Create a DataFrame to hold predictions
        predictions_df = pd.DataFrame({'date': dates, 'actual': y})

        # Train models and get predictions
        for model_type in ['random_forest', 'gradient_boosting', 'xgboost']:
            model = train_model(X_train_scaled, y_train, model_type)
            save_model(model, table, model_type)

            # Predict using the trained model on the entire dataset (train + test)
            predictions = model.predict(scaler.transform(X))
            predictions_df[f'predicted_{model_type}'] = predictions

            # Get 5th and 95th percentiles for each data point
            p5, p95 = get_percentiles_for_data_points(model, scaler.transform(X))
            predictions_df[f'predicted_{model_type}_5th'] = p5
            predictions_df[f'predicted_{model_type}_95th'] = p95

        # Save predictions to the predictions database
        save_predictions_to_db(table, predictions_df)

if __name__ == "__main__":
    main()
