import os
import sqlite3
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import requests

# Set paths
DB_URL = 'https://raw.githubusercontent.com/chiragpalan/final_project/main/database/joined_data.db'
DATA_DB = 'joined_data.db'
PREDICTIONS_DB = 'data/predictions.db'
MODELS_DIR = 'models'

def download_database():
    """Download the joined database from the GitHub repository."""
    response = requests.get(DB_URL)
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

def save_predictions_to_db(predictions_df, table_name):
    """Save predictions to the predictions database."""
    os.makedirs(os.path.dirname(PREDICTIONS_DB), exist_ok=True)  # Ensure 'data/' folder exists
    conn = sqlite3.connect(PREDICTIONS_DB)
    predictions_df.to_sql(table_name, conn, if_exists='replace', index=False)
    conn.close()
    print(f"Saved predictions for {table_name} to {PREDICTIONS_DB}")

def extract_predictions_from_estimators(model, X_scaled):
    """Extract predictions from individual estimators and calculate percentiles."""
    all_preds = []

    # Loop through each estimator directly (list format)
    for est in model.estimators_:
        if hasattr(est, 'predict'):
            all_preds.append(est.predict(X_scaled))
        else:
            raise AttributeError("Estimator object does not support 'predict'.")

    # Convert predictions to DataFrame and return 5th/95th percentiles
    all_preds_df = pd.DataFrame(all_preds).T
    return (
        all_preds_df.quantile(0.05, axis=1),
        all_preds_df.quantile(0.95, axis=1),
    )

def main():
    # Download the joined database
    download_database()

    # Get all table names from the database
    tables = get_table_names(DATA_DB)

    # Process each table for predictions
    for table in tables:
        print(f"Processing table: {table}")

        # Load the full dataset
        df = load_data_from_table(DATA_DB, table).dropna()

        # Ensure 'Date' column exists
        if 'Date' not in df.columns:
            raise KeyError("The 'Date' column is missing from the data.")

        # Prepare features and output columns
        X = df.drop(columns=['Date', 'target_n7d'], errors='ignore')
        y_actual = df['target_n7d']
        dates = df['Date']

        # Initialize DataFrame to store predictions
        predictions_df = pd.DataFrame({'Date': dates, 'Actual': y_actual})

        # Scale the features
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        # Loop through each model type and generate predictions
        for model_type in ['random_forest', 'gradient_boosting', 'xgboost']:
            model_path = os.path.join(MODELS_DIR, f"{table}_{model_type}.joblib")
            print(f"Loading model from: {model_path}")

            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")

            model = joblib.load(model_path)
            predictions = model.predict(X_scaled)
            predictions_df[f'Predicted_{model_type}'] = predictions

            # Calculate percentiles for ensemble models
            if hasattr(model, 'estimators_'):
                p5, p95 = extract_predictions_from_estimators(model, X_scaled)
                predictions_df[f'5th_Percentile_{model_type}'] = p5
                predictions_df[f'95th_Percentile_{model_type}'] = p95

        # Save the predictions DataFrame to the predictions database
        save_predictions_to_db(predictions_df, table)

if __name__ == "__main__":
    main()
