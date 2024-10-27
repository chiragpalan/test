import os
import sqlite3
import pandas as pd
import numpy as np
import joblib

DATA_DB = 'joined_data.db'
MODELS_DIR = 'models'
PREDICTIONS_DB = 'data/predictions.db'

os.makedirs('data', exist_ok=True)  # Ensure data directory exists

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

def load_model_and_scaler(table_name, model_type):
    """Load the trained model and scaler."""
    model = joblib.load(f"{MODELS_DIR}/{table_name}_{model_type}.joblib")
    scaler = joblib.load(f"{MODELS_DIR}/{table_name}_{model_type}_scaler.joblib")
    return model, scaler

def get_percentiles(model, X):
    """Calculate 5th and 95th percentiles for ensemble models."""
    predictions = np.array([tree.predict(X) for tree in model.estimators_])
    return np.percentile(predictions, 5, axis=0), np.percentile(predictions, 95, axis=0)

def save_predictions_to_db(table_name, predictions_df):
    """Save predictions to the SQLite database."""
    conn = sqlite3.connect(PREDICTIONS_DB)
    predictions_df.to_sql(f'predictions_{table_name}', conn, if_exists='replace', index=False)
    conn.close()
    print(f"Saved predictions for {table_name} to {PREDICTIONS_DB}")

def main():
    if not os.path.exists(DATA_DB):
        raise FileNotFoundError(f"Database not found at {DATA_DB}")

    tables = get_table_names(DATA_DB)

    for table in tables:
        print(f"Processing table: {table}")
        df = load_data_from_table(DATA_DB, table).dropna()

        if 'date' in df.index.names:
            df.reset_index(inplace=True)

        X = df.drop(columns=['Date', 'target_n7d'])
        y = df['target_n7d']
        dates = df['Date']

        predictions_df = pd.DataFrame({'date': dates, 'actual': y})

        for model_type in ['random_forest', 'gradient_boosting', 'xgboost']:
            model, scaler = load_model_and_scaler(table, model_type)
            X_scaled = scaler.transform(X)
            predictions = model.predict(X_scaled)

            predictions_df[f'predicted_{model_type}'] = predictions

            if hasattr(model, 'estimators_'):
                p5, p95 = get_percentiles(model, X_scaled)
                predictions_df[f'{model_type}_5th_percentile'] = p5
                predictions_df[f'{model_type}_95th_percentile'] = p95

        save_predictions_to_db(table, predictions_df)

if __name__ == "__main__":
    main()
