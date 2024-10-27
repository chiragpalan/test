import os
import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import requests

DATA_DB = 'joined_data.db'
MODELS_DIR = 'models'

os.makedirs(MODELS_DIR, exist_ok=True)  # Ensure models directory exists

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
    """Train a regression model with n_estimators=800."""
    if model_type == 'random_forest':
        model = RandomForestRegressor(n_estimators=800)
    elif model_type == 'gradient_boosting':
        model = GradientBoostingRegressor(n_estimators=800)
    elif model_type == 'xgboost':
        model = XGBRegressor(n_estimators=800)
    else:
        raise ValueError(f"Unsupported model type: {model_type}")
    model.fit(X_train, y_train)
    return model

def save_model_and_scaler(model, scaler, table_name, model_type):
    """Save the trained model and scaler."""
    model_path = f"{MODELS_DIR}/{table_name}_{model_type}.joblib"
    scaler_path = f"{MODELS_DIR}/{table_name}_{model_type}_scaler.joblib"
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)
    print(f"Saved model and scaler: {model_path}, {scaler_path}")

def main():
    download_database()

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

        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, shuffle=True)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        for model_type in ['random_forest', 'gradient_boosting', 'xgboost']:
            model = train_model(X_train_scaled, y_train, model_type)
            save_model_and_scaler(model, scaler, table, model_type)

if __name__ == "__main__":
    main()
