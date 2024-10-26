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
DATA_DB = 'joined_data.db'
MODELS_DIR = 'models'
PREDICTIONS_DB = 'data/predictions.db'

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs('data', exist_ok=True)

def download_database():
    url = 'https://raw.githubusercontent.com/chiragpalan/final_project/main/database/joined_data.db'
    response = requests.get(url)
    if response.status_code == 200:
        with open(DATA_DB, 'wb') as f:
            f.write(response.content)
        print("Database downloaded successfully.")
    else:
        raise Exception("Failed to download the database.")

def get_table_names(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    return tables

def load_data_from_table(db_path, table_name):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query(f"SELECT * FROM {table_name}", conn)
    conn.close()
    return df

def train_model(X_train, y_train, model_type):
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
    filename = f"{MODELS_DIR}/{table_name}_{model_type}.joblib"
    joblib.dump(model, filename)
    print(f"Saved model: {filename}")

def get_percentiles_for_data_points(model, X):
    """Compute 5th and 95th percentiles for each data point."""
    if hasattr(model, "estimators_"):
        # For ensemble models like RandomForest and GradientBoosting
        predictions = np.array([est.predict(X) for est in model.estimators_]).T
    elif isinstance(model, XGBRegressor):
        # For XGBoost: Extract individual trees
        predictions = np.array([model.get_booster().predict(X, iteration=i) for i in range(model.n_estimators)]).T
    else:
        raise ValueError("Model type not supported for percentile extraction.")
    
    p5 = np.percentile(predictions, 5, axis=1)
    p95 = np.percentile(predictions, 95, axis=1)
    return p5, p95

def save_predictions_to_db(table_name, predictions_df):
    conn = sqlite3.connect(PREDICTIONS_DB)
    predictions_df.to_sql(f'predictions_{table_name}', conn, if_exists='replace', index=False)
    conn.close()
    print(f"Saved predictions for {table_name} to {PREDICTIONS_DB}")

def main():
    download_database()

    if not os.path.exists(DATA_DB):
        raise FileNotFoundError(f"Database not found at {DATA_DB}")

    tables = get_table_names(DATA_DB)

    for table in tables:
        print(f"Processing table: {table}")
        df = load_data_from_table(DATA_DB, table)
        df = df.dropna()

        if 'date' in df.index.names:
            df.reset_index(inplace=True)

        X = df.drop(columns=['Date', 'target_n7d'])
        y = df['target_n7d']
        dates = df['Date']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        predictions_df = pd.DataFrame({'date': dates, 'actual': y})

        for model_type in ['random_forest', 'gradient_boosting', 'xgboost']:
            model = train_model(X_train_scaled, y_train, model_type)
            save_model(model, table, model_type)

            predictions = model.predict(scaler.transform(X))
            predictions_df[f'predicted_{model_type}'] = predictions

            p5, p95 = get_percentiles_for_data_points(model, scaler.transform(X))
            predictions_df[f'predicted_{model_type}_5th'] = p5
            predictions_df[f'predicted_{model_type}_95th'] = p95

        save_predictions_to_db(table, predictions_df)

if __name__ == "__main__":
    main()
