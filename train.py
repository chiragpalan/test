import os
import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.preprocessing import StandardScaler
import joblib
import requests
import numpy as np

# Set paths
DATA_DB = 'joined_data.db'
MODELS_DIR = 'models_v1'

# Ensure the models folder exists
os.makedirs(MODELS_DIR, exist_ok=True)

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

def check_and_clean_data(X):
    """Check for infinity, NaN, or very large values in X and clean them."""
    if not np.isfinite(X.values).all():
        print("Warning: X contains NaN, infinity, or very large values. Cleaning data...")
        X = X.replace([np.inf, -np.inf], np.nan)
        X = X.dropna()
    return X

def train_random_forest(X_train, y_train, table_name):
    # Hyperparameter tuning for Random Forest
    print("Training Random Forest model")
    param_grid = {
        'n_estimators': [200, 400, 600, 800],
        'max_depth': [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }
    model = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=5, verbose=1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    save_model(best_model, table_name, 'random_forest')

def train_gradient_boosting(X_train, y_train, table_name):
    print("Training Gradient Boosting model")
    # Hyperparameter tuning for Gradient Boosting
    param_grid = {
        'n_estimators': [200, 400, 600, 800],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0]
    }
    model = GradientBoostingRegressor(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=5, verbose=1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    save_model(best_model, table_name, 'gradient_boosting')

def train_xgboost(X_train, y_train, table_name):
    print("Training XGB model")
    # Hyperparameter tuning for XGBoost
    param_grid = {
        'n_estimators': [200, 400, 600, 800],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'subsample': [0.8, 1.0]
    }
    model = XGBRegressor(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=5, verbose=1)
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    save_model(best_model, table_name, 'xgboost')

def save_model(model, table_name, model_type):
    filename = f"{MODELS_DIR}/{table_name}_{model_type}.joblib"
    joblib.dump(model, filename)
    print(f"Saved model: {filename}")

def main():
    download_database()
    tables = get_table_names(DATA_DB)

    for table in tables:
        print(f"Processing table: {table}")
        df = load_data_from_table(DATA_DB, table).dropna()

        if 'date' in df.index.names:
            df.reset_index(inplace=True)

        X = df.drop(columns=['Date', 'target_n7d'], errors='ignore')
        y = df['target_n7d']

        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, shuffle=True)
        scaler = StandardScaler()

        # Clean and scale X_train
        X_train_cleaned = check_and_clean_data(X_train)
        X_train_scaled = scaler.fit_transform(X_train_cleaned)

        # Train and save each model separately
        train_random_forest(X_train_scaled, y_train, table)
        train_gradient_boosting(X_train_scaled, y_train, table)
        train_xgboost(X_train_scaled, y_train, table)

if __name__ == "__main__":
    main()