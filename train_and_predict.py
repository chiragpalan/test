import sqlite3
import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Paths to databases and models directory
DATA_DB = '../final_project/database/joined_data.db'  # Path to source data database
PREDICTIONS_DB = 'predictions.db'  # Local predictions database
MODELS_DIR = 'models'  # Directory to store trained models

# Ensure the models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

# Function to get all table names from the database
def get_table_names(db_path):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row[0] for row in cursor.fetchall()]
    conn.close()
    return tables

# Load data from a specific table
def load_data(table_name):
    conn = sqlite3.connect(DATA_DB)
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

# Split data into train and test sets based on date
def split_data(df, date_col='date', test_size=0.2):
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col)
    split_idx = int(len(df) * (1 - test_size))
    return df.iloc[:split_idx], df.iloc[split_idx:]

# Train and save a model
def train_model(model, X_train, y_train, model_name):
    model.fit(X_train, y_train)
    joblib.dump(model, f"{MODELS_DIR}/{model_name}.pkl")

# Predict using a trained model
def make_predictions(model, X, model_name):
    predictions = model.predict(X)
    return pd.DataFrame({'model': model_name, 'prediction': predictions})

# Store predictions in SQLite database
def save_predictions_to_db(df, table_name):
    conn = sqlite3.connect(PREDICTIONS_DB)
    df.to_sql(table_name, conn, if_exists='append', index=False)
    conn.close()

# Main function to train models on all tables
def main():
    tables = get_table_names(DATA_DB)  # Get all table names from the database
    models = {
        'random_forest': RandomForestRegressor(),
        'gradient_boosting': GradientBoostingRegressor(),
        'xgboost': xgb.XGBRegressor()
    }

    for table in tables:
        print(f"Processing table: {table}")
        df = load_data(table)
        df = df.dropna()  # Drop rows with missing values

        # Separate features and target
        X = df.drop(['date', 'target'], axis=1)  # Modify according to your columns
        y = df['target']

        # Split into train and test sets
        X_train, X_test = X.iloc[:len(X)//2], X.iloc[len(X)//2:]
        y_train, y_test = y.iloc[:len(y)//2], y.iloc[len(y)//2:]

        # Train models and save them
        for name, model in models.items():
            model_name = f"{table}_{name}"
            train_model(model, X_train, y_train, model_name)

            # Predict on the full dataset (train + test)
            predictions_df = make_predictions(model, X, model_name)
            predictions_df['actual'] = y.values  # Include actual values

            # Save predictions to the predictions database
            save_predictions_to_db(predictions_df, table)

if __name__ == '__main__':
    main()
