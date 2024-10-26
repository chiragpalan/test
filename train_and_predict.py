import sqlite3
import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb

# Paths
DATA_DB = 'final_project/database/joined_data.db'  # Database with source data
PREDICTIONS_DB = 'predictions.db'  # Local predictions storage
MODELS_DIR = 'models'  # Folder to store trained models

# Ensure models directory exists
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

# Train and save model
def train_model(model, X_train, y_train, model_name):
    model.fit(X_train, y_train)
    joblib.dump(model, f"{MODELS_DIR}/{model_name}.pkl")

# Make predictions using trained model
def make_predictions(model, X):
    return model.predict(X)

# Save predictions to SQLite database
def save_predictions_to_db(predictions_df, table_name):
    conn = sqlite3.connect(PREDICTIONS_DB)
    predictions_df.to_sql(table_name, conn, if_exists='append', index=False)
    conn.close()

# Main logic for training models on all tables
def main():
    tables = get_table_names(DATA_DB)  # Get all tables
    models = {
        'random_forest': RandomForestRegressor(),
        'gradient_boosting': GradientBoostingRegressor(),
        'xgboost': xgb.XGBRegressor()
    }

    for table in tables:
        print(f"Processing table: {table}")
        df = load_data(table).dropna()  # Load data and drop missing values

        # Split features and target
        X = df.drop(['date', 'target'], axis=1)  # Adjust based on your data
        y = df['target']

        # Train models and save predictions
        for model_name, model in models.items():
            trained_model_name = f"{table}_{model_name}"
            train_model(model, X, y, trained_model_name)  # Train and save model

            # Make predictions on entire dataset (train + test)
            predictions = make_predictions(model, X)

            # Store predictions along with actual values
            predictions_df = pd.DataFrame({
                'model': trained_model_name,
                'prediction': predictions,
                'actual': y.values
            })

            # Save predictions to database
            save_predictions_to_db(predictions_df, table)

if __name__ == '__main__':
    main()
