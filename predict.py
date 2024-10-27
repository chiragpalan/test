import os
import sqlite3
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib

# Set paths to the models and predictions database
MODELS_DIR = 'models'  # Folder where models are stored
PREDICTIONS_DB = 'data/predictions.db'  # SQLite DB for predictions

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

def main():
    # Get all table names from the predictions database
    tables = get_table_names(PREDICTIONS_DB)

    # Process each table
    for table in tables:
        print(f"Processing table: {table}")
        df_test = load_data_from_table(PREDICTIONS_DB, table)

        # Debugging: Check the columns of df_test
        print("Columns in df_test:", df_test.columns.tolist())

        # Ensure 'Date' column is present and handle accordingly
        if 'date' not in df_test.columns:
            raise KeyError("The 'date' column is missing from the DataFrame.")

        # Store dates and actual values
        dates = df_test['date']
        y_actual = df_test['actual']

        # Prepare predictions DataFrame
        predictions_df = pd.DataFrame({'date': dates, 'actual': y_actual})

        # Load models and generate predictions
        table_prefix = table.replace('predictions_', '')  # Adjust table name to match model filenames
        for model_type in ['random_forest', 'gradient_boosting', 'xgboost']:
            model_path = os.path.join(MODELS_DIR, f"{table_prefix}_{model_type}.joblib")

            # Check if the model file exists
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Model file not found: {model_path}")

            model = joblib.load(model_path)

            # Prepare test data for prediction
            X_test = df_test.drop(columns=['date', 'actual'], errors='ignore')
            scaler = StandardScaler()
            X_test_scaled = scaler.fit_transform(X_test)

            # Predict and store results
            predictions = model.predict(X_test_scaled)
            predictions_df[f'predicted_{model_type}'] = predictions

            # Calculate 5th and 95th percentiles if the model has multiple estimators
            if hasattr(model, 'estimators_'):
                preds = [est.predict(X_test_scaled) for est in model.estimators_]
                preds_df = pd.DataFrame(preds).T
                predictions_df[f'5th_percentile_{model_type}'] = preds_df.quantile(0.05, axis=1)
                predictions_df[f'95th_percentile_{model_type}'] = preds_df.quantile(0.95, axis=1)

        # Save predictions to a new table in the predictions database
        conn = sqlite3.connect(PREDICTIONS_DB)
        predictions_df.to_sql(table, conn, if_exists='replace', index=False)
        conn.close()
        print(f"Saved predictions for {table} to {PREDICTIONS_DB}")

if __name__ == "__main__":
    main()
