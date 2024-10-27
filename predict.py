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
        print("Columns in df_test:", df_test.columns.tolist())  # List the available columns

        # Ensure 'Date' is present and handle it accordingly
        if 'date' not in df_test.columns:
            raise KeyError("The 'Date' column is missing from the DataFrame.")

        # Store dates and actual values
        dates = df_test['date']
        y_actual = df_test['actual']  # Adjust according to your actual target column name

        # Prepare predictions DataFrame
        predictions_df = pd.DataFrame({'Date': dates, 'actual': y_actual})

        # Load models
        for model_type in ['random_forest', 'gradient_boosting', 'xgboost']:
            model_path = os.path.join(MODELS_DIR, f"{table}_{model_type}.joblib")
            model = joblib.load(model_path)

            # Scale features from the test DataFrame
            # Assuming the input features are the same as used during training
            X_test = df_test.drop(columns=['Date', 'target_n7d'], errors='ignore')  # Use errors='ignore' to avoid KeyErrors
            scaler = StandardScaler()  # Initialize the scaler
            
            # Fit on train data if scaling is needed; for predict, usually just transform
            X_test_scaled = scaler.fit_transform(X_test)  # Use the same scaler if you saved it during training

            # Predict using the trained model
            predictions = model.predict(X_test_scaled)
            predictions_df[f'predicted_{model_type}'] = predictions

            # Calculate the 5th and 95th percentiles for each weak learner
            if hasattr(model, 'estimators_'):
                preds = [tree.predict(X_test_scaled) for tree in model.estimators_]
                preds = pd.DataFrame(preds).T
                predictions_df[f'5th_percentile_{model_type}'] = preds.quantile(0.05, axis=1)
                predictions_df[f'95th_percentile_{model_type}'] = preds.quantile(0.95, axis=1)

        # Save predictions to a new table in the predictions database
        conn = sqlite3.connect(PREDICTIONS_DB)
        predictions_df.to_sql(f'predictions_{table}', conn, if_exists='replace', index=False)
        conn.close()
        print(f"Saved predictions for {table} to {PREDICTIONS_DB}")

if __name__ == "__main__":
    main()
