import os
import sqlite3
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Set paths to the models and output directories
MODELS_DIR = 'models'  # Folder where trained models are saved
PREDICTIONS_DB = 'data/predictions.db'  # SQLite DB for predictions

def load_model(model_type, table_name):
    """Load a trained model from the models directory."""
    filename = f"{MODELS_DIR}/{table_name}_{model_type}.joblib"
    model = joblib.load(filename)
    return model

def save_predictions_to_db(table_name, predictions_df):
    """Save predictions and actual values to a new table in the predictions database."""
    conn = sqlite3.connect(PREDICTIONS_DB)
    # Save to the predictions database, create a new table for each original table
    predictions_df.to_sql(f'predictions_{table_name}', conn, if_exists='replace', index=False)  # Replace table with new data
    conn.close()
    print(f"Saved predictions for {table_name} to {PREDICTIONS_DB}")

def main():
    # Assuming you have a way to retrieve the table names and the data for each table
    # For demonstration purposes, let's say you have a list of tables
    tables = ['asian_data', 'nifty_data', 'reliance_data', 'tcs_data']  # Replace with actual table names

    for table in tables:
        print(f"Processing table: {table}")

        # Load the test data
        # Here, you will need to implement your logic to retrieve the test data as a DataFrame
        # For example, using a method similar to load_data_from_table
        # df_test = load_data_from_table(DATA_DB, table)
        # df_test should contain the columns used for prediction
        
        # Placeholder for actual test data loading
        df_test = pd.DataFrame()  # Replace with actual loading code

        # Drop rows with missing values if any
        df_test = df_test.dropna()

        # Store dates for predictions
        dates = df_test['Date']  # Assuming 'Date' column exists
        X_test = df_test.drop(columns=['Date', 'target_n7d'])  # Adjust based on your actual target column

        # Scale the test data
        scaler = StandardScaler()
        X_test_scaled = scaler.fit_transform(X_test)

        # Initialize DataFrame for predictions
        predictions_df = pd.DataFrame({'date': dates})

        # Store percentiles for each model type
        for model_type in ['random_forest', 'gradient_boosting', 'xgboost']:
            model = load_model(model_type, table)

            # Get predictions from each weak learner
            if model_type == 'random_forest':
                predictions = model.predict(X_test_scaled)
                weak_predictions = [tree.predict(X_test_scaled) for tree in model.estimators_]
            elif model_type == 'gradient_boosting':
                predictions = model.predict(X_test_scaled)
                weak_predictions = model.decision_function(X_test_scaled)  # For GBM
            elif model_type == 'xgboost':
                predictions = model.predict(X_test_scaled)
                weak_predictions = model.predict(X_test_scaled, ntree_limit=model.best_ntree_limit)
            else:
                continue  # Unsupported model type

            # Calculate percentiles
            weak_predictions = pd.DataFrame(weak_predictions).T  # Shape: (n_estimators, n_samples)
            percentile_5th = weak_predictions.quantile(0.05, axis=0)
            percentile_95th = weak_predictions.quantile(0.95, axis=0)

            # Store results in the DataFrame
            predictions_df[f'predicted_{model_type}'] = predictions
            predictions_df[f'5th_percentile_{model_type}'] = percentile_5th
            predictions_df[f'95th_percentile_{model_type}'] = percentile_95th

        # Save predictions to a new table in the predictions database
        save_predictions_to_db(table, predictions_df)

if __name__ == "__main__":
    main()
