import pandas as pd
import sqlite3
import requests
from io import BytesIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
import joblib

def get_database_from_github(url):
    """Fetch the SQLite database from GitHub as a BytesIO stream."""
    response = requests.get(url)
    response.raise_for_status()  # Raise an error for bad responses
    return BytesIO(response.content)

def get_table_names(conn):
    """Retrieve all table names from the SQLite database."""
    query = "SELECT name FROM sqlite_master WHERE type='table';"
    return pd.read_sql(query, conn)['name'].tolist()

def main():
    # URL to the joined_data.db file in your GitHub repository
    database_url = 'https://raw.githubusercontent.com/chiragpalan/final_project/main/database/joined_data.db'
    
    # Get the database from GitHub
    db_stream = get_database_from_github(database_url)
    
    # Connect to the SQLite database using the BytesIO stream
    conn = sqlite3.connect(db_stream)
    
    # Get all table names in the database
    tables = get_table_names(conn)

    # Iterate through each table in the database
    for table in tables:
        # Load data from the current table
        df = pd.read_sql(f'SELECT * FROM {table}', conn)
        
        # Drop missing values
        df = df.dropna()
        
        # Prepare the data (modify this according to your requirements)
        # Ensure you have a target variable
        if 'target' not in df.columns:
            print(f"Target column 'target' not found in {table}. Skipping this table.")
            continue
        
        X = df.drop(columns=['target_n7d'])  # Replace 'target' with your actual target variable
        y = df['target_n7d']  # Replace with your actual target variable

        # Split the data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Scale the features
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # Initialize models
        models = {
            'RandomForest': RandomForestRegressor(),
            'GradientBoosting': GradientBoostingRegressor(),
            'XGBoost': XGBRegressor()
        }

        predictions = {}

        # Train each model and predict
        for name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            predictions[name] = y_pred
            
            # Save the model
            joblib.dump(model, f'models/{table}_{name}.joblib')

        # Store predictions in a DataFrame
        prediction_df = pd.DataFrame(predictions)
        prediction_df['actual'] = y_test.values

        # Save predictions to the predictions database
        with sqlite3.connect('data/predictions.db') as pred_conn:
            prediction_df.to_sql(name=table + '_predictions', con=pred_conn, if_exists='replace', index=False)

    # Close the connection
    conn.close()

if __name__ == "__main__":
    main()
