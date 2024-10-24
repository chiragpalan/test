def fetch_and_store_data(ticker, table_name):
    initialize_table(table_name)  # Ensure the table schema is correct

    # Fetch the latest available date from the table
    query = f"SELECT MAX(Date) FROM {table_name}"
    result = pd.read_sql(query, conn).iloc[0, 0]

    # Set the start date for fetching new data
    start_date = (pd.to_datetime(result) + pd.Timedelta(days=1)).strftime('%Y-%m-%d') if result else '2019-01-01'

    # Download data from Yahoo Finance
    data = yf.download(ticker, start=start_date)
    print(f"Downloaded data for {ticker}:")
    print(data.head())  # Debug: Verify what data was downloaded

    # Check if the DataFrame is empty
    if data.empty:
        print(f"No new data available for {ticker}.")
        return

    # Ensure 'Date' is a column after resetting the index
    data.reset_index(inplace=True)
    print(f"Columns after reset_index: {data.columns}")  # Debug: Check column names

    # Rename columns to match the database schema
    data = data.rename(columns={'Adj Close': 'Adj_Close'})

    # Drop any rows where 'Date' is NaN
    if 'Date' in data.columns:
        data = data.dropna(subset=['Date'])
    else:
        print(f"Error: 'Date' column not found in data for {ticker}.")
        return

    # Insert data into the database
    data.to_sql(table_name, conn, if_exists="append", index=False)
    print(f"Data for {ticker} stored successfully in {table_name}.")
