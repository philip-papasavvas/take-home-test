# Added on Friday 9 Feb by Philip P
# Exploratory data analysis of parquet files: market data, executions, reference data

import pandas as pd

pd.options.display.max_columns = 10

ref_data_raw_df = pd.read_parquet('data/refdata.parquet', engine='pyarrow')
market_data_raw_df = pd.read_parquet('data/marketdata.parquet', engine='pyarrow')
executions_raw_df = pd.read_parquet('data/executions.parquet', engine='pyarrow')


def return_lowercase_column_names_df(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Utility function to make all column names lowercase for ease of calling functions"""
    dataframe.columns = [x.lower() for x in dataframe.columns]
    return dataframe


ref_data_df = return_lowercase_column_names_df(dataframe=ref_data_raw_df)
market_data_df = return_lowercase_column_names_df(dataframe=market_data_raw_df)
executions_df = return_lowercase_column_names_df(dataframe=executions_raw_df)

# garbage clearance/free up memory
del ref_data_raw_df, market_data_raw_df, executions_raw_df

# Tasks
# 1 a.	Count the number of executions within the executions.parquet file,
print(f"Number of executions: {executions_df['trade_id'].nunique():,.0f}")

# determine the unique number of [‘Venue’]s and the date of executions. Log output this information.
# new column for the date to get the datetime
executions_df['date'] = pd.to_datetime(executions_df['tradetime']).dt.date

print(f"Number of venues for execution: {executions_df['venue'].nunique()}")

# date of executions for each venue, sorted by alphabetical order
output_df = pd.pivot_table(
    executions_df,
    index=['venue','date'],
    values='trade_id',
    aggfunc='count'
)
