# Added on Friday 9 Feb by Philip P
# Exploratory data analysis of parquet files: market data, executions, reference data
from typing import Dict, Union

import numpy as np
import pandas as pd

pd.options.display.max_columns = 14
pd.options.display.width = 800

def load_and_preprocess_data(file_path: str) -> pd.DataFrame:
    """
    Loads a parquet file into a DataFrame and converts column names to lowercase.
    
    Parameters:
        file_path (str): The file path to the parquet file.
    
    Returns:
        pd.DataFrame: Preprocessed DataFrame with lowercase column names. 
    """
    df = pd.read_parquet(file_path, engine='pyarrow')
    df.columns = [x.lower() for x in df.columns]
    return df

# Load and preprocess the data
ref_data_df = load_and_preprocess_data('data/refdata.parquet')
market_data_df = load_and_preprocess_data('data/marketdata.parquet')
executions_df = load_and_preprocess_data('data/executions.parquet')

# Task 1a: Exploratory Daa Analysis
def count_executions_and_venues(executions_df: pd.DataFrame) -> Dict[str, Union[int, pd.DataFrame]]:
    """
    Counts the number of unique executions and venues, and summarizes executions by venue and date.

    Parameters:
        executions_df (pd.DataFrame): The executions DataFrame.

    Returns:
        Dict[str, Union[int, pd.DataFrame]: A tuple containing the number of executions, the number of unique venues,
                                       and a DataFrame summarizing the count of executions by venue and date.
    """
    num_executions = executions_df['trade_id'].nunique()
    num_venues = executions_df['venue'].nunique()
    executions_df['date'] = pd.to_datetime(executions_df['tradetime']).dt.date
    summary_df = pd.pivot_table(
        executions_df,
        index=['venue', 'date'],
        values='trade_id',
        aggfunc='count'
    ).reset_index()
    out_dict = {
        'num_executions': num_executions,
        'num_venues': num_venues,
        'summary_df': summary_df
    }
    return out_dict


executions_venues_result_dct = count_executions_and_venues(executions_df)

print(f"Number of executions: {executions_venues_result_dct['num_executions']:,}")
print(f"Number of unique venues: {executions_venues_result_dct['num_venues']}")
print(f"Executions by venue and date:\n{executions_venues_result_dct['summary_df']}\n")


# 2.a. Filter executions.paraquet for only CONTINUOUS_TRADING trades.
filter_to_apply = 'CONTINUOUS_TRADING'
continuous_executions_df = executions_df.loc[executions_df['phase'] == filter_to_apply].copy()

# check for duplicates
assert len(continuous_executions_df) == len(continuous_executions_df.drop_duplicates()), f"There are duplicates in the dataset"

# b.	Log output the # of executions.
print(f"Number of {filter_to_apply} executions: {continuous_executions_df['trade_id'].nunique():,.0f}")

# 3.	Data Transformation
# a.	Add column [‘side’], if quantity is negative, side = 2, if quantity is positive side = 1.
continuous_executions_df['side'] = np.where(
    continuous_executions_df['quantity'] < 0,
    2,
    1
)

# b.	Complement the data with refdata.parquet - add primary ticker and primary mic
continuous_executions_df_extra = pd.merge(
    left=continuous_executions_df,
    right=ref_data_df[['isin','primary_ticker', 'primary_mic']],
    left_on='isin',
    right_on='isin',
    how='left'
)



# 3. Calculations
# a. Best bid price and best ask at execution - look at executions for each time, and then look at
# market data
# round to the nearest second
continuous_executions_df_extra['time'] = continuous_executions_df_extra['tradetime'].dt.round('S')

# drop all non-continuous trades in market data
market_data_df = market_data_df.loc[market_data_df['market_state'] == 'CONTINUOUS_TRADING']
market_data_df['datetime'] = pd.to_datetime(market_data_df['event_timestamp']).dt.round('S')

# enrich the market data with isins
market_data_enriched_df = pd.merge(
    left=market_data_df,
    right=ref_data_df[['isin', 'id', 'currency', 'primary_ticker']],
    left_on='listing_id',
    right_on='id',
    how='left'
)

market_data_enriched_df.drop(
    columns=['best_bid_size', 'best_ask_size'],
    inplace=True
)

print(continuous_executions_df_extra.head())
print(market_data_enriched_df.head())

continuous_executions_df_extra['isin_venue'] = continuous_executions_df_extra['isin'] + '_' + \
                                               continuous_executions_df_extra['venue']
market_data_enriched_df['isin_venue'] = market_data_enriched_df['isin'] + '_' + market_data_enriched_df['primary_mic']

# match the isin and venue for the execution of trades in the market data
unique_traded_isin_venues = continuous_executions_df_extra['isin_venue'].unique()
located_isin_in_market_data = unique_traded_isin_venues[np.in1d(unique_traded_isin_venues,
                                                                market_data_enriched_df['isin_venue'].unique())]

# cut down the market data dataframe for only the executed ISIN on each venue
market_data_df_subset = market_data_enriched_df.loc[market_data_enriched_df['isin_venue'].isin(unique_traded_isin_venues)]

# steps to ensure we can get the market data at the time of execution and one second before and after
# 1. ensure the isin_venue_trade_df['time'] column is the same type as market_data_isin_venue_df['datetime'],
# i.e. both 'datetime64[ns]' and rounded to the nearest second
print("Step 1, check time columns are equal")
assert market_data_df_subset['datetime'].dtype == continuous_executions_df_extra['time'].dtype, 'Time columns are not the same datatype'

filtered_market_data = []
for isin_venue in located_isin_in_market_data:
    print(f"Looking up {isin_venue}...")
    # 2. For each trade execution time in isin_venue_trade_df, find the corresponding market data in market_data_isin_venue_df
    isin_venue_trade_df = continuous_executions_df_extra.loc[continuous_executions_df_extra['isin_venue'] == isin_venue]
    market_data_isin_venue_df = market_data_df_subset.loc[market_data_df_subset['isin_venue'] == isin_venue]

    # create a dataframe with unique trade execution times
    unique_execution_times = pd.DataFrame(
        isin_venue_trade_df['time'].unique(), columns = ['time']
    )
    # add one second before and after
    unique_execution_times['time_minus_one'] = unique_execution_times['time'] - pd.Timedelta(seconds=1)
    unique_execution_times['time_plus_one'] = unique_execution_times['time'] + pd.Timedelta(seconds=1)

    times_to_extract = pd.concat([
        unique_execution_times['time_minus_one'],
        unique_execution_times['time'],
        unique_execution_times['time_plus_one']
    ]).drop_duplicates().reset_index(drop=True)

    filtered_market_data = market_data_isin_venue_df.loc[market_data_isin_venue_df['datetime'].isin(times_to_extract)]

    aggregated_market_data = filtered_market_data.groupby('datetime').agg({
        'best_bid_price': 'max',  # The best bid price is the maximum bid price
        'best_ask_price': 'min'  # The best ask price is the minimum ask price
    }).reset_index()

    aggregated_market_data['best_bid_price_minus_one'] = aggregated_market_data['best_bid_price'].shift(1)
    aggregated_market_data['best_bid_price_plus_one'] = aggregated_market_data['best_bid_price'].shift(-1)

    resulting_df = pd.merge(
        left=isin_venue_trade_df,
        right=aggregated_market_data,
        left_on='time',
        right_on='datetime',
        how='left'
    )



    # 3. Filter market data_enriched_df based on these times and extract 'best_bid_price' and 'best_ask_price'
    for trade_execution_time in isin_venue_trade_df['time'].unique():
        execution_times = trade_execution_time
        # add the one second before and after
        execution_plus_one_second = [x + pd.Timedelta(seconds=1) for x in execution_times]
        execution_minus_one_second = [x - pd.Timedelta(seconds=1) for x in execution_times]


    # filter for only the trades in this isin_venue name...
    # filter the market data to get the best bid/offer price at time of execution, and 1 second before and after
    # excecution
    time_of_execution = isin_venue_trade_df['time'].unique()
    # add and take away one second:
    times_to_filter = time_of_execution + [time_of_execution + pd.Timedelta(seconds=1)] + [time_of_execution - pd.Timedelta(seconds=1)]
    pd.concat(time_of_execution,
              [time_of_execution + pd.Timedelta(seconds=1)],
              [time_of_execution - pd.Timedelta(seconds=1)])

    market_data_isin_venue_df.loc[market_data_isin_venue_df['datetime'] == isin_venue_trade_df['time']]

for listing_id in market_data_df['listing_id'].unique():
    listing_id_df = market_data_df.loc[market_data_df['listing_id'] == listing_id]
    listing_id_df.loc[listing_id_df['best_bid_price'] == listing_id_df['best_bid_price'].max()]

# look at the execution of each isin at each venue
continuous_executions_df_extra['isin_venue'] =

dct = dict(zip(ref_data_df['primary_ticker'], ref_data_df['primary_mic']))
continuous_executions_df_extra['temp'] = continuous_executions_df_extra['venue'].map(dct)