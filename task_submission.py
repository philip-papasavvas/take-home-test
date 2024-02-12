# Added on Friday 9 Feb by Philip P
# Exploratory data analysis of parquet files: market data, executions, reference data
import logging
from typing import Dict, Union

import numpy as np
import pandas as pd

pd.options.display.max_columns = 14
pd.options.display.width = 800

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='data_analysis.log',  # Log file path
    filemode='w'  # 'w' to overwrite the log file each time, 'a' to append
)


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


# Task 2a: Data Cleaning - Filter for CONTINUOUS_TRADING trades
def filter_continuous_trading(executions_df: pd.DataFrame,
                              phase: str = 'CONTINUOUS_TRADING') -> pd.DataFrame:
    """
    Filters the executions DataFrame for rows where the phase is CONTINUOUS_TRADING.

    Parameters:
        executions_df (pd.DataFrame): The executions DataFrame.
        phase (str): The trading phase to filter by. Default is 'CONTINUOUS_TRADING'.

    Returns:
        pd.DataFrame: A DataFrame containing only executions during the continuous trading phase.
    """
    continuous_df = executions_df.loc[executions_df['phase'] == phase].copy()
    # Check for duplicates
    assert continuous_df['trade_id'].is_unique, "There are duplicate trade IDs in the dataset."

    assert len(continuous_df) == len(continuous_df.drop_duplicates()), f"There are duplicates in the dataset"

    return continuous_df


def add_side_and_venue_information(
        executions_clean_df: pd.DataFrame,
        reference_data_df: pd.DataFrame
):
    """
    Transforms the executions DataFrame by adding a 'side' column, then merges it
    with the reference data to add 'primary_ticker' and 'primary_mic'.

    Parameters:
        executions_clean_df (pd.DataFrame): The DataFrame containing executions data.
        reference_data_df (pd.DataFrame): The DataFrame containing reference data.

    Returns:
        pd.DataFrame: The transformed and merged DataFrame.
    """
    # Add 'side' column based on 'quantity'
    executions_clean_df['side'] = np.where(executions_clean_df['quantity'] < 0,
                                           2,
                                           1)

    # Merge with reference data to add 'primary_ticker' and 'primary_mic'
    merged_df = pd.merge(
        left=executions_clean_df,
        right=reference_data_df[['isin', 'primary_ticker', 'primary_mic']],
        on='isin',
        how='left'
    )

    return merged_df


if __name__ == '__main__':
    # Load and preprocess the data
    ref_data_df = load_and_preprocess_data('data/refdata.parquet')
    market_data_df = load_and_preprocess_data('data/marketdata.parquet')
    executions_df = load_and_preprocess_data('data/executions.parquet')

    # Task 1 - Count executions and venues
    executions_venues_result_dct = count_executions_and_venues(executions_df)
    logging.info(f"Number of executions: {executions_venues_result_dct['num_executions']:,}")
    logging.info(f"Number of unique venues: {executions_venues_result_dct['num_venues']}")
    logging.info(f"Executions by venue and date:\n{executions_venues_result_dct['summary_df']}\n")

    # Task 2 - Filter for CONTINUOUS TRADING and count
    filter_to_apply = 'CONTINUOUS_TRADING'
    continuous_executions_df = filter_continuous_trading(executions_df)
    logging.info(f"Number of {filter_to_apply} executions: {len(continuous_executions_df):,.0f}")

    # Task 3 - Data transformation and adding reference data
    executions_df = add_side_and_venue_information(
        executions_clean_df=continuous_executions_df,
        reference_data_df=ref_data_df
    )

    # 4. Calculations
    # a. Best bid price and best ask at execution - look at executions for each time, and then look at
    # market data
    # round to the nearest second
    def round_data_times(
        dataframe: pd.DataFrame,
        time_column_name: str
    ) -> pd.DataFrame:
        """
        Round market data/trade execution times to the nearest second.

        Parameters:
        - dataframe (DataFrame): Dataframe containing trade execution data.

        Returns:
        - DataFrame: A modified dataframe with rounded trade execution times.
        """
        dataframe[time_column_name] = pd.to_datetime(dataframe[time_column_name])
        dataframe['time'] = dataframe[time_column_name].dt.round('S')
        return dataframe

    executions_df = round_data_times(dataframe=executions_df, time_column_name='tradetime')

    # drop all non-continuous trades in market data
    market_data_df = market_data_df.loc[market_data_df['market_state'] == 'CONTINUOUS_TRADING'].copy()
    market_data_df = round_data_times(dataframe=market_data_df,
                                      time_column_name='event_timestamp')

    def add_isin_to_reference_data(
        market_data_df: pd.DataFrame,
        reference_data_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Enrich market data with reference data ISINs.

        Parameters:
            market_data_df (DataFrame): Dataframe containing market data.
            reference_data_df (DataFrame): Dataframe containing reference data.

        Returns:
            DataFrame: An enriched dataframe with ISINs and other reference data merged.
        """
        market_data_merged_df = pd.merge(
            left=market_data_df,
            right=reference_data_df[['isin', 'id', 'currency', 'primary_ticker']],
            left_on='listing_id',
            right_on='id',
            how='left'
        )
        # drop these columns as we're not interested in size here
        market_data_merged_df.drop(columns=['best_bid_size', 'best_ask_size'], inplace=True)
        return market_data_merged_df

    market_data_df = add_isin_to_reference_data(
        market_data_df=market_data_df,
        reference_data_df=ref_data_df)
    
    executions_df['isin_venue'] = executions_df['isin'] + '_' + executions_df['venue']
    market_data_df['isin_venue'] = market_data_df['isin'] + '_' + market_data_df['primary_mic']

    # match the isin and venue for the execution of trades in the market data
    unique_traded_isin_venues = executions_df['isin_venue'].unique()
    located_isin_in_market_data = unique_traded_isin_venues[np.in1d(unique_traded_isin_venues,
                                                                    market_data_df['isin_venue'].unique())]

    # cut down the market data dataframe for only the executed ISIN on each venue
    market_data_df_subset = market_data_df.loc[
        market_data_df['isin_venue'].isin(unique_traded_isin_venues)]

    # steps to ensure we can get the market data at the time of execution and one second before and after
    # 1. ensure the isin_venue_trade_df['time'] column is the same type as market_data_isin_venue_df['datetime'],
    # i.e. both 'datetime64[ns]' and rounded to the nearest second
    print("Step 1, check time columns are equal")
    assert market_data_df_subset['datetime'].dtype == executions_df[
        'time'].dtype, 'Time columns are not the same datatype'

    merged_isin_venue_data = []
    for isin_venue in located_isin_in_market_data:
        print(f"Looking up {isin_venue}...")
        # 2. For each trade execution time in isin_venue_trade_df, find the corresponding market data in market_data_isin_venue_df
        isin_venue_trade_df = executions_df.loc[executions_df['isin_venue'] == isin_venue]
        market_data_isin_venue_df = market_data_df_subset.loc[market_data_df_subset['isin_venue'] == isin_venue]

        # create a dataframe with unique trade execution times
        unique_execution_times = pd.DataFrame(
            isin_venue_trade_df['time'].unique(), columns=['time']
        )
        # add one second before and after
        unique_execution_times['time_minus_one'] = unique_execution_times['time'] - pd.Timedelta(seconds=1)
        unique_execution_times['time_plus_one'] = unique_execution_times['time'] + pd.Timedelta(seconds=1)

        times_to_extract = pd.concat([
            unique_execution_times['time_minus_one'],
            unique_execution_times['time'],
            unique_execution_times['time_plus_one']
        ]).drop_duplicates().reset_index(drop=True)

        filtered_market_data = market_data_isin_venue_df.loc[
            market_data_isin_venue_df['datetime'].isin(times_to_extract)]

        aggregated_market_data = filtered_market_data.groupby('datetime').agg({
            'best_bid_price': 'max',  # The best bid price is the maximum bid price
            'best_ask_price': 'min'  # The best ask price is the minimum ask price
        }).reset_index().rename(columns={'best_bid_price': 'best_bid',
                                         'best_ask_price': 'best_ask'})

        aggregated_market_data['best_bid_min_1s'] = aggregated_market_data['best_bid'].shift(1)
        aggregated_market_data['best_bid_1s'] = aggregated_market_data['best_bid'].shift(-1)
        aggregated_market_data['best_ask_min_1s'] = aggregated_market_data['best_ask'].shift(1)
        aggregated_market_data['best_ask_1s'] = aggregated_market_data['best_ask'].shift(-1)

        out_df = pd.merge(
            left=isin_venue_trade_df,
            right=aggregated_market_data,
            left_on='time',
            right_on='datetime',
            how='left'
        )

        # mid price calculations
        out_df['mid_price'] = (out_df['best_bid'] + out_df['best_ask']) / 2
        out_df['mid_price_min_1s'] = (out_df['best_bid_min_1s'] + out_df['best_ask_min_1s']) / 2
        out_df['mid_price_1s'] = (out_df['best_bid_1s'] + out_df['best_ask_1s']) / 2

        # slippage
        out_df['slippage'] = np.where(
            out_df['side'] == 2,
            (out_df['price'] - out_df['best_bid']) / (out_df['best_ask'] - out_df['best_bid']),
            (out_df['best_ask'] - out_df['price']) / (out_df['best_ask'] - out_df['best_bid'])
        )
        # where there are inf, fill in with na.
        out_df.replace(np.inf, np.nan, inplace=True)

        merged_isin_venue_data.append(out_df)

    result_df = pd.concat(merged_isin_venue_data)