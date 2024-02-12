# Added on Friday 9 Feb by Philip P
# Take Home Test Submissions for Financial Company

import logging
import os
import time
from typing import Dict, Union

import numpy as np
import pandas as pd

pd.options.display.max_columns = 14
pd.options.display.width = 800

# setup timer log for performance metrics - Task 5
def timed_log(func):
    """
    Decorator that logs the duration of the function call.
    """

    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logging.info(f"Function {func.__name__} took {end_time - start_time:.2f} seconds to complete.")
        return result

    return wrapper


@timed_log
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
@timed_log
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
@timed_log
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


@timed_log
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
    return market_data_merged_df


def isolate_traded_stocks_venues(
        executions_df: pd.DataFrame,
        market_data_enriched_df: pd.DataFrame
) -> np.ndarray:
    """
   Identify ISINs and venues that are traded and have market data.

   Parameters:
       executions_df (DataFrame): Dataframe containing trade execution data.
       market_data_enriched_df (DataFrame): Dataframe containing enriched market data.

   Returns:
       ndarray: An array of unique ISIN_venue identifiers present in both executions and market data.
   """
    unique_traded_isin_venues = executions_df['isin_venue'].unique()
    isin_traded_in_market_data = unique_traded_isin_venues[
        np.in1d(unique_traded_isin_venues, market_data_enriched_df['isin_venue'].unique())
    ]
    logging.info(f"Located {len(isin_traded_in_market_data)} stocks traded corresponding to matching"
                 f"market data out of a total possible {market_data_enriched_df['isin_venue'].nunique()} stocks. "
                 f"Dropping those that could not be matched.")
    return isin_traded_in_market_data


def return_market_data_by_execution_time(
        single_name_trade_df: pd.DataFrame,
        market_data_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Aggregates market data around the trade execution times from a single-name trade DataFrame.

    This function takes a DataFrame of trades for a single financial instrument and a DataFrame
    of market data. It extracts market data at one second before, at, and one second after each
    unique trade execution time found in the trades DataFrame. The function then aggregates this
    market data to find the best bid and ask prices, calculates the mid price at each time point,
    and computes the slippage based on the trade side.

    Parameters:
        single_name_trade_df (pd.DataFrame): DataFrame containing trades of a single name,
                                           must include a 'time' column with trade execution times.
        market_data_df (pd.DataFrame): DataFrame containing market data with a 'datetime' column.

    Returns:
        pd.DataFrame: A DataFrame with the aggregated market data, and additional metrics such as best_bid, best_ask,
        mid-price (and the respective values plus and minus one second), and the slippage calculations
    """

    # create a dataframe with unique trade execution times
    unique_execution_times = pd.DataFrame(
        single_name_trade_df['time'].unique(), columns=['time']
    )
    # add one second before and after
    unique_execution_times['time_minus_one'] = unique_execution_times['time'] - pd.Timedelta(seconds=1)
    unique_execution_times['time_plus_one'] = unique_execution_times['time'] + pd.Timedelta(seconds=1)

    times_to_extract = pd.concat([
        unique_execution_times['time_minus_one'],
        unique_execution_times['time'],
        unique_execution_times['time_plus_one']
    ]).drop_duplicates().reset_index(drop=True)

    filtered_market_data = market_data_df.loc[market_data_df['time'].isin(times_to_extract)]

    mkt_data_agg_df = filtered_market_data.groupby('time').agg({
        'best_bid_price': 'max',  # The best bid price is the maximum bid price
        'best_ask_price': 'min'  # The best ask price is the minimum ask price
    }).reset_index().rename(columns={'best_bid_price': 'best_bid',
                                     'best_ask_price': 'best_ask'})

    mkt_data_agg_df['best_bid_min_1s'] = mkt_data_agg_df['best_bid'].shift(1)
    mkt_data_agg_df['best_bid_1s'] = mkt_data_agg_df['best_bid'].shift(-1)
    mkt_data_agg_df['best_ask_min_1s'] = mkt_data_agg_df['best_ask'].shift(1)
    mkt_data_agg_df['best_ask_1s'] = mkt_data_agg_df['best_ask'].shift(-1)

    # mid price calculations
    mkt_data_agg_df['mid_price'] = (mkt_data_agg_df['best_bid'] + mkt_data_agg_df['best_ask']) / 2
    mkt_data_agg_df['mid_price_min_1s'] = \
        (mkt_data_agg_df['best_bid_min_1s'] + mkt_data_agg_df['best_ask_min_1s']) / 2
    mkt_data_agg_df['mid_price_1s'] = (mkt_data_agg_df['best_bid_1s'] + mkt_data_agg_df['best_ask_1s']) / 2

    return mkt_data_agg_df


@timed_log
def run_stock_calculations(
        market_executions_df: pd.DataFrame,
        mkt_data_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Wrap up all code to run the calculations for the best bid and offer using the market data, reference data,
    execution data. Additionally, metrics such as the mid price are also calculated.

    Parameters:
        market_executions_df: A dataframe providing data on the executed trades
        mkt_data_df: A dataframe providing market data

    Returns:
        pd.DataFrame: A consolidated dataframe containing details such as the execution price, time, size of the
        market trades, and the requested calculations for the best bid and offer, mid price, and slippage (at
        execution price).
    """

    market_trades_executed_df = round_data_times(dataframe=market_executions_df,
                                                 time_column_name='tradetime')

    # drop all non-continuous trades in market data
    mkt_data_df = mkt_data_df.loc[mkt_data_df['market_state'] == 'CONTINUOUS_TRADING'].copy()
    mkt_data_df = round_data_times(dataframe=mkt_data_df,
                                   time_column_name='event_timestamp')

    mkt_data_df = add_isin_to_reference_data(market_data_df=mkt_data_df,
                                             reference_data_df=ref_data_df)

    market_trades_executed_df['isin_venue'] = market_trades_executed_df['isin'] + '_' + \
                                              market_trades_executed_df['venue']
    mkt_data_df['isin_venue'] = mkt_data_df['isin'] + '_' + mkt_data_df['primary_mic']

    located_isin_in_market_data = isolate_traded_stocks_venues(
        executions_df=market_trades_executed_df,
        market_data_enriched_df=mkt_data_df
    )

    # filter market data for only the executed ISIN on each venue
    market_data_df_subset = mkt_data_df.loc[
        mkt_data_df['isin_venue'].isin(located_isin_in_market_data)]

    # Check time columns in market data and execution data are equal.
    assert market_data_df_subset['time'].dtype == market_trades_executed_df['time'].dtype, \
        f'Time columns are not the same datatype'

    merged_isin_venue_data = []
    for isin_venue in located_isin_in_market_data:
        logging.info(f"Calculating best bid and offer/mid-price/slippage for: {isin_venue}...")
        # Match each trade execution timestamp from isin_venue_trade_df with the corresponding
        # market data entry in market_data_isin_venue_df.
        isin_venue_trade_df = market_trades_executed_df.loc[market_trades_executed_df['isin_venue'] == isin_venue]
        isin_venue_market_data_df = market_data_df_subset.loc[market_data_df_subset['isin_venue'] == isin_venue]

        # calculate desired metrics
        market_data_agg_df_stock = return_market_data_by_execution_time(
            single_name_trade_df=isin_venue_trade_df,
            market_data_df=isin_venue_market_data_df
        )

        out_df = pd.merge(
            left=isin_venue_trade_df,
            right=market_data_agg_df_stock,
            on='time',
            how='left'
        )

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

    return result_df


if __name__ == '__main__':
    if not os.path.exists('output'):
        os.mkdir('output')

    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        filename='output/data_analysis.log',  # Log file path
        filemode='w'  # 'w' to overwrite the log file each time, 'a' to append
    )

    # Load and preprocess the data
    ref_data_df = load_and_preprocess_data('data/refdata.parquet')
    market_data_df = load_and_preprocess_data('data/marketdata.parquet')
    executions_df = load_and_preprocess_data('data/executions.parquet')

    #   ######################
    # Task 1 - Count executions and venues
    executions_venues_result_dct = count_executions_and_venues(executions_df)
    logging.info(f"Number of executions: {executions_venues_result_dct['num_executions']:,}")
    logging.info(f"Number of unique venues: {executions_venues_result_dct['num_venues']}")
    logging.info(f"Executions by venue and date:\n{executions_venues_result_dct['summary_df']}\n")
    #   ######################

    #   ######################
    # Task 2 - Filter for CONTINUOUS TRADING and count
    filter_to_apply = 'CONTINUOUS_TRADING'
    continuous_executions_df = filter_continuous_trading(executions_df)
    logging.info(f"Number of {filter_to_apply} executions: {len(continuous_executions_df):,.0f}")
    #   ######################

    #   ######################
    # Task 3 - Data transformation and adding reference data
    executions_df = add_side_and_venue_information(
        executions_clean_df=continuous_executions_df,
        reference_data_df=ref_data_df
    )
    #   ######################

    #   ######################
    # Task 4. Calculations
    output_df = run_stock_calculations(
        market_executions_df=executions_df,
        mkt_data_df=market_data_df
    )
    logging.info("Completed calculations for traded stocks with market data corresponding to trade times.")
    output_df.to_csv('output/output_file.csv', index=False)
