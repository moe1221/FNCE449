import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf


def get_prices(tickers, start_date, end_date):
    """
    Fetches historical price data for given tickers between two dates.
    """
    prices = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    prices.ffill().bfill().dropna(axis=1, inplace=True)
    return prices
def calculate_returns(prices):
    """
    Calculates daily returns of the prices.
    """
    return prices / prices.shift(1) - 1

def calculate_mean(returns):
    """
    Calculates the average of returns
    """
    return returns.mean()

def calculate_std_dev(returns):
    """
    Calculates the standard deviation of returns.
    """
    return returns.std(ddof=1)

def calculate_sharpe_ratio(returns, std_dev):
    """
    Calculates the modified Sharpe ratio using returns and standard deviation.
    """
    mean_returns = returns.mean()
    return abs(mean_returns) / std_dev
def normalize_prices(prices):
    norm = prices/prices.iloc[0]
    return norm
def renormalize_from_point(data, point):
    """
    Renormalizes the data from a given point in the DataFrame.
    """
    new_base = data.iloc[point]
    data.iloc[point:] = data.iloc[point:] / new_base * 1
    return data
def sensitivity_analysis(leveraged_etf_mapping, start, end, x_values):
    """
    Performs sensitivity analysis for a given set of leveraged ETFs, thresholds, and date range.
    
    Parameters:
        leveraged_etf_mapping (dict): Mapping of ETFs with Long 2x and Short 2x leveraged ETFs.
        start (str): Start date for the price data.
        end (str): End date for the price data.
        x_values (list): List of thresholds for sensitivity analysis.
    
    Returns:
        Sensitivity DataFrame with final M2M Portfolio values for each threshold.
    """
    
    # Initialize the sensitivity DataFrame
    sensitivity_df = pd.DataFrame(index=x_values, columns=list(leveraged_etf_mapping.keys()))
    
    # Gather all unique ETFs for download
    all_etfs = set()
    for mapping in leveraged_etf_mapping.values():
        all_etfs.update([mapping['Long 2x'], mapping['Short 2x']])
    
    # Download the prices for all leveraged ETFs
    all_prices = get_prices(list(all_etfs),start,end)
    
    # Prepare price data dictionary for each ticker
    price_data = {}
    for ticker, mapping in leveraged_etf_mapping.items():
        Long_2x = mapping['Long 2x']
        Short_2x = mapping['Short 2x']
        price_data[ticker] = all_prices[[Long_2x, Short_2x]]
    
    # Perform sensitivity analysis across x_values
    for x in x_values:
        min_threshold = 1 - x
        max_threshold = 1 + x
        m2m_rebalancing_portfolio_df = pd.DataFrame()
        
        for ticker in leveraged_etf_mapping.keys():
            Long_2x = leveraged_etf_mapping[ticker]['Long 2x']
            Short_2x = leveraged_etf_mapping[ticker]['Short 2x']
            lev_prices = price_data[ticker]
            
            # Normalize prices and initialize tracking columns
            lev_norm = lev_prices / lev_prices.iloc[0] * 1
            lev_norm['daily profit'] = 0.0
            lev_norm['M2M Portfolio'] = 0.0
            
            # Iterate through each day and apply renormalization based on thresholds
            for i in range(1, len(lev_norm)):
                if (lev_norm.iloc[i][[Long_2x, Short_2x]] < min_threshold).any() or \
                   (lev_norm.iloc[i][[Long_2x, Short_2x]] > max_threshold).any():
                    lev_norm = renormalize_from_point(lev_norm, i)
                
                # Calculate daily profit/loss and accumulate M2M Portfolio
                if (lev_norm[Long_2x].iloc[i] == 1) and (lev_norm[Short_2x].iloc[i] == 1):
                    lev_norm.at[lev_norm.index[i], 'daily profit'] = 0.0
                else:
                    lev_norm.at[lev_norm.index[i], 'daily profit'] = (
                        (lev_norm[Long_2x].iloc[i-1] + lev_norm[Short_2x].iloc[i-1]) - 
                        (lev_norm[Long_2x].iloc[i] + lev_norm[Short_2x].iloc[i])
                    )
                
                lev_norm.at[lev_norm.index[i], 'M2M Portfolio'] = (
                    lev_norm['M2M Portfolio'].iloc[i-1] + lev_norm['daily profit'].iloc[i]
                )
            
            # Store final M2M Portfolio values
            m2m_rebalancing_portfolio_df[ticker] = lev_norm['M2M Portfolio']
        
        # Append final values for each ticker at the current threshold to sensitivity DataFrame
        sensitivity_df.loc[x] = m2m_rebalancing_portfolio_df.iloc[-1]
    return sensitivity_df

def calculate_m2m_portfolio(leveraged_etf_mapping, testing_start_date, testing_end_date):
    """
    Calculates the M2M Portfolio for each ticker in the leveraged ETF mapping within the specified date range.
    
    Parameters:
        leveraged_etf_mapping (dict): A dictionary with tickers as keys and a dictionary of 'Long 2x' and 'Short 2x' ETFs as values.
        testing_start_date (str): Start date for the test period.
        testing_end_date (str): End date for the test period.
    
    Returns:
        DataFrame with the M2M Portfolio for each ticker.
    """
    
    m2m_portfolio_df = pd.DataFrame()
    
    for ticker in leveraged_etf_mapping.keys():
        Long_2x = leveraged_etf_mapping[ticker]['Long 2x']
        Short_2x = leveraged_etf_mapping[ticker]['Short 2x']
        
        # Download prices for the leveraged ETFs within the date range
        lev_prices = yf.download([Long_2x, Short_2x], start=testing_start_date, end=testing_end_date)['Adj Close']
        
        # Normalize prices and calculate M2M Portfolio
        lev_norm = lev_prices / lev_prices.iloc[0] * 1
        lev_norm['M2M Portfolio'] = (lev_norm[Long_2x].iloc[0] + lev_norm[Short_2x].iloc[0]) - (lev_norm[Long_2x] + lev_norm[Short_2x])
        
        # Store M2M Portfolio in the result DataFrame
        m2m_portfolio_df[ticker] = lev_norm['M2M Portfolio']
        """plt.figure(figsize=(14, 7))
        plt.plot(lev_norm[Long_2x], label=f'{Long_2x} (2x Long)', color='blue')
        plt.plot(lev_norm[Short_2x], label=f'{Short_2x} (2x Short)', color='red')
        plt.plot(lev_norm['M2M Portfolio'], label='M2M Portfolio', color='green')
        plt.title(f'Strategy 1 Long vs Short Leveraged ETFs and M2M Portfolio for {ticker}')
        plt.xlabel('Date')
        plt.ylabel('Normalized Price')
        plt.legend(loc='best')
        plt.grid(False)
        plt.savefig(f"{ticker}_start1.png", dpi=1200)
        plt.show()"""   # this is the figures used for the report uncoment if you would like to view them
    
    return m2m_portfolio_df


def calculate_and_plot_m2m_portfolio(leveraged_etf_mapping, sensitivity_df, testing_start_date, testing_end_date):
    """
    Calculates and plots the M2M portfolio for each ticker based on optimal rebalancing thresholds.
    
    Parameters:
        leveraged_etf_mapping (dict): Mapping of tickers to their respective Long and Short leveraged ETFs.
        sensitivity_df (pd.DataFrame): DataFrame with sensitivity analysis results to determine optimal threshold.
        testing_start_date (str): Start date for the testing period.
        testing_end_date (str): End date for the testing period.
    
    Returns:
        A DataFrame containing M2M portfolio values over time for each ticker.
    """
    
    # Prepare data storage
    price_data = {}
    all_etfs = set()
    for mapping in leveraged_etf_mapping.values():
        all_etfs.update([mapping['Long 2x'], mapping['Short 2x']])

    # Download prices for all leveraged ETFs
    all_prices = yf.download(list(all_etfs), start=testing_start_date, end=testing_end_date)['Adj Close']
    
    # Store prices for each ticker
    for ticker, mapping in leveraged_etf_mapping.items():
        Long_2x = mapping['Long 2x']
        Short_2x = mapping['Short 2x']
        price_data[ticker] = all_prices[[Long_2x, Short_2x]]

    # Initialize DataFrame for M2M portfolio values
    m2m_rebalancing_portfolio_df = pd.DataFrame()

    # Calculate M2M Portfolio with rebalancing for each ticker
    for ticker in leveraged_etf_mapping.keys():
        Long_2x = leveraged_etf_mapping[ticker]['Long 2x']
        Short_2x = leveraged_etf_mapping[ticker]['Short 2x']
        x_optimal = float(sensitivity_df[ticker].idxmax())
        min_threshold = 1 - x_optimal
        max_threshold = 1 + x_optimal

        # Get and normalize ETF prices
        lev_prices = price_data[ticker]
        lev_norm = lev_prices / lev_prices.iloc[0] * 1
        lev_norm['daily profit'] = 0.0
        lev_norm['M2M Portfolio'] = 0.0

        # Iterate through days to apply rebalancing strategy
        for i in range(1, len(lev_norm)):
            if (lev_norm.iloc[i][[Long_2x, Short_2x]] < min_threshold).any() or (lev_norm.iloc[i][[Long_2x, Short_2x]] > max_threshold).any():
                lev_norm = renormalize_from_point(lev_norm, i)

            # Calculate daily profit or set to zero if normalized
            if (lev_norm[Long_2x].iloc[i] == 1) and (lev_norm[Short_2x].iloc[i] == 1):
                lev_norm.at[lev_norm.index[i], 'daily profit'] = 0.0
            else:
                lev_norm.at[lev_norm.index[i], 'daily profit'] = (
                    (lev_norm[Long_2x].iloc[i-1] + lev_norm[Short_2x].iloc[i-1]) - 
                    (lev_norm[Long_2x].iloc[i] + lev_norm[Short_2x].iloc[i])
                )
            lev_norm.at[lev_norm.index[i], 'M2M Portfolio'] = (
                lev_norm['M2M Portfolio'].iloc[i-1] + lev_norm['daily profit'].iloc[i]
            )
        
        # Store M2M Portfolio in the main DataFrame
        m2m_rebalancing_portfolio_df[ticker] = lev_norm['M2M Portfolio']
        
        # Plot M2M Portfolio and ETF Prices
        """axs = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

        # Plot M2M Portfolio
        axs[0].plot(lev_norm.index, lev_norm['M2M Portfolio'], color='blue', label='M2M Portfolio')
        axs[0].set_title(f'Strategy 2 M2M Portfolio for {ticker}')
        axs[0].set_ylabel('M2M Portfolio Value')
        axs[0].legend()
        axs[0].grid(False)

        # Plot ETF Prices
        axs[1].plot(lev_norm.index, lev_norm[Long_2x], color='green', label=f'{Long_2x} Price')
        axs[1].plot(lev_norm.index, lev_norm[Short_2x], color='red', label=f'{Short_2x} Price')
        axs[1].set_title(f'ETF Position for {ticker}')
        axs[1].set_xlabel('Date')
        axs[1].set_ylabel('Normalized Price')
        axs[1].legend()
        axs[1].grid(False)

        plt.tight_layout()
        plt.savefig(f"{ticker}_start2.png", dpi=1200)
        plt.show()""" # this is the figures used for the report uncoment if you would like to view them
    
    return m2m_rebalancing_portfolio_df

if __name__ == "__main__":
    # Define the leveraged ETF mapping
    leveraged_etf_mapping = {
        "GC=F": {"Long 2x": "UGL", "Short 2x": "GLL"},    # Gold
        "SI=F": {"Long 2x": "AGQ", "Short 2x": "ZSL"},    # Silver
        "CL=F": {"Long 2x": "UCO", "Short 2x": "SCO"},    # Crude Oil
        "NG=F": {"Long 2x": "BOIL", "Short 2x": "KOLD"},  # Natural Gas
        "GDX" : {"Long 2x": "NUGT", "Short 2x": "DUST"},  # Gold Miners
        "^SPX": {"Long 2x": "SSO", "Short 2x": "SDS"}     # S&P 500
    }

    # Define sensitivity analysis thresholds
    x_values = [0.05, 0.075, 0.1, 0.125, 0.15, 0.175, 0.2, 0.225, 0.25]
    training_start_date = '2012-01-01'
    training_end_date = '2022-01-01'
    testing_start_date = '2022-01-01'
    testing_end_date = '2024-01-01'

    analyzed_securities_prices = get_prices(list(leveraged_etf_mapping.keys()),training_start_date,training_end_date)
    analyzed_securities_returns = calculate_returns(analyzed_securities_prices)
    analyzed_securities_mean = calculate_mean(analyzed_securities_returns)
    analyzed_securities_stdev = calculate_std_dev(analyzed_securities_returns)
    analyzed_securities_modified_sharpe = calculate_sharpe_ratio(analyzed_securities_mean,analyzed_securities_stdev)
    print("Mean Returns:", analyzed_securities_mean)
    print("Standard Deviation:", analyzed_securities_stdev)
    print("Sharpe Ratio:", analyzed_securities_modified_sharpe)

    # Perform sensitivity analysis to determine optimal thresholds
    sensitivity_df = sensitivity_analysis(leveraged_etf_mapping, training_start_date, training_end_date, x_values)
    print("Sensitivity Analysis Results:\n", sensitivity_df)

    # Calculate M2M Portfolio without rebalancing
    m2m_portfolio_df = calculate_m2m_portfolio(leveraged_etf_mapping, testing_start_date, testing_end_date)
    print("\nM2M Portfolio (without rebalancing):\n", m2m_portfolio_df)

    # Calculate and plot M2M Portfolio with optimal rebalancing thresholds
    m2m_rebalancing_portfolio_df = calculate_and_plot_m2m_portfolio(
        leveraged_etf_mapping, 
        sensitivity_df, 
        testing_start_date, 
        testing_end_date
    )
    print("\nM2M Portfolio with Rebalancing:\n", m2m_rebalancing_portfolio_df)