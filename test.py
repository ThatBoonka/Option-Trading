import pandas as pd
import yfinance as yf
from datetime import datetime
import numpy as np
from scipy.stats import norm
from scipy.optimize import newton
import plotly.graph_objects as go

# SECTION 1: BLACK-SCHOLES AND IMPLIED VOLATILITY SOLVER

def black_scholes(S, K, T, r, sigma, q=0, otype='call'):
    """ Calculates the theoretical price of a European option using the Black-Scholes model.
            S: The current price of the underlying asset.
            K: The strike price of the option.
            T: Time to expiration in years.
            r: The risk-free interest rate.
            Sigma: The implied volatility.
            q: The dividend yield (default is 0 for non-dividend stocks).
        Returns theoretical price of the option as a float or numpy array. """
    
    T = np.maximum(T, 1e-9)
    sigma = np.maximum(sigma, 1e-9)
    
    # Calculate the d1 and d2 parameters, which are central to the Black-Scholes formula.
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Use the d1 and d2 values with the standard normal CDF (norm.cdf) to compute price based on the option type
    if otype == 'call':
        return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)

def solve_for_iv_scipy(S, K, T, r, price, otype='call', sigma_guess=0.5):
    #Solves IV of an option using Newton-Raphson method to find volatility that makes Black-Scholes price = market price
    #Returns implied volatility as a float
    
    def loss(sigma):
        return black_scholes(S, K, T, r, sigma, otype=otype) - price

    try:
        # `scipy.optimize.newton` takes loss function and an initial sigma_guess to find the root, which is our implied volatility.
        iv = newton(loss, sigma_guess, tol=1e-5, maxiter=100)
        
        if iv < 0:
            return np.nan
        return iv
    except RuntimeError:
        return np.nan


# SECTION 2: DATA FETCHING AND PROCESSING 

def dte_to_years(dte_str):
    dte = datetime.strptime(dte_str, '%Y-%m-%d')
    today = datetime.now()
    return (dte - today).days / 365.0

def get_options_data(ticker_symbol, r=0.0423, otype='call'):
    # Processes options data for a given ticker from Yahoo Finance.
    # Retrieves the current stock price and then iterates through all available expiration dates to build a DataFrame of options contracts.
    # Returns a DataFrame with the data needed for IV calculation and plotting. """

    ticker = yf.Ticker(ticker_symbol)
    
    try:
        S = ticker.history(period="1d")['Close'].iloc[-1]
        # Get a list of all available expiration dates for the ticker.
        expirations = ticker.options
    except IndexError:
        print('Invail stock ticker')
        return pd.DataFrame()
    except Exception as e:
        print(e)
        return pd.DataFrame()

    all_options_data = []

    # Loop through each expiration date to build the full options chain.
    for exp_date in expirations:
        chain = ticker.option_chain(exp_date)
        if otype == 'call':
            options_df = chain.calls 
        else:
            options_df = chain.puts
        
        options_df['T'] = dte_to_years(exp_date)
        options_df['S'] = S
        options_df['r'] = r
        options_df['otype'] = otype
        options_df['price'] = options_df[['bid', 'ask']].mean(axis=1) # market price is midpoint of  bid and ask price
        options_df['moneyness'] = options_df['S'] / options_df['strike']
        all_options_data.append(options_df)
      
    if not all_options_data:
        return pd.DataFrame()

    # Combine all the individual DataFrames into a single, comprehensive one.
    # Perform final data cleaning and filtering to remove bad data points.
    full_df = pd.concat(all_options_data, ignore_index=True)
    full_df = full_df[full_df['volume'].notna() & (full_df['volume'] > 0)]
    full_df = full_df[(full_df['price'] > 0) & (full_df['strike'] > 0) & (full_df['T'] > 0)]
    full_df = full_df.dropna(subset=['strike', 'price', 'T', 'S'])
    
    # Rename the 'strike' column to 'K' for consistency with the Black-Scholes formula.
    return full_df.rename(columns={'strike': 'K'})

# SECTION 3: MAIN EXECUTION AND PLOTTING

def main():
    ticker_symbol = input('Enter Ticker: ')
    
    print('Fetching options data for {}...'.format(ticker_symbol))
    options_df = get_options_data(ticker_symbol, otype='call')

    print('Data retrieved successfully. Found {} contracts. Calculating Implied Volatility...'.format(len(options_df)))
    
    options_df['ivs'] = options_df.apply(
        lambda row: solve_for_iv_scipy(
            row['S'], row['K'], row['T'], row['r'], row['price'], otype=row['otype']),axis=1)
    
    options_df = options_df[options_df['ivs'].notna() & (options_df['ivs'] > 0)]

    print('IV calculation complete. {} contracts processed. Plotting data...'.format(len(options_df)))

    moneyness = options_df['moneyness'].values
    T_days = options_df['T'].values * 365
    ivs = options_df['ivs'].values

    fig = go.Figure(data=[go.Mesh3d(x=moneyness, y=T_days, z=ivs, intensity=ivs,
        colorscale='Inferno', colorbar=dict(title='Implied Volatility'),
        showscale=True, opacity=0.8, name='IV Surface (Mesh)')])

    # Add the raw data points on top of the mesh for reference
    fig.add_trace(go.Scatter3d(x=moneyness, y=T_days, z=ivs, mode='markers',
        marker=dict(size=2, color='black', opacity=0.4), name='Raw Data Points' ))

    fig.update_layout(
        title=f'Implied Volatility Surface for {ticker_symbol}',
        scene=dict(
            xaxis_title='Moneyness (S/K)',
            yaxis_title='Time to Expiration (Days)',
            zaxis_title='Implied Volatility',
            xaxis=dict(range=[.82, 1.18]),
            yaxis=dict(range=[-5, 100]),
            zaxis=dict(range=[0, max(ivs)*1.2])),
            margin=dict(l=0, r=0, b=0, t=50))
    fig.show()

if __name__ == '__main__':
    main()