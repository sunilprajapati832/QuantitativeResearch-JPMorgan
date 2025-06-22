import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load the CSV file
df = pd.read_csv('natural_gas_prices.csv')
print(df.columns)


# Preview data
print(df.head())
print(df.info())

df['Dates'] = pd.to_datetime(df['Dates'], format='%m/%d/%y')

# Convert the date column and set it as index
df['Dates'] = pd.to_datetime(df['Dates'])
df.set_index('Dates', inplace=True)

# Visualize the data to find patterns and seasonality
plt.figure(figsize=(12,6))
plt.plot(df.index, df['Prices'], marker='o')
plt.title('Monthly Natural Gas Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.grid(True)
plt.show()

# Analyze seasonality by month
df['month'] = df.index.month
monthly_avg = df.groupby('month')['Prices'].mean()

plt.figure(figsize=(8,5))
monthly_avg.plot(kind='bar')
plt.title('Average Price by Month (Seasonality)')
plt.xlabel('Month')
plt.ylabel('Average Price')
plt.show()

# Interpolate to estimate prices on any date within data range
# Resample data to daily frequency and interpolate missing days linearly
daily_prices = df['Prices'].resample('D').interpolate(method='linear')

# Extrapolate prices for next 1 year (Oct 2024 to Sept 2025)
# For simple extrapolation (seasonal decomposition or seasonal naive method)



# Repeat the last 12 months (seasonal naive) for extrapolation
last_year = daily_prices['2023-10-01':'2024-09-30']
extrapolated_dates = pd.date_range(start='2024-10-01', end='2025-09-30', freq='D')

# To keep seasonality, repeat last year's prices
extrapolated_prices = np.tile(last_year.values, int(np.ceil(len(extrapolated_dates)/len(last_year))))[:len(extrapolated_dates)]
extrapolated_series = pd.Series(data=extrapolated_prices, index=extrapolated_dates)

# Combine original and extrapolated data
full_series = pd.concat([daily_prices, extrapolated_series])

# Create a function to return price estimate for any date
def estimate_price(date_str):
    date = pd.to_datetime(date_str)
    if date in full_series.index:
        return full_series.loc[date]
    else:
        return "Date out of range. Please choose between 2020-10-31 and 2025-09-30."

# Example usage:
print(estimate_price('2021-05-15'))
print(estimate_price('2025-06-20'))

# Visualize full data with extrapolation
plt.figure(figsize=(14,6))
plt.plot(full_series.index, full_series.values, label='Price Estimate')
plt.axvline(pd.Timestamp('2024-09-30'), color='red', linestyle='--', label='Data End')
plt.title('Natural Gas Price Estimate with Extrapolation')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid(True)
plt.show()


