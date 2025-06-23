import pandas as pd
import numpy as np
import logging
import matplotlib.pyplot as plt

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

# Load the CSV file
df = pd.read_csv('natural_gas_prices.csv')
df['Dates'] = pd.to_datetime(df['Dates'], format='%m/%d/%y')
df.set_index('Dates', inplace=True)

# Interpolate daily prices
daily_prices = df['Prices'].resample('D').interpolate(method='linear')

# Extrapolate prices from Oct 2024 to Sept 2025
last_year = daily_prices['2023-10-01':'2024-09-30']
extrapolated_dates = pd.date_range(start='2024-10-01', end='2025-09-30', freq='D')
extrapolated_prices = np.tile(last_year.values,
                              int(np.ceil(len(extrapolated_dates)/
                                          len(last_year))))[:len(extrapolated_dates)]
extrapolated_series = pd.Series(data=extrapolated_prices, index=extrapolated_dates)

# Combine full series
full_series = pd.concat([daily_prices, extrapolated_series])

# --- Price Lookup Function ---
def estimate_price(date_str):
    date = pd.to_datetime(date_str)
    if date in full_series.index:
        return full_series.loc[date]
    else:
        return "Date out of range. Please choose between 2020-10-31 and 2025-09-30."

# --- Enhanced Pricing Model Function ---
def price_generalized_storage_contract(
    injection_dates, withdrawal_dates,
    injection_rates, withdrawal_rates,
    max_storage_volume,
    storage_cost_per_day,
    plot_storage=True
):
    # --- Validation ---
    if not (len(injection_dates) == len(injection_rates)):
        raise ValueError("Mismatch in number of injection dates and rates")
    if not (len(withdrawal_dates) == len(withdrawal_rates)):
        raise ValueError("Mismatch in number of withdrawal dates and rates")

    injection_dates = pd.to_datetime(injection_dates)
    withdrawal_dates = pd.to_datetime(withdrawal_dates)

    all_dates = sorted(set(injection_dates.tolist() + withdrawal_dates.tolist()))
    start_date = min(all_dates)
    end_date = max(all_dates)
    timeline = pd.date_range(start=start_date, end=end_date, freq='D')

    # Initialize trackers
    storage_balance = 0
    storage_tracker = {}
    total_purchase_cost = 0
    total_revenue = 0
    total_storage_cost = 0
    storage_levels = []

    # --- Injections ---
    for date, rate in zip(injection_dates, injection_rates):
        price = estimate_price(date)
        if isinstance(price, str): raise ValueError(f"Invalid price on injection date {date}")
        if storage_balance + rate > max_storage_volume:
            logging.warning(f"Storage limit exceeded on {date}. Skipping injection.")
            continue
        total_purchase_cost += price * rate
        storage_tracker[date] = {'volume': rate, 'days_stored': 0}
        storage_balance += rate
        logging.info(f"Injected {rate} on {date.date()} at price ${price:.2f}")

    # --- Daily Tracking ---
    for current_date in timeline:
        for inj_date in list(storage_tracker.keys()):
            storage_tracker[inj_date]['days_stored'] += 1

        total_storage_cost += storage_cost_per_day * storage_balance
        storage_levels.append(storage_balance)

        # --- Withdrawals ---
        if current_date in withdrawal_dates:
            idx = withdrawal_dates.tolist().index(current_date)
            withdraw_volume = withdrawal_rates[idx]
            if withdraw_volume > storage_balance:
                logging.warning(f"Withdrawal on {current_date.date()} "
                                f"exceeds storage. Adjusting.")
                withdraw_volume = storage_balance
            remaining_to_withdraw = withdraw_volume
            price = estimate_price(current_date)
            if isinstance(price, str): raise ValueError(
                f"Invalid price on withdrawal date {current_date}")
            total_revenue += price * withdraw_volume
            logging.info(f"Withdrew {withdraw_volume} on "
                         f"{current_date.date()} at price ${price:.2f}")

            # FIFO Deduction
            for inj_date in sorted(storage_tracker.keys()):
                stored = storage_tracker[inj_date]['volume']
                if stored <= 0: continue
                if remaining_to_withdraw <= 0: break
                deduction = min(remaining_to_withdraw, stored)
                storage_tracker[inj_date]['volume'] -= deduction
                storage_balance -= deduction
                remaining_to_withdraw -= deduction

    # --- Plot storage level ---
    if plot_storage:
        plt.figure(figsize=(12,5))
        plt.plot(timeline, storage_levels, label="Storage Level (MMBtu)")
        plt.title("Storage Balance Over Time")
        plt.xlabel("Date")
        plt.ylabel("Volume")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # --- Final Report ---
    contract_value = total_revenue - total_purchase_cost - total_storage_cost
    return {
        'Total Purchase Cost': round(total_purchase_cost, 2),
        'Total Revenue': round(total_revenue, 2),
        'Total Storage Cost': round(total_storage_cost, 2),
        'Net Contract Value': round(contract_value, 2)
    }

# --- Sample Test Case ---
if __name__ == '__main__':
    result = price_generalized_storage_contract(
        injection_dates=['2024-06-01', '2024-07-01'],
        withdrawal_dates=['2024-11-01', '2024-12-15'],
        injection_rates=[500_000, 500_000],
        withdrawal_rates=[600_000, 400_000],
        max_storage_volume=1_000_000,
        storage_cost_per_day=1000
    )

    print("\n---- Contract Evaluation Summary ----")
    for k, v in result.items():
        print(f"{k}: ${v:,.2f}")
