import tkinter as tk
from tkinter import messagebox
import yfinance as yf
import threading
import time

# Function to fetch live BTC price with retry mechanism
def fetch_btc_price():
    def task():
        try:
            button_fetch_price.config(state='disabled')
            label_result.config(text="Fetching live BTC price...")
            retries = 3
            for attempt in range(retries):
                try:
                    btc = yf.Ticker("BTC-USD")
                    btc_data = btc.history(period="1d")
                    live_price = btc_data["Close"].iloc[-1]
                    entry_btc_price.delete(0, tk.END)
                    entry_btc_price.insert(0, f"{live_price:.2f}")
                    label_result.config(text="Fetched live BTC price!")
                    break
                except Exception as fetch_exception:
                    if attempt < retries - 1:
                        time.sleep(2)  # Wait before retrying
                        continue
                    else:
                        raise fetch_exception
        except Exception as e:
            messagebox.showerror("Error", f"Failed to fetch BTC price:\n{e}")
            label_result.config(text="Failed to fetch BTC price.")
        finally:
            button_fetch_price.config(state='normal')

    threading.Thread(target=task).start()

# Function to calculate rupture price
def calculate_rupture():
    try:
        # Retrieve and validate user inputs
        balance = float(entry_balance.get())
        entry_price = float(entry_btc_price.get())
        spread_cost = float(entry_spread.get())
        lot_size = float(entry_lot_size.get())
        maintenance_margin = float(entry_maintenance_margin.get())

        if balance <= 0:
            raise ValueError("Balance must be greater than 0.")
        if spread_cost < 0:
            raise ValueError("Spread cost cannot be negative.")
        if lot_size <= 0:
            raise ValueError("Lot size must be greater than 0.")
        if not (0 <= maintenance_margin < 100):
            raise ValueError("Maintenance margin must be between 0 and 100 percent.")

        # Get selected leverage
        leverage = leverage_var.get()
        if leverage not in [1, 10, 50, 100]:
            raise ValueError("Invalid leverage selected.")

        # Calculate maximum allowable loss considering maintenance margin
        max_loss = balance - spread_cost - (balance * (maintenance_margin / 100))
        if max_loss <= 0:
            raise ValueError("Balance is too low to cover the spread cost and maintenance margin.")

        # Calculate price movement considering leverage and lot size
        price_movement = max_loss / (leverage * lot_size)

        # Calculate rupture price
        rupture_price = entry_price - price_movement

        # Calculate percentage drop
        price_drop_percentage = (price_movement / entry_price) * 100

        label_result.config(
            text=f"Rupture Price (1:{leverage}): ${rupture_price:.2f}\n"
                 f"Price Drop: {price_drop_percentage:.2f}%"
        )
    except ValueError as ve:
        messagebox.showerror("Input Error", f"{ve}")
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred:\n{e}")

# Set up Tkinter window
root = tk.Tk()
root.title("RUPTURA APP")
root.geometry("600x730")  # Increased height for additional UI elements

# Main Title
tk.Label(root, text="RUPTURA APP", font=("Arial", 20, "bold")).pack(pady=10)

# Banner Note
note_text = "Note: This calculation assumes that you have bought the minimum lot size in a leveraged account."
banner = tk.Label(
    root,
    text=note_text,
    font=("Arial", 10),
    fg="white",
    bg="#2E86C1",  # A distinct blue color for the banner
    wraplength=580,  # Adjusted to fit within the window width
    justify="left",
    padx=10,
    pady=5
)
banner.pack(pady=(0, 20), fill="x")  # Added padding below the banner

# Create a frame for inputs
input_frame = tk.Frame(root)
input_frame.pack(pady=10, padx=20, fill="x")

# BTC Live Price Section
tk.Label(input_frame, text="BTC LIVE PRICE ($)", font=("Arial", 12)).grid(row=0, column=0, sticky="w", pady=5)
entry_btc_price = tk.Entry(input_frame, font=("Arial", 12))
entry_btc_price.grid(row=0, column=1, pady=5, sticky="e")
button_fetch_price = tk.Button(
    input_frame,
    text="Live BTC Price",
    command=fetch_btc_price,
    font=("Arial", 12)
)
button_fetch_price.grid(row=0, column=2, padx=10, pady=5)

# User Balance Section
tk.Label(input_frame, text="YOUR BALANCE ($)", font=("Arial", 12)).grid(row=1, column=0, sticky="w", pady=5)
entry_balance = tk.Entry(input_frame, font=("Arial", 12))
entry_balance.insert(0, "370")  # Example starting balance
entry_balance.grid(row=1, column=1, pady=5, sticky="e")

# Spread Cost Section
tk.Label(input_frame, text="SPREAD COST ($)", font=("Arial", 12)).grid(row=2, column=0, sticky="w", pady=5)
entry_spread = tk.Entry(input_frame, font=("Arial", 12))
entry_spread.insert(0, "2")  # Default spread cost
entry_spread.grid(row=2, column=1, pady=5, sticky="e")

# Lot Size Section
tk.Label(input_frame, text="LOT SIZE (BTC)", font=("Arial", 12)).grid(row=3, column=0, sticky="w", pady=5)
entry_lot_size = tk.Entry(input_frame, font=("Arial", 12))
entry_lot_size.insert(0, "1")  # Default lot size
entry_lot_size.grid(row=3, column=1, pady=5, sticky="e")

# Maintenance Margin Section
tk.Label(input_frame, text="MAINTENANCE MARGIN (%)", font=("Arial", 12)).grid(row=4, column=0, sticky="w", pady=5)
entry_maintenance_margin = tk.Entry(input_frame, font=("Arial", 12))
entry_maintenance_margin.insert(0, "50")  # Default maintenance margin
entry_maintenance_margin.grid(row=4, column=1, pady=5, sticky="e")

# Leverage Selection Section
tk.Label(input_frame, text="SELECT LEVERAGE", font=("Arial", 12)).grid(row=5, column=0, sticky="w", pady=(20,5))

leverage_var = tk.IntVar()
leverage_var.set(10)  # Default leverage

# Define leverage options
leverage_options = [("1:1", 1), ("1:10", 10), ("1:50", 50), ("1:100", 100)]

# Create a frame for leverage radio buttons
leverage_frame = tk.Frame(input_frame)
leverage_frame.grid(row=5, column=1, pady=(20,5), sticky="w")

# Create radio buttons for leverage selection
for text, value in leverage_options:
    tk.Radiobutton(
        leverage_frame,
        text=text,
        variable=leverage_var,
        value=value,
        font=("Arial", 12)
    ).pack(anchor="w")

# Start Calculation Button
button_calculate = tk.Button(
    root,
    text="START CALCULATION",
    command=calculate_rupture,
    font=("Arial", 12, "bold"),
    bg="#28B463",
    fg="white",
    width=25,
    height=2
)
button_calculate.pack(pady=20)

# Display the result
label_result = tk.Label(root, text="PRICE FOR RUPTURE CALCULATION: ", font=("Arial", 12))
label_result.pack(pady=20)

# Run the app
root.mainloop()
