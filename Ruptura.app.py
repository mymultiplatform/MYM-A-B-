import tkinter as tk
from tkinter import messagebox
import yfinance as yf

# Function to fetch live BTC price
def fetch_btc_price():
    btc = yf.Ticker("BTC-USD")
    btc_data = btc.history(period="1d")
    live_price = btc_data["Close"].iloc[-1]
    entry_btc_price.delete(0, tk.END)
    entry_btc_price.insert(0, f"{live_price:.2f}")
    label_result.config(text="Fetched live BTC price!")

# Function to calculate rupture price
def calculate_rupture():
    try:
        balance = float(entry_balance.get())
        entry_price = float(entry_btc_price.get())
        spread_cost = 2  # Assuming a $2 spread
        lot_size = 1  # 1 BTC position size
        leverage = 10  # 1:10 leverage
        
        # Calculate maximum allowable loss (balance minus spread)
        max_loss = balance - spread_cost
        
        # Calculate rupture price
        rupture_price = entry_price - (max_loss / lot_size)
        
        # Display result
        label_result.config(text=f"Rupture Price: ${rupture_price:.2f}")
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numbers for balance and BTC price.")

# Set up Tkinter window
root = tk.Tk()
root.title("RUPTURA APP")
root.geometry("400x400")

# Labels and Entries
tk.Label(root, text="RUPTURA APP", font=("Arial", 16)).pack(pady=10)

tk.Label(root, text="BTC LIVE PRICE").pack()
entry_btc_price = tk.Entry(root)
entry_btc_price.pack()

# Button to fetch live BTC price
button_fetch_price = tk.Button(root, text="Fetch Live BTC Price", command=fetch_btc_price)
button_fetch_price.pack(pady=5)

tk.Label(root, text="YOUR BALANCE").pack()
entry_balance = tk.Entry(root)
entry_balance.insert(0, "370")  # Example starting balance
entry_balance.pack()

# Start Calculation Button
button_calculate = tk.Button(root, text="START CALCULATION", command=calculate_rupture)
button_calculate.pack(pady=20)

# Display the result
label_result = tk.Label(root, text="PRICE FOR RUPTURE CALCULATION: ", font=("Arial", 12))
label_result.pack(pady=20)

# Run the app
root.mainloop()
