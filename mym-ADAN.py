import tkinter as tk
from tkinter import messagebox
import MetaTrader5 as mt5
import threading
import time

def main():
    global root, connect_button
    root = tk.Tk()
    root.title("MT5 Debug")
    root.geometry("400x300")

    login_label = tk.Label(root, text="Login:", font=("Helvetica", 12))
    login_label.pack(pady=5)
    login_entry = tk.Entry(root, font=("Helvetica", 12))
    login_entry.pack(pady=5)
    login_entry.insert(0, "312128713")

    password_label = tk.Label(root, text="Password:", font=("Helvetica", 12))
    password_label.pack(pady=5)
    password_entry = tk.Entry(root, show="*", font=("Helvetica", 12))
    password_entry.pack(pady=5)
    password_entry.insert(0, "restore")

    server_label = tk.Label(root, text="Server:", font=("Helvetica", 12))
    server_label.pack(pady=5)
    server_entry = tk.Entry(root, font=("Helvetica", 12))
    server_entry.pack(pady=5)
    server_entry.insert(0, "XMGlobal-MT5 7")

    connect_button = tk.Button(root, text="Connect", font=("Helvetica", 12),
                               command=lambda: connect_to_mt5(login_entry.get(), password_entry.get(), server_entry.get()))
    connect_button.pack(pady=20)

    root.mainloop()

def connect_to_mt5(login, password, server):
    if not mt5.initialize():
        messagebox.showerror("Error", "initialize() failed")
        return

    authorized = mt5.login(login=int(login), password=password, server=server)
    if authorized:
        messagebox.showinfo("Success", "Connected to MT5!")
        threading.Thread(target=delayed_trade, daemon=True).start()
    else:
        messagebox.showerror("Error", "Failed to connect to MT5")

def delayed_trade():
    time.sleep(15)  # Wait for 15 seconds before placing trade
    place_trade()

def place_trade():
    symbol = "BTCUSD"
    lot_size = 0.1
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        messagebox.showerror("Error", "Failed to get symbol info")
        return

    price = tick.ask
    take_profit = price + 500  # Example TP
    stop_loss = price - 500  # Example SL

    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": mt5.ORDER_TYPE_BUY,
        "price": price,
        "sl": stop_loss,
        "tp": take_profit,
        "deviation": 10,
        "magic": 234000,
        "comment": "Debug Trade",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        messagebox.showinfo("Success", "Trade placed successfully!")
    else:
        messagebox.showerror("Trade Error", f"Failed to place trade: {result.retcode}")

if __name__ == "__main__":
    main()
