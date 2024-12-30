from tkinter import Tk, Frame, messagebox, ttk
import MetaTrader5 as mt5
import threading
import time

# Global variables
trade_data = []  # Stores all trade data
profits_live = []  # Stores live profits for all trades
def initialize_profits():
    global profits_live
    profits_live = [0] * len(trade_data)

def fetch_live_profit(position_id):
    """
    Fetch live profit for an open position from MT5 API.
    """
    position_info = mt5.positions_get(ticket=position_id)
    if position_info and len(position_info) > 0:
        return position_info[0].profit  # Direct live profit from MT5 API
    return 0.0

def fetch_live_price(symbol):
    """
    Fetch the latest price for a given symbol from MT5 API.
    """
    tick = mt5.symbol_info_tick(symbol)
    if tick:
        return tick.ask, tick.bid
    return None, None

def update_live_profits_and_prices():
    """
    Periodically update live profits and prices, then refresh the UI table.
    """
    while True:
        for i, trade_id in enumerate(trade_data):
            trade_profit = fetch_live_profit(trade_id)  # Fetch live profit for this trade
            if i < len(profits_live):
                profits_live[i] = trade_profit
            else:
                profits_live.append(trade_profit)

            # Update the profit column in the UI
            if i < len(all_trades_table.get_children()):
                all_trades_table.set(
                    all_trades_table.get_children()[i],
                    column="Profit",
                    value=f"{trade_profit:.2f}"
                )
        
        # Update live prices
        symbol = "BTCUSD"
        ask, bid = fetch_live_price(symbol)
        if ask and bid:
            price_label.config(text=f"Ask: {ask:.2f} | Bid: {bid:.2f}")

        time.sleep(1)  # Refresh every second

def place_trade(order_type):
    """
    Place a trade and store its ID.
    """
    symbol = "BTCUSD"
    lot_size = 0.1
    tick = mt5.symbol_info_tick(symbol)
    if not tick:
        messagebox.showerror("Trade Error", "Symbol info not available")
        return

    price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
    request = {
        "action": mt5.TRADE_ACTION_DEAL,
        "symbol": symbol,
        "volume": lot_size,
        "type": order_type,
        "price": price,
        "deviation": 10,
        "magic": 234000,
        "comment": "Automated trade",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    result = mt5.order_send(request)
    if result.retcode == mt5.TRADE_RETCODE_DONE:
        trade_data.append(result.order)
        initialize_profits()
        all_trades_table.insert("", "end", values=(len(trade_data), result.order, "0.00"))
    else:
        messagebox.showerror("Trade Error", f"Failed to place trade: {result.retcode}")

def start_reinforcement_learning():
    """
    Simulate reinforcement learning logic, placing trades and updating UI.
    """
    num_trades = 10  # Reduced for testing
    for trade_idx in range(num_trades):
        order_type = mt5.ORDER_TYPE_BUY if trade_idx % 2 == 0 else mt5.ORDER_TYPE_SELL
        place_trade(order_type)
    
    # Start live profit updates in a separate thread
    threading.Thread(target=update_live_profits_and_prices, daemon=True).start()

def create_ui():
    """
    Create the main trading bot UI.
    """
    global all_trades_table, price_label

    root = Tk()
    root.title("MT5 Trading Bot Debug UI - Live Trade Supervision")
    root.geometry("800x600")

    left_frame = Frame(root, width=400, height=600)
    left_frame.pack(side="left", fill="both", expand=True)

    right_frame = Frame(root, width=400, height=600)
    right_frame.pack(side="right", fill="both", expand=True)

    # Left Frame - All Trades
    ttk.Label(left_frame, text="Live Trades (100)").pack()
    all_trades_table = ttk.Treeview(left_frame, columns=("Trade #", "Trade ID", "Profit"), show="headings", height=20)
    all_trades_table.heading("Trade #", text="Trade #")
    all_trades_table.heading("Trade ID", text="Trade ID")
    all_trades_table.heading("Profit", text="Profit")
    all_trades_table.pack(fill="both", expand=True)

    # Right Frame - Live Prices
    ttk.Label(right_frame, text="Live Prices (BTCUSD)").pack()
    price_label = ttk.Label(right_frame, text="Fetching prices...")
    price_label.pack()

    start_reinforcement_learning()
    root.mainloop()

def connect_mt5():
    """
    Connect to the MT5 platform.
    """
    if not mt5.initialize(login=312128713, server="XMGlobal-MT5 7", password="Sexo247420@"):
        messagebox.showerror("Error", "Failed to connect to MT5")
        return False
    return True

def shutdown_mt5():
    """
    Shutdown the MT5 platform.
    """
    mt5.shutdown()

# Entry Point
if connect_mt5():
    create_ui()
    shutdown_mt5()
