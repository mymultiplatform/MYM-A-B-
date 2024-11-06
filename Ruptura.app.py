import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import yfinance as yf
import threading
import time
from decimal import Decimal, getcontext, ROUND_HALF_UP

# Set decimal precision
getcontext().prec = 10
getcontext().rounding = ROUND_HALF_UP

# Function to fetch live BTC price with retry mechanism
def fetch_btc_price():
    def task():
        try:
            button_fetch_price.config(state='disabled')
            label_status.config(text="Fetching live BTC price...")
            retries = 3
            for attempt in range(retries):
                try:
                    btc = yf.Ticker("BTC-USD")
                    btc_data = btc.history(period="1d")
                    live_price = Decimal(btc_data["Close"].iloc[-1])
                    entry_btc_price.delete(0, tk.END)
                    entry_btc_price.insert(0, f"{live_price:.2f}")
                    label_status.config(text="Fetched live BTC price!")
                    break
                except Exception as fetch_exception:
                    if attempt < retries - 1:
                        time.sleep(2)  # Wait before retrying
                        continue
                    else:
                        raise fetch_exception
        except Exception as e:
            messagebox.showerror("Error", f"Failed to fetch BTC price:\n{e}")
            label_status.config(text="Failed to fetch BTC price.")
        finally:
            button_fetch_price.config(state='normal')

    threading.Thread(target=task).start()

# Function to calculate rupture price
def calculate_rupture():
    try:
        # Retrieve and validate user inputs
        balance = Decimal(entry_balance.get())
        entry_price = Decimal(entry_btc_price.get())
        spread_cost = Decimal(entry_spread.get())
        lot_size = Decimal(entry_lot_size.get())
        maintenance_margin = Decimal(entry_maintenance_margin.get())

        # Input Validations
        if balance <= 0:
            raise ValueError("Balance must be greater than 0.")
        if spread_cost < 0:
            raise ValueError("Spread cost cannot be negative.")
        if lot_size <= 0:
            raise ValueError("Lot size must be greater than 0.")
        if not (0 <= maintenance_margin < 100):
            raise ValueError("Maintenance margin must be between 0 and 100 percent.")

        # Get selected leverage
        leverage = Decimal(leverage_var.get())
        if leverage not in [Decimal(1), Decimal(10), Decimal(50), Decimal(100)]:
            raise ValueError("Invalid leverage selected.")

        # Calculate initial margin and required maintenance margin
        initial_margin = balance / leverage
        required_margin = initial_margin * (maintenance_margin / Decimal(100))

        # Calculate maximum allowable loss before hitting maintenance margin
        max_loss = balance - required_margin - spread_cost

        if max_loss <= 0:
            raise ValueError("Balance is too low to cover the spread cost and maintenance margin.")

        # Calculate price movement considering leverage and lot size
        price_movement = max_loss / (leverage * lot_size)

        # Calculate rupture price
        rupture_price = entry_price - price_movement

        # Ensure rupture price is not negative
        rupture_price = max(rupture_price, Decimal(0))

        # Calculate percentage drop
        price_drop_percentage = (price_movement / entry_price) * Decimal(100)

        label_result.config(
            text=f"Rupture Price (1:{leverage}): ${rupture_price:.2f}\n"
                 f"Price Drop: {price_drop_percentage:.2f}%"
        )

        # Update the Thermal Graph with the new prices
        update_thermal_graph(entry_price, rupture_price, balance, leverage, lot_size, spread_cost)

    except ValueError as ve:
        messagebox.showerror("Input Error", f"{ve}")
    except Exception as e:
        messagebox.showerror("Error", f"An unexpected error occurred:\n{e}")

# Function to interpolate between two colors channel-wise
def interpolate_color(color_start, color_end, factor: float):
    """
    Interpolates between two hex colors channel-wise.
    :param color_start: Starting color in hex (e.g., '#00FF00')
    :param color_end: Ending color in hex (e.g., '#FF0000')
    :param factor: Float between 0 and 1 indicating the interpolation factor
    :return: Interpolated color in hex
    """
    factor = max(0.0, min(1.0, factor))  # Clamp factor between 0 and 1
    # Convert hex to RGB tuples
    start_rgb = tuple(int(color_start[i:i+2], 16) for i in (1, 3, 5))
    end_rgb = tuple(int(color_end[i:i+2], 16) for i in (1, 3, 5))
    # Interpolate each channel
    interp_rgb = tuple(int(s + (e - s) * factor) for s, e in zip(start_rgb, end_rgb))
    # Convert back to hex
    return f'#{interp_rgb[0]:02x}{interp_rgb[1]:02x}{interp_rgb[2]:02x}'

# Function to update the Thermal Graph
def update_thermal_graph(entry_price, rupture_price, balance, leverage, lot_size, spread_cost):
    # Clear the previous drawings
    thermal_canvas.delete("all")

    # Define the fixed step size
    step_size = Decimal('10')  # $10 steps

    # Define canvas dimensions
    canvas_width = 400  # Increased width to accommodate dual labels
    level_height = 30  # Height of each price level

    # Calculate the current loss per $1 drop in BTC price
    loss_per_dollar = leverage * lot_size

    # Calculate corresponding balance for each BTC price level
    # Starting from entry_price down to rupture_price in $10 steps
    price_levels = []
    current_price = entry_price
    while current_price >= rupture_price:
        price_levels.append(current_price)
        current_price -= step_size
    if price_levels[-1] > rupture_price:
        price_levels.append(rupture_price)
    if price_levels[-1] > Decimal(0):
        price_levels.append(Decimal(0))  # Ensure $0 is included

    num_levels = len(price_levels)

    # Calculate total height needed
    total_height = num_levels * level_height

    # Update the scrollable region
    thermal_canvas.config(scrollregion=(0, 0, canvas_width, total_height))

    # Define colors
    color_start = "#00FF00"  # Green
    color_end = "#FF0000"    # Red

    for i, price in enumerate(price_levels):
        # Calculate color based on the level (from green to red)
        factor = float(i) / (num_levels - 1) if num_levels > 1 else 1.0
        color = interpolate_color(color_start, color_end, factor)

        # Calculate corresponding balance
        price_drop = entry_price - price
        current_balance = balance - (price_drop * loss_per_dollar)
        current_balance = max(current_balance, Decimal(0))  # Avoid negative balance

        # Draw rectangle for the level
        thermal_canvas.create_rectangle(
            10, i * level_height, canvas_width - 10, (i + 1) * level_height - 5,
            fill=color, outline="black"
        )

        # Add BTC price text on the left side
        thermal_canvas.create_text(
            60, i * level_height + level_height / 2 - 5,
            text=f"${price:.2f}",
            font=("Arial", 10, "bold"),
            fill="black" if factor > 0.5 else "white"  # Ensure text is visible
        )

        # Add corresponding USD balance text on the right side
        thermal_canvas.create_text(
            300, i * level_height + level_height / 2 - 5,
            text=f"${current_balance:.2f}",
            font=("Arial", 10, "bold"),
            fill="black" if factor > 0.5 else "white"  # Ensure text is visible
        )

    # Highlight the rupture price
    if rupture_price > Decimal(0) and rupture_price < entry_price:
        # Find the closest price level to the rupture price
        rupture_level = None
        for idx, price in enumerate(price_levels):
            if price <= rupture_price:
                rupture_level = idx
                break
        if rupture_level is not None:
            y1 = rupture_level * level_height
            y2 = y1 + level_height - 5
            thermal_canvas.create_rectangle(
                10, y1, canvas_width - 10, y2,
                fill="#0000FF", outline="black", width=2  # Blue highlight
            )
            thermal_canvas.create_text(
                60, y1 + level_height / 2 - 5,
                text=f"Rupture: ${rupture_price:.2f}",
                font=("Arial", 10, "bold"),
                fill="white"
            )
            thermal_canvas.create_text(
                300, y1 + level_height / 2 - 5,
                text=f"${0.00:.2f}",
                font=("Arial", 10, "bold"),
                fill="white"
            )

# Function to provide tooltips
def create_tooltip(widget, text):
    tooltip = tk.Toplevel(widget)
    tooltip.withdraw()
    tooltip.overrideredirect(True)
    label = tk.Label(tooltip, text=text, background="#FFFFE0", relief="solid", borderwidth=1, font=("Arial", 10))
    label.pack(ipadx=1)

    def enter(event):
        x = event.x_root + 20
        y = event.y_root + 10
        tooltip.geometry(f"+{x}+{y}")
        tooltip.deiconify()

    def leave(event):
        tooltip.withdraw()

    widget.bind("<Enter>", enter)
    widget.bind("<Leave>", leave)

# Set up Tkinter window
root = tk.Tk()
root.title("RUPTURA APP")
root.geometry("1000x700")  # Increased width to accommodate side-by-side layout
root.resizable(True, True)  # Allow window to be resizable

# Main Title
title_label = tk.Label(root, text="RUPTURA APP", font=("Arial", 24, "bold"))
title_label.pack(pady=10)

# Banner Note
note_text = "Note: This calculation assumes that you have bought the minimum lot size in a leveraged account."
banner = tk.Label(
    root,
    text=note_text,
    font=("Arial", 12),
    fg="white",
    bg="#2E86C1",  # A distinct blue color for the banner
    wraplength=950,  # Adjusted to fit within the window width
    justify="left",
    padx=10,
    pady=10
)
banner.pack(pady=(0, 20), fill="x")

# Create the main frame to hold input and thermal graph frames side by side
main_frame = tk.Frame(root)
main_frame.pack(padx=20, pady=10, fill="both", expand=True)

# Left Frame for Inputs and Results
left_frame = tk.Frame(main_frame)
left_frame.pack(side="left", fill="both", expand=True)

# BTC Live Price Section
btc_frame = tk.Frame(left_frame)
btc_frame.pack(fill="x", pady=5)

tk.Label(btc_frame, text="BTC LIVE PRICE ($)", font=("Arial", 12)).pack(side="left", anchor="w")
entry_btc_price = tk.Entry(btc_frame, font=("Arial", 12), width=15)
entry_btc_price.pack(side="left", padx=10)
button_fetch_price = tk.Button(
    btc_frame,
    text="Live BTC Price",
    command=fetch_btc_price,
    font=("Arial", 12)
)
button_fetch_price.pack(side="left", padx=10)

# Tooltip for BTC Live Price
create_tooltip(button_fetch_price, "Click to fetch the latest BTC price.")

# Status Label for fetching BTC price
label_status = tk.Label(left_frame, text="", font=("Arial", 10), fg="blue")
label_status.pack(fill="x", pady=(0, 10))

# User Balance Section
balance_frame = tk.Frame(left_frame)
balance_frame.pack(fill="x", pady=5)

tk.Label(balance_frame, text="YOUR BALANCE ($)", font=("Arial", 12)).pack(side="left", anchor="w")
entry_balance = tk.Entry(balance_frame, font=("Arial", 12), width=15)
entry_balance.insert(0, "370")  # Example starting balance
entry_balance.pack(side="left", padx=10)

# Tooltip for Balance
create_tooltip(entry_balance, "Enter your current account balance in USD.")

# Spread Cost Section
spread_frame = tk.Frame(left_frame)
spread_frame.pack(fill="x", pady=5)

tk.Label(spread_frame, text="SPREAD COST ($)", font=("Arial", 12)).pack(side="left", anchor="w")
entry_spread = tk.Entry(spread_frame, font=("Arial", 12), width=15)
entry_spread.insert(0, "2")  # Default spread cost
entry_spread.pack(side="left", padx=10)

# Tooltip for Spread Cost
create_tooltip(entry_spread, "Enter the spread cost per BTC trade in USD.")

# Lot Size Section
lot_size_frame = tk.Frame(left_frame)
lot_size_frame.pack(fill="x", pady=5)

tk.Label(lot_size_frame, text="LOT SIZE (BTC)", font=("Arial", 12)).pack(side="left", anchor="w")
entry_lot_size = tk.Entry(lot_size_frame, font=("Arial", 12), width=15)
entry_lot_size.insert(0, "1")  # Default lot size
entry_lot_size.pack(side="left", padx=10)

# Tooltip for Lot Size
create_tooltip(entry_lot_size, "Enter the number of BTC units in your position.")

# Maintenance Margin Section
maintenance_frame = tk.Frame(left_frame)
maintenance_frame.pack(fill="x", pady=5)

tk.Label(maintenance_frame, text="MAINTENANCE MARGIN (%)", font=("Arial", 12)).pack(side="left", anchor="w")
entry_maintenance_margin = tk.Entry(maintenance_frame, font=("Arial", 12), width=15)
entry_maintenance_margin.insert(0, "50")  # Default maintenance margin
entry_maintenance_margin.pack(side="left", padx=10)

# Tooltip for Maintenance Margin
create_tooltip(entry_maintenance_margin, "Enter the maintenance margin percentage required by your broker.")

# Leverage Selection Section
leverage_label = tk.Label(left_frame, text="SELECT LEVERAGE", font=("Arial", 12))
leverage_label.pack(anchor="w", pady=(20, 5))

leverage_var = tk.IntVar()
leverage_var.set(10)  # Default leverage

# Define leverage options
leverage_options = [("1:1", 1), ("1:10", 10), ("1:50", 50), ("1:100", 100)]

# Create a frame for leverage radio buttons
leverage_frame = tk.Frame(left_frame)
leverage_frame.pack(anchor="w")

# Create radio buttons for leverage selection
for text, value in leverage_options:
    tk.Radiobutton(
        leverage_frame,
        text=text,
        variable=leverage_var,
        value=value,
        font=("Arial", 12)
    ).pack(anchor="w")

# Tooltip for Leverage
create_tooltip(leverage_frame, "Select the leverage ratio for your position.")

# Start Calculation Button
button_calculate = tk.Button(
    left_frame,
    text="START CALCULATION",
    command=calculate_rupture,
    font=("Arial", 14, "bold"),
    bg="#28B463",
    fg="white",
    width=20,
    height=2
)
button_calculate.pack(pady=20)

# Tooltip for Calculate Button
create_tooltip(button_calculate, "Click to calculate the rupture price based on your inputs.")

# Display the result
label_result = tk.Label(left_frame, text="PRICE FOR RUPTURE CALCULATION: ", font=("Arial", 12))
label_result.pack(pady=10)

# Right Frame for Thermal Graph
right_frame = tk.Frame(main_frame)
right_frame.pack(side="right", fill="both", expand=True)

# Thermal Graph Section Label
thermal_section_label = tk.Label(right_frame, text="THERMAL GRAPH", font=("Arial", 16, "bold"))
thermal_section_label.pack(pady=(10, 5))

# Create a frame for the scrollable canvas
canvas_frame = tk.Frame(right_frame)
canvas_frame.pack(fill="both", expand=True, pady=10, padx=10)

# Create a vertical scrollbar
v_scrollbar = tk.Scrollbar(canvas_frame, orient=tk.VERTICAL)
v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

# Create the canvas
thermal_canvas = tk.Canvas(canvas_frame, width=400, height=500, bg="#F0F0F0",
                           highlightthickness=1, highlightbackground="black",
                           yscrollcommand=v_scrollbar.set)
thermal_canvas.pack(side=tk.LEFT, fill="both", expand=True)

# Configure the scrollbar
v_scrollbar.config(command=thermal_canvas.yview)

# Initialize the Thermal Graph with default values
def initialize_thermal_graph():
    try:
        entry_price = Decimal(entry_btc_price.get()) if entry_btc_price.get() else Decimal(0.0)
    except:
        entry_price = Decimal(0.0)
    rupture_price = Decimal(0.0)  # Default rupture price before calculation
    try:
        balance = Decimal(entry_balance.get())
    except:
        balance = Decimal(0.0)
    try:
        lot_size = Decimal(entry_lot_size.get())
    except:
        lot_size = Decimal(1.0)
    try:
        spread_cost = Decimal(entry_spread.get())
    except:
        spread_cost = Decimal(0.0)
    try:
        leverage = Decimal(leverage_var.get())
    except:
        leverage = Decimal(10)
    update_thermal_graph(entry_price, rupture_price, balance, leverage, lot_size, spread_cost)

# Call initialize function on startup
initialize_thermal_graph()

# Real-time update of thermal graph when inputs change
def on_input_change(event):
    initialize_thermal_graph()

# Bind events to update thermal graph in real-time
entry_btc_price.bind("<KeyRelease>", on_input_change)
entry_balance.bind("<KeyRelease>", on_input_change)
entry_spread.bind("<KeyRelease>", on_input_change)
entry_lot_size.bind("<KeyRelease>", on_input_change)
entry_maintenance_margin.bind("<KeyRelease>", on_input_change)
leverage_var.trace_add('write', lambda *args: initialize_thermal_graph())

# Run the app
root.mainloop()
