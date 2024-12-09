# MYM-A Automated Trading & Dice Simulation

A Python application that combines **MetaTrader 5 (MT5) trading automation** with a fun **dice rolling simulation**.

---

## Key Features

### 🎲 **Dice Game**
- Interactive 3-faced dice rolling game with animations.
- Automatically triggers MetaTrader connection upon rolling a "3."

### 💹 **Trading Automation**
- Connects to MT5 using provided credentials.
- Automates:
  - Historical data fetching.
  - Price predictions using **LSTM neural networks**.
  - Bullish/Bearish trend detection.
  - Automated trade placement and monitoring.

---

## Tech Stack
- **GUI**: `tkinter`
- **Trading**: `MetaTrader5`
- **ML**: `tensorflow`, `LSTM`
- **Data**: `pandas`, `numpy`, `sklearn`

---

## Usage
1. Run the script to launch the GUI.
2. Roll the dice for fun!
3. Upon rolling a "3," the app connects to MT5 and starts trading automation.
4. Trades are based on **BTCUSD** predictions using LSTM models.

---

### **Disclaimer**
This is for educational purposes only. Always test in a demo account before using in live trading. Use at your own risk!
