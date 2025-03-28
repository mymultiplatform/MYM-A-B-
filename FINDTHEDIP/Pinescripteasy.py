//@version=4
strategy(shorttitle = "BB + EMA + RSI + ADX + ATR Reversal", title = "Bollinger Bands Reversal", overlay = true, default_qty_type = strategy.cash, default_qty_value = 100, initial_capital = 1000, currency = currency.USD, commission_type = strategy.commission.percent, commission_value = 0.036)

// Inputs
strategyinput       = input(title = "Active Strategies",                type = input.string,    defval = "Both",                options = ["Both", "Reversal", "Breakout"],     group = "Strategy")
atrMultiplier       = input(title = "ATR Multiplier",                   type = input.float,     defval = 1.0,   minval = 0.1,                   step = 0.25,                    group = "Strategy")
ema1Input           = input(title = "EMA1 Input",                       type = input.integer,   defval = 200,   minval = 10,    maxval = 400,   step = 10,                      group = "Indicators")
ema2Input           = input(title = "EMA2 Input",                       type = input.integer,   defval = 100,   minval = 10,    maxval = 400,   step = 10,                      group = "Indicators")
stochMaxLongEntry   = input(title = "Stochastic Max Long",              type = input.integer,   defval = 60,    minval = 0,     maxval = 100,                                   group = "Ranging Indicators")
stochMinLongEntry   = input(title = "Stochastic Min Long",              type = input.integer,   defval = 40,    minval = 0,     maxval = 100,                                   group = "Ranging Indicators")
stochMaxShortEntry  = input(title = "Stochastic Max Short",             type = input.integer,   defval = 60,    minval = 0,     maxval = 100,                                   group = "Ranging Indicators")
stochMinShortEntry  = input(title = "Stochastic Min Short",             type = input.integer,   defval = 40,    minval = 0,     maxval = 100,                                   group = "Ranging Indicators")
rsiMaxLongEntry     = input(title = "RSI Max Long",                     type = input.integer,   defval = 60,    minval = 0,     maxval = 100,                                   group = "Ranging Indicators")
rsiMinLongEntry     = input(title = "RSI Min Long",                     type = input.integer,   defval = 50,    minval = 0,     maxval = 100,                                   group = "Ranging Indicators")
rsiMaxShortEntry    = input(title = "RSI Max Short",                    type = input.integer,   defval = 40,    minval = 0,     maxval = 100,                                   group = "Ranging Indicators")
rsiMinShortEntry    = input(title = "RSI Min Short",                    type = input.integer,   defval = 50,    minval = 0,     maxval = 100,                                   group = "Ranging Indicators")
sigMaxValue         = input(title = "ADX Max Value",                    type = input.integer,   defval = 30,    minval = 0,     maxval = 100,   step = 1,                       group = "Ranging Indicators")
rsiTrendingLongMax  = input(title = "RSI Max Trending Long",            type = input.integer,   defval = 100,   minval = 0,     maxval = 100,                                   group = "Trending Indicators")
rsiTrendingLongMin  = input(title = "RSI Min Trending Long",            type = input.integer,   defval = 65,    minval = 0,     maxval = 100,                                   group = "Trending Indicators")
rsiTrendingShortMax = input(title = "RSI Max Trending Short",           type = input.integer,   defval = 35,    minval = 0,     maxval = 100,                                   group = "Trending Indicators")
rsiTrendingShortMin = input(title = "RSI Min Trending Short",           type = input.integer,   defval = 0,     minval = 0,     maxval = 100,                                   group = "Trending Indicators")
sigMinValue         = input(title = "ADX Min Value",                    type = input.integer,   defval = 50,    minval = 0,     maxval = 100,   step = 1,                       group = "Trending Indicators")
adxlen              = input(title = "ADX Smoothing",                    type = input.integer,   defval = 14,                                                                    group = "ADX Indicator")
dilen               = input(title = "DI Length",                        type = input.integer,   defval = 14,                                                                    group = "ADX Indicator")
atrLength           = input(title = "ATR Length",                       type = input.integer,   defval = 14,    minval = 1,                                                     group = "ATR Indicator")
useStructure        = input(title = "Use Trailing Stop?",               type = input.bool,      defval = true,                                                                  group = "ATR Indicator")
atrlookback         = input(title = "ATR Lookback Period",              type = input.integer,   defval = 7,     minval = 1,                                                     group = "ATR Indicator")
length              = input(title = "BB Length",                        type = input.integer,   defval = 20,    minval = 1,                                                     group = "Bollinger Band Indicator")
bbsrc               = input(title = "BB Source",                        type = input.source,    defval = close,                                                                 group = "Bollinger Band Indicator")
mult                = input(title = "BB Standard Deviation",            type = input.float,     defval = 2.0,   minval = 0.001, maxval = 50,                                    group = "Bollinger Band Indicator")
offset              = input(title = "BB Offset",                        type = input.integer,   defval = 0,     minval = -500,  maxval = 500,                                   group = "Bollinger Band Indicator")
stochLength         = input(title = "Stochastic Length",                type = input.integer,   defval = 14,    minval = 1,                                                     group = "Stochastic Indicator")
rsilen              = input(title = "RSI Length",                       type = input.integer,   defval = 14,    minval = 1,                                                     group = "RSI Indicator")
rsisrc              = input(title = "RSI Source",                       type = input.source,    defval = close,                                                                 group = "RSI Indicator")

// Date input
fromMonth       = input(defval = 1,    title = "From Month",        type = input.integer,   minval = 1,     maxval = 12,    group = "Backtest Date Range")
fromDay         = input(defval = 1,    title = "From Day",          type = input.integer,   minval = 1,     maxval = 31,    group = "Backtest Date Range")
fromYear        = input(defval = 2000, title = "From Year",         type = input.integer,   minval = 1970,                  group = "Backtest Date Range")
thruMonth       = input(defval = 1,    title = "Thru Month",        type = input.integer,   minval = 1,     maxval = 12,    group = "Backtest Date Range")
thruDay         = input(defval = 1,    title = "Thru Day",          type = input.integer,   minval = 1,     maxval = 31,    group = "Backtest Date Range")
thruYear        = input(defval = 2099, title = "Thru Year",         type = input.integer,   minval = 1970,                  group = "Backtest Date Range")
inDataRange     = (time >= timestamp(syminfo.timezone, fromYear, fromMonth, fromDay, 0, 0)) and (time < timestamp(syminfo.timezone, thruYear, thruMonth, thruDay, 0, 0))

// Built in Bollinger Band
basis           = sma(bbsrc, length)
dev             = mult * stdev(bbsrc, length)
upper           = basis + dev
lower           = basis - dev
// Built in RSI
up              = rma(max(change(rsisrc), 0), rsilen)
down            = rma(-min(change(rsisrc), 0), rsilen)
rsi             = down == 0 ? 100 : up == 0 ? 0 : 100 - (100 / (1 + up / down))
// Built in ADX
dirmov(len) =>
	up = change(high)
	down = -change(low)
	plusDM = na(up) ? na : (up > down and up > 0 ? up : 0)
	minusDM = na(down) ? na : (down > up and down > 0 ? down : 0)
	truerange = rma(tr, len)
	plus = fixnan(100 * rma(plusDM, len) / truerange)
	minus = fixnan(100 * rma(minusDM, len) / truerange)
	[plus, minus]
adx(dilen, adxlen) =>
	[plus, minus] = dirmov(dilen)
	sum = plus + minus
	adx = 100 * rma(abs(plus - minus) / (sum == 0 ? 1 : sum), adxlen)
sig = adx(dilen, adxlen)

// Variables
RangingStrategy     = strategyinput == "Reversal" or strategyinput == "Both"
BreakoutStrategy    = strategyinput == "Breakout" or strategyinput == "Both"
ema1                = ema(close, ema1Input)
ema2                = ema(close, ema2Input)
atr                 = atr(atrLength)
lookback            = 1
var StopLong        = 0.0
var StopShort       = 0.0
emaCheckLong        = close > ema1 and close > ema2
emaCheckShort       = close < ema1 and close < ema2
ReverseCandleLong   = close[1] < open[1] and close > open
ReverseCandleShort  = close[1] > open[1] and close < open
RangingLongStoch    = stoch(close, high, low, stochLength) > stochMinLongEntry and stoch(close, high, low, stochLength) < stochMaxLongEntry
RangingShortStoch   = stoch(close, high, low, stochLength) > stochMinShortEntry and stoch(close, high, low, stochLength) < stochMaxShortEntry
RangingLongRSI      = rsi < rsiMaxLongEntry and rsi > rsiMinLongEntry
RangingShortRSI     = rsi > rsiMaxShortEntry and rsi < rsiMinShortEntry
TrendingLongRSI     = rsi < rsiTrendingLongMax and rsi > rsiTrendingLongMin
TrendingShortRSI    = rsi < rsiTrendingShortMax and rsi > rsiTrendingShortMin
RangingADX          = sig < sigMaxValue
TrendingADX         = sig > sigMinValue
BreakoutLong        = close >= upper 
BreakoutShort       = close <= lower
PriceBelowLower     = false
for i = 0 to lookback
    if close[i] <= lower[i]
        PriceBelowLower := true
PriceAboveUpper    = false
for i = 0 to lookback
    if close[i] >= upper[i]
        PriceAboveUpper := true

//Entry
CrossLongEntry      = PriceBelowLower and ReverseCandleLong and RangingADX and RangingStrategy and RangingLongStoch  and emaCheckLong and sig < sigMaxValue and inDataRange and strategy.position_size == 0 // and RangingLongRSI
CrossShortEntry     = PriceAboveUpper and ReverseCandleShort and RangingADX and RangingStrategy and RangingShortStoch and emaCheckShort and sig < sigMaxValue and inDataRange and strategy.position_size == 0 // and RangingShortRSI

TrendLongEntry      = BreakoutLong and emaCheckLong and TrendingLongRSI and TrendingADX and BreakoutStrategy and inDataRange and strategy.position_size == 0
TrendShortEntry     = BreakoutShort and emaCheckShort and TrendingShortRSI and TrendingADX and BreakoutStrategy and inDataRange and strategy.position_size == 0

// Calculating ATR stop loss and letting it trail
StopLossLong        = (useStructure ? lowest(low, atrlookback) : close) - atr * atrMultiplier
StopLossShort       = (useStructure ? highest(high, atrlookback) : close) + atr * atrMultiplier
if CrossLongEntry or TrendLongEntry
    StopLong := StopLossLong
if CrossShortEntry or TrendShortEntry
    StopShort := StopLossShort
if strategy.position_size > 0 and StopLossLong > StopLong
    StopLong := StopLossLong
if strategy.position_size < 0 and StopLossShort < StopShort
    StopShort := StopLossShort
    
//Exit signals
CrossLongExit       = (PriceAboveUpper and ReverseCandleShort and inDataRange and strategy.position_size > 0) or close < StopLong and RangingStrategy
CrossShortExit      = (PriceBelowLower and ReverseCandleLong and inDataRange and strategy.position_size < 0) or close > StopShort and RangingStrategy

TrendLongExit       = crossover(close, basis) or BreakoutStrategy and close <= StopLong and strategy.position_size > 0  
TrendShortExit      = crossunder(close, basis) or BreakoutStrategy and close >= StopShort and strategy.position_size < 0 

// Strategy
strategy.entry("Cross Long", strategy.long, when = CrossLongEntry, comment = "Cross Long")
strategy.close("Cross Long", when = CrossLongExit, comment = "Cross Long Exit")

strategy.entry("Trend Long", strategy.long, when = TrendLongEntry, comment = "Trend Long")
strategy.close("Trend Long", when = TrendLongExit, comment = "Trend Long Exit")

strategy.entry("Cross Short", strategy.short, when = CrossShortEntry, comment = "Cross Short")
strategy.close("Cross Short", when = CrossShortExit, comment = "Cross Short Exit")

strategy.entry("Trend Short", strategy.short, when = TrendShortEntry, comment = "Trend Short")
strategy.close("Trend Short", when = TrendShortExit, comment = "Trend Short Exit")

// Plots the Bollinger Band
plot(basis, "Basis", color=#872323, offset = offset)
p1 = plot(upper, "Upper", color=color.teal, offset = offset)
p2 = plot(lower, "Lower", color=color.teal, offset = offset)
fill(p1, p2, title = "Background", color=#198787, transp=95)

// Use this if you want to see the stoploss visualised, be aware though plotting these can be confusing
// plot(StopLong)
// plot(StopShort)
