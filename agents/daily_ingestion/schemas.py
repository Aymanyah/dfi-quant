"""Pyarrow schema for daily OHLCV bars. Immutable.

If Binance adds a field, do NOT mutate the existing schema. Define a new
versioned schema (e.g. OHLCV_1D_V2) and update the writer to emit a new
partition path.
"""
import pyarrow as pa

OHLCV_1D_V1 = pa.schema([
    pa.field("ts_open",       pa.int64()),    # bar open,  microseconds UTC
    pa.field("ts_close",      pa.int64()),    # bar close, microseconds UTC
    pa.field("open",          pa.float64()),
    pa.field("high",          pa.float64()),
    pa.field("low",           pa.float64()),
    pa.field("close",         pa.float64()),
    pa.field("volume",        pa.float64()),  # base asset (e.g. BTC)
    pa.field("quote_vol",     pa.float64()),  # USDT notional
    pa.field("n_trades",      pa.int64()),
    pa.field("taker_buy_vol", pa.float64()),  # taker buy volume in base asset
    pa.field("exchange",      pa.string()),
    pa.field("symbol",        pa.string()),
])

SCHEMAS = {
    "ohlcv_1d": OHLCV_1D_V1,
}
