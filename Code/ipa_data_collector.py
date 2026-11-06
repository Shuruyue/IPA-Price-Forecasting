"""
IPA Price Prediction Model - Data Collection Module
Data Collector for Isopropyl Alcohol Price Prediction

This module is responsible for collecting data from multiple sources:
- Yahoo Finance (yfinance): Crude oil, natural gas, exchange rates
- Event indicators: Geopolitical and macro shock proxies
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Data collection packages
try:
    import yfinance as yf
except ImportError:
    yf = None

# ---------------------------------------------------------------------------
# Ticker configuration
# ---------------------------------------------------------------------------
TICKER_MAP: Dict[str, str] = {
    'WTI_Price': 'CL=F',
    'Brent_Price': 'BZ=F',
    'NatGas_Price': 'NG=F',
    'USD_TWD': 'USDTWD=X',
    'DXY': 'DX-Y.NYB',
    'TWII': '^TWII',
    'TSM_Price': 'TSM',
}

# ---------------------------------------------------------------------------
# Event date ranges (start, end or None for ongoing)
# ---------------------------------------------------------------------------
EVENT_RANGES: Dict[str, Tuple[str, Optional[str]]] = {
    'COVID19': ('2020-01-01', '2023-05-31'),
    'Ukraine_War': ('2022-02-24', None),
    'RedSea_Crisis': ('2023-11-01', None),
    'US_China_Trade': ('2018-03-01', None),
}


class IPADataCollector:
    """Isopropyl Alcohol Price Prediction - Data Collector."""

    MARKET_COLUMNS: List[str] = list(TICKER_MAP.keys())
    EVENT_COLUMNS: List[str] = list(EVENT_RANGES.keys()) + ['Geopolitical_Risk']
    REQUIRED_CACHE_COLUMNS: List[str] = MARKET_COLUMNS + EVENT_COLUMNS

    def __init__(
        self,
        start_date: str = '2012-01-01',
        end_date: str = '2024-12-31',
        cache_dir: str = 'Data',
        use_cache: bool = True,
        refresh_cache: bool = False,
    ) -> None:
        self.start_date = start_date
        self.end_date = end_date
        self.cache_dir = cache_dir
        self.use_cache = use_cache
        self.refresh_cache = refresh_cache
        self.data: Dict[str, pd.DataFrame] = {}

        safe_start = self.start_date.replace('-', '')
        safe_end = self.end_date.replace('-', '')
        self.cache_file = os.path.join(
            self.cache_dir,
            f"market_data_{safe_start}_{safe_end}.csv",
        )

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _validate_cached_data(self, df: pd.DataFrame) -> Tuple[bool, str]:
        if df is None or df.empty:
            return False, "cached data is empty"
        if not isinstance(df.index, pd.DatetimeIndex):
            return False, "cache index is not DatetimeIndex"

        missing = [c for c in self.REQUIRED_CACHE_COLUMNS if c not in df.columns]
        if missing:
            return False, f"missing columns: {missing}"

        return True, "ok"

    def _normalize_output_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure merged output keeps stable schema for downstream steps."""
        result = df.copy()
        for col in self.REQUIRED_CACHE_COLUMNS:
            if col not in result.columns:
                result[col] = 0.0 if col in self.EVENT_COLUMNS else np.nan

        result = result[self.REQUIRED_CACHE_COLUMNS]
        result = result.apply(pd.to_numeric, errors='coerce')
        result[self.EVENT_COLUMNS] = result[self.EVENT_COLUMNS].fillna(0).astype(int)
        result = result.sort_index().ffill()
        return result

    # ------------------------------------------------------------------
    # Download helpers
    # ------------------------------------------------------------------

    def _download_close_price(self, ticker: str, column_name: str) -> Optional[pd.DataFrame]:
        """Download close price series from Yahoo Finance.

        Handles both normal and MultiIndex yfinance responses.
        """
        if yf is None:
            return None

        raw = yf.download(
            ticker,
            start=self.start_date,
            end=self.end_date,
            progress=False,
        )
        if raw is None or raw.empty:
            raise ValueError(f"No data returned for {ticker}")

        if isinstance(raw.columns, pd.MultiIndex):
            close_cols = [c for c in raw.columns if str(c[0]).lower() == 'close']
            if not close_cols:
                series = raw.iloc[:, 0]
            else:
                selected = raw[close_cols[0]]
                series = selected.iloc[:, 0] if isinstance(selected, pd.DataFrame) else selected
        else:
            close_col = 'Close' if 'Close' in raw.columns else raw.columns[0]
            series = raw[close_col]

        result = series.to_frame(name=column_name)
        result.index = pd.to_datetime(result.index)
        result = result.loc[~result.index.duplicated(keep='first')].sort_index()
        return result

    def _fetch_tickers(self, tickers: Dict[str, str], label: str) -> Optional[pd.DataFrame]:
        """Fetch a group of tickers and merge into one DataFrame."""
        logger.info("Fetching %s...", label)
        print(f"Fetching {label}...")

        if yf is None:
            print(f"  [WARN] yfinance unavailable, {label} skipped")
            return None

        try:
            frames = []
            for col_name, ticker in tickers.items():
                df = self._download_close_price(ticker, col_name)
                if df is not None and not df.empty:
                    frames.append(df)
            if not frames:
                raise ValueError(f"No data returned for {label}")
            merged = pd.concat(frames, axis=1)
            self.data[label] = merged
            print(f"  [OK] {label}: {len(merged)} records")
            return merged
        except Exception as e:
            logger.warning("Failed to fetch %s: %s", label, e)
            print(f"  [ERROR] Failed to fetch {label}: {e}")
            return None

    # ------------------------------------------------------------------
    # Public fetch methods
    # ------------------------------------------------------------------

    def fetch_crude_oil_prices(self) -> Optional[pd.DataFrame]:
        """Fetch crude oil prices (WTI and Brent)."""
        return self._fetch_tickers(
            {'WTI_Price': TICKER_MAP['WTI_Price'], 'Brent_Price': TICKER_MAP['Brent_Price']},
            'crude oil prices',
        )

    def fetch_natural_gas_prices(self) -> Optional[pd.DataFrame]:
        """Fetch natural gas prices (Henry Hub)."""
        return self._fetch_tickers(
            {'NatGas_Price': TICKER_MAP['NatGas_Price']},
            'natural gas prices',
        )

    def fetch_exchange_rates(self) -> Optional[pd.DataFrame]:
        """Fetch exchange rate data (USD/TWD, DXY)."""
        return self._fetch_tickers(
            {'USD_TWD': TICKER_MAP['USD_TWD'], 'DXY': TICKER_MAP['DXY']},
            'exchange rate data',
        )

    def fetch_stock_indices(self) -> Optional[pd.DataFrame]:
        """Fetch stock indices (^TWII, TSM)."""
        return self._fetch_tickers(
            {'TWII': TICKER_MAP['TWII'], 'TSM_Price': TICKER_MAP['TSM_Price']},
            'stock indices',
        )

    def create_event_indicators(self) -> pd.DataFrame:
        """Create geopolitical and major event indicators.

        Events are configured via the module-level ``EVENT_RANGES`` dict.
        """
        print("Creating event indicators...")

        date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        events = pd.DataFrame(index=date_range)

        for event_name, (start, end) in EVENT_RANGES.items():
            events[event_name] = 0
            if end is not None:
                events.loc[start:end, event_name] = 1
            else:
                events.loc[start:, event_name] = 1

        # Composite geopolitical risk indicator
        risk_components = ['COVID19', 'Ukraine_War', 'RedSea_Crisis']
        events['Geopolitical_Risk'] = events[[c for c in risk_components if c in events.columns]].sum(axis=1)

        self.data['events'] = events
        print(f"  [OK] Event indicators: {len(events)} records")
        return events

    # ------------------------------------------------------------------
    # Collect & merge
    # ------------------------------------------------------------------

    def collect_all_data(self) -> pd.DataFrame:
        """Collect all data and merge into a single DataFrame."""
        if self.use_cache and not self.refresh_cache and os.path.exists(self.cache_file):
            try:
                cached = pd.read_csv(self.cache_file, parse_dates=['Date'], index_col='Date')
                cached = cached.sort_index()
                valid, reason = self._validate_cached_data(cached)
                if not valid:
                    raise ValueError(f"invalid cache ({reason})")
                print("\n" + "=" * 50)
                print("Loading cached market data...")
                print("=" * 50 + "\n")
                print(f"[OK] Loaded cache: {self.cache_file}")
                print(f"  Total records: {len(cached)}")
                print(f"  Number of columns: {len(cached.columns)}")
                print(f"  Time range: {cached.index.min()} ~ {cached.index.max()}")
                return self._normalize_output_schema(cached)
            except Exception as e:
                logger.warning("Failed to load cache, fallback to remote: %s", e)
                print(f"[WARN] Failed to load cache, fallback to remote collection: {e}")

        print("\n" + "=" * 50)
        print("Starting to collect all data...")
        print("=" * 50 + "\n")

        self.fetch_crude_oil_prices()
        self.fetch_natural_gas_prices()
        self.fetch_exchange_rates()
        self.fetch_stock_indices()
        self.create_event_indicators()

        print("\nMerging data...")
        dfs_to_merge: List[pd.DataFrame] = []
        for _name, df in self.data.items():
            if df is not None and not df.empty and isinstance(df.index, pd.DatetimeIndex):
                dfs_to_merge.append(df.loc[~df.index.duplicated(keep='first')])

        if not dfs_to_merge:
            print("  [ERROR] No available data")
            return pd.DataFrame()

        all_data = pd.concat(dfs_to_merge, axis=1)
        all_data = all_data.loc[:, ~all_data.columns.duplicated()]
        all_data = self._normalize_output_schema(all_data)

        print("\n[OK] Data collection complete!")
        print(f"  Total records: {len(all_data)}")
        print(f"  Number of columns: {len(all_data.columns)}")
        print(f"  Time range: {all_data.index.min()} ~ {all_data.index.max()}")

        if self.use_cache:
            try:
                os.makedirs(self.cache_dir, exist_ok=True)
                all_data.to_csv(self.cache_file, index_label='Date')
                print(f"[OK] Cache saved: {self.cache_file}")
            except Exception as e:
                logger.warning("Failed to write cache: %s", e)
                print(f"[WARN] Failed to write cache: {e}")

        return all_data

    def resample_to_quarterly(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert daily frequency data to quarterly averages."""
        print("\nConverting to quarterly data...")

        quarterly = df.resample('QE').mean()
        quarterly['Year'] = quarterly.index.year
        quarterly['Quarter'] = quarterly.index.quarter
        quarterly['Quarter_Label'] = quarterly.apply(
            lambda x: f"{int(x['Year'])}Q{int(x['Quarter'])}", axis=1,
        )
        print(f"[OK] Quarterly data: {len(quarterly)} records")
        return quarterly

    def save_data(self, df: pd.DataFrame, filename: str) -> None:
        """Save data to CSV."""
        df.to_csv(filename)
        print(f"[OK] Data saved to: {filename}")


def load_ipa_prices_from_csv(filepath: str) -> pd.DataFrame:
    """Load IPA price data from CSV.

    Expected format: Date (index) + Price column (TWD/KG).
    """
    return pd.read_csv(filepath, parse_dates=['Date'], index_col='Date')


# Main test program
if __name__ == "__main__":
    collector = IPADataCollector(
        start_date='2012-01-01',
        end_date='2024-12-31',
    )
    data = collector.collect_all_data()

    print("\n" + "=" * 50)
    print("Data Summary")
    print("=" * 50)
    print(data.describe())

    quarterly_data = collector.resample_to_quarterly(data)
    print("\nQuarterly Data:")
    print(quarterly_data.tail(10))
