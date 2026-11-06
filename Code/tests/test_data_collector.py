"""
Unit tests for IPA Data Collector Module
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ipa_data_collector import IPADataCollector, EVENT_RANGES, TICKER_MAP


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def collector():
    return IPADataCollector(
        start_date='2020-01-01',
        end_date='2022-12-31',
        use_cache=False,
    )


# ---------------------------------------------------------------------------
# Configuration constants
# ---------------------------------------------------------------------------

class TestModuleConstants:
    def test_ticker_map_has_all_market_columns(self):
        for col in IPADataCollector.MARKET_COLUMNS:
            assert col in TICKER_MAP

    def test_event_ranges_valid(self):
        for name, (start, end) in EVENT_RANGES.items():
            assert pd.to_datetime(start) is not None
            if end is not None:
                assert pd.to_datetime(end) >= pd.to_datetime(start)

    def test_required_cache_columns_includes_both(self):
        cols = IPADataCollector.REQUIRED_CACHE_COLUMNS
        assert 'WTI_Price' in cols
        assert 'COVID19' in cols
        assert 'Geopolitical_Risk' in cols


# ---------------------------------------------------------------------------
# Cache validation
# ---------------------------------------------------------------------------

class TestCacheValidation:
    def test_empty_df_invalid(self, collector):
        valid, reason = collector._validate_cached_data(pd.DataFrame())
        assert not valid
        assert "empty" in reason

    def test_missing_columns_invalid(self, collector):
        df = pd.DataFrame({'WTI_Price': [1.0]}, index=pd.to_datetime(['2020-01-01']))
        valid, reason = collector._validate_cached_data(df)
        assert not valid
        assert "missing columns" in reason

    def test_non_datetime_index_invalid(self, collector):
        data = {col: [0.0] for col in IPADataCollector.REQUIRED_CACHE_COLUMNS}
        df = pd.DataFrame(data, index=[0])
        valid, reason = collector._validate_cached_data(df)
        assert not valid
        assert "DatetimeIndex" in reason

    def test_valid_data_passes(self, collector):
        data = {col: [1.0] for col in IPADataCollector.REQUIRED_CACHE_COLUMNS}
        df = pd.DataFrame(data, index=pd.to_datetime(['2020-01-01']))
        valid, reason = collector._validate_cached_data(df)
        assert valid
        assert reason == "ok"


# ---------------------------------------------------------------------------
# Schema normalization
# ---------------------------------------------------------------------------

class TestNormalizeOutputSchema:
    def test_missing_columns_filled(self, collector):
        df = pd.DataFrame(
            {'WTI_Price': [70.0, 72.0]},
            index=pd.to_datetime(['2020-01-01', '2020-01-02']),
        )
        result = collector._normalize_output_schema(df)
        assert set(IPADataCollector.REQUIRED_CACHE_COLUMNS).issubset(result.columns)
        # Event columns filled with 0
        assert result['COVID19'].iloc[0] == 0

    def test_event_columns_are_int(self, collector):
        data = {col: [1.0] for col in IPADataCollector.REQUIRED_CACHE_COLUMNS}
        df = pd.DataFrame(data, index=pd.to_datetime(['2020-01-01']))
        result = collector._normalize_output_schema(df)
        for event_col in IPADataCollector.EVENT_COLUMNS:
            assert result[event_col].dtype in (np.int32, np.int64, int)


# ---------------------------------------------------------------------------
# Event indicators
# ---------------------------------------------------------------------------

class TestEventIndicators:
    def test_event_indicators_created(self, collector):
        events = collector.create_event_indicators()
        assert 'COVID19' in events.columns
        assert 'Geopolitical_Risk' in events.columns
        assert len(events) > 0

    def test_covid_active_in_2020(self, collector):
        events = collector.create_event_indicators()
        assert events.loc['2020-06-01', 'COVID19'] == 1

    def test_covid_inactive_before_2020(self):
        c = IPADataCollector(start_date='2018-01-01', end_date='2019-12-31', use_cache=False)
        events = c.create_event_indicators()
        assert events['COVID19'].sum() == 0

    def test_geopolitical_risk_composite(self, collector):
        events = collector.create_event_indicators()
        # Geopolitical_Risk should be sum of COVID19 + Ukraine_War + RedSea_Crisis
        expected = events['COVID19'] + events['Ukraine_War'] + events['RedSea_Crisis']
        pd.testing.assert_series_equal(events['Geopolitical_Risk'], expected, check_names=False)


# ---------------------------------------------------------------------------
# Quarterly resampling
# ---------------------------------------------------------------------------

class TestResampleToQuarterly:
    def test_quarterly_output(self, collector):
        dates = pd.date_range('2020-01-01', periods=365, freq='D')
        df = pd.DataFrame({'value': np.random.rand(365)}, index=dates)
        quarterly = collector.resample_to_quarterly(df)
        assert 'Year' in quarterly.columns
        assert 'Quarter' in quarterly.columns
        assert 'Quarter_Label' in quarterly.columns
        assert len(quarterly) <= 5  # ~4-5 quarters in 2020
