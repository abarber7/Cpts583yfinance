import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import sys
import os

# Adjust the path to include the directory containing base.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from yfinance.base import TickerBase  # Importing TickerBase from base.py within yfinance package

class TestTickerBase(unittest.TestCase):
    def setUp(self):
        self.ticker = 'AAPL'
        self.ticker_base = TickerBase(self.ticker)

    @patch('yfinance.base.utils.is_isin', return_value=False)
    def test_init(self, mock_is_isin):
        ticker = TickerBase('MSFT')
        self.assertEqual(ticker.ticker, 'MSFT')

    @patch('yfinance.base.PriceHistory')
    def test_lazy_load_price_history(self, mock_price_history):
        # Simulate the lazy load function
        mock_instance = mock_price_history.return_value

        # Invoke the method we are testing
        self.ticker_base._lazy_load_price_history()

        # Assert that PriceHistory was called
        mock_price_history.assert_called_once()
        # Ensure that the _price_history attribute of TickerBase is assigned
        self.assertEqual(self.ticker_base._price_history, mock_instance)

    @patch('yfinance.base.TickerBase._fetch_ticker_tz')
    def test_get_ticker_tz(self, mock_fetch_tz):
        mock_fetch_tz.return_value = 'America/New_York'
        tz = self.ticker_base._get_ticker_tz(proxy=None, timeout=10)
        self.assertEqual(tz, 'America/New_York')

    @patch('yfinance.base.TickerBase._fetch_ticker_tz')
    @patch('yfinance.base.PriceHistory.history', return_value=pd.DataFrame())
    def test_history(self, mock_history, mock_fetch_tz):
        mock_fetch_tz.return_value = 'America/New_York'
        result = self.ticker_base.history()
        self.assertIsInstance(result, pd.DataFrame)

    @patch('yfinance.base.Quote')
    def test_get_recommendations(self, mock_quote):
        mock_quote.return_value.recommendations = pd.DataFrame({'period': ['2024Q1'], 'strongBuy': [10]})
        self.ticker_base._quote = mock_quote.return_value
        result = self.ticker_base.get_recommendations()
        self.assertIsInstance(result, pd.DataFrame)

    @patch('yfinance.base.Quote')
    def test_get_calendar(self, mock_quote):
        mock_quote.return_value.calendar = {'Event': 'Earnings Call'}
        self.ticker_base._quote = mock_quote.return_value
        result = self.ticker_base.get_calendar()
        self.assertEqual(result, {'Event': 'Earnings Call'})

    @patch('yfinance.base.Quote')
    def test_get_sec_filings(self, mock_quote):
        mock_quote.return_value.sec_filings = {'Filing': '10-Q'}
        self.ticker_base._quote = mock_quote.return_value
        result = self.ticker_base.get_sec_filings()
        self.assertEqual(result, {'Filing': '10-Q'})

    @patch('yfinance.base.Holders')
    def test_get_major_holders(self, mock_holders):
        mock_holders.return_value.major = pd.DataFrame({'Holder': ['BlackRock']})
        self.ticker_base._holders = mock_holders.return_value
        result = self.ticker_base.get_major_holders()
        self.assertIsInstance(result, pd.DataFrame)

    @patch('yfinance.base.Fundamentals')
    def test_get_income_stmt(self, mock_fundamentals):
        mock_fundamentals.return_value.financials.get_income_time_series.return_value = pd.DataFrame({'Year': ['2021']})
        self.ticker_base._fundamentals = mock_fundamentals.return_value
        result = self.ticker_base.get_income_stmt()
        self.assertIsInstance(result, pd.DataFrame)

    @patch('yfinance.base.TickerBase._lazy_load_price_history')
    def test_get_dividends(self, mock_price_history):
        mock_price_history.return_value.get_dividends.return_value = pd.Series([0.22, 0.23, 0.25])
        result = self.ticker_base.get_dividends()
        self.assertIsInstance(result, pd.Series)

    @patch('yfinance.base.TickerBase._lazy_load_price_history')
    def test_get_splits(self, mock_price_history):
        mock_price_history.return_value.get_splits.return_value = pd.Series([1, 2, 4])
        result = self.ticker_base.get_splits()
        self.assertIsInstance(result, pd.Series)

    @patch('yfinance.base.Analysis')
    def test_get_eps_trend(self, mock_analysis):
        mock_analysis.return_value.eps_trend = pd.DataFrame({'0q': ['trend1']})
        self.ticker_base._analysis = mock_analysis.return_value
        result = self.ticker_base.get_eps_trend()
        self.assertIsInstance(result, pd.DataFrame)

    @patch('yfinance.base.TickerBase.get_news')
    def test_get_news(self, mock_get_news):
        mock_get_news.return_value = [{'title': 'Stock rises'}]
        result = self.ticker_base.get_news()
        self.assertIsInstance(result, list)
        self.assertEqual(result[0]['title'], 'Stock rises')

    @patch('yfinance.base.TickerBase.get_shares_full')
    def test_get_shares_full(self, mock_get_shares_full):
        mock_get_shares_full.return_value = pd.Series([1.5e9, 1.6e9], index=pd.to_datetime(['2021-01-01', '2022-01-01']))
        result = self.ticker_base.get_shares_full()
        self.assertIsInstance(result, pd.Series)

    @patch('yfinance.base.Fundamentals')
    def test_get_earnings(self, mock_fundamentals):
        mock_fundamentals.return_value.earnings = {'yearly': pd.DataFrame({'Year': ['2021'], 'Earnings': [1000]})}
        self.ticker_base._fundamentals = mock_fundamentals.return_value
        result = self.ticker_base.get_earnings()
        self.assertIsInstance(result, pd.DataFrame)

    @patch('yfinance.base.TickerBase.get_earnings_dates')
    def test_get_earnings_dates(self, mock_get_earnings_dates):
        mock_get_earnings_dates.return_value = pd.DataFrame({'Earnings Date': ['2021-10-20']})
        result = self.ticker_base.get_earnings_dates()
        self.assertIsInstance(result, pd.DataFrame)

if __name__ == '__main__':
    unittest.main()
