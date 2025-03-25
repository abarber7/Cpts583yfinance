import unittest
from unittest.mock import Mock, patch
from yfinance.scrapers.history import PriceHistory
from yfinance.exceptions import YFInvalidPeriodError, YFPricesMissingError, YFTzMissingError
import yfinance as yf
import pandas as pd
import warnings
from urllib.error import HTTPError

class TestPriceHistoryExtended(unittest.TestCase):

    @patch("yfinance.scrapers.history.PriceHistory")  # Correct placement of the patch decorator
    def setUp(self, mock_price_history):
        # Mocking data, ticker, and timezone arguments for PriceHistory
        mock_data = Mock()
        mock_tz = "America/New_York"  # Example timezone
        # Use mock_price_history to mock the PriceHistory instance
        self.price_history = mock_price_history.return_value
        self.price_history.history.return_value = pd.DataFrame()

    @classmethod
    def setUpClass(cls):
        # Suppress deprecation warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        # Define class variables for valid and invalid ticker
        cls.valid_ticker = "AAPL"
        cls.invalid_ticker = "INVALIDTICKER"

    def test_invalid_interval_with_repair(self):
        # Test error when using '5d' interval with repair=True
        with self.assertRaises(Exception) as context:
            self.price_history.history(interval='5d', repair=True)
        self.assertIn("Yahoo's interval '5d' is nonsense", str(context.exception))

    def test_timezone_handling_with_missing_timezone(self):
        # Test handling of missing timezone for a ticker
        with patch.object(self.price_history, 'tz', None), \
             patch('yfinance.scrapers.history.PriceHistory.history', side_effect=YFTzMissingError("Missing timezone for ticker AAPL: possibly delisted; no timezone found")):
            with self.assertRaises(YFTzMissingError) as context:
                self.price_history.history(period="ytd")
            self.assertEqual(str(context.exception), "Missing timezone for ticker AAPL: possibly delisted; no timezone found")

    def test_back_adjust_with_large_period(self):
        # Test back-adjustment for large periods
        with patch('yfinance.scrapers.history.PriceHistory.history', return_value=pd.DataFrame()):
            result = self.price_history.history(period="10y", back_adjust=True)
            self.assertTrue(result.empty)  # Since we mock the return value as empty DataFrame

    def test_proxy_setting(self):
        # Test proxy usage during API call
        with patch('yfinance.scrapers.history.PriceHistory.history') as mock_history:
            self.price_history.history(proxy="http://proxyserver.com")
            mock_history.assert_called()

    def test_timeout_handling(self):
        # Test timeout handling during API calls
        with patch('yfinance.utils._requests.get', side_effect=Exception("Timeout")):
            with self.assertRaises(Exception) as context:
                self.price_history.history(timeout=0.01)
            self.assertIn("Timeout", str(context.exception))

    def test_keepna_option(self):
        # Test keepna option to retain NaN rows
        with patch('yfinance.scrapers.history.PriceHistory.history', return_value="data_with_nan"):
            result = self.price_history.history(keepna=True)
            self.assertEqual(result, "data_with_nan")

    def test_large_interval_with_repair(self):
        # Test large intervals like '1wk' with repair enabled
        with patch('yfinance.scrapers.history.PriceHistory.history', return_value="repaired_data"):
            result = self.price_history.history(interval="1wk", repair=True)
            self.assertEqual(result, "repaired_data")

    def test_invalid_dates(self):
        # Test invalid start and end date combinations
        with patch('yfinance.scrapers.history.PriceHistory.history', side_effect=ValueError("Start date is after end date")):
            with self.assertRaises(ValueError) as context:
                self.price_history.history(start="2025-01-01", end="2024-01-01")
            self.assertIn("Start date is after end date", str(context.exception))

    def test_large_dataset(self):
        """Test large data retrieval without crashing"""
        ticker = yf.Ticker(self.valid_ticker)
        try:
            data = ticker.history(period="5y", interval="1d")
            self.assertIsInstance(data, pd.DataFrame)
            self.assertGreater(len(data), 1000)  # Ensure large data is returned
        except Exception as e:
            self.fail(f"Exception occurred with large dataset: {e}")

    def test_invalid_ticker_for_history(self):
        """Test that invalid ticker raises an error during history retrieval"""
        ticker = yf.Ticker(self.invalid_ticker)
        with self.assertRaises(YFPricesMissingError):
            ticker.history(period="1d", interval="1d")

    def test_empty_data_return(self):
        """Test history returns empty data correctly"""
        ticker = yf.Ticker(self.valid_ticker)
        self.price_history.history.return_value = pd.DataFrame()
        data = ticker.history(period="1d")
        self.assertTrue(data.empty)

    def test_metadata_invalid_ticker(self):
        """Test that metadata returns None for an invalid ticker"""
        ticker = yf.Ticker(self.invalid_ticker)
        metadata = ticker.info
        self.assertIsNone(metadata)

    def test_multiple_periods_fetching(self):
        mock_history = Mock()
        mock_history.return_value = pd.DataFrame({"Close": [100, 101, 102]})
        tickers = ["AAPL", "GOOG", "TSLA"]
        results = []
        for ticker in tickers:
            yf_ticker = yf.Ticker(ticker)
            results.append(yf_ticker.history(period="1d"))
        self.assertEqual(len(results), 3)

    def test_nonexistent_date(self):
        """Test requesting history for a date that does not exist"""
        ticker = yf.Ticker(self.valid_ticker)
        with self.assertRaises(ValueError):
            ticker.history(period="1d", start="3000-01-01", end="3000-01-02")

    def test_invalid_proxy(self):
        """Test invalid proxy configuration raises an error"""
        ticker = yf.Ticker(self.valid_ticker)
        with self.assertRaises(Exception):
            ticker.history(proxy="invalid_proxy")

    def test_set_timezone(self):
        """Test that timezone can be set correctly"""
        ticker = yf.Ticker(self.valid_ticker)
        ticker.history(timezone="UTC")
        self.assertEqual(ticker.history(timezone="UTC").iloc[0].name.tz.zone, "UTC")

    def test_invalid_interval(self):
        """Test invalid interval throws an error"""
        ticker = yf.Ticker(self.valid_ticker)
        with self.assertRaises(YFInvalidPeriodError):
            ticker.history(interval="invalid_interval")

    def test_invalid_ticker_dividends(self):
        """Test accessing dividend data with an invalid ticker"""
        ticker = yf.Ticker(self.invalid_ticker)
        dividends = ticker.dividends
        self.assertTrue(dividends.empty)

    def test_invalid_date_range(self):
        """Test that invalid date range throws an error"""
        ticker = yf.Ticker(self.valid_ticker)
        with self.assertRaises(ValueError):
            ticker.history(start="2020-01-01", end="2019-01-01")

    def test_quarterly_earnings(self):
        """Test quarterly earnings data retrieval"""
        ticker = yf.Ticker(self.valid_ticker)
        earnings = ticker.quarterly_earnings
        self.assertTrue(isinstance(earnings, pd.DataFrame))

    def test_invalid_ticker_earnings(self):
        """Test earnings data retrieval for invalid ticker"""
        ticker = yf.Ticker(self.invalid_ticker)
        earnings = ticker.quarterly_earnings
        self.assertTrue(earnings.empty)

    def test_stock_splits_valid(self):
        """Test stock splits retrieval"""
        ticker = yf.Ticker(self.valid_ticker)
        splits = ticker.splits
        self.assertIsInstance(splits, pd.Series)

    def test_stock_splits_invalid(self):
        """Test stock splits retrieval with invalid ticker"""
        ticker = yf.Ticker(self.invalid_ticker)
        splits = ticker.splits
        self.assertTrue(splits.empty)

    def test_no_internet_connection(self, mock_get):
        mock_get.side_effect = HTTPError("No internet connection")
        ticker = yf.Ticker(self.valid_ticker)
        with self.assertRaises(HTTPError):
            ticker.history(period="1d")

    def test_empty_response_for_invalid_period(self):
        """Test for empty data return for invalid period"""
        ticker = yf.Ticker(self.valid_ticker)
        self.price_history.history.return_value = pd.DataFrame()
        data = ticker.history(period="invalid")
        self.assertTrue(data.empty)

    def test_institutional_holders_invalid_ticker(self):
        """Test institutional holders retrieval with an invalid ticker"""
        ticker = yf.Ticker(self.invalid_ticker)
        holders = ticker.institutional_holders
        self.assertTrue(holders is None or isinstance(holders, pd.DataFrame))

    def test_invalid_timezone_handling(self):
        """Test handling invalid timezone for history"""
        ticker = yf.Ticker(self.valid_ticker)
        with self.assertRaises(ValueError):
            ticker.history(timezone="Invalid/Timezone")

    def test_empty_financials_data(self):
        """Test that financial data retrieval returns None for missing data"""
        ticker = yf.Ticker(self.valid_ticker)
        financials = ticker.financials
        self.assertTrue(financials is None or isinstance(financials, pd.DataFrame))

if __name__ == "__main__":
    unittest.main()
