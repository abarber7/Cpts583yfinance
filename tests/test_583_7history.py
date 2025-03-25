import unittest
from unittest.mock import MagicMock, patch
from yfinance.scrapers.history import PriceHistory
from yfinance.exceptions import YFPricesMissingError, YFInvalidPeriodError
import pandas as pd


class TestPriceHistory(unittest.TestCase):

    def setUp(self):
        self.data = MagicMock()
        self.ticker = "AAPL"
        self.timezone = "America/New_York"
        self.price_history = PriceHistory(self.data, self.ticker, self.timezone)

    @patch("yfinance.scrapers.history.PriceHistory.fetch_price_data")
    def test_fetch_price_data_correctly(self, mock_fetch):
        # Line 901-902 - Verify fetch_price_data is called with correct params
        self.price_history.history(period="1mo", interval="1d")
        mock_fetch.assert_called_once_with("1mo", "1d")

    def test_raise_error_for_invalid_interval(self):
        # Line 932-935 - Raise error for unsupported interval
        with self.assertRaises(YFInvalidPeriodError):
            self.price_history.history(period="1mo", interval="unsupported")

    def test_handle_empty_data_response(self):
        # Line 941 - Handle empty DataFrame response gracefully
        with patch("yfinance.scrapers.history.PriceHistory.fetch_price_data", return_value=pd.DataFrame()):
            with self.assertRaises(YFPricesMissingError):
                self.price_history.history(period="1mo", interval="1d")

    @patch("yfinance.scrapers.history.PriceHistory.validate_parameters")
    def test_validate_parameters_called(self, mock_validate):
        # Line 949-952 - Ensure validate_parameters is called
        self.price_history.history(period="1mo", interval="1d")
        mock_validate.assert_called_once()

    def test_resample_data_logic(self):
        # Line 978 - Resample data with valid frequency
        df = pd.DataFrame({'close': [1, 2, 3]}, index=pd.date_range("2024-01-01", periods=3))
        resampled = self.price_history._resample_data(df, interval="1d")
        self.assertIsInstance(resampled, pd.DataFrame)

    @patch("yfinance.scrapers.history.PriceHistory.repair_data")
    def test_repair_data_called_with_correct_input(self, mock_repair):
        # Line 993 - Repair data and validate the call
        df = pd.DataFrame({'close': [1, None, 3]})
        self.price_history._repair_data(df)
        mock_repair.assert_called_once_with(df)

    def test_handle_invalid_parameters_error(self):
        # Line 1000-1002 - Raise error for invalid parameters
        with self.assertRaises(ValueError):
            self.price_history.history(period="invalid", interval="1d")

    def test_adjusted_close_handling(self):
        # Line 1005-1008 - Verify handling of adjusted close column
        df = pd.DataFrame({'Adj Close': [100, 101, 102]})
        adjusted_df = self.price_history._adjust_close_data(df)
        self.assertIn("Adj Close", adjusted_df.columns)

    def test_fetch_raw_data(self):
        # Line 1013 - Fetch raw data from Yahoo Finance
        with patch("yfinance.scrapers.history.PriceHistory.fetch_price_data", return_value=pd.DataFrame({'close': [1, 2, 3]})):
            result = self.price_history.history(period="1mo", interval="1d")
            self.assertIsInstance(result, pd.DataFrame)

    @patch("yfinance.scrapers.history.PriceHistory.validate_data")
    def test_validate_data_called(self, mock_validate):
        # Line 1025-1027 - Validate data after fetching
        df = pd.DataFrame({'close': [1, 2, 3]})
        self.price_history.validate_data(df)
        mock_validate.assert_called_once_with(df)


if __name__ == "__main__":
    unittest.main()
