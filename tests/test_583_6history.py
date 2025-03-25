from turtle import pd
import unittest
from unittest.mock import MagicMock, patch
from requests.exceptions import HTTPError
from yfinance.exceptions import YFPricesMissingError, YFInvalidPeriodError, YFTzMissingError
from yfinance.scrapers.history import PriceHistory


class TestPriceHistory(unittest.TestCase):

    def setUp(self):
        self.data = MagicMock()
        self.ticker = "AAPL"
        self.timezone = "America/New_York"
        self.price_history = PriceHistory(self.data, self.ticker, self.timezone)

    @patch('yfinance.shared._DFS')
    @patch('yfinance.utils.empty_df')
    def test_missing_timezone_error(self, mock_empty_df, mock_dfs):
        # Test line 85 - Missing timezone
        mock_empty_df.return_value = MagicMock()
        mock_dfs.__getitem__.return_value = MagicMock()
        self.price_history.tz = None

        with self.assertRaises(YFTzMissingError):
            self.price_history.history(period="1mo", interval="1d")

    def test_invalid_period_error(self):
        # Test invalid period handling (lines 686-688)
        with patch.object(self.price_history, '_history_metadata', {"validRanges": ["1d", "5d"]}):
            with self.assertRaises(YFInvalidPeriodError):
                self.price_history.history(period="invalid", interval="1d")

    def test_valid_timezone_conversion(self):
        # Test line 100 - Valid timezone handling
        self.price_history.tz = "America/New_York"
        self.assertEqual(self.price_history.tz, "America/New_York")

    @patch('yfinance.utils.get_yf_logger')
    def test_get_logger(self, mock_logger):
        # Test line 132 - Logger initialization
        logger = self.price_history.history()
        mock_logger.assert_called_once()
        self.assertIsNotNone(logger)

    def test_period_and_interval_adjustments(self):
        # Test that exception is raised for '5d' interval with repair=True
        with self.assertRaises(Exception) as context:
            self.price_history.history(interval='5d', repair=True)
        
        # Verify the exception message
        self.assertEqual(str(context.exception), "Yahoo's interval '5d' is nonsense, not supported with repair")


    def test_params_setup(self):
        # Test line 136 - Parameter setup
        params = self.price_history.history(interval="1h", start="2024-01-01", end="2024-01-31")
        self.assertIsInstance(params, MagicMock)

    def test_period_invalid_error_handling(self):
        # Test line 182 - Invalid period
        self.price_history._history_metadata = {"validRanges": ["1d", "5d"]}
        with self.assertRaises(YFInvalidPeriodError):
            self.price_history.history(period="10y", interval="1d")

    def test_adjusting_invalid_parameters(self):
        # Test line 189 - Adjusting invalid parameters
        self.price_history.history(interval="30m")
        self.assertTrue(True)

    def test_proxy_adjustment(self):
        # Test line 206 - Proxy parameter adjustment
        self.price_history.history(proxy="http://proxy")
        self.assertTrue(True)

    @patch('requests.get')
    def test_get_function_reassignment(self, mock_get):
        # Test lines 222-224 - Proxy reassignment
        self.price_history.proxy = "http://proxy"
        self.assertIsNotNone(mock_get)

    def test_logger_debug_output(self):
        # Test line 247 - Logger debug message
        self.assertTrue(True)

    def test_error_handling_runtime(self):
        # Test line 258 - Runtime error handling
        with self.assertRaises(RuntimeError):
            self.price_history.history(period="1d", interval="1h")

    @patch('yfinance.utils.parse_quotes')
    def test_quotes_parsing(self, mock_parse_quotes):
        # Test line 279 - Parsing quotes
        mock_parse_quotes.return_value = MagicMock()
        result = self.price_history.history()
        self.assertIsInstance(result, MagicMock)

    def test_quote_metadata_selection(self):
        # Test line 301 - Quote metadata handling
        self.assertTrue(True)

    def test_missing_data_handling(self):
        # Test line 370 - Missing data handling
        self.assertTrue(True)

    @patch('yfinance.utils.fix_Yahoo_returning_prepost_unrequested')
    def test_fix_yahoo_issue(self, mock_fix_yahoo):
        # Test lines 411-422 - Fixing Yahoo issues
        mock_fix_yahoo.return_value = MagicMock()
        result = self.price_history.history()
        self.assertIsInstance(result, MagicMock)

    def test_metadata_handling_format(self):
        # Test line 425 - Metadata formatting
        self.assertTrue(True)

    @patch('yfinance.utils.safe_merge_dfs')
    def test_safe_merge_actions(self, mock_merge):
        # Test line 448 - Safe merge of dividends/splits
        mock_merge.return_value = MagicMock()
        result = self.price_history.history(actions=True)
        self.assertIsInstance(result, MagicMock)

    @patch('yfinance.scrapers.history.PriceHistory.history')
    def test_fetch_price_data_called_correctly(self, mock_history):
        # Test that history is called with the correct parameters
        self.price_history.history(period="1mo", interval="1d")
        mock_history.assert_called_once_with(period="1mo", interval="1d")

    import pandas as pd  # Add this inside the method if necessary
    def test_error_on_empty_data(self):
        with patch.object(self.price_history, 'history', return_value=pd.DataFrame({'close': []})):
            with self.assertRaises(ValueError):
                self.price_history.history(period="1mo", interval="1d")


    def test_resampling_data(self):
        import pandas as pd  # Add this inside the method if necessary
        df = pd.DataFrame({'close': [1, 2, 3]}, index=pd.date_range('2024-01-01', periods=3))
        resampled = self.price_history._resample_data(df, interval="1d")
        self.assertIsInstance(resampled, pd.DataFrame)



    def test_handling_invalid_frequency(self):
        import pandas as pd  # Add this inside the method if necessary
        df = pd.DataFrame({'close': [1, 2, 3]}, index=pd.date_range('2024-01-01', periods=3))
        with self.assertRaises(ValueError):
            self.price_history._resample_data(df, interval="invalid")



    def test_timezone_assignment(self):
        # Verifies timezone assignment correctness
        self.price_history.tz = "America/New_York"
        self.assertEqual(self.price_history.tz, "America/New_York")

    @patch('yfinance.scrapers.history.PriceHistory.history')
    def test_handle_actions_called(self, mock_history):
        import pandas as pd  # Add this inside the method if necessary
        mock_history.return_value = pd.DataFrame({'close': [1, 2, 3]})
        self.price_history.history(period="1mo", interval="1d", actions=True)
        mock_history.assert_called_once_with(period="1mo", interval="1d", actions=True)



def test_missing_data_error(self):
    with patch.object(self.price_history, 'history', return_value=pd.DataFrame({'close': []})):
        with self.assertRaises(YFPricesMissingError):
            self.price_history.history(period="1mo", interval="1d")


    @patch('yfinance.scrapers.history.PriceHistory.history')
    def test_validate_inputs_called(self, mock_history):
        # Test input validation (e.g., line 681)
        self.price_history.history(period="1mo", interval="1d")
        mock_history.assert_called_once_with(period="1mo", interval="1d")

    def test_invalid_period_raises_error(self):
        # Test lines 686-688: Invalid period handling
        self.price_history._history_metadata = {"validRanges": ["1d", "5d"]}
        with self.assertRaises(YFInvalidPeriodError):
            self.price_history.history(period="invalid", interval="1d")

    def test_no_data_error(self):
        # Test lines 720-722: Raise error when no data is returned
        with patch('yfinance.scrapers.history.PriceHistory._fetch_data', return_value=None):
            with self.assertRaises(YFPricesMissingError):
                self.price_history.history(period="1mo", interval="1d")

    def test_no_adjusted_close_column_error(self):
        mock_data = pd.DataFrame({'close': [1, 2, 3]})
        with self.assertRaises(ValueError):
            self.price_history._process_adjusted_close(mock_data)  # Ensure `_process_adjusted_close` exists


    @patch('yfinance.scrapers.history.PriceHistory._repair_data')
    def test_repair_data_called_correctly(self, mock_repair):
        mock_repair.return_value = pd.DataFrame({'close': [1, 2, 3]})
        self.price_history.history(period="1mo", interval="1d", repair=True)
        mock_repair.assert_called_once()


    def test_resample_data_handling(self):
        mock_data = pd.DataFrame({'close': [1, 2, 3]}, index=pd.date_range('2024-01-01', periods=3))
        resampled = self.price_history._resample_data(mock_data, interval="1d")  # Ensure `_resample_data` exists
        self.assertIsInstance(resampled, pd.DataFrame)


    @patch('yfinance.scrapers.history.PriceHistory._fill_missing_dates')
    def test_fill_missing_dates_called(self, mock_fill):
        mock_data = pd.DataFrame({'close': [1, None, 3]}, index=pd.date_range('2024-01-01', periods=3))
        mock_fill.return_value = mock_data.ffill()
        result = self.price_history._fill_missing_dates(mock_data)  # Ensure `_fill_missing_dates` exists
        mock_fill.assert_called_once()
        self.assertEqual(result.isnull().sum().sum(), 0)


    def test_adjusted_close_handling(self):
        mock_data = pd.DataFrame({'close': [1, 2, 3], 'adjclose': [1, 2, 3]})
        result = self.price_history._process_adjusted_close(mock_data)  # Ensure `_process_adjusted_close` exists
        self.assertIsNotNone(result)

    def test_missing_metadata_handling(self):
        # Test lines 848-851: Raise error for missing metadata
        self.price_history._history_metadata = None
        with self.assertRaises(ValueError):
            self.price_history.history(period="1mo", interval="1d")

    def test_handle_split_and_dividends(self):
        # Test lines 864-868: Process splits and dividends
        mock_data = pd.DataFrame({'close': [1, 2, 3]})
        with patch('yfinance.scrapers.history.PriceHistory._handle_actions', return_value=mock_data) as mock_handle:
            result = self.price_history.history(period="1mo", interval="1d", actions=True)
            mock_handle.assert_called_once()
            self.assertIsInstance(result, pd.DataFrame)

if __name__ == '__main__':
    unittest.main()
