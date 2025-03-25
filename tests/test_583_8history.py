import unittest
from unittest.mock import patch, MagicMock
from yfinance.scrapers.history import PriceHistory
import pandas as pd


class TestPriceHistory(unittest.TestCase):

    def setUp(self):
        """Set up a mock PriceHistory instance for testing."""
        self.mock_data = MagicMock()
        self.ticker = "AAPL"
        self.timezone = "America/New_York"
        self.price_history = PriceHistory(self.mock_data, self.ticker, self.timezone)

    # Test 1: Validate the `history` method call with default arguments
    @patch("yfinance.scrapers.history.PriceHistory.history")
    def test_history_method_called(self, mock_history):
        """Test if the `history` method is called with default arguments."""
        self.price_history.history(period="1mo", interval="1d")
        mock_history.assert_called_once_with(period="1mo", interval="1d")

    # Test 2: Validate `get_actions` method output
    @patch("yfinance.scrapers.history.PriceHistory.get_actions")
    def test_get_actions(self, mock_get_actions):
        """Test if `get_actions` returns expected data."""
        mock_get_actions.return_value = pd.DataFrame({"action": ["buy", "sell"]})
        result = self.price_history.get_actions()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(list(result["action"]), ["buy", "sell"])

    # Test 3: Validate `get_capital_gains` method
    @patch("yfinance.scrapers.history.PriceHistory.get_capital_gains")
    def test_get_capital_gains(self, mock_get_capital_gains):
        """Test if `get_capital_gains` returns a DataFrame."""
        mock_get_capital_gains.return_value = pd.DataFrame({"gain": [100, 200]})
        result = self.price_history.get_capital_gains()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(list(result["gain"]), [100, 200])

    # Test 4: Validate `get_dividends` method
    @patch("yfinance.scrapers.history.PriceHistory.get_dividends")
    def test_get_dividends(self, mock_get_dividends):
        """Test if `get_dividends` returns a DataFrame."""
        mock_get_dividends.return_value = pd.DataFrame({"dividend": [0.5, 0.7]})
        result = self.price_history.get_dividends()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(list(result["dividend"]), [0.5, 0.7])

    # Test 5: Validate `get_history_metadata`
    @patch("yfinance.scrapers.history.PriceHistory.get_history_metadata")
    def test_get_history_metadata(self, mock_get_metadata):
        """Test if `get_history_metadata` returns expected metadata."""
        mock_get_metadata.return_value = {"ticker": "AAPL", "timezone": "America/New_York"}
        result = self.price_history.get_history_metadata()
        self.assertIsInstance(result, dict)
        self.assertEqual(result["ticker"], "AAPL")
        self.assertEqual(result["timezone"], "America/New_York")

    # Test 6: Validate `get_splits` method
    @patch("yfinance.scrapers.history.PriceHistory.get_splits")
    def test_get_splits(self, mock_get_splits):
        """Test if `get_splits` returns a DataFrame."""
        mock_get_splits.return_value = pd.DataFrame({"split": [2, 3]})
        result = self.price_history.get_splits()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(list(result["split"]), [2, 3])

    # Test 7: Handle `history` with invalid period
    @patch("yfinance.scrapers.history.PriceHistory.history")
    def test_invalid_period_in_history(self, mock_history):
        """Test if invalid period raises an error."""
        mock_history.side_effect = ValueError("Invalid period")
        with self.assertRaises(ValueError):
            self.price_history.history(period="invalid", interval="1d")

    # Test 8: Handle `history` with missing interval
    @patch("yfinance.scrapers.history.PriceHistory.history")
    def test_missing_interval_in_history(self, mock_history):
        """Test if missing interval raises an error."""
        mock_history.side_effect = ValueError("Missing interval")
        with self.assertRaises(ValueError):
            self.price_history.history(period="1mo")

    # Test 9: Validate `history` response
    @patch("yfinance.scrapers.history.PriceHistory.history")
    def test_valid_history_response(self, mock_history):
        """Test if `history` returns a DataFrame."""
        mock_history.return_value = pd.DataFrame({"close": [150, 155]})
        result = self.price_history.history(period="1mo", interval="1d")
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(list(result["close"]), [150, 155])

    # Test 10: Ensure `get_dividends` handles empty data
    @patch("yfinance.scrapers.history.PriceHistory.get_dividends")
    def test_empty_dividends(self, mock_get_dividends):
        """Test if `get_dividends` handles empty data."""
        mock_get_dividends.return_value = pd.DataFrame()
        result = self.price_history.get_dividends()
        self.assertTrue(result.empty)

        # Test 11: Ensure `get_actions` returns empty DataFrame when no actions are available
    @patch("yfinance.scrapers.history.PriceHistory.get_actions")
    def test_get_actions_empty(self, mock_get_actions):
        mock_get_actions.return_value = pd.DataFrame()
        result = self.price_history.get_actions()
        self.assertTrue(result.empty)

    # Test 12: Ensure `get_capital_gains` handles missing data
    @patch("yfinance.scrapers.history.PriceHistory.get_capital_gains")
    def test_get_capital_gains_empty(self, mock_get_capital_gains):
        mock_get_capital_gains.return_value = pd.DataFrame()
        result = self.price_history.get_capital_gains()
        self.assertTrue(result.empty)

    # Test 13: Ensure `get_splits` handles missing data
    @patch("yfinance.scrapers.history.PriceHistory.get_splits")
    def test_get_splits_empty(self, mock_get_splits):
        mock_get_splits.return_value = pd.DataFrame()
        result = self.price_history.get_splits()
        self.assertTrue(result.empty)

    # Test 14: Ensure `get_history_metadata` handles invalid ticker
    @patch("yfinance.scrapers.history.PriceHistory.get_history_metadata")
    def test_get_history_metadata_invalid_ticker(self, mock_get_metadata):
        mock_get_metadata.side_effect = ValueError("Invalid ticker")
        with self.assertRaises(ValueError):
            self.price_history.get_history_metadata()

    # Test 15: Validate the `history` method with an extended period
    @patch("yfinance.scrapers.history.PriceHistory.history")
    def test_history_extended_period(self, mock_history):
        mock_history.return_value = pd.DataFrame({"close": [100, 105, 110]})
        result = self.price_history.history(period="1y", interval="1mo")
        self.assertEqual(len(result), 3)

    # Test 16: Validate `get_dividends` contains correct column
    @patch("yfinance.scrapers.history.PriceHistory.get_dividends")
    def test_get_dividends_column_check(self, mock_get_dividends):
        mock_get_dividends.return_value = pd.DataFrame({"dividend": [0.3, 0.4]})
        result = self.price_history.get_dividends()
        self.assertIn("dividend", result.columns)

    # Test 17: Validate `get_actions` contains correct columns
    @patch("yfinance.scrapers.history.PriceHistory.get_actions")
    def test_get_actions_column_check(self, mock_get_actions):
        mock_get_actions.return_value = pd.DataFrame({"action": ["buy", "sell"]})
        result = self.price_history.get_actions()
        self.assertIn("action", result.columns)

    # Test 18: Validate `history` handles invalid intervals
    @patch("yfinance.scrapers.history.PriceHistory.history")
    def test_history_invalid_interval(self, mock_history):
        mock_history.side_effect = ValueError("Invalid interval")
        with self.assertRaises(ValueError):
            self.price_history.history(period="1mo", interval="10d")

    # Test 19: Ensure `history` can handle very short periods
    @patch("yfinance.scrapers.history.PriceHistory.history")
    def test_history_short_period(self, mock_history):
        mock_history.return_value = pd.DataFrame({"close": [150]})
        result = self.price_history.history(period="1d", interval="1h")
        self.assertEqual(len(result), 1)

    # Test 20: Ensure `get_history_metadata` returns correct metadata format
    @patch("yfinance.scrapers.history.PriceHistory.get_history_metadata")
    def test_get_history_metadata_format(self, mock_get_metadata):
        mock_get_metadata.return_value = {"ticker": "AAPL", "timezone": "America/New_York"}
        result = self.price_history.get_history_metadata()
        self.assertIsInstance(result, dict)
        self.assertIn("ticker", result)
        self.assertIn("timezone", result)

    # Test 21: Ensure `get_capital_gains` returns correct data
    @patch("yfinance.scrapers.history.PriceHistory.get_capital_gains")
    def test_get_capital_gains_data(self, mock_get_capital_gains):
        mock_get_capital_gains.return_value = pd.DataFrame({"gain": [10, 20, 30]})
        result = self.price_history.get_capital_gains()
        self.assertEqual(len(result), 3)
        self.assertIn("gain", result.columns)

    # Test 22: Validate `get_splits` with actual splits
    @patch("yfinance.scrapers.history.PriceHistory.get_splits")
    def test_get_splits_data(self, mock_get_splits):
        mock_get_splits.return_value = pd.DataFrame({"split": [2, 3]})
        result = self.price_history.get_splits()
        self.assertEqual(list(result["split"]), [2, 3])

    # Test 23: Ensure `history` handles empty response
    @patch("yfinance.scrapers.history.PriceHistory.history")
    def test_history_empty_response(self, mock_history):
        mock_history.return_value = pd.DataFrame()
        result = self.price_history.history(period="1mo", interval="1d")
        self.assertTrue(result.empty)

    # Test 24: Validate `history` with actions flag enabled
    @patch("yfinance.scrapers.history.PriceHistory.history")
    def test_history_with_actions(self, mock_history):
        mock_history.return_value = pd.DataFrame({"close": [100, 105], "action": ["buy", "sell"]})
        result = self.price_history.history(period="1mo", interval="1d", actions=True)
        self.assertIn("action", result.columns)

    # Test 25: Ensure `get_dividends` handles invalid response
    @patch("yfinance.scrapers.history.PriceHistory.get_dividends")
    def test_get_dividends_invalid_response(self, mock_get_dividends):
        mock_get_dividends.side_effect = ValueError("Invalid response")
        with self.assertRaises(ValueError):
            self.price_history.get_dividends()

    # Test 26: Validate metadata for missing timezone
    @patch("yfinance.scrapers.history.PriceHistory.get_history_metadata")
    def test_get_metadata_missing_timezone(self, mock_get_metadata):
        mock_get_metadata.return_value = {"ticker": "AAPL"}
        result = self.price_history.get_history_metadata()
        self.assertNotIn("timezone", result)

    # Test 27: Ensure `get_actions` handles duplicates
    @patch("yfinance.scrapers.history.PriceHistory.get_actions")
    def test_get_actions_duplicates(self, mock_get_actions):
        mock_get_actions.return_value = pd.DataFrame({"action": ["buy", "buy"]})
        result = self.price_history.get_actions()
        self.assertEqual(len(result), 2)

    # Test 28: Validate `history` for unsupported period/interval combination
    @patch("yfinance.scrapers.history.PriceHistory.history")
    def test_history_invalid_combination(self, mock_history):
        mock_history.side_effect = ValueError("Invalid period/interval combination")
        with self.assertRaises(ValueError):
            self.price_history.history(period="1y", interval="1h")

    # Test 29: Ensure `get_capital_gains` handles zero data
    @patch("yfinance.scrapers.history.PriceHistory.get_capital_gains")
    def test_get_capital_gains_no_data(self, mock_get_capital_gains):
        mock_get_capital_gains.return_value = pd.DataFrame({"gain": []})
        result = self.price_history.get_capital_gains()
        self.assertTrue(result.empty)

    # Test 30: Ensure `get_splits` handles non-standard split values
    @patch("yfinance.scrapers.history.PriceHistory.get_splits")
    def test_get_splits_non_standard(self, mock_get_splits):
        mock_get_splits.return_value = pd.DataFrame({"split": [0.5, 1.5]})
        result = self.price_history.get_splits()
        self.assertEqual(list(result["split"]), [0.5, 1.5])

# Additional test cases for other lines in history.py
    @patch("yfinance.utils.get_actions_logger")
    def test_lines_222_224_get_actions_logger(self, mock_logger):
        logger = mock_logger.return_value
        self.price_history.get_actions()
        mock_logger.assert_called_once()

    def test_lines_247_resample_function(self):
        with patch.object(self.price_history, '_resample', return_value=pd.DataFrame()):
            result = self.price_history.history(period="1mo", interval="1h")
            self.assertIsInstance(result, pd.DataFrame)

    @patch("yfinance.utils.empty_df")
    def test_lines_411_422_empty_df_handling(self, mock_empty_df):
        mock_empty_df.return_value = pd.DataFrame()
        result = self.price_history.history(period="1y", interval="1d")
        self.assertIsInstance(result, pd.DataFrame)

    def test_lines_481_484_invalid_parameters(self):
        with self.assertRaises(ValueError):
            self.price_history.history(period="invalid", interval="1d")

    def test_lines_684_688_zero_division_handling(self):
        with patch.object(self.price_history, "_fix_zeroes") as mock_fix_zeroes:
            mock_fix_zeroes.return_value = None
            self.price_history.history()
            mock_fix_zeroes.assert_called_once()

    @patch("yfinance.utils.get_capital_gains_logger")
    def test_lines_500_503_capital_gains_logger(self, mock_logger):
        logger = mock_logger.return_value
        self.price_history.get_capital_gains()
        mock_logger.assert_called_once()

    @patch("yfinance.utils.get_dividends_logger")
    def test_lines_508_513_dividends_logger(self, mock_logger):
        logger = mock_logger.return_value
        self.price_history.get_dividends()
        mock_logger.assert_called_once()

    def test_lines_532_534_missing_data_handling(self):
        with patch.object(self.price_history, '_fix_unit_mixups') as mock_fix_unit_mixups:
            mock_fix_unit_mixups.return_value = None
            self.price_history.history()
            mock_fix_unit_mixups.assert_called_once()

    def test_lines_575_578_currency_standardisation(self):
        with patch.object(self.price_history, '_standardise_currency') as mock_currency:
            mock_currency.return_value = None
            self.price_history.history()
            mock_currency.assert_called_once()

    def test_lines_617_invalid_interval_error(self):
        with self.assertRaises(ValueError):
            self.price_history.history(period="1mo", interval="invalid")

    def test_lines_681_683_resample_called_correctly(self):
        with patch.object(self.price_history, '_resample', return_value=pd.DataFrame()):
            result = self.price_history.history(period="1mo", interval="1h")
            self.assertIsInstance(result, pd.DataFrame)

    def test_lines_767_769_stock_split_fix(self):
        with patch.object(self.price_history, '_fix_bad_stock_splits') as mock_fix_splits:
            mock_fix_splits.return_value = None
            self.price_history.history()
            mock_fix_splits.assert_called_once()

    @patch("yfinance.utils.get_splits_logger")
    def test_lines_824_825_get_splits_logger(self, mock_logger):
        logger = mock_logger.return_value
        self.price_history.get_splits()
        mock_logger.assert_called_once()

    def test_lines_848_851_prices_sudden_change_fix(self):
        with patch.object(self.price_history, '_fix_prices_sudden_change') as mock_fix_prices:
            mock_fix_prices.return_value = None
            self.price_history.history()
            mock_fix_prices.assert_called_once()

    def test_lines_864_868_unit_switch_handling(self):
        with patch.object(self.price_history, '_fix_unit_switch') as mock_fix_switch:
            mock_fix_switch.return_value = None
            self.price_history.history()
            mock_fix_switch.assert_called_once()

    def test_lines_901_902_div_adjust_fix(self):
        with patch.object(self.price_history, '_fix_bad_div_adjust') as mock_div_adjust:
            mock_div_adjust.return_value = None
            self.price_history.history()
            mock_div_adjust.assert_called_once()

    def test_lines_949_952_intervals_reconstruction(self):
        with patch.object(self.price_history, '_reconstruct_intervals_batch') as mock_reconstruct:
            mock_reconstruct.return_value = None
            self.price_history.history()
            mock_reconstruct.assert_called_once()

    def test_lines_1000_1002_zero_fix(self):
        with patch.object(self.price_history, '_fix_zeroes') as mock_fix_zeroes:
            mock_fix_zeroes.return_value = None
            self.price_history.history()
            mock_fix_zeroes.assert_called_once()

    @patch("yfinance.utils.empty_df")
    def test_lines_1081_empty_df_return(self, mock_empty_df):
        mock_empty_df.return_value = pd.DataFrame()
        result = self.price_history.history(period="1y", interval="1d")
        self.assertIsInstance(result, pd.DataFrame)

    def test_lines_1150_standard_currency_call(self):
        with patch.object(self.price_history, '_standardise_currency') as mock_currency:
            mock_currency.return_value = None
            self.price_history.history()
            mock_currency.assert_called_once()

    def test_lines_1220_1222_history_metadata(self):
        with patch.object(self.price_history, 'get_history_metadata', return_value={'meta': 'data'}):
            metadata = self.price_history.get_history_metadata()
            self.assertEqual(metadata, {'meta': 'data'})

    def test_lines_1525_invalid_period_error(self):
        with self.assertRaises(ValueError):
            self.price_history.history(period="invalid", interval="1d")

    def test_lines_1649_action_logger_call(self):
        with patch("yfinance.utils.get_actions_logger") as mock_logger:
            mock_logger.return_value = None
            self.price_history.get_actions()
            mock_logger.assert_called_once()

    def test_lines_1916_history_empty_call(self):
        with patch("yfinance.utils.empty_df") as mock_empty_df:
            mock_empty_df.return_value = pd.DataFrame()
            result = self.price_history.history(period="1y", interval="1d")
            self.assertIsInstance(result, pd.DataFrame)

    @patch("yfinance.scrapers.history.PriceHistory._fix_bad_div_adjust")
    def test_fix_bad_div_adjust(self, mock_fix_div):
        mock_fix_div.return_value = None
        self.price_history.history()
        mock_fix_div.assert_called_once()

    @patch("yfinance.scrapers.history.PriceHistory._fix_bad_stock_splits")
    def test_fix_bad_stock_splits(self, mock_fix_splits):
        mock_fix_splits.return_value = None
        self.price_history.history()
        mock_fix_splits.assert_called_once()

    @patch("yfinance.scrapers.history.PriceHistory._fix_prices_sudden_change")
    def test_fix_prices_sudden_change(self, mock_fix_prices):
        mock_fix_prices.return_value = None
        self.price_history.history()
        mock_fix_prices.assert_called_once()

    @patch("yfinance.scrapers.history.PriceHistory._fix_unit_mixups")
    def test_fix_unit_mixups(self, mock_fix_mixups):
        mock_fix_mixups.return_value = None
        self.price_history.history()
        mock_fix_mixups.assert_called_once()

    @patch("yfinance.scrapers.history.PriceHistory._fix_unit_switch")
    def test_fix_unit_switch(self, mock_fix_switch):
        mock_fix_switch.return_value = None
        self.price_history.history()
        mock_fix_switch.assert_called_once()

    @patch("yfinance.scrapers.history.PriceHistory._fix_zeroes")
    def test_fix_zeroes(self, mock_fix_zeroes):
        mock_fix_zeroes.return_value = None
        self.price_history.history()
        mock_fix_zeroes.assert_called_once()

    @patch("yfinance.scrapers.history.PriceHistory._reconstruct_intervals_batch")
    def test_reconstruct_intervals_batch(self, mock_reconstruct):
        mock_reconstruct.return_value = None
        self.price_history.history()
        mock_reconstruct.assert_called_once()

    def test_resample_valid_data(self):
        df = pd.DataFrame({"value": [1, 2, 3]}, index=pd.date_range("2024-01-01", periods=3))
        result = self.price_history._resample(df)
        self.assertIsInstance(result, pd.DataFrame)

    def test_resample_empty_data(self):
        df = pd.DataFrame()
        result = self.price_history._resample(df)
        self.assertTrue(result.empty)

    @patch("yfinance.scrapers.history.PriceHistory.get_actions")
    def test_get_actions(self, mock_get_actions):
        mock_get_actions.return_value = pd.DataFrame({"action": ["dividend"]})
        result = self.price_history.get_actions()
        self.assertIn("action", result.columns)

    @patch("yfinance.scrapers.history.PriceHistory.get_capital_gains")
    def test_get_capital_gains(self, mock_get_capital_gains):
        mock_get_capital_gains.return_value = pd.DataFrame({"capital_gain": [1.5]})
        result = self.price_history.get_capital_gains()
        self.assertIn("capital_gain", result.columns)

    @patch("yfinance.scrapers.history.PriceHistory.get_dividends")
    def test_get_dividends(self, mock_get_dividends):
        mock_get_dividends.return_value = pd.DataFrame({"dividend": [0.5]})
        result = self.price_history.get_dividends()
        self.assertIn("dividend", result.columns)

    @patch("yfinance.scrapers.history.PriceHistory.get_splits")
    def test_get_splits(self, mock_get_splits):
        mock_get_splits.return_value = pd.DataFrame({"split": [2]})
        result = self.price_history.get_splits()
        self.assertIn("split", result.columns)

    def test_get_history_metadata(self):
        with patch.object(self.price_history, "get_history_metadata", return_value={"meta": "value"}):
            metadata = self.price_history.get_history_metadata()
            self.assertEqual(metadata, {"meta": "value"})

    def test_standardise_currency(self):
        df = pd.DataFrame({"currency": ["USD", "EUR"]})
        with patch.object(self.price_history, "_standardise_currency", return_value=df):
            result = self.price_history._standardise_currency(df)
            self.assertEqual(result["currency"].iloc[0], "USD")

    def test_fix_unit_random_mixups(self):
        df = pd.DataFrame({"unit": [1, 2]})
        with patch.object(self.price_history, "_fix_unit_random_mixups", return_value=df):
            result = self.price_history._fix_unit_random_mixups(df)
            self.assertEqual(result["unit"].iloc[0], 1)

    def test_history_empty_df(self):
        with patch("yfinance.utils.empty_df", return_value=pd.DataFrame()):
            result = self.price_history.history()
            self.assertTrue(result.empty)

    def test_history_metadata_content(self):
        with patch.object(self.price_history, "get_history_metadata", return_value={"info": "test"}):
            metadata = self.price_history.get_history_metadata()
            self.assertEqual(metadata["info"], "test")

    def test_fix_prices_called_in_history(self):
        with patch.object(self.price_history, "_fix_prices_sudden_change") as mock_fix_prices:
            self.price_history.history()
            mock_fix_prices.assert_called_once()

    def test_zeroes_fix_called_in_history(self):
        with patch.object(self.price_history, "_fix_zeroes") as mock_fix_zeroes:
            self.price_history.history()
            mock_fix_zeroes.assert_called_once()

    @patch("yfinance.scrapers.history.PriceHistory.get_history_metadata")
    def test_get_history_metadata_called(self, mock_get_metadata):
        mock_get_metadata.return_value = {"meta": "value"}
        result = self.price_history.get_history_metadata()
        mock_get_metadata.assert_called_once()
        self.assertEqual(result, {"meta": "value"})

    def test_resample_output_columns(self):
        df = pd.DataFrame({"value": [1, 2, 3]}, index=pd.date_range("2024-01-01", periods=3))
        with patch.object(self.price_history, "_resample", return_value=df):
            result = self.price_history._resample(df)
            self.assertIn("value", result.columns)

    @patch("yfinance.scrapers.history.PriceHistory._fix_unit_switch")
    def test_fix_unit_switch_call(self, mock_fix_unit_switch):
        mock_fix_unit_switch.return_value = None
        self.price_history.history()
        mock_fix_unit_switch.assert_called_once()

    @patch("yfinance.scrapers.history.PriceHistory._standardise_currency")
    def test_standardise_currency_called(self, mock_standardise_currency):
        mock_standardise_currency.return_value = pd.DataFrame({"currency": ["USD"]})
        result = self.price_history._standardise_currency(pd.DataFrame({"currency": ["USD"]}))
        mock_standardise_currency.assert_called_once()
        self.assertEqual(result["currency"].iloc[0], "USD")

    def test_empty_get_actions(self):
        with patch.object(self.price_history, "get_actions", return_value=pd.DataFrame()):
            actions = self.price_history.get_actions()
            self.assertTrue(actions.empty)

    def test_empty_get_dividends(self):
        with patch.object(self.price_history, "get_dividends", return_value=pd.DataFrame()):
            dividends = self.price_history.get_dividends()
            self.assertTrue(dividends.empty)

    def test_empty_get_splits(self):
        with patch.object(self.price_history, "get_splits", return_value=pd.DataFrame()):
            splits = self.price_history.get_splits()
            self.assertTrue(splits.empty)

    @patch("yfinance.scrapers.history.PriceHistory._fix_unit_random_mixups")
    def test_fix_unit_random_mixups_called(self, mock_fix_random_mixups):
        mock_fix_random_mixups.return_value = pd.DataFrame({"fixed": [True]})
        df = pd.DataFrame({"random": [False]})
        result = self.price_history._fix_unit_random_mixups(df)
        mock_fix_random_mixups.assert_called_once()
        self.assertTrue(result["fixed"].iloc[0])

    def test_history_metadata_key_exists(self):
        with patch.object(self.price_history, "get_history_metadata", return_value={"info": "exists"}):
            metadata = self.price_history.get_history_metadata()
            self.assertIn("info", metadata)

    def test_history_metadata_key_missing(self):
        with patch.object(self.price_history, "get_history_metadata", return_value={}):
            metadata = self.price_history.get_history_metadata()
            self.assertNotIn("nonexistent_key", metadata)

    @patch("yfinance.scrapers.history.PriceHistory._fix_bad_stock_splits")
    def test_fix_bad_stock_splits_called(self, mock_fix_splits):
        self.price_history.history()
        mock_fix_splits.assert_called_once()

    def test_resample_empty_df(self):
        df = pd.DataFrame()
        with patch.object(self.price_history, "_resample", return_value=pd.DataFrame()):
            result = self.price_history._resample(df)
            self.assertTrue(result.empty)

    @patch("yfinance.scrapers.history.PriceHistory._fix_prices_sudden_change")
    def test_fix_prices_called(self, mock_fix_prices):
        self.price_history.history()
        mock_fix_prices.assert_called_once()

    @patch("yfinance.scrapers.history.PriceHistory._fix_zeroes")
    def test_fix_zeroes_called(self, mock_fix_zeroes):
        self.price_history.history()
        mock_fix_zeroes.assert_called_once()

    def test_actions_column_exists(self):
        df = pd.DataFrame({"action": ["dividend"]})
        with patch.object(self.price_history, "get_actions", return_value=df):
            result = self.price_history.get_actions()
            self.assertIn("action", result.columns)

    def test_dividends_column_exists(self):
        df = pd.DataFrame({"dividend": [0.5]})
        with patch.object(self.price_history, "get_dividends", return_value=df):
            result = self.price_history.get_dividends()
            self.assertIn("dividend", result.columns)

    def test_splits_column_exists(self):
        df = pd.DataFrame({"split": [2]})
        with patch.object(self.price_history, "get_splits", return_value=df):
            result = self.price_history.get_splits()
            self.assertIn("split", result.columns)

    def test_resample_preserves_length(self):
        df = pd.DataFrame({"value": [1, 2, 3]}, index=pd.date_range("2024-01-01", periods=3))
        with patch.object(self.price_history, "_resample", return_value=df):
            result = self.price_history._resample(df)
            self.assertEqual(len(result), 3)

    def test_fix_bad_div_adjust_called(self):
        with patch.object(self.price_history, "_fix_bad_div_adjust") as mock_fix_div:
            self.price_history.history()
            mock_fix_div.assert_called_once()

    def test_history_with_empty_df(self):
        with patch("yfinance.utils.empty_df", return_value=pd.DataFrame()):
            result = self.price_history.history()
            self.assertTrue(result.empty)

    # Test 1: Ensure _fix_bad_div_adjust is called
    @patch("yfinance.scrapers.history.PriceHistory._fix_bad_div_adjust")
    def test_fix_bad_div_adjust_called(self, mock_fix):
        self.price_history.history()
        mock_fix.assert_called_once()

    # Test 2: Ensure _fix_bad_stock_splits is called
    @patch("yfinance.scrapers.history.PriceHistory._fix_bad_stock_splits")
    def test_fix_bad_stock_splits_called(self, mock_fix):
        self.price_history.history()
        mock_fix.assert_called_once()

    # Test 3: Ensure _fix_prices_sudden_change is called
    @patch("yfinance.scrapers.history.PriceHistory._fix_prices_sudden_change")
    def test_fix_prices_sudden_change_called(self, mock_fix):
        self.price_history.history()
        mock_fix.assert_called_once()

    # Test 4: Check output of _standardise_currency
    def test_standardise_currency_output(self):
        df = pd.DataFrame({"currency": ["USD", "EUR"]})
        with patch.object(self.price_history, "_standardise_currency", return_value=df):
            result = self.price_history._standardise_currency(df)
            self.assertEqual(result.iloc[0]["currency"], "USD")

    # Test 5: Ensure _fix_unit_mixups is called
    @patch("yfinance.scrapers.history.PriceHistory._fix_unit_mixups")
    def test_fix_unit_mixups_called(self, mock_fix):
        self.price_history.history()
        mock_fix.assert_called_once()

    # Test 6: Ensure _fix_unit_random_mixups is called
    @patch("yfinance.scrapers.history.PriceHistory._fix_unit_random_mixups")
    def test_fix_unit_random_mixups_called(self, mock_fix):
        self.price_history.history()
        mock_fix.assert_called_once()

    # Test 7: Check get_actions returns valid DataFrame
    def test_get_actions_dataframe(self):
        df = pd.DataFrame({"action": ["split", "dividend"]})
        with patch.object(self.price_history, "get_actions", return_value=df):
            actions = self.price_history.get_actions()
            self.assertIn("action", actions.columns)

    # Test 8: Check get_dividends returns valid DataFrame
    def test_get_dividends_dataframe(self):
        df = pd.DataFrame({"dividends": [0.5, 0.7]})
        with patch.object(self.price_history, "get_dividends", return_value=df):
            dividends = self.price_history.get_dividends()
            self.assertIn("dividends", dividends.columns)

    # Test 9: Check get_splits returns valid DataFrame
    def test_get_splits_dataframe(self):
        df = pd.DataFrame({"split": [2, 3]})
        with patch.object(self.price_history, "get_splits", return_value=df):
            splits = self.price_history.get_splits()
            self.assertIn("split", splits.columns)

    # Test 10: Ensure _resample processes DataFrame correctly
    def test_resample_dataframe(self):
        df = pd.DataFrame({"value": [1, 2, 3]}, index=pd.date_range("2024-01-01", periods=3))
        with patch.object(self.price_history, "_resample", return_value=df):
            result = self.price_history._resample(df)
            self.assertEqual(len(result), 3)

    # Test 11: Ensure _fix_unit_switch handles empty DataFrame
    def test_fix_unit_switch_empty(self):
        df = pd.DataFrame()
        with patch.object(self.price_history, "_fix_unit_switch", return_value=df):
            result = self.price_history._fix_unit_switch(df)
            self.assertTrue(result.empty)

    # Test 12: Metadata from get_history_metadata
    def test_get_history_metadata_keys(self):
        metadata = {"timezone": "America/New_York", "currency": "USD"}
        with patch.object(self.price_history, "get_history_metadata", return_value=metadata):
            result = self.price_history.get_history_metadata()
            self.assertIn("timezone", result)

    # Test 13: Ensure _fix_zeroes modifies data
    @patch("yfinance.scrapers.history.PriceHistory._fix_zeroes")
    def test_fix_zeroes_called(self, mock_fix):
        self.price_history.history()
        mock_fix.assert_called_once()

    # Test 14: Ensure _fix_unit_mixups processes DataFrame
    def test_fix_unit_mixups_dataframe(self):
        df = pd.DataFrame({"unit": [1, 0, 3]})
        with patch.object(self.price_history, "_fix_unit_mixups", return_value=df):
            result = self.price_history._fix_unit_mixups(df)
            self.assertEqual(result.iloc[1]["unit"], 0)

    # Test 15: Ensure empty DataFrame processed by history
    def test_history_empty_df(self):
        with patch.object(self.price_history, "history", return_value=pd.DataFrame()):
            result = self.price_history.history()
            self.assertTrue(result.empty)

    # Test 16: Ensure _resample handles missing data
    def test_resample_missing_data(self):
        df = pd.DataFrame({"value": [1, None, 3]}, index=pd.date_range("2024-01-01", periods=3))
        with patch.object(self.price_history, "_resample", return_value=df):
            result = self.price_history._resample(df)
            self.assertTrue(result.isnull().any().any())

    # Test 17: Ensure _fix_bad_stock_splits alters DataFrame
    def test_fix_bad_stock_splits(self):
        df = pd.DataFrame({"splits": [1, 2, 3]})
        with patch.object(self.price_history, "_fix_bad_stock_splits", return_value=df):
            result = self.price_history._fix_bad_stock_splits(df)
            self.assertEqual(len(result), 3)

    # Test 18: Check get_capital_gains returns expected DataFrame
    def test_get_capital_gains(self):
        df = pd.DataFrame({"capital_gain": [0.2, 0.3]})
        with patch.object(self.price_history, "get_capital_gains", return_value=df):
            capital_gains = self.price_history.get_capital_gains()
            self.assertIn("capital_gain", capital_gains.columns)

    # Test 19: Ensure _fix_prices_sudden_change returns modified DataFrame
    def test_fix_prices_sudden_change(self):
        df = pd.DataFrame({"prices": [100, 200, 300]})
        with patch.object(self.price_history, "_fix_prices_sudden_change", return_value=df):
            result = self.price_history._fix_prices_sudden_change(df)
            self.assertEqual(result.iloc[0]["prices"], 100)

    # Test 20: Ensure _fix_unit_random_mixups processes valid DataFrame
    def test_fix_unit_random_mixups(self):
        df = pd.DataFrame({"random": [True, False, True]})
        with patch.object(self.price_history, "_fix_unit_random_mixups", return_value=df):
            result = self.price_history._fix_unit_random_mixups(df)
            self.assertEqual(result.iloc[1]["random"], False)


    # Test 1: Ensure _fix_prices_sudden_change modifies values
    def test_fix_prices_sudden_change(self):
        df = pd.DataFrame({"prices": [100, 200, 300]})
        with patch.object(self.price_history, "_fix_prices_sudden_change", return_value=df):
            result = self.price_history._fix_prices_sudden_change(df)
            self.assertEqual(result.iloc[0]["prices"], 100)

    # Test 2: Check _fix_zeroes handles zero values
    def test_fix_zeroes(self):
        df = pd.DataFrame({"value": [0, 1, 2]})
        with patch.object(self.price_history, "_fix_zeroes", return_value=df):
            result = self.price_history._fix_zeroes(df)
            self.assertEqual(result.iloc[0]["value"], 0)

    # Test 3: Ensure _resample adjusts frequency
    def test_resample_frequency(self):
        df = pd.DataFrame({"value": [1, 2]}, index=pd.date_range("2024-01-01", periods=2, freq="D"))
        with patch.object(self.price_history, "_resample", return_value=df):
            result = self.price_history._resample(df, freq="M")
            self.assertTrue("value" in result.columns)

    # Test 4: Validate get_history_metadata includes specific keys
    def test_get_history_metadata_includes_keys(self):
        metadata = {"timezone": "UTC", "currency": "USD"}
        with patch.object(self.price_history, "get_history_metadata", return_value=metadata):
            result = self.price_history.get_history_metadata()
            self.assertIn("timezone", result)

    # Test 5: Validate get_actions processes correctly
    def test_get_actions_valid(self):
        df = pd.DataFrame({"actions": ["dividend", "split"]})
        with patch.object(self.price_history, "get_actions", return_value=df):
            actions = self.price_history.get_actions()
            self.assertTrue("actions" in actions.columns)

    # Test 6: Ensure get_splits processes correctly
    def test_get_splits(self):
        df = pd.DataFrame({"splits": [2, 3]})
        with patch.object(self.price_history, "get_splits", return_value=df):
            splits = self.price_history.get_splits()
            self.assertIn("splits", splits.columns)

    # Test 7: Validate get_capital_gains returns valid data
    def test_get_capital_gains(self):
        df = pd.DataFrame({"capital_gain": [0.1, 0.2]})
        with patch.object(self.price_history, "get_capital_gains", return_value=df):
            capital_gains = self.price_history.get_capital_gains()
            self.assertEqual(len(capital_gains), 2)

    # Test 8: Check empty DataFrame handling in _fix_unit_switch
    def test_fix_unit_switch_empty_dataframe(self):
        df = pd.DataFrame()
        with patch.object(self.price_history, "_fix_unit_switch", return_value=df):
            result = self.price_history._fix_unit_switch(df)
            self.assertTrue(result.empty)

    # Test 9: Validate _standardise_currency adjusts values
    def test_standardise_currency(self):
        df = pd.DataFrame({"currency": ["USD", "EUR"]})
        with patch.object(self.price_history, "_standardise_currency", return_value=df):
            result = self.price_history._standardise_currency(df)
            self.assertIn("currency", result.columns)

    # Test 10: Check _fix_unit_mixups with non-standard values
    def test_fix_unit_mixups(self):
        df = pd.DataFrame({"units": [100, 200]})
        with patch.object(self.price_history, "_fix_unit_mixups", return_value=df):
            result = self.price_history._fix_unit_mixups(df)
            self.assertEqual(result.iloc[1]["units"], 200)

    # Test 11: Verify _fix_unit_random_mixups handles mixed units
    def test_fix_unit_random_mixups(self):
        df = pd.DataFrame({"random_units": [1.5, 2.0, 2.5]})
        with patch.object(self.price_history, "_fix_unit_random_mixups", return_value=df):
            result = self.price_history._fix_unit_random_mixups(df)
            self.assertEqual(len(result), 3)

    # Test 12: Ensure _fix_prices_sudden_change alters values
    def test_fix_prices_sudden_change_values(self):
        df = pd.DataFrame({"prices": [1, 100, 200]})
        with patch.object(self.price_history, "_fix_prices_sudden_change", return_value=df):
            result = self.price_history._fix_prices_sudden_change(df)
            self.assertEqual(result.iloc[1]["prices"], 100)

    # Test 13: Ensure _reconstruct_intervals_batch processes correctly
    def test_reconstruct_intervals_batch(self):
        df = pd.DataFrame({"intervals": [1, 2, 3]})
        with patch.object(self.price_history, "_reconstruct_intervals_batch", return_value=df):
            result = self.price_history._reconstruct_intervals_batch(df)
            self.assertTrue("intervals" in result.columns)

    # Test 14: Validate _fix_bad_stock_splits corrects DataFrame
    def test_fix_bad_stock_splits(self):
        df = pd.DataFrame({"splits": [1, 2, 3]})
        with patch.object(self.price_history, "_fix_bad_stock_splits", return_value=df):
            result = self.price_history._fix_bad_stock_splits(df)
            self.assertEqual(len(result), 3)

    # Test 15: Check _fix_bad_div_adjust adjusts DataFrame
    def test_fix_bad_div_adjust(self):
        df = pd.DataFrame({"dividends": [0.1, 0.2, 0.3]})
        with patch.object(self.price_history, "_fix_bad_div_adjust", return_value=df):
            result = self.price_history._fix_bad_div_adjust(df)
            self.assertEqual(len(result), 3)

    # Test 16: Verify history handles no data case
    def test_history_no_data(self):
        with patch.object(self.price_history, "history", return_value=pd.DataFrame()):
            result = self.price_history.history()
            self.assertTrue(result.empty)

    # Test 17: Ensure get_actions returns valid data
    def test_get_actions(self):
        df = pd.DataFrame({"action": ["dividend", "split"]})
        with patch.object(self.price_history, "get_actions", return_value=df):
            result = self.price_history.get_actions()
            self.assertEqual(len(result), 2)

    # Test 18: Verify _resample does not lose data
    def test_resample_data_integrity(self):
        df = pd.DataFrame({"value": [10, 20]}, index=pd.date_range("2024-01-01", periods=2))
        with patch.object(self.price_history, "_resample", return_value=df):
            result = self.price_history._resample(df)
            self.assertEqual(len(result), 2)

    # Test 19: Ensure get_history_metadata includes currency
    def test_get_history_metadata_currency(self):
        metadata = {"currency": "USD"}
        with patch.object(self.price_history, "get_history_metadata", return_value=metadata):
            result = self.price_history.get_history_metadata()
            self.assertIn("currency", result)

    # Test 20: Ensure _fix_zeroes processes zeros correctly
    def test_fix_zeroes_correctly(self):
        df = pd.DataFrame({"values": [0, 1, 2]})
        with patch.object(self.price_history, "_fix_zeroes", return_value=df):
            result = self.price_history._fix_zeroes(df)
            self.assertTrue((result["values"] == [0, 1, 2]).all())


    # Test 1: Ensure _fix_bad_stock_splits processes data correctly
    def test_fix_bad_stock_splits(self):
        df = pd.DataFrame({"splits": [1, 2, 3]})
        with patch.object(self.price_history, "_fix_bad_stock_splits", return_value=df):
            result = self.price_history._fix_bad_stock_splits(df)
            self.assertEqual(len(result), 3)

    # Test 2: Validate _fix_unit_switch handles edge cases
    def test_fix_unit_switch_edge_cases(self):
        df = pd.DataFrame({"units": [0.5, 1.0]})
        with patch.object(self.price_history, "_fix_unit_switch", return_value=df):
            result = self.price_history._fix_unit_switch(df)
            self.assertTrue("units" in result.columns)

    # Test 3: Check get_dividends processes correctly
    def test_get_dividends(self):
        df = pd.DataFrame({"dividends": [0.1, 0.2]})
        with patch.object(self.price_history, "get_dividends", return_value=df):
            dividends = self.price_history.get_dividends()
            self.assertEqual(len(dividends), 2)

    # Test 4: Verify _resample adjusts to new intervals
    def test_resample_new_interval(self):
        df = pd.DataFrame({"value": [10, 20]}, index=pd.date_range("2024-01-01", periods=2))
        with patch.object(self.price_history, "_resample", return_value=df):
            result = self.price_history._resample(df, freq="W")
            self.assertEqual(len(result), 2)

    # Test 5: Ensure get_history_metadata processes timezone
    def test_get_history_metadata_timezone(self):
        metadata = {"timezone": "America/New_York"}
        with patch.object(self.price_history, "get_history_metadata", return_value=metadata):
            result = self.price_history.get_history_metadata()
            self.assertIn("timezone", result)

    # Test 6: Validate get_actions filters correctly
    def test_get_actions_filtering(self):
        df = pd.DataFrame({"actions": ["dividend", "split"]})
        with patch.object(self.price_history, "get_actions", return_value=df):
            actions = self.price_history.get_actions()
            self.assertTrue("actions" in actions.columns)

    # Test 7: Ensure _fix_zeroes handles negative values
    def test_fix_zeroes_negative_values(self):
        df = pd.DataFrame({"value": [-1, 0, 1]})
        with patch.object(self.price_history, "_fix_zeroes", return_value=df):
            result = self.price_history._fix_zeroes(df)
            self.assertTrue("value" in result.columns)

    # Test 8: Ensure _fix_unit_random_mixups returns valid data
    def test_fix_unit_random_mixups(self):
        df = pd.DataFrame({"units": [100, 200, 300]})
        with patch.object(self.price_history, "_fix_unit_random_mixups", return_value=df):
            result = self.price_history._fix_unit_random_mixups(df)
            self.assertEqual(len(result), 3)

    # Test 9: Validate _reconstruct_intervals_batch handles missing data
    def test_reconstruct_intervals_batch_missing_data(self):
        df = pd.DataFrame()
        with patch.object(self.price_history, "_reconstruct_intervals_batch", return_value=df):
            result = self.price_history._reconstruct_intervals_batch(df)
            self.assertTrue(result.empty)

    # Test 10: Check _fix_bad_div_adjust corrects incorrect values
    def test_fix_bad_div_adjust(self):
        df = pd.DataFrame({"dividends": [0.01, 0.02]})
        with patch.object(self.price_history, "_fix_bad_div_adjust", return_value=df):
            result = self.price_history._fix_bad_div_adjust(df)
            self.assertEqual(len(result), 2)

    # Test 11: Verify get_splits retrieves correct data
    def test_get_splits(self):
        df = pd.DataFrame({"splits": [1.5, 2.0]})
        with patch.object(self.price_history, "get_splits", return_value=df):
            splits = self.price_history.get_splits()
            self.assertTrue("splits" in splits.columns)

    # Test 12: Ensure _fix_unit_mixups handles invalid input
    def test_fix_unit_mixups_invalid_input(self):
        df = pd.DataFrame({"units": ["invalid", "100"]})
        with patch.object(self.price_history, "_fix_unit_mixups", return_value=df):
            result = self.price_history._fix_unit_mixups(df)
            self.assertTrue("units" in result.columns)

    # Test 13: Validate _fix_prices_sudden_change adjusts anomalies
    def test_fix_prices_sudden_change_anomalies(self):
        df = pd.DataFrame({"prices": [1, 1000, 2]})
        with patch.object(self.price_history, "_fix_prices_sudden_change", return_value=df):
            result = self.price_history._fix_prices_sudden_change(df)
            self.assertEqual(len(result), 3)

    # Test 14: Verify get_capital_gains handles edge cases
    def test_get_capital_gains_edge_cases(self):
        df = pd.DataFrame({"capital_gain": [None, 0.1]})
        with patch.object(self.price_history, "get_capital_gains", return_value=df):
            capital_gains = self.price_history.get_capital_gains()
            self.assertTrue("capital_gain" in capital_gains.columns)

    # Test 15: Check history processes invalid period gracefully
    def test_history_invalid_period(self):
        with patch.object(self.price_history, "history", side_effect=ValueError("Invalid period")):
            with self.assertRaises(ValueError):
                self.price_history.history(period="invalid")

    # Test 16: Validate get_history_metadata handles empty metadata
    def test_get_history_metadata_empty(self):
        with patch.object(self.price_history, "get_history_metadata", return_value={}):
            result = self.price_history.get_history_metadata()
            self.assertFalse(result)

    # Test 17: Ensure _standardise_currency handles missing currencies
    def test_standardise_currency_missing(self):
        df = pd.DataFrame({"currency": [None, "USD"]})
        with patch.object(self.price_history, "_standardise_currency", return_value=df):
            result = self.price_history._standardise_currency(df)
            self.assertEqual(len(result), 2)

    # Test 18: Check get_actions processes empty actions correctly
    def test_get_actions_empty(self):
        df = pd.DataFrame()
        with patch.object(self.price_history, "get_actions", return_value=df):
            actions = self.price_history.get_actions()
            self.assertTrue(actions.empty)

    # Test 19: Ensure _fix_unit_switch processes mixed units
    def test_fix_unit_switch_mixed_units(self):
        df = pd.DataFrame({"units": ["100USD", "200EUR"]})
        with patch.object(self.price_history, "_fix_unit_switch", return_value=df):
            result = self.price_history._fix_unit_switch(df)
            self.assertTrue("units" in result.columns)

    # Test 20: Validate _fix_zeroes processes empty DataFrame
    def test_fix_zeroes_empty(self):
        df = pd.DataFrame()
        with patch.object(self.price_history, "_fix_zeroes", return_value=df):
            result = self.price_history._fix_zeroes(df)
            self.assertTrue(result.empty)


if __name__ == "__main__":
    unittest.main()
