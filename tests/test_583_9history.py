import unittest
import pandas as pd
from unittest.mock import patch
from yfinance.scrapers.history import PriceHistory


class TestPriceHistoryAdditionalCoverage(unittest.TestCase):
    def setUp(self):
        """Set up a mock PriceHistory instance for testing."""
        self.price_history = PriceHistory(None, "AAPL", "America/New_York")

    def create_mock_dataframe(self, columns, rows):
        """Helper function to create a mock DataFrame."""
        return pd.DataFrame(rows, columns=columns)

    # Test for _fix_bad_div_adjust
    def test_fix_bad_div_adjust(self):
        df = self.create_mock_dataframe(["Dividends"], [[0.1], [0], [0.2]])
        result = self.price_history._fix_bad_div_adjust(df, interval="1d", currency="USD")
        self.assertEqual(result["Dividends"].iloc[1], 0.0, "Dividends should be corrected.")

    # Test for _fix_bad_stock_splits
    def test_fix_bad_stock_splits(self):
        df = self.create_mock_dataframe(["Stock Splits"], [[1.0], [0], [0.5]])
        result = self.price_history._fix_bad_stock_splits(df, interval="1d", tz_exchange="America/New_York")
        self.assertTrue((result["Stock Splits"] > 0).all(), "Stock splits should remain positive.")

    # Test for _fix_prices_sudden_change
    def test_fix_prices_sudden_change(self):
        df = self.create_mock_dataframe(["Price"], [[100], [200], [1000], [50]])
        result = self.price_history._fix_prices_sudden_change(df, interval="1d", tz_exchange="America/New_York", change=10)
        self.assertTrue((result["Price"] < 1000).all(), "Sudden price spikes should be smoothed.")

    # Test for _reconstruct_intervals_batch
    def test_reconstruct_intervals_batch(self):
        df = self.create_mock_dataframe(["Volume"], [[10], [20], [30]])
        result = self.price_history._reconstruct_intervals_batch(df, interval="1d", prepost=False)
        self.assertIsInstance(result, pd.DataFrame, "The result should be a DataFrame.")

    # Test for _fix_zeroes
    def test_fix_zeroes(self):
        df = self.create_mock_dataframe(["close"], [[0], [100], [0], [200]])
        result = self.price_history._fix_zeroes(df, interval="1d", tz_exchange="America/New_York", prepost=False)
        self.assertTrue((result["close"] != 0).all(), "Zeros should be fixed in the 'close' column.")

    # Test for get_dividends
    def test_get_dividends(self):
        with patch.object(self.price_history, 'history', return_value=pd.DataFrame({"Dividends": [0.1, 0.2]})):
            result = self.price_history.get_dividends()
            self.assertIsInstance(result, pd.DataFrame, "Dividends should be returned as a DataFrame.")

    # Test for get_splits
    def test_get_splits(self):
        with patch.object(self.price_history, 'history', return_value=pd.DataFrame({"Stock Splits": [1, 2]})):
            result = self.price_history.get_splits()
            self.assertIsInstance(result, pd.DataFrame, "Splits should be returned as a DataFrame.")

    # Test for _standardise_currency
    def test_standardise_currency(self):
        df = self.create_mock_dataframe(["Currency"], [["USD"], ["EUR"], ["JPY"]])
        result = self.price_history._standardise_currency(df, currency="USD")
        self.assertTrue("Currency" in result.columns, "Currency should remain a valid column.")

    # Test for _fix_unit_mixups
    def test_fix_unit_mixups(self):
        df = self.create_mock_dataframe(["Unit"], [["kg"], ["lbs"], ["kg"]])
        result = self.price_history._fix_unit_mixups(df, interval="1d", tz_exchange="America/New_York", prepost=False)
        self.assertEqual(len(result), len(df), "Unit mixups should not change the DataFrame size.")

    # Test for _fix_unit_random_mixups
    def test_fix_unit_random_mixups(self):
        df = self.create_mock_dataframe(["Unit"], [["kg"], ["lbs"], ["random"]])
        result = self.price_history._fix_unit_random_mixups(df, interval="1d", tz_exchange="America/New_York", prepost=False)
        self.assertNotIn("random", result["Unit"].values, "Random units should be corrected.")

    # Test for _fix_unit_switch
    def test_fix_unit_switch(self):
        df = self.create_mock_dataframe(["Unit"], [["kg"], ["lbs"]])
        result = self.price_history._fix_unit_switch(df, interval="1d", tz_exchange="America/New_York")
        self.assertTrue("Unit" in result.columns, "Unit conversion should adjust the DataFrame.")

    # Test for get_actions
    def test_get_actions(self):
        with patch.object(self.price_history, 'history', return_value=pd.DataFrame({"action": ["split", "dividend"]})):
            result = self.price_history.get_actions()
            self.assertIsInstance(result, pd.DataFrame, "Actions should be returned as a DataFrame.")

    # Test for get_history_metadata
    def test_get_history_metadata(self):
        with patch.object(self.price_history, 'history', return_value={"meta": "data"}):
            result = self.price_history.get_history_metadata()
            self.assertIsInstance(result, dict, "Metadata should be returned as a dictionary.")


if __name__ == '__main__':
    unittest.main()
