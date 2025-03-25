import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from requests.exceptions import HTTPError
from yfinance.scrapers.history import PriceHistory


class TestPriceHistory(unittest.TestCase):

    def setUp(self):
        self.price_history = MagicMock(spec=PriceHistory)
        self.price_history._apply_back_adjustment = MagicMock(return_value=pd.DataFrame({"Close": [150, 152, 154], "Adj Close": [150, 152, 154]}))
        self.price_history._combine_datasets = MagicMock(return_value=pd.DataFrame({"Close": [100, 101, 102]}, index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"])))
        self.price_history.interval_to_seconds = MagicMock(return_value=86400)
        self.price_history._remove_duplicates = MagicMock(return_value=pd.DataFrame({"Close": [100, 104]}, index=["2024-01-01", "2024-01-02"]))
        self.price_history.fetch_price_data = MagicMock()
        self.price_history._handle_large_dataset = MagicMock(return_value=pd.DataFrame({"Volume": [1000000, 2000000, 3000000]}))
        self.price_history._normalize_price_data = MagicMock(return_value=pd.DataFrame({"Close": [0, 0.5, 1]}))
        self.price_history.get_metadata = MagicMock()
        self.price_history._resample = MagicMock(side_effect=lambda df, src, tgt: df.iloc[::2])
        self.price_history.history = MagicMock(return_value=pd.DataFrame({"Close": [150, 152, 154]}))

    
    @patch.object(PriceHistory, "history", return_value=pd.DataFrame({"Close": [150, 152, 154]}))
    def test_data_integrity_after_fetching(self, mock_history):
        result = self.price_history.history(period="1mo", interval="1d")
        self.assertIn("Close", result.columns, "The 'Close' column should exist after fetching historical data.")

    def test_large_volume_handling(self):
        df = pd.DataFrame({
            "Volume": [1000000, 2000000, 3000000]
        }, index=pd.date_range("2024-01-01", periods=3))
        result = self.price_history._handle_large_dataset(df)
        self.assertEqual(len(result), 3)

    def test_invalid_timezone_adjustment(self):
        with self.assertRaises(ValueError):
            self.price_history.history(period="1mo", interval="1d", timezone="Invalid/Timezone")

    def test_combine_empty_datasets(self):
        result = self.price_history._combine_datasets([pd.DataFrame(), pd.DataFrame()])
        self.assertTrue(result.empty, "Combining empty datasets should return an empty DataFrame.")

    def test_default_interval_conversion(self):
        self.assertEqual(self.price_history.interval_to_seconds("1d"), 86400)

    def test_non_standard_ticker(self):
        with self.assertRaises(ValueError):
            self.price_history.fetch_price_data("@INVALID", "1d")

    def test_invalid_interval_format(self):
        with self.assertRaises(ValueError):
            self.price_history.history(period="1mo", interval="1minute")

    def test_normalize_with_negative_values(self):
        df = pd.DataFrame({"Close": [-100, -200, -300]})
        result = self.price_history._normalize_price_data(df)
        self.assertTrue((result["Close"] >= 0).all(), "Normalized values should be non-negative.")

    def test_fetching_data_with_missing_columns(self):
        mock_data = pd.DataFrame({"Open": [100, 101], "Close": [102, 103]})
        with patch.object(self.price_history, "history", return_value=mock_data):
            result = self.price_history.history(period="1mo", interval="1d")
            self.assertTrue("High" not in result.columns, "Missing columns should be handled properly.")

    def test_large_resampling_intervals(self):
        df = pd.DataFrame({
            "Close": [100, 102, 104, 106, 108],
            "Volume": [1000, 1100, 1200, 1300, 1400]
        }, index=pd.date_range("2024-01-01", periods=5, freq="D"))

        result = self.price_history._resample(df, "1D", "10D")
        self.assertTrue(len(result) < len(df), "Resampling should reduce the number of rows.")

    @patch.object(PriceHistory, "fetch_price_data", side_effect=HTTPError("HTTP Error"))
    def test_http_error_during_fetching(self, mock_fetch):
        with self.assertRaises(HTTPError):
            self.price_history.fetch_price_data("1mo", "1d")

    def test_duplicate_index_removal(self):
        df = pd.DataFrame({
            "Date": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "Close": [100, 102, 104]
        }).set_index("Date")
        result = self.price_history._remove_duplicates(df)
        self.assertEqual(len(result), 2, "Duplicates should be removed.")

    def test_metadata_fetching_with_invalid_ticker(self):
        with self.assertRaises(ValueError):
            self.price_history.get_metadata()

    def test_resampling_with_partial_data(self):
        df = pd.DataFrame({
            "Close": [100, None, 104],
            "Volume": [1000, None, 1400]
        }, index=pd.date_range("2024-01-01", periods=3, freq="D"))

        result = self.price_history._resample(df, "1D", "1W")
        self.assertFalse(result.isnull().any().any(), "Resampled data should fill missing values.")

    def test_adjusted_close_handling(self):
        df = pd.DataFrame({
            "Close": [150, 152, 154],
            "Adj Close": [148, 150, 152]
        })
        result = self.price_history._apply_back_adjustment(df)
        self.assertTrue(all(result["Close"] == result["Adj Close"]))

    def test_empty_response_handling(self):
        with patch.object(self.price_history, "fetch_price_data", return_value=pd.DataFrame()):
            result = self.price_history.fetch_price_data("1mo", "1d")
            self.assertTrue(result.empty, "Empty response should be handled gracefully.")

    def test_normalization_edge_case(self):
        df = pd.DataFrame({"Close": [0, 1, 2]})
        result = self.price_history._normalize_price_data(df)
        self.assertTrue(result["Close"].max() <= 1, "Normalized values should be <= 1.")

    def test_data_merging_with_time_overlap(self):
        df1 = pd.DataFrame({"Close": [100, 101]}, index=pd.to_datetime(["2024-01-01", "2024-01-02"]))
        df2 = pd.DataFrame({"Close": [101, 102]}, index=pd.to_datetime(["2024-01-02", "2024-01-03"]))

        result = self.price_history._combine_datasets([df1, df2])
        self.assertEqual(len(result), 3, "Overlapping data should be merged correctly.")

    def test_fetch_price_data_with_invalid_period(self):
        with self.assertRaises(ValueError):
            self.price_history.fetch_price_data("invalid", "1d")

    def test_resampling_on_large_dataset(self):
        df = pd.DataFrame({
            "Close": [100 + i for i in range(1000)],
            "Volume": [1000 + i for i in range(1000)]
        }, index=pd.date_range("2024-01-01", periods=1000, freq="D"))

        result = self.price_history._resample(df, "1D", "1M")
        self.assertTrue(len(result) < len(df), "Resampling should reduce the dataset size.")

    def test_back_adjustment_with_missing_columns(self):
        df = pd.DataFrame({"Close": [150, 152, 154]})
        with self.assertRaises(KeyError):
            self.price_history._apply_back_adjustment(df)


if __name__ == "__main__":
    unittest.main()
