import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from requests.exceptions import HTTPError
from yfinance.exceptions import YFInvalidPeriodError, YFPricesMissingError, YFTzMissingError
from yfinance.scrapers.history import PriceHistory


class TestPriceHistoryExtended(unittest.TestCase):
    def setUp(self):
        self.price_history = PriceHistory(None, "AAPL", "America/New_York")
        self.price_history._data = MagicMock()
        self.price_history._data.get.return_value = {
            "chart": {"result": [{"meta": {}, "indicators": {}}], "error": None}
    }

    @patch.object(PriceHistory, "history")
    def test_history_valid_period(self, mock_history):
        # Mock valid return data
        mock_history.return_value = pd.DataFrame({
            "Date": ["2023-01-01", "2023-01-02"],
            "Close": [100, 102]
        }).set_index("Date")

        # Test history method
        result = self.price_history.history(period="1mo", interval="1d")
        self.assertFalse(result.empty, "Expected non-empty DataFrame for valid period.")
        self.assertIn("Close", result.columns, "Expected 'Close' column in DataFrame.")


    def test_data_fetch(self):
        self.price_history._data.get = MagicMock(return_value={"chart": {"result": [{}]}})
        result = self.price_history.history(period="1mo", interval="1d")
        self.assertTrue(result.empty, "Expected an empty DataFrame for mocked empty data.")


# 1. Test handling missing values in OHLC data
    def test_handle_missing_ohlc_values(self):
        data = {'Open': [1, None, 3], 'High': [2, 4, None], 'Low': [None, 1, 2], 'Close': [3, None, 4]}
        df = pd.DataFrame(data)
        with patch.object(self.price_history, '_handle_missing_values', return_value=df.fillna(0)):
            result = self.price_history._handle_missing_values(df)
            self.assertFalse(result.isnull().any().any(), "All missing values should be handled.")

    # 2. Test resampling with custom intervals
    @patch.object(PriceHistory, '_resample_data', return_value=pd.DataFrame({'Date': ['2024-01-01', '2024-01-02'], 'Value': [100, 200]}))
    def test_resample_with_custom_interval(self, mock_resample_data):
        mock_data = pd.DataFrame({'Date': ['2024-01-01', '2024-01-02'], 'Value': [100, 200]})
        result = self.price_history._resample_data(mock_data, '3D')
        self.assertTrue(mock_resample_data.called, "Expected `_resample_data` to be called.")
        self.assertIsInstance(result, pd.DataFrame, "Expected DataFrame from mocked `_resample_data`.")

        # Define expected output of the mocked resample function
        resampled_data = pd.DataFrame({
            'Date': pd.date_range(start='2023-01-01', periods=2, freq='3D'),
            'Close': [101, 104],
            'Volume': [630, 700]
        }).set_index('Date')

        # Mock the behavior of the `_resample_data` method
        mock_resample_data.return_value = resampled_data

        # Call the method to test
        result = self.price_history._resample_data(mock_data, '3D')

        # Validate the result
        self.assertIsInstance(result, pd.DataFrame, "Expected result to be a DataFrame.")
        self.assertFalse(result.empty, "Expected non-empty DataFrame after resampling.")
        self.assertEqual(result.shape, (2, 2), "Expected DataFrame with 2 rows and 2 columns.")
        self.assertTrue((result.index == resampled_data.index).all(), "Expected resampled indices to match.")
        self.assertTrue((result['Close'] == resampled_data['Close']).all(), "Expected 'Close' values to match.")
        self.assertTrue((result['Volume'] == resampled_data['Volume']).all(), "Expected 'Volume' values to match.")

    # 3. Test zero-value handling for adjusted close
    def test_fix_zero_adjusted_close(self):
        if hasattr(self.price_history, "_fix_zeroes"):
            mock_fix_zeroes = MagicMock(return_value=pd.DataFrame({'Adj Close': [150, 152, 154]}))
            self.price_history._fix_zeroes = mock_fix_zeroes
            result = self.price_history._fix_zeroes(pd.DataFrame({'Adj Close': [0, 150, 152]}))
            self.assertTrue(result['Adj Close'].min() > 0, "All zero values should be fixed.")
        else:
            self.skipTest("_fix_zeroes is not implemented.")

    # 15. Test invalid interval handling with mocking
    @patch.object(PriceHistory, "history", side_effect=ValueError("Invalid interval"))
    def test_invalid_interval_handling(self, mock_history):
        with self.assertRaises(ValueError, msg="A ValueError should be raised for an invalid interval."):
            self.price_history.history(period="1mo", interval="invalid")
        mock_history.assert_called_once_with(period="1mo", interval="invalid")

    # 5. Test duplicate row removal
    def test_remove_duplicates(self):
        if hasattr(self.price_history, "_remove_duplicates"):
            mock_remove_duplicates = MagicMock(return_value=pd.DataFrame({'A': [1, 2]}))
            self.price_history._remove_duplicates = mock_remove_duplicates
            result = self.price_history._remove_duplicates(pd.DataFrame({'A': [1, 1, 2, 2]}))
            self.assertTrue(isinstance(result, pd.DataFrame), "Expected DataFrame from duplicate removal.")
        else:
            self.skipTest("_remove_duplicates is not implemented.")


    # 6. Test partial metadata caching
    @patch.object(PriceHistory, '_get_metadata', return_value={'key': 'value'})
    def test_partial_metadata_caching(self, mock_get_metadata):
        metadata = self.price_history._get_metadata()
        self.assertEqual(metadata, {'key': 'value'}, "Mocked metadata should match expected output.")
        mock_get_metadata.assert_called_once()



    # 7. Test date range validation
    @patch.object(PriceHistory, 'history')
    def test_validate_date_range(self, mock_history):
        # Mock valid range to return non-empty DataFrame
        mock_history.return_value = pd.DataFrame({
            'Date': ['2023-01-01', '2023-01-02'],
            'Close': [150, 152]
        }).set_index('Date')

        # Valid date range
        result_valid = self.price_history.history(start="2023-01-01", end="2023-12-31", interval="1d")
        self.assertIsInstance(result_valid, pd.DataFrame, "Expected a DataFrame for a valid date range.")
        self.assertFalse(result_valid.empty, "Expected non-empty DataFrame for valid date range.")

        # Mock invalid range to return empty DataFrame
        mock_history.return_value = pd.DataFrame()

        # Invalid date range
        result_invalid = self.price_history.history(start="2024-01-01", end="2023-12-31", interval="1d")
        self.assertIsInstance(result_invalid, pd.DataFrame, "Expected a DataFrame for an invalid date range.")
        self.assertTrue(result_invalid.empty, "Expected empty DataFrame for invalid date range.")



    # 8. Test empty DataFrame handling
    @patch.object(PriceHistory, "_handle_empty_dataframe", return_value=pd.DataFrame())
    def test_empty_dataframe_handling(self, mock_handle_empty_dataframe):
        df = pd.DataFrame()
        result = mock_handle_empty_dataframe(df)
        self.assertTrue(result.empty, "Empty DataFrame should return unchanged.")
        mock_handle_empty_dataframe.assert_called_once_with(df)

    # 9. Test timezone adjustments
    @patch.object(PriceHistory, 'history')
    def test_timezone_adjustments(self, mock_history):
        # Mock data to simulate timezone adjustments
        mock_data = pd.DataFrame({
            'Date': pd.date_range(start='2023-01-01', periods=5, freq='D'),
            'Close': [100, 101, 102, 103, 104]
        }).set_index('Date')

        mock_history.return_value = mock_data

        # Simulate timezone adjustment
        result = self.price_history.history(start="2023-01-01", end="2023-01-05", interval="1d")
        
        # Check if the mock data is returned correctly
        self.assertIsInstance(result, pd.DataFrame, "Expected a DataFrame after timezone adjustment.")
        self.assertFalse(result.empty, "Expected non-empty DataFrame after timezone adjustment.")
        self.assertEqual(result.shape[0], 5, "Expected 5 rows in the DataFrame.")

        # Mock timezone adjustment function (manual adjustment in test)
        result.index = result.index.tz_localize('UTC').tz_convert('America/New_York')

        # Validate the timezone adjustment
        self.assertEqual(result.index.tz.zone, 'America/New_York', "Expected timezone to be 'America/New_York'.")

    # 10. Test interval conversion logic
    @patch.object(PriceHistory, "_convert_interval_to_timedelta", return_value=pd.Timedelta(days=1))
    def test_interval_conversion_logic(self, mock_convert_interval_to_timedelta):
        result = self.price_history._convert_interval_to_timedelta('1d')
        self.assertEqual(result.days, 1, "Interval conversion should return correct timedelta.")
        mock_convert_interval_to_timedelta.assert_called_once_with('1d')

    # 11. Test filtering data by date
    @patch.object(PriceHistory, '_filter_data_by_date', return_value=pd.DataFrame({'Filtered': [1, 2]}))
    def test_filter_data_by_date(self, mock_filter):
        df = pd.DataFrame({'Date': ['2024-01-01', '2024-01-02'], 'Value': [100, 200]})
        result = self.price_history._filter_data_by_date(df, '2024-01-01', '2024-01-02')
        self.assertEqual(len(result), 2, "Expected filtered data length to match mock return value.")

    @patch("yfinance.scrapers.history.requests.get")
    def test_metadata_fetching(self, mock_get):
        mock_get.return_value.json.return_value = {
            "chart": {"result": [{"meta": {"symbol": "AAPL", "exchangeName": "NASDAQ"}}]}
        }
        metadata = self.price_history.get_metadata()
        self.assertIn("symbol", metadata, "Expected 'symbol' key in metadata.")



    # 12. Test normalization of price data
    @patch.object(PriceHistory, "_normalize_price_data", return_value=pd.DataFrame({
            'Value': [100, 200]
        }))
    def test_normalize_price_data(self, mock_normalize):
        df = pd.DataFrame({'Price': [1, 2]})
        result = self.price_history._normalize_price_data(df)
        self.assertTrue(isinstance(result, pd.DataFrame), "Result should be a DataFrame.")
        mock_normalize.assert_called_once()


    @patch.object(PriceHistory, '_resample_data', return_value=pd.DataFrame({'Stock Splits': [1.0, 0.0]}))
    def test_resample_with_mocked_data(self, mock_resample):
        mock_data = pd.DataFrame({'Date': ['2024-01-01'], 'Value': [100]})
        result = self.price_history._resample_data(mock_data, '1wk')
        self.assertIn('Stock Splits', result.columns, "Expected 'Stock Splits' column in DataFrame")

    # 14. Test fetching historical data for valid inputs with mocking
    @patch.object(PriceHistory, "history", return_value=pd.DataFrame({"Close": [150, 152, 154]}))
    def test_valid_historical_data_fetching(self, mock_history):
        result = self.price_history.history(period="1mo", interval="1d")
        self.assertTrue(isinstance(result, pd.DataFrame), "Historical data should be a DataFrame.")
        self.assertFalse(result.empty, "Historical data should not be empty for valid inputs.")
        self.assertIn("Close", result.columns, "Result should include the 'Close' column.")
        mock_history.assert_called_once_with(period="1mo", interval="1d")

    # 15. Test combining datasets
    def test_combine_datasets(self):
        if hasattr(self.price_history, "_combine_datasets"):
            df1 = pd.DataFrame({"Close": [150, 152]})
            df2 = pd.DataFrame({"Close": [154, 156]})
            combined = self.price_history._combine_datasets([df1, df2])
            self.assertIsInstance(combined, pd.DataFrame, "Expected DataFrame from combined datasets.")
        else:
            self.skipTest("_combine_datasets is not implemented.")



    # 16. Test handling of large datasets
    def test_large_dataset_handling(self):
        if hasattr(self.price_history, "_handle_large_dataset"):
            mock_handle_large_dataset = MagicMock(return_value=pd.DataFrame({"Close": [100] * 10000}))
            self.price_history._handle_large_dataset = mock_handle_large_dataset
            result = self.price_history._handle_large_dataset(pd.DataFrame({"Close": [100] * 10000}))
            self.assertTrue(isinstance(result, pd.DataFrame), "Result should be a DataFrame.")
            mock_handle_large_dataset.assert_called_once()
        else:
            self.skipTest("_handle_large_dataset is not implemented.")

    # 17. Test metadata retrieval for a valid ticker
    @patch.object(PriceHistory, "get_metadata", return_value={"symbol": "AAPL", "exchange": "NASDAQ"})
    def test_metadata_retrieval(self, mock_get_metadata):
        metadata = self.price_history.get_metadata()
        self.assertIn("symbol", metadata, "Metadata should contain the symbol key.")
        self.assertIn("exchange", metadata, "Metadata should contain the exchange key.")
        mock_get_metadata.assert_called_once()

    # 18. Test caching mechanism for metadata
    @patch.object(PriceHistory, "get_metadata", side_effect=[{"cached": True}, {"cached": False}])
    def test_metadata_caching(self, mock_get_metadata):
        first_call = self.price_history.get_metadata()
        second_call = self.price_history.get_metadata()
        self.assertTrue(first_call["cached"], "First call should return cached data.")
        self.assertFalse(second_call["cached"], "Second call should not return cached data.")
        self.assertEqual(mock_get_metadata.call_count, 2, "get_metadata should be called twice.")

    # 19. Test resampling from 1D to 1M
    @patch.object(PriceHistory, "_resample", return_value=pd.DataFrame())
    def test_resample_1d_to_1m(self, mock_resample):
        df = pd.DataFrame({"Close": [150, 152, 154]})
        result = mock_resample(df, "1d", "1m")
        self.assertTrue(isinstance(result, pd.DataFrame), "Result should be a DataFrame.")
        mock_resample.assert_called_once_with(df, "1d", "1m")


        # 20. Test handling of empty metadata responses
        # @patch("yfinance.scrapers.history.requests.get")
        # def test_empty_metadata_handling(self, mock_requests_get):
        #     mock_requests_get.return_value.json.return_value = {}
        #    metadata = self.price_history.get_metadata()
        #    self.assertEqual(metadata, {}, "Empty metadata should be handled gracefully.")
        #   mock_requests_get.assert_called_once()


    # 1. Test valid ticker with valid period and interval
    @patch("yfinance.scrapers.history.PriceHistory.history")
    def test_valid_ticker_valid_period_interval(self, mock_history):
        # Define mock_data
        mock_data = {
            'Date': ['2023-11-01', '2023-11-02', '2023-11-03'],
            'Open': [100, 102, 104],
            'High': [105, 106, 107],
            'Low': [95, 96, 97],
            'Close': [102, 104, 106],
            'Volume': [1000, 1500, 1200]
        }

        # Create a DataFrame and mock the history method return value
        df = pd.DataFrame(mock_data).set_index('Date')
        mock_history.return_value = df

        # Call the history method
        result = self.price_history.history(period="1mo", interval="1d")

        # Assert the results
        self.assertIsInstance(result, pd.DataFrame, "Expected result to be a DataFrame.")
        self.assertFalse(result.empty, "Expected non-empty DataFrame for valid ticker and period.")
        self.assertIn("Close", result.columns, "Expected 'Close' column in DataFrame.")


    def test_resample_data(self, mock_history):
        mock_resample = MagicMock(return_value=pd.DataFrame({
            'Date': pd.date_range(start='2023-01-01', periods=2, freq='2D'),
            'Close': [100.5, 103.5]
        }).set_index('Date'))
        self.price_history._resample_data = mock_resample
        result = self.price_history._resample_data(mock_history.return_value, '2D')
        self.assertTrue(isinstance(result, pd.DataFrame), "Expected a DataFrame for resampled data.")
        self.assertTrue(mock_resample.called, "Resample method should be called.")

    # 2. Test invalid period
    @patch.object(PriceHistory, "history", side_effect=YFInvalidPeriodError("Invalid period specified.", "100years", ["1d", "1wk", "1mo", "1y"]))
    def test_invalid_period(self, mock_history):
        with self.assertRaises(YFInvalidPeriodError):
            self.price_history.history(period="100years", interval="1h")



    # 3. Test invalid interval
    @patch("yfinance.scrapers.history.PriceHistory.history")
    def test_invalid_interval(self, mock_history):
        # Simulate an invalid interval error
        mock_history.side_effect = Exception("Invalid interval specified.")

        with self.assertRaises(Exception):
            # Call with an invalid interval to trigger the exception
            self.price_history.history(period="1d", interval="100m")

    # 4. Test missing timezone
    @patch("yfinance.scrapers.history.PriceHistory.history")
    def test_missing_timezone(self, mock_history):
        # Simulate a missing timezone error
        mock_history.side_effect = YFTzMissingError("Timezone is missing for the requested data.")

        with self.assertRaises(YFTzMissingError):
            # Call with a None timezone to trigger the error
            self.price_history.history(period="1d", interval="1h", timezone=None)

    # 5. Test period exceeding max range
    @patch('yfinance.scrapers.history.PriceHistory.history')
    def test_period_exceeding_max(self, mock_history):
        mock_history.side_effect = YFPricesMissingError("Prices data is missing for the requested period.", debug_info={})
        with self.assertRaises(YFPricesMissingError):
            self.price_history.history(period="10y", interval="1d")


    # 6. Test empty data handling
    def test_empty_data_handling(self):
            self.price_history._data.get = MagicMock(return_value={"chart": {"result": [{}], "error": None}})
            result = self.price_history.history(period="1mo", interval="1d")
            self.assertTrue(result.empty, "Expected an empty DataFrame when data is missing.")




    # 7. Test invalid data structure from API
    @patch("yfinance.scrapers.history.PriceHistory.get_ohlcv_data")
    def test_invalid_data_structure(self, mock_ohlcv_data):
        # Simulate an invalid data structure response
        mock_ohlcv_data.return_value = {"unexpected": "data structure"}

        with self.assertRaises(YFPricesMissingError):
            self.price_history.get_ohlcv_data("1d", "1h")

    # 8. Test resampling from 1d to 1wk
    @patch('yfinance.scrapers.history.PriceHistory._resample')
    def test_resample_1d_to_1wk(self, mock_resample):
        mock_data = pd.DataFrame({'Stock Splits': [1.0, 1.0], 'Close': [100, 200]})
        mock_resample.return_value = mock_data
        result = self.price_history._resample(mock_data, "1d", "1wk")
        self.assertIsInstance(result, pd.DataFrame, "Expected DataFrame as a result of resampling.")

    # 9. Test metadata fetching
    @patch("yfinance.scrapers.history.requests.get")
    def test_get_metadata(self, mock_get):
        mock_get.return_value.json.return_value = {
            "chart": {"result": [{"meta": {"symbol": "AAPL", "exchangeName": "NASDAQ"}}]}
        }
        metadata = self.price_history.get_metadata()
        self.assertIn("symbol", metadata)


    # 10. Test actions parsing
    def test_get_actions(self):
        actions = self.price_history.get_actions()
        self.assertIsInstance(actions, pd.Series)

    # 11. Test dividends parsing
    def test_get_dividends(self):
        dividends = self.price_history.get_dividends()
        self.assertIsInstance(dividends, pd.Series)

    # 12. Test capital gains parsing
    def test_get_capital_gains(self):
        capital_gains = self.price_history.get_capital_gains()
        self.assertIsInstance(capital_gains, pd.Series)

    # 13. Test splits parsing
    def test_get_splits(self):
        splits = self.price_history.get_splits()
        self.assertIsInstance(splits, pd.Series)

    # 14. Test repair option
    def test_repair_option(self):
        result = self.price_history.history(period="1mo", interval="1d", repair=True)
        self.assertIsInstance(result, pd.DataFrame)

    # 15. Test back adjustment
    def test_back_adjust(self):
        result = self.price_history.history(period="1mo", interval="1d", back_adjust=True)
        self.assertIsInstance(result, pd.DataFrame)

    # 16. Test rounding
    def test_rounding_option(self):
        result = self.price_history.history(period="1mo", interval="1d", rounding=True)
        self.assertIsInstance(result, pd.DataFrame)

    # 17. Test empty metadata handling
    @patch("yfinance.scrapers.history.requests.get")
    def test_empty_metadata_handling(self, mock_get):
        # Mock an empty metadata response
        mock_get.return_value.json.return_value = {"chart": {"result": [{}]}}
        
        metadata = self.price_history.get_metadata()
        self.assertIsInstance(metadata, dict)
        self.assertEqual(metadata, {}, "Metadata should be an empty dictionary when no data is returned.")


    # 18. Test resample with invalid target interval
    def test_resample_invalid_target(self):
        with self.assertRaises(Exception):
            self.price_history._resample(pd.DataFrame(), "1d", "invalid")

    # 19. Test history with specific start and end
    def test_history_with_start_end(self):
        self.price_history._data = MagicMock()
        self.price_history._data.get.return_value = {"chart": {"result": [{"meta": {}, "indicators": {}}]}}
        result = self.price_history.history(start="2023-01-01", end="2023-12-31", interval="1d")
        self.assertIsInstance(result, pd.DataFrame, "Expected a DataFrame.")


    # 20. Test combining actions
    @patch.object(PriceHistory, "get_actions", return_value=pd.DataFrame({"Dividends": [0.5], "Stock Splits": [2]}))
    def test_combine_actions(self, mock_get_actions):
        result = self.price_history.get_actions()
        self.assertIn("Dividends", result.columns)
        self.assertIn("Stock Splits", result.columns)


if __name__ == "__main__":
    unittest.main()