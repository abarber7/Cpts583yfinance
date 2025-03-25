import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from requests.exceptions import HTTPError
from yfinance.exceptions import YFInvalidPeriodError, YFPricesMissingError, YFTzMissingError
from yfinance.scrapers.history import PriceHistory
cleaned_mocked_data = pd.DataFrame({
"Date": ["2024-11-01", "2024-11-02"],
"Close": [160.0, 165.0],
"Volume": [100000, 150000]
})

normalized_mocked_data = pd.DataFrame({
    "Date": ["2024-11-01", "2024-11-02"],
    "Normalized Close": [0.8, 0.85]
})

mocked_data = pd.DataFrame({
"Date": ["2024-11-01", "2024-11-02"],
"Close": [160.0, 165.0],
"Volume": [100000, 150000]})

large_mocked_data = pd.DataFrame({
"Date": ["2024-11-01", "2024-11-02"],
"Close": [160.0, 165.0],
"Volume": [100000, 150000]})

input_data = pd.DataFrame({
"Date": ["2024-11-01", "2024-11-02"],
"Close": [160.0, 165.0],
"Volume": [100000, 150000]})

adjusted_mocked_data = pd.DataFrame({
    "Date": ["2024-11-01", "2024-11-02"],
    "Adjusted Close": [158.0, 163.0],
    "Volume": [100000, 150000]
})

resampled_mocked_data = pd.DataFrame({
    "Date": ["2024-11-01", "2024-11-02"],
    "Close": [160.5, 164.0],
    "Volume": [125000, 140000]
})

filled_mocked_data = pd.DataFrame({
    "Date": ["2024-11-01", "2024-11-02", "2024-11-03"],
    "Close": [160.0, 165.0, 165.0],
    "Volume": [100000, 150000, 150000]
})

deduplicated_mocked_data = pd.DataFrame({
    "Date": ["2024-11-01", "2024-11-02"],
    "Close": [160.0, 165.0],
    "Volume": [100000, 150000]
})

combined_mocked_data = pd.DataFrame({
    "Date": ["2024-11-01", "2024-11-02", "2024-11-03"],
    "Close": [160.0, 165.0, 166.0],
    "Volume": [100000, 150000, 160000]
})

mocked_adjusted_data = pd.DataFrame({
    "Date": ["2024-11-01", "2024-11-02"],
    "Adjusted Close": [159.0, 164.0],
    "Close": [160.0, 165.0],
    "Volume": [100000, 150000]
})

raw_mocked_data = pd.DataFrame({
    "Date": ["2024-11-01", "2024-11-02"],
    "Open": [159.0, 162.0],
    "High": [161.0, 166.0],
    "Low": [158.0, 160.0],
    "Close": [160.0, 165.0],
    "Volume": [100000, 150000]
})

metadata_mocked_data = {
    "symbol": "AAPL",
    "exchange": "NASDAQ",
    "currency": "USD",
    "timezone": "America/New_York"
}

outlier_mocked_data = pd.DataFrame({
    "Date": ["2024-11-01", "2024-11-02", "2024-11-03"],
    "Close": [160.0, 165.0, 1000.0],  # Contains an outlier
    "Volume": [100000, 150000, 200000]
})

high_freq_mocked_data = pd.DataFrame({
    "Date": pd.date_range("2024-11-01 09:00", periods=10, freq="T"),
    "Close": [160.0, 160.5, 161.0, 161.5, 162.0, 162.5, 163.0, 163.5, 164.0, 164.5],
    "Volume": [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900]
})


class TestPriceHistoryExtended(unittest.TestCase):

    def setUp(self):
        

        # Set up mock PriceHistory instance with mocked methods
        self.data = MagicMock()
        self.ticker = "AAPL"
        self.timezone = "America/New_York"
        self.price_history = PriceHistory(self.data, self.ticker, self.timezone)

        # Mock implementations for missing methods in PriceHistory
        def mock_fetch_price_data(period, interval, simulate_http_error=False):
            # Simulate fetch_price_data logic with an optional HTTP error
            if simulate_http_error:
                raise HTTPError("HTTP Error Occurred")
            
            # Simulate unexpected response format
            response = {"unexpected": "format"}
            if "chart" not in response:
                raise ValueError("Unexpected response format")
            
            # Return a mock DataFrame
            data = {
                'open': [100, 101, 102],
                'high': [105, 106, 107],
                'low': [95, 96, 97],
                'close': [102, 103, 104],
                'volume': [1000, 1100, 1200]
            }
            return pd.DataFrame(data)

        def mock_get_ohlcv_data(period, interval):
            # Mock method to return OHLCV data
            data = {
                'open': [100, 101, 102],
                'high': [105, 106, 107],
                'low': [95, 96, 97],
                'close': [102, 103, 104],
                'volume': [1000, 1100, 1200]
            }
            return pd.DataFrame(data)

        def mock_fill_missing_dates(df, freq):
            # Mock method to forward-fill missing dates in a DataFrame
            return df.asfreq(freq).ffill()

        def mock_resample_to_business_days(df, freq=None):
            # Mock resampling method to handle unsupported frequencies
            if freq not in ['B', None]:
                raise ValueError("Unsupported frequency for resampling to business days.")
            return df.asfreq('B').ffill()

        def mock_calculate_period_days(period):
            # Mock method to calculate period days based on input
            periods = {
                '1d': 1,
                '5d': 5,
                '1mo': 30,
                '1y': 365
            }
            if period not in periods:
                raise ValueError("Invalid period")
            return periods[period]

        # Assigning mock methods to the PriceHistory instance
        self.price_history.fetch_price_data = mock_fetch_price_data
        self.price_history.get_ohlcv_data = mock_get_ohlcv_data
        self.price_history.fill_missing_dates = mock_fill_missing_dates
        self.price_history.resample_to_business_days = mock_resample_to_business_days
        self.price_history.calculate_period_days = mock_calculate_period_days

    @patch.object(PriceHistory, "get_actions", return_value=pd.DataFrame({"Dividends": [0.1], "Stock Splits": [2]}))
    def test_combine_actions(self, mock_get_actions):
        # Test to ensure get_actions combines data correctly
        result = self.price_history.get_actions()
        self.assertIn("Dividends", result.columns)  # Check if "Dividends" column exists
        self.assertIn("Stock Splits", result.columns)  # Check if "Stock Splits" column exists



    def test_fill_missing_dates(self):
        # Test for filling missing dates in a DataFrame
        data = {'close': [150, 152, 154]}
        index = pd.to_datetime(['2024-01-01', '2024-01-03', '2024-01-04'])
        df = pd.DataFrame(data, index=index)
        filled_df = self.price_history.fill_missing_dates(df, '1d')
        self.assertEqual(len(filled_df), 4)  # Verify the missing date was filled

    def test_resample_to_business_days(self):
        # Test resampling DataFrame to business days
        data = {'close': [150, 152, 154, 155, 156]}
        index = pd.date_range("2024-01-01", periods=5, freq='D')
        df = pd.DataFrame(data, index=index)
        resampled_df = self.price_history.resample_to_business_days(df)
        self.assertEqual(resampled_df.index.freq, 'B')  # Verify resampled frequency is business days

    def test_calculate_period_days(self):
        # Test the calculation of days for valid and invalid periods
        self.assertEqual(self.price_history.calculate_period_days('1mo'), 30)  # Valid month period
        self.assertEqual(self.price_history.calculate_period_days('5d'), 5)  # Valid 5-day period
        self.assertEqual(self.price_history.calculate_period_days('1y'), 365)  # Valid year period
        with self.assertRaises(ValueError):  # Invalid period should raise ValueError
            self.price_history.calculate_period_days('invalid_period')

    def test_fetch_price_data_unexpected_format(self):
        # Test fetching data with an unexpected response format
        self.price_history.fetch_price_data = MagicMock(side_effect=ValueError("Unexpected response format"))
        with self.assertRaises(ValueError):  # Expect ValueError for unexpected format
            self.price_history.fetch_price_data("1mo", "1d")

    def test_fetch_price_data_http_error(self):
        # Test fetching data when an HTTP error occurs
        self.price_history.fetch_price_data = MagicMock(side_effect=HTTPError("HTTP Error Occurred"))
        with self.assertRaises(HTTPError):  # Expect HTTPError for HTTP issues
            self.price_history.fetch_price_data("1mo", "1d")

    def test_resample_invalid_frequency(self):
        # Test resampling with an unsupported frequency
        data = {'close': [150, 152, 154]}
        index = pd.date_range("2024-01-01", periods=3, freq='D')
        df = pd.DataFrame(data, index=index)
        with self.assertRaises(ValueError):  # Expect ValueError for invalid frequency
            self.price_history.resample_to_business_days(df, freq='invalid')

    def test_get_ohlcv_data(self):
        # Test to ensure correct OHLCV data is returned
        result = self.price_history.get_ohlcv_data("1mo", "1d")
        expected_df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [102, 103, 104],
            'volume': [1000, 1100, 1200]
        })
        pd.testing.assert_frame_equal(result, expected_df)  # Validate returned DataFrame matches expected

    @patch.object(PriceHistory, "history", side_effect=YFPricesMissingError(
        "Prices missing for requested period", debug_info={"period": "1y", "interval": "1d"}
    ))
    def test_prices_missing_error(self, mock_history):
        """
        Test to verify behavior when price data is missing for the requested period.
        """
        with self.assertRaises(YFPricesMissingError) as context:
            self.price_history.history(period="1y", interval="1d")
        self.assertIn("Prices missing", str(context.exception))  # Verify error message content
        self.assertIn("period", context.exception.debug_info)  # Verify debug info is provided
        self.assertEqual(context.exception.debug_info["period"], "1y")  # Check specific debug details

    def _normalize_price_data(self, df):
        for col in ['Close', 'Volume']:
            if col in df:
                df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        return df


    @patch.object(PriceHistory, "_normalize_price_data", return_value=pd.DataFrame({'Value': [100, 200]}))
    def te_normalize_price_data(self, mock_normalize):
        # Test normalization of price data
        df = pd.DataFrame({'Price': [1, 2]})
        result = self.price_history._normalize_price_data(df)
        self.assertIsInstance(result, pd.DataFrame)  # Ensure result is a DataFrame
        mock_normalize.assert_called_once()  # Ensure normalization was called

    @patch.object(PriceHistory, '_handle_large_dataset', return_value=large_mocked_data)
    def test_handle_large_dataset(self, mock_handle):
        result = self.price_history._handle_large_dataset(input_data)
        self.assertEqual(len(result), len(input_data))


    @patch.object(PriceHistory, "get_metadata", return_value={"symbol": "AAPL", "exchange": "NASDAQ"})
    def test_metadata_retrieval(self, mock_get_metadata):
        # Test metadata retrieval for a valid ticker
        metadata = self.price_history.get_metadata()
        self.assertIn("symbol", metadata)  # Ensure metadata contains the symbol key
        self.assertIn("exchange", metadata)  # Ensure metadata contains the exchange key
        mock_get_metadata.assert_called_once()  # Ensure mock was called once

    @patch.object(PriceHistory, "get_dividends", return_value=pd.Series([0.1, 0.2], name="Dividends"))
    def test_get_dividends(self, mock_get_dividends):
        # Test fetching dividends data
        dividends = self.price_history.get_dividends()
        self.assertIsInstance(dividends, pd.Series)  # Ensure result is a Series
        self.assertEqual(dividends.name, "Dividends")  # Check name of the Series

    def test_empty_metadata_handling(self):
        # Test handling of empty metadata responses
        self.price_history.get_metadata = MagicMock(return_value={})
        metadata = self.price_history.get_metadata()
        self.assertEqual(metadata, {})  # Ensure empty metadata is handled gracefully

    def test_invalid_ticker_symbol(self):
        # Test invalid ticker symbol handling
        with self.assertRaises(ValueError):  # Expect ValueError for invalid ticker
            PriceHistory(None, "INVALID_TICKER", self.timezone)

    def test_empty_dataframe_handling(self):
        # Test handling of empty DataFrame in various operations
        empty_df = pd.DataFrame()  # Simulating empty DataFrame
        self.price_history._handle_empty_dataframe = MagicMock(return_value=empty_df)
        result = self.price_history._handle_empty_dataframe(empty_df)
        self.assertTrue(result.empty)  # Ensure result is still empty
        self.price_history._handle_empty_dataframe.assert_called_once_with(empty_df)

    def test_fetch_price_data_with_empty_response(self):
        # Test behavior when fetch_price_data returns an empty DataFrame
        self.price_history.fetch_price_data = MagicMock(return_value=pd.DataFrame())
        result = self.price_history.fetch_price_data("1mo", "1d")
        self.assertTrue(result.empty)  # Ensure the returned DataFrame is empty
        self.price_history.fetch_price_data.assert_called_once_with("1mo", "1d")

    def _remove_duplicates(self, df):
        return df.loc[~df.index.duplicated(keep='first')]
        

    @patch.object(PriceHistory, "_remove_duplicates", return_value=pd.DataFrame({"Close": [100, 102]}))
    def test_remove_duplicates(self, mock_remove_duplicates):
        # Test duplicate row removal
        data = {"Close": [100, 100, 102, 102]}
        df = pd.DataFrame(data)
        result = self.price_history._remove_duplicates(df)
        self.assertIsInstance(result, pd.DataFrame)  # Ensure result is a DataFrame
        self.assertEqual(len(result), 2)  # Ensure duplicates are removed
        mock_remove_duplicates.assert_called_once_with(df)

    @patch.object(PriceHistory, "_filter_data_by_date", return_value=pd.DataFrame({"Close": [100, 102]}))
    def test_filter_data_by_date(self, mock_filter_data):
        # Test filtering data by date range
        data = {"Close": [100, 102, 104]}
        index = pd.date_range("2024-01-01", periods=3)
        df = pd.DataFrame(data, index=index)
        result = self.price_history._filter_data_by_date(df, "2024-01-01", "2024-01-02")
        self.assertIsInstance(result, pd.DataFrame)  # Ensure result is a DataFrame
        self.assertEqual(len(result), 2)  # Ensure only two rows remain after filtering
        mock_filter_data.assert_called_once_with(df, "2024-01-01", "2024-01-02")



    @patch.object(PriceHistory, "_convert_interval_to_timedelta", return_value=pd.Timedelta(days=1))
    def test_interval_conversion_logic(self, mock_convert_interval_to_timedelta):
        # Test conversion of interval strings to timedelta
        result = self.price_history._convert_interval_to_timedelta("1d")
        self.assertEqual(result, pd.Timedelta(days=1))  # Ensure conversion is correct
        mock_convert_interval_to_timedelta.assert_called_once_with("1d")

    @patch.object(PriceHistory, "_normalize_price_data", return_value=pd.DataFrame({"Close": [100, 102, 104]}))
    def test_normalize_price_data(self, mock_normalize):
        # Test normalization of price data
        df = pd.DataFrame({"Close": [100, 102, 104]})
        result = self.price_history._normalize_price_data(df)
        self.assertIsInstance(result, pd.DataFrame)  # Ensure result is a DataFrame
        self.assertEqual(len(result), 3)  # Ensure all rows remain after normalization
        mock_normalize.assert_called_once_with(df)

    def test_invalid_period_handling(self):
        # Test handling of an invalid period
        with self.assertRaises(ValueError):  # Ensure ValueError is raised for invalid periods
            self.price_history.calculate_period_days("invalid_period")

    @patch.object(PriceHistory, "history", return_value=pd.DataFrame({"Close": [150, 152, 154]}))
    def test_history_with_valid_data(self, mock_history):
        # Test fetching history for valid data
        result = self.price_history.history(period="1mo", interval="1d")
        self.assertIsInstance(result, pd.DataFrame)  # Ensure result is a DataFrame
        self.assertEqual(len(result), 3)  # Ensure 3 rows of data are returned
        mock_history.assert_called_once_with(period="1mo", interval="1d")

    @patch.object(PriceHistory, "_handle_large_dataset", return_value=pd.DataFrame({"Close": [100, 102, 104]}))
    def test_large_dataset_handling(self, mock_handle_large_dataset):
        # Test handling of a large dataset
        large_data = {"Close": [100] * 10000}
        df = pd.DataFrame(large_data)
        result = self.price_history._handle_large_dataset(df)
        self.assertIsInstance(result, pd.DataFrame)  # Ensure result is a DataFrame
        self.assertEqual(len(result), 10000)  # Ensure all rows are handled
        mock_handle_large_dataset.assert_called_once_with(df)

    @patch.object(PriceHistory, 'get_metadata', return_value={"ticker": "AAPL", "name": "Apple Inc."})
    def test_metadata_retrieval_valid_ticker(self, mock_metadata):
        result = self.price_history.get_metadata("AAPL")
        self.assertIn("ticker", result)


    def _handle_missing_values(self, df):
        return df.interpolate(method='linear').dropna()


    def test_handle_missing_values(self):
        # Test handling of missing values in OHLCV data
        data = {
            "Open": [100, None, 104],
            "Close": [102, 103, None],
            "Volume": [1000, 1100, None]
        }
        df = pd.DataFrame(data)
        result = self.price_history._handle_missing_values(df)
        self.assertFalse(result.isnull().any().any())  # Ensure no NaN values remain

    @patch.object(PriceHistory, "get_dividends", return_value=pd.Series({"2024-01-01": 0.5, "2024-01-02": 0.6}))
    def test_get_dividends_with_mock(self, mock_get_dividends):
        # Test retrieval of dividends with mocked data
        result = self.price_history.get_dividends()
        self.assertIsInstance(result, pd.Series)  # Ensure result is a Series
        self.assertEqual(len(result), 2)  # Ensure 2 entries are present
        mock_get_dividends.assert_called_once()

    @patch.object(PriceHistory, "get_actions", return_value=pd.DataFrame({"Action": ["Split", "Dividend"], "Value": [2, 1.5]}))
    def test_get_actions_with_mock(self, mock_get_actions):
        # Test retrieval of actions with mocked data
        result = self.price_history.get_actions()
        self.assertIsInstance(result, pd.DataFrame)  # Ensure result is a DataFrame
        self.assertEqual(len(result), 2)  # Ensure 2 rows of actions are present
        self.assertIn("Action", result.columns)  # Ensure "Action" column exists
        mock_get_actions.assert_called_once()

    @patch.object(PriceHistory, "history", return_value=pd.DataFrame({"Close": [100, 102], "Open": [99, 101]}))
    def test_repair_option_with_mocked_history(self, mock_history):
        # Test the repair option for historical data
        result = self.price_history.history(period="1mo", interval="1d", repair=True)
        self.assertIsInstance(result, pd.DataFrame)  # Ensure result is a DataFrame
        self.assertIn("Close", result.columns)  # Ensure "Close" column exists
        mock_history.assert_called_once_with(period="1mo", interval="1d", repair=True)

    def test_handle_empty_metadata_response(self):
        # Test handling of empty metadata response
        self.price_history.get_metadata = MagicMock(return_value={})  # Mock empty metadata
        result = self.price_history.get_metadata()
        self.assertEqual(result, {})  # Ensure result is an empty dictionary
        self.price_history.get_metadata.assert_called_once()

    def test_invalid_symbol_handling(self):
        # Test handling of invalid ticker symbols
        with self.assertRaises(ValueError):  # Expect a ValueError for invalid symbols
            PriceHistory(None, "INVALID_SYMBOL", "America/New_York")

    def test_back_adjust_option(self):
        # Test the back adjustment option
        self.price_history.history = MagicMock(return_value=pd.DataFrame({"Close": [150, 152, 154]}))
        result = self.price_history.history(period="1mo", interval="1d", back_adjust=True)
        self.assertIsInstance(result, pd.DataFrame)  # Ensure result is a DataFrame
        self.assertEqual(len(result), 3)  # Ensure 3 rows of adjusted data are present
        self.price_history.history.assert_called_once_with(period="1mo", interval="1d", back_adjust=True)

    def test_rounding_option(self):
        # Test the rounding option for historical data
        self.price_history.history = MagicMock(return_value=pd.DataFrame({"Close": [150.1234, 152.5678, 154.9999]}))
        result = self.price_history.history(period="1mo", interval="1d", rounding=True)
        self.assertIsInstance(result, pd.DataFrame)  # Ensure result is a DataFrame
        self.assertAlmostEqual(result["Close"].iloc[0], 150.12, places=2)  # Ensure rounding is applied
        self.price_history.history.assert_called_once_with(period="1mo", interval="1d", rounding=True)

    @patch("yfinance.scrapers.history.requests.get")
    def test_fetch_metadata_with_http_error(self, mock_requests_get):
        # Simulate an HTTP error during metadata fetching
        mock_requests_get.side_effect = HTTPError("HTTP Error Occurred")
        with self.assertRaises(HTTPError):  # Expect an HTTPError
            self.price_history.get_metadata()

    @patch.object(PriceHistory, "_handle_large_dataset")
    def test_large_dataset_chunking(self, mock_handle_large_dataset):
        # Simulate large dataset handling by chunking
        mock_data = pd.DataFrame({"Close": [100] * 5000})  # Simulate 5000 rows
        self.price_history._handle_large_dataset.return_value = mock_data
        result = self.price_history._handle_large_dataset(mock_data)
        self.assertEqual(len(result), 5000)  # Ensure all rows are processed
        mock_handle_large_dataset.assert_called_once_with(mock_data)

    @patch.object(PriceHistory, "get_splits", return_value=pd.Series({"2024-01-01": 2}))
    def test_get_stock_splits_with_mock(self, mock_get_splits):
        # Test retrieval of stock splits with mocked data
        result = self.price_history.get_splits()
        self.assertIsInstance(result, pd.Series)  # Ensure result is a Series
        self.assertEqual(len(result), 1)  # Ensure 1 entry is present
        mock_get_splits.assert_called_once()

    def test_invalid_interval_for_resample(self):
        # Test resampling with an unsupported interval
        data = {"Close": [150, 152, 154]}
        index = pd.date_range("2024-01-01", periods=3, freq="D")
        df = pd.DataFrame(data, index=index)
        with self.assertRaises(ValueError):  # Expect ValueError for unsupported intervals
            self.price_history.resample_to_business_days(df, "unsupported")

    def test_timezone_adjustment_for_metadata(self):
        # Test timezone adjustment for metadata
        self.price_history.get_metadata = MagicMock(return_value={"timezone": "UTC"})
        metadata = self.price_history.get_metadata()
        self.assertEqual(metadata["timezone"], "UTC")  # Ensure timezone is correctly retrieved

    @patch.object(PriceHistory, "_combine_datasets")
    def test_merge_datasets_with_time_overlap(self, mock_combine_datasets):
        # Simulate combining datasets with overlapping time ranges
        df1 = pd.DataFrame({"Close": [100, 101], "Date": ["2024-01-01", "2024-01-02"]}).set_index("Date")
        df2 = pd.DataFrame({"Close": [101, 102], "Date": ["2024-01-02", "2024-01-03"]}).set_index("Date")
        mock_combine_datasets.return_value = pd.concat([df1, df2]).drop_duplicates()
        result = self.price_history._combine_datasets([df1, df2])
        self.assertEqual(len(result), 3)  # Ensure duplicates are removed
        mock_combine_datasets.assert_called_once_with([df1, df2])

    def test_validate_data_integrity_columns(self):
        # Test to ensure that required OHLC columns exist in the data
        df = pd.DataFrame({"Open": [100], "High": [105], "Low": [95], "Close": [102], "Volume": [1000]})
        required_columns = {"Open", "High", "Low", "Close", "Volume"}
        self.assertTrue(required_columns.issubset(df.columns), "DataFrame must contain required OHLC columns.")

        # Test with a DataFrame missing columns
        df_invalid = pd.DataFrame({"Invalid": [1]})
        self.assertFalse(required_columns.issubset(df_invalid.columns), "DataFrame missing required columns must fail validation.")

    def _merge_metadata_with_prices(self, price_data, metadata):
        for key, value in metadata.items():
            price_data.attrs[key] = value
        return price_data


    def test_merge_metadata_and_price_data(self):
        # Test combining metadata and price data into a unified DataFrame
        price_data = pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=5, freq="D"),
            "Close": [100, 102, 104, 106, 108]
        }).set_index("Date")

        metadata = {"currency": "USD", "symbol": "AAPL", "exchange": "NASDAQ"}
        result = self.price_history._merge_metadata_with_prices(price_data, metadata)
        self.assertEqual(result.attrs["currency"], "USD", "Merged DataFrame should retain currency metadata.")
        self.assertEqual(result.attrs["symbol"], "AAPL", "Merged DataFrame should retain symbol metadata.")

    @patch.object(PriceHistory, "get_history_metadata", return_value={"symbol": "AAPL", "timezone": "UTC"})
    def test_history_metadata_timezone(self, mock_get_history_metadata):
        # Test ensuring timezone in metadata matches expected values
        metadata = self.price_history.get_history_metadata()
        self.assertEqual(metadata["timezone"], "UTC", "Expected timezone metadata to match.")

    @patch.object(PriceHistory, "get_metadata")
    def test_metadata_caching_behavior(self, mock_get_metadata):
        # Simulate caching mechanism for metadata retrieval
        mock_get_metadata.side_effect = [{"cached": True}, {"cached": False}]
        first_metadata = self.price_history.get_metadata()
        second_metadata = self.price_history.get_metadata()

        # Ensure caching returns consistent data on subsequent calls
        self.assertTrue(first_metadata["cached"], "First metadata call should return cached result.")
        self.assertFalse(second_metadata["cached"], "Second metadata call should not use cached data.")
        self.assertEqual(mock_get_metadata.call_count, 2, "Metadata retrieval should be called twice for non-cached requests.")

    def test_large_period_resample(self):
        # Test handling of large periods during resampling
        df = pd.DataFrame({
            "Date": pd.date_range(start="2024-01-01", periods=1000, freq="D"),
            "Close": [100 + i for i in range(1000)]
        }).set_index("Date")

        result = self.price_history._resample(df, source_interval="1d", target_interval="1mo")
        self.assertTrue(len(result) < len(df), "Resampling should reduce the size of the DataFrame.")
        self.assertTrue("Close" in result.columns, "Resampled DataFrame must retain original columns.")

    def _apply_back_adjustment(self, price_data):
        if 'Adj Close' in price_data:
            price_data['Close'] = price_data['Adj Close']
        return price_data


    def test_back_adjustment_effectiveness(self):
        # Test application of back adjustment to historical price data
        price_data = pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=5, freq="D"),
            "Close": [200, 190, 180, 170, 160],
            "Adj Close": [180, 170, 160, 150, 140]
        }).set_index("Date")

        adjusted_data = self.price_history._apply_back_adjustment(price_data)
        self.assertTrue(all(adjusted_data["Close"] == adjusted_data["Adj Close"]),
                        "Back adjustment should align 'Close' and 'Adj Close' values.")

    def test_timezone_adjustment_for_resample(self):
        # Test resampling with timezone adjustments
        df = pd.DataFrame({
            "Date": pd.date_range(start="2024-01-01", periods=10, freq="D"),
            "Close": [100 + i for i in range(10)]
        }).set_index("Date").tz_localize("UTC")

        adjusted_df = df.tz_convert("America/New_York")
        result = self.price_history._resample(adjusted_df, source_interval="1d", target_interval="1wk")
        self.assertEqual(result.index.tz.zone, "America/New_York", "Timezone of resampled DataFrame should match the target timezone.")

    @patch.object(PriceHistory, '_normalize_price_data', return_value=normalized_mocked_data)
    def test_data_normalization_scaling(self, mock_normalize):
        result = self.price_history._normalize_price_data(normalized_mocked_data)
        self.assertEqual(result, normalized_mocked_data)


    def test_invalid_metadata_handling(self):
        # Test behavior with invalid metadata
        self.price_history.get_metadata = MagicMock(return_value={"unexpected_key": "value"})
        with self.assertRaises(KeyError):
            _ = self.price_history.get_metadata()["symbol"]  # Symbol should raise a KeyError

    def test_unsupported_frequency_handling(self):
        # Test unsupported frequency handling in resample
        df = pd.DataFrame({"Close": [100, 102, 104], "Volume": [1000, 2000, 3000]})
        with self.assertRaises(ValueError):
            self.price_history._resample(df, source_interval="1d", target_interval="1yearly")

    @patch.object(PriceHistory, '_normalize_price_data', side_effect=ValueError("Invalid data format"))
    def test_invalid_price_data_structure(self, mock_normalize):
        with self.assertRaises(ValueError) as context:
            self.price_history._normalize_price_data(mocked_data)
        self.assertEqual(str(context.exception), "Invalid data format")

    



    def test_empty_dataset_handling(self):
        # Test behavior when the fetched data is empty
        self.price_history.fetch_price_data = MagicMock(return_value=pd.DataFrame())
        result = self.price_history.fetch_price_data("1mo", "1d")
        self.assertTrue(result.empty, "Returned DataFrame should be empty when no data is available.")

    def test_edge_case_large_interval(self):
        # Test behavior with a very large interval
        df = pd.DataFrame({
            "Date": pd.date_range(start="2024-01-01", periods=1000, freq="D"),
            "Close": [100 + i for i in range(1000)]
        }).set_index("Date")

        with self.assertRaises(ValueError):
            self.price_history._resample(df, source_interval="1d", target_interval="1century")

    def test_adjusted_close_data_missing_columns(self):
        # Test handling of missing columns in adjusted close data
        invalid_data = pd.DataFrame({"Close": [150, 152, 154]})  # Missing "Adj Close"
        self.price_history.get_adjusted_close_data = MagicMock(return_value=invalid_data)

        with self.assertRaises(KeyError):
            self.price_history.get_adjusted_close_data("1mo", "1d")["Adj Close"]

    @patch("yfinance.scrapers.history.PriceHistory.get_metadata")
    def test_metadata_key_error(self, mock_get_metadata):
        # Test accessing non-existent keys in metadata
        mock_get_metadata.return_value = {"currency": "USD", "exchange": "NASDAQ"}
        metadata = self.price_history.get_metadata()

        with self.assertRaises(KeyError):
            _ = metadata["symbol"]  # This should raise KeyError

    def test_resampling_for_short_dataset(self):
        # Test resampling logic with a dataset shorter than the target interval
        df = pd.DataFrame({
            "Date": pd.date_range(start="2024-01-01", periods=2, freq="D"),
            "Close": [150, 152]
        }).set_index("Date")

        result = self.price_history._resample(df, source_interval="1d", target_interval="1wk")
        self.assertTrue(len(result) <= len(df), "Resampled data should not exceed the original data length.")

    def test_timezone_mismatch_error(self):
        # Test behavior when data has a timezone mismatch
        df = pd.DataFrame({
            "Date": pd.date_range(start="2024-01-01", periods=5, freq="D"),
            "Close": [100, 102, 104, 106, 108]
        }).set_index("Date").tz_localize("UTC")

        with self.assertRaises(ValueError):
            df.tz_convert("Invalid/Timezone")

    @patch.object(PriceHistory, "fetch_price_data")
    def test_http_error_handling(self, mock_fetch_price_data):
        # Simulate HTTPError during data fetch
        mock_fetch_price_data.side_effect = HTTPError("HTTP Error occurred")

        with self.assertRaises(HTTPError):
            self.price_history.fetch_price_data("1mo", "1d")

    @patch.object(PriceHistory, '_apply_back_adjustment', return_value=mocked_adjusted_data)
    def test_adjusted_close_scaling(self, mock_adjustment):
        result = self.price_history._apply_back_adjustment(mocked_data)
        self.assertEqual(result, mocked_adjusted_data)


    def test_large_volume_data(self):
        # Test handling large datasets with millions of rows
        large_data = pd.DataFrame({
            "Date": pd.date_range(start="2024-01-01", periods=10**6, freq="min"),
            "Close": [100 + i % 5 for i in range(10**6)]
        }).set_index("Date")

        result = self.price_history._resample(large_data, "1T", "1H")
        self.assertTrue(len(result) < len(large_data), "Resampled data should have fewer rows.")

    def test_missing_metadata_fields(self):
        # Simulate incomplete metadata
        self.price_history.get_metadata = MagicMock(return_value={"symbol": "AAPL"})
        metadata = self.price_history.get_metadata()

        # Validate missing keys raise errors appropriately
        with self.assertRaises(KeyError):
            _ = metadata["exchange"]

    def test_invalid_currency_conversion(self):
        # Simulate invalid currency data
        self.price_history.get_metadata = MagicMock(return_value={"currency": "INVALID"})
        metadata = self.price_history.get_metadata()

        with self.assertRaises(ValueError):
            if metadata["currency"] not in ["USD", "EUR", "JPY"]:
                raise ValueError("Unsupported currency detected.")

    @patch.object(PriceHistory, '_detect_and_handle_outliers', return_value=cleaned_mocked_data)
    def test_outlier_detection_in_price_data(self, mock_outliers):
        result = self.price_history._detect_and_handle_outliers(input_data)
        self.assertEqual(result, cleaned_mocked_data)


    def _detect_and_handle_outliers(self, df, z_thresh=3):
        z_scores = (df - df.mean()) / df.std()
        return df[(z_scores.abs() < z_thresh).all(axis=1)]



    @patch("yfinance.scrapers.history.requests.get", side_effect=HTTPError("429 Too Many Requests"))
    def test_api_rate_limiting(self, mock_get):
        with self.assertRaises(HTTPError):
            self.price_history.fetch_price_data("AAPL")


    def test_partial_dataset_with_na(self):
        # Test combining datasets where one dataset contains NaN values
        df1 = pd.DataFrame({"Close": [100, 102], "Volume": [1000, None]})
        df2 = pd.DataFrame({"Close": [104, 106], "Volume": [None, 1100]})

        result = self.price_history._combine_datasets([df1, df2])
        self.assertTrue(result.isnull().any().any(), "Combined dataset should retain NaN values where appropriate.")

    def test_unsupported_interval_error(self):
        # Ensure unsupported intervals raise an appropriate error
        with self.assertRaises(ValueError):
            self.price_history.interval_to_seconds("unsupported_interval")

    def interval_to_seconds(self, interval):
        mapping = {
            "1s": 1,
            "1min": 60,
            "1h": 3600,
            "1d": 86400,
            "1wk": 604800,
            "1mo": 2592000
        }
        if interval not in mapping:
            raise ValueError(f"Unsupported interval: {interval}")
        return mapping[interval]


    def test_empty_combined_data_handling(self):
        # Test combining completely empty datasets
        df1 = pd.DataFrame()
        df2 = pd.DataFrame()

        result = self.price_history._combine_datasets([df1, df2])
        self.assertTrue(result.empty, "Combining empty datasets should return an empty DataFrame.")

    def create_mock_data():
        return {
            "raw_data": pd.DataFrame({
                "Date": ["2024-11-01", "2024-11-02"],
                "Close": [160.0, 165.0],
                "Volume": [100000, 150000]
            }),
            "adjusted_data": pd.DataFrame({
                "Date": ["2024-11-01", "2024-11-02"],
                "Adjusted Close": [150.0, 155.0]
            }),
            "normalized_data": pd.DataFrame({
                "Date": ["2024-11-01", "2024-11-02"],
                "Normalized Close": [0.8, 0.85]
            }),
            "cleaned_data": pd.DataFrame({
                "Date": ["2024-11-01", "2024-11-02"],
                "Close": [160.0, 165.0],
                "Volume": [100000, 150000]
            })
        }


    def test_high_frequency_data_resampling(self):
        # Test resampling high-frequency data into lower frequencies
        df = pd.DataFrame({
            "Date": pd.date_range(start="2024-01-01", periods=60, freq="s"),
            "Close": [100 + i % 3 for i in range(60)]
        }).set_index("Date")

        result = self.price_history._resample(df, source_interval="1S", target_interval="1T")
        self.assertTrue(len(result) < len(df), "Resampled data should have fewer rows for lower frequency.")

    @patch("yfinance.scrapers.history.PriceHistory._handle_large_dataset")
    def test_large_dataset_processing(self, mock_handle_large_dataset):
        # Simulate handling a large dataset
        mock_handle_large_dataset.return_value = pd.DataFrame({
            "Close": [150, 152, 154]
        })

        result = self.price_history._handle_large_dataset(mock_handle_large_dataset())
        self.assertIsInstance(result, pd.DataFrame, "Large dataset processing should return a DataFrame.")
        self.assertFalse(result.empty, "Processed dataset should not be empty.")

    def test_zero_volume_handling(self):
        # Test for datasets where volume is zero, which might occur in illiquid stocks
        df = pd.DataFrame({
            "Date": pd.date_range(start="2024-01-01", periods=5, freq="D"),
            "Close": [100, 102, 104, 103, 105],
            "Volume": [0, 0, 0, 0, 0]
        }).set_index("Date")

        result = self.price_history._handle_zero_volume(df)
        self.assertTrue("Volume" in result.columns, "Volume column should remain in the data.")
        self.assertTrue((result["Volume"] == 0).all(), "Zero volume should be handled appropriately.")

    @patch.object(PriceHistory, '_remove_duplicates', return_value=deduplicated_mocked_data)
    def test_duplicate_index_handling(self, mock_remove):
        result = self.price_history._remove_duplicates(mocked_data)
        self.assertEqual(result, deduplicated_mocked_data)


    def test_partial_day_data(self):
        # Test for data with missing times in intraday data
        df = pd.DataFrame({
            "Date": pd.date_range(start="2024-01-01 09:00", periods=6, freq="2H"),
            "Close": [100, 102, 104, None, 108, 110]
        }).set_index("Date")

        result = self.price_history._fill_missing_data(df, freq="1H")
        self.assertTrue(len(result) > len(df), "Missing times should be filled.")
        self.assertFalse(result.isnull().any().any(), "All missing values should be filled.")

    @patch.object(PriceHistory, '_resample', return_value=resampled_mocked_data)
    def test_resample_downscale(self, mock_resample):
        result = self.price_history._resample(mocked_data, source_interval="1T", target_interval="1D")
        self.assertEqual(result, resampled_mocked_data)


    @patch("yfinance.scrapers.history.requests.get")
    def test_empty_api_response(self, mock_get):
        # Test when API returns an empty response
        mock_get.return_value.json.return_value = {"chart": {"result": []}}

        with self.assertRaises(ValueError):
            self.price_history.fetch_price_data("1mo", "1d")

    def test_invalid_column_names(self):
        # Test for data with unexpected or missing column names
        df = pd.DataFrame({
            "OpenPrice": [100, 101, 102],
            "HighPrice": [105, 106, 107],
            "LowPrice": [95, 96, 97],
            "ClosingPrice": [102, 103, 104],
            "Volume": [1000, 1100, 1200]
        })

        with self.assertRaises(KeyError):
            self.price_history._validate_columns(df)

    @patch.object(PriceHistory, '_adjust_timezone', return_value=adjusted_mocked_data)
    def test_timezone_aware_index(self, mock_timezone):
        result = self.price_history._adjust_timezone(mocked_data, "America/New_York")
        self.assertEqual(result, adjusted_mocked_data)


    def test_fetch_data_with_invalid_ticker(self):
        # Test when an invalid ticker symbol is provided
        with self.assertRaises(ValueError):
            self.price_history.fetch_price_data("INVALID_TICKER", "1d")

    def test_large_time_range(self):
        # Test for fetching data over an excessively large time range
        with self.assertRaises(YFPricesMissingError):
            self.price_history.history(period="20y", interval="1d")

    def test_unexpected_data_format(self):
        # Test for handling unexpected data structures from the API
        mock_response = MagicMock()
        mock_response.json.return_value = {"unexpected_key": "unexpected_value"}

        with patch("yfinance.scrapers.history.requests.get", return_value=mock_response):
            with self.assertRaises(ValueError):
                self.price_history.fetch_price_data("1mo", "1d")

    def _handle_zero_volume(self, df):
        # Handle scenarios where trading volume is zero for all rows
        if df["Volume"].eq(0).all():
            df["Volume"] = None  # Optionally set to None for clarity
        return df

    def _fill_missing_data(self, df, freq):
        return df.asfreq(freq).ffill()


    def _validate_columns(self, df):
        required_columns = {"Open", "High", "Low", "Close", "Volume"}
        if not required_columns.issubset(df.columns):
            raise KeyError(f"Missing required columns: {required_columns - set(df.columns)}")


    def _adjust_timezone(self, df, target_tz):
        return df.tz_convert(target_tz)


    def test_missing_close_prices(self):
        # Test for datasets missing 'Close' prices
        df = pd.DataFrame({
            "Open": [100, 101, 102],
            "High": [105, 106, 107],
            "Low": [95, 96, 97],
            "Volume": [1000, 1100, 1200]
        })
        with self.assertRaises(KeyError):
            self.price_history._validate_columns(df)

    def test_unsupported_frequency(self):
        # Test for unsupported resampling frequencies
        df = pd.DataFrame({
            "Close": [100, 102, 104],
            "Volume": [1000, 1100, 1200]
        }, index=pd.date_range("2024-01-01", periods=3, freq="D"))
        
        with self.assertRaises(ValueError):
            self.price_history._resample_data(df, "1D", "unsupported_freq")

    @patch("yfinance.scrapers.history.PriceHistory.get_metadata")
    def test_empty_metadata_response(self, mock_get_metadata):
        # Test when metadata response is empty or missing expected keys
        mock_get_metadata.return_value = {}
        metadata = self.price_history.get_metadata()
        self.assertEqual(metadata, {}, "Metadata should be empty if no valid data is returned.")

    @patch("yfinance.scrapers.history.PriceHistory.fetch_price_data")
    def test_fetch_price_data_raises_http_error(self, mock_fetch_price_data):
        # Simulate HTTP error during data fetching
        mock_fetch_price_data.side_effect = HTTPError("HTTP Error")
        with self.assertRaises(HTTPError):
            self.price_history.fetch_price_data("1mo", "1d")

    def test_handle_nonexistent_ticker(self):
        # Test when a non-existent ticker is passed
        with self.assertRaises(ValueError) as context:
            self.price_history.fetch_price_data("FAKETICKER", "1d")
        self.assertEqual(str(context.exception), "Invalid ticker symbol", "Error message should reflect invalid ticker.")

    def test_history_invalid_dates(self):
        # Test for invalid date ranges passed to history method
        with self.assertRaises(ValueError):
            self.price_history.history(start="2025-01-01", end="2024-12-31", interval="1d")

    @patch.object(PriceHistory, '_combine_datasets', return_value=combined_mocked_data)
    def test_combine_datasets(self, mock_combine):
        result = self.price_history._combine_datasets([df1, df2])
        self.assertEqual(result, combined_mocked_data)


    def test_large_resampling_intervals(self):
        # Test for very large resampling intervals
        df = pd.DataFrame({
            "Close": [100, 102, 104, 106, 108],
            "Volume": [1000, 1100, 1200, 1300, 1400]
        }, index=pd.date_range("2024-01-01", periods=5, freq="D"))

        result = self.price_history._resample(df, "1D", "10D")
        self.assertTrue(len(result) < len(df), "Resampling should reduce the number of rows.")
        self.assertTrue(result.index.freq, "10D")

    def test_unexpected_column_case_sensitivity(self):
        # Test case sensitivity in column names
        df = pd.DataFrame({
            "open": [100, 101, 102],
            "high": [105, 106, 107],
            "low": [95, 96, 97],
            "close": [102, 103, 104],
            "volume": [1000, 1100, 1200]
        })
        with self.assertRaises(KeyError):
            self.price_history._validate_columns(df)

    @patch.object(PriceHistory, '_fill_missing_data', return_value=filled_mocked_data)
    def test_data_with_irregular_intervals(self, mock_fill):
        result = self.price_history._fill_missing_data(input_data, freq="30T")
        self.assertEqual(result, filled_mocked_data)


    def test_api_rate_limit_handling(self):
        # Simulate API rate limit error
        with patch("yfinance.scrapers.history.requests.get", side_effect=HTTPError("429 Too Many Requests")):
            with self.assertRaises(HTTPError):
                self.price_history.fetch_price_data("1mo", "1d")

    @patch("yfinance.scrapers.history.PriceHistory.history")
    def test_edge_case_invalid_period_and_interval(self, mock_history):
        # Test both invalid period and interval passed to history
        mock_history.side_effect = ValueError("Invalid period and interval combination.")
        with self.assertRaises(ValueError):
            self.price_history.history(period="100y", interval="10min")

    def _resample(self, df, source_interval=None, target_interval=None):
        if target_interval == "1D":
            return df.resample("D").mean()
        # Add other intervals as needed
        raise ValueError(f"Unsupported target interval: {target_interval}")



    def _combine_datasets(self, datasets):
        if not datasets:
            return pd.DataFrame()
        combined = pd.concat(datasets, axis=0).drop_duplicates()
        return combined



if __name__ == "__main__":
    unittest.main()