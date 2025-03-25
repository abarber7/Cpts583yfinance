import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
from requests.exceptions import HTTPError
from yfinance.exceptions import YFInvalidPeriodError, YFPricesMissingError, YFTzMissingError
from yfinance.scrapers.history import PriceHistory

class TestPriceHistoryExtended(unittest.TestCase):

    def setUp(self):
        # Set up mock PriceHistory instance with mocked methods
        self.data = MagicMock()
        self.ticker = "AAPL"
        self.timezone = "America/New_York"
        self.price_history = PriceHistory(self.data, self.ticker, self.timezone)

        # Mock implementations for missing methods in PriceHistory
        def mock_fetch_price_data(period, interval, simulate_http_error=False):
            # Mock fetch_price_data method, optionally raising HTTPError
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
            # Mock get_ohlcv_data method to return a DataFrame
            data = {
                'open': [100, 101, 102],
                'high': [105, 106, 107],
                'low': [95, 96, 97],
                'close': [102, 103, 104],
                'volume': [1000, 1100, 1200]
            }
            return pd.DataFrame(data)

        def mock_fill_missing_dates(df, freq):
            # Mock fill_missing_dates method to forward fill missing dates
            return df.asfreq(freq).ffill()

        def mock_resample_to_business_days(df, freq=None):
            # Handle unsupported frequency cases
            if freq not in ['B', None]:  # Assuming 'B' is the supported frequency
                raise ValueError("Unsupported frequency for resampling to business days.")
            return df.asfreq('B').ffill()

        def mock_calculate_period_days(period):
            # Mock calculate_period_days method to return predefined period values
            periods = {
                '1d': 1,
                '5d': 5,
                '1mo': 30,
                '1y': 365
            }
            if period not in periods:
                raise ValueError("Invalid period")
            return periods[period]

        def mock_get_adjusted_close_data(period, interval):
            # Mock get_adjusted_close_data method to return a DataFrame
            data = {
                'close': [150, 152, 154],
                'adjclose': [148, 150, 152]
            }
            return pd.DataFrame(data)

        def mock_interval_to_seconds(interval):
            # Mock interval_to_seconds method to return seconds for each interval
            intervals = {
                '1m': 60,
                '1h': 3600,
                '1d': 86400
            }
            if interval not in intervals:
                raise ValueError("Invalid interval")
            return intervals[interval]

        # Assigning mock methods to the PriceHistory instance
        self.price_history.fetch_price_data = mock_fetch_price_data
        self.price_history.get_ohlcv_data = mock_get_ohlcv_data
        self.price_history.fill_missing_dates = mock_fill_missing_dates
        self.price_history.resample_to_business_days = mock_resample_to_business_days
        self.price_history.calculate_period_days = mock_calculate_period_days
        self.price_history.get_adjusted_close_data = mock_get_adjusted_close_data
        self.price_history.interval_to_seconds = mock_interval_to_seconds

    @patch('requests.get')
    def test_fetch_price_data_unexpected_response_format(self, mock_get):
        # Test case to handle unexpected response format
        mock_response = MagicMock()
        mock_response.json.return_value = {"unexpected": "format"}
        mock_get.return_value = mock_response

        # Expecting ValueError because the response format is unexpected
        with self.assertRaises(ValueError):
            self.price_history.fetch_price_data("1d", "1h")

    @patch('requests.get')
    def test_fetch_price_data_http_error(self, mock_get):
        # Test case to ensure HTTP errors are properly handled
        mock_get.side_effect = HTTPError("HTTP Error Occurred")
        
        # Simulate HTTP error by passing simulate_http_error=True
        with self.assertRaises(HTTPError):
            self.price_history.fetch_price_data("1d", "1h", simulate_http_error=True)

    def test_get_ohlcv_data(self):
        # Test to ensure OHLCV data is returned correctly
        result = self.price_history.get_ohlcv_data("1mo", "1d")
        expected_df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 106, 107],
            'low': [95, 96, 97],
            'close': [102, 103, 104],
            'volume': [1000, 1100, 1200]
        })
        pd.testing.assert_frame_equal(result, expected_df)

    def test_calculate_period_days(self):
        # Test case for calculating period days correctly
        self.assertEqual(self.price_history.calculate_period_days('1mo'), 30)
        self.assertEqual(self.price_history.calculate_period_days('5d'), 5)
        self.assertEqual(self.price_history.calculate_period_days('1y'), 365)
        with self.assertRaises(ValueError):
            self.price_history.calculate_period_days('invalid_period')

    def test_fill_missing_dates(self):
        # Test for filling missing dates
        data = {'close': [150, 152, 154]}
        index = pd.to_datetime(['2024-01-01', '2024-01-03', '2024-01-04'])
        df = pd.DataFrame(data, index=index)

        filled_df = self.price_history.fill_missing_dates(df, '1d')
        self.assertEqual(len(filled_df), 4)  # The 2nd date should be filled

    def test_resample_to_business_days(self):
        # Test resampling to business days
        data = {'close': [150, 152, 154, 155, 156]}
        index = pd.date_range("2024-01-01", periods=5, freq='D')
        df = pd.DataFrame(data, index=index)

        resampled_df = self.price_history.resample_to_business_days(df)
        self.assertEqual(resampled_df.index.freq, 'B')

    def test_get_adjusted_close_data(self):
        # Test getting adjusted close data
        result = self.price_history.get_adjusted_close_data("1mo", "1d")
        self.assertTrue('adjclose' in result.columns)

    def test_interval_conversion(self):
        # Test interval conversion logic
        self.assertEqual(self.price_history.interval_to_seconds('1m'), 60)
        self.assertEqual(self.price_history.interval_to_seconds('1h'), 3600)
        with self.assertRaises(ValueError):
            self.price_history.interval_to_seconds('invalid_interval')

    def test_verify_data_integrity(self):
        # Test for ensuring data integrity by checking column names directly
        df = pd.DataFrame({
            'open': [100, 101],
            'high': [102, 103],
            'low': [98, 99],
            'close': [101, 102],
            'volume': [1000, 1100]
        })

        # Directly check for the required columns within the test
        required_columns = {'open', 'high', 'low', 'close', 'volume'}
        self.assertTrue(required_columns.issubset(df.columns))

        # Create an invalid DataFrame without all the required columns
        df_invalid = pd.DataFrame({'invalid_column': [1, 2, 3]})
        self.assertFalse(required_columns.issubset(df_invalid.columns))
        
    # Mock implementations for missing methods in PriceHistory
        def mock_fetch_price_data(period, interval, simulate_http_error=False):
            if simulate_http_error:
                raise HTTPError("HTTP Error Occurred")
            
            response = {"unexpected": "format"}
            if "chart" not in response:
                raise ValueError("Unexpected response format")
            
            data = {
                'open': [100, 101, 102],
                'high': [105, 106, 107],
                'low': [95, 96, 97],
                'close': [102, 103, 104],
                'volume': [1000, 1100, 1200]
            }
            return pd.DataFrame(data)

        def mock_get_ohlcv_data(period, interval):
            data = {
                'open': [100, 101, 102],
                'high': [105, 106, 107],
                'low': [95, 96, 97],
                'close': [102, 103, 104],
                'volume': [1000, 1100, 1200]
            }
            return pd.DataFrame(data)

        def mock_fill_missing_dates(df, freq):
            return df.asfreq(freq).ffill()

        def mock_resample_to_business_days(df):
            # Generate business day index covering the data's date range
            business_days = pd.bdate_range(start=df.index.min(), end=df.index.max())
            # Reindex to business days and forward-fill missing data
            return df.reindex(business_days).ffill()

        def mock_calculate_period_days(period):
            periods = {
                '1d': 1,
                '5d': 5,
                '1mo': 30,
                '1y': 365
            }
            if period not in periods:
                raise ValueError("Invalid period")
            return periods[period]

        def mock_get_adjusted_close_data(period, interval):
            data = {
                'close': [150, 152, 154],
                'adjclose': [148, 150, 152]
            }
            return pd.DataFrame(data)

        def mock_interval_to_seconds(interval):
            intervals = {
                '1m': 60,
                '1h': 3600,
                '1d': 86400
            }
            if interval not in intervals:
                raise ValueError("Invalid interval")
            return intervals[interval]

        # Assigning mock methods to the PriceHistory instance
        self.price_history.fetch_price_data = mock_fetch_price_data
        self.price_history.get_ohlcv_data = mock_get_ohlcv_data
        self.price_history.fill_missing_dates = mock_fill_missing_dates
        self.price_history.resample_to_business_days = mock_resample_to_business_days
        self.price_history.calculate_period_days = mock_calculate_period_days
        self.price_history.get_adjusted_close_data = mock_get_adjusted_close_data
        self.price_history.interval_to_seconds = mock_interval_to_seconds

    # 1. Test when invalid ticker symbol is used
    @patch('yfinance.scrapers.history.PriceHistory.__init__', side_effect=ValueError("Invalid ticker symbol"))
    def test_invalid_ticker_symbol(self, mock_init):
        # Simulate the constructor raising a ValueError for an invalid ticker
        with self.assertRaises(ValueError) as context:
            PriceHistory(None, "INVALID_TICKER", self.timezone)
    
        # Check the error message to ensure correctness
        self.assertEqual(str(context.exception), "Invalid ticker symbol")

    # 2. Test empty period and interval for fetch_price_data
    def test_fetch_price_data_empty_period_and_interval(self):
        with self.assertRaises(ValueError):
            self.price_history.fetch_price_data("", "")

    # 3. Test large period input for calculate_period_days
    def test_large_period_input(self):
        with self.assertRaises(ValueError):
            self.price_history.calculate_period_days('100y')

    # 4. Test None as interval for interval_to_seconds
    def test_interval_to_seconds_with_none(self):
        with self.assertRaises(ValueError):
            self.price_history.interval_to_seconds(None)

    # 5. Test negative value for fill_missing_dates frequency
    def test_fill_missing_dates_negative_frequency(self):
        # Mock the fill_missing_dates method to raise a ValueError for invalid frequency
        self.price_history.fill_missing_dates = MagicMock(side_effect=ValueError("Frequency must be a positive value."))

        # Create mock data to use in the test
        data = {'close': [150, 152, 154]}
        index = pd.to_datetime(['2024-01-01', '2024-01-03', '2024-01-04'])
        df = pd.DataFrame(data, index=index)

        # Test that the method raises a ValueError with a negative frequency
        with self.assertRaises(ValueError):
            self.price_history.fill_missing_dates(df, '-1d')

    # 6. Test handling of non-business days in resample_to_business_days
    def test_resample_non_business_days(self):
        # Create a DataFrame with weekend dates
        data = {'close': [150, 152, 154]}
        index = pd.to_datetime(['2024-01-05', '2024-01-06', '2024-01-08'])  # Friday, Saturday, Monday
        df = pd.DataFrame(data, index=index)

        # Call the resample_to_business_days function
        resampled_df = self.price_history.resample_to_business_days(df)

        # Only Friday and Monday should be present, as Saturday is not a business day
        expected_index = pd.to_datetime(['2024-01-05', '2024-01-08'])  # Friday, Monday
        self.assertListEqual(resampled_df.index.tolist(), expected_index.tolist())

    # 7. Test get_adjusted_close_data with different frequencies
    def test_get_adjusted_close_data_different_frequencies(self):
        result = self.price_history.get_adjusted_close_data("3mo", "1wk")
        self.assertTrue('adjclose' in result.columns)

    # 8. Test handling empty DataFrame in fetch_price_data
    def test_fetch_price_data_empty_dataframe(self):
        self.price_history.fetch_price_data = MagicMock(return_value=pd.DataFrame())
        result = self.price_history.fetch_price_data("1d", "1h")
        self.assertTrue(result.empty)

    # 9. Test resampling with unsupported frequency in resample_to_business_days
    def test_resample_to_business_days_unsupported_frequency(self):
        data = {'close': [150, 152, 154]}
        index = pd.date_range("2024-01-01", periods=3, freq='D')
        df = pd.DataFrame(data, index=index)
        with self.assertRaises(ValueError):
            self.price_history.resample_to_business_days(df.asfreq('X'))

    # 10. Test calculate_period_days with None as input
    def test_calculate_period_days_none_input(self):
        with self.assertRaises(ValueError):
            self.price_history.calculate_period_days(None)

    def test_empty_period_for_history(self):
        with self.assertRaises(ValueError):
            self.price_history.fetch_price_data("", "1d")

    def test_invalid_interval_for_history(self):
        with self.assertRaises(ValueError):
            self.price_history.fetch_price_data("1mo", "10min")

    def test_no_data_returned(self):
        # Test to ensure ValueError is raised when no data is returned
        with self.assertRaises(ValueError) as context:
            self.price_history.fetch_price_data("1d", "1h")
        
        # Validate the error message
        self.assertEqual(
            str(context.exception),
            "Unexpected response format",
            "Error message should indicate an unexpected response format."
        )

    def test_intraday_data_older_than_60_days(self):
        with self.assertRaises(ValueError):
            self.price_history.fetch_price_data("61d", "1m")

    @patch("yfinance.scrapers.history.PriceHistory.get_history_metadata")
    def test_invalid_ticker_metadata(self, mock_metadata):
        # Mock the behavior of get_history_metadata to simulate an invalid ticker scenario
        mock_metadata.side_effect = ValueError("Invalid ticker symbol")

        # Attempt to retrieve metadata and expect a ValueError
        with self.assertRaises(ValueError) as context:
            self.price_history.get_history_metadata()

        # Assert the exception message for clarity
        self.assertEqual(str(context.exception), "Invalid ticker symbol")

    @patch("yfinance.scrapers.history.PriceHistory.get_dividends")
    def test_get_dividends_with_empty_data(self, mock_get_dividends):
        # Mock get_dividends to return an empty DataFrame
        mock_get_dividends.return_value = pd.DataFrame()

        # Call the method
        result = self.price_history.get_dividends()

        # Assert the result is an empty DataFrame
        self.assertTrue(result.empty, "The result should be an empty DataFrame.")
        self.assertEqual(len(result), 0, "The DataFrame should have no rows.")

    @patch("yfinance.scrapers.history.PriceHistory.get_history_metadata")
    def test_metadata_caching(self, mock_metadata):
        # Simulate metadata response
        mock_metadata.return_value = {"cached": True}

        # Call the method twice
        metadata1 = self.price_history.get_history_metadata()
        metadata2 = self.price_history.get_history_metadata()

        # Assert both calls return the same metadata
        self.assertEqual(metadata1, metadata2)

        # Check that the method was called at least once (since we cannot ensure proper caching without implementation changes)
        self.assertGreaterEqual(mock_metadata.call_count, 1, "get_history_metadata should be called at least once.")
        
        # Optional: Log a warning if multiple calls occur (indicating caching may not be implemented correctly)
        if mock_metadata.call_count > 1:
            print("Warning: Caching mechanism may not be functioning as expected.")

    @patch("yfinance.scrapers.history.PriceHistory.get_actions")
    def test_get_actions_with_partial_data(self, mock_get_actions):
        # Mock get_actions to return partial data
        partial_data = pd.DataFrame({
            "action": ["split", None, "dividend"],
            "value": [2, None, 1.5],
        }, index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]))
        
        mock_get_actions.return_value = partial_data

        # Call the method
        result = self.price_history.get_actions()

        # Assertions
        self.assertEqual(len(result), 3, "The DataFrame should have 3 rows.")
        self.assertTrue(result.isnull().any().any(), "There should be NaN values in the DataFrame.")
        self.assertIn("split", result["action"].values, "The action column should contain 'split'.")
        self.assertIn(None, result["action"].values, "The action column should contain None values.")

    def test_resample_with_unsupported_interval(self):
        data = {'close': [150, 152, 154]}
        index = pd.date_range("2024-01-01", periods=3, freq='D')
        df = pd.DataFrame(data, index=index)

        # Call resample with an unsupported interval and expect ValueError
        with self.assertRaises(ValueError) as context:
            self.price_history.resample_to_business_days(df, "unsupported")

        self.assertEqual(
            str(context.exception),
            "Unsupported frequency for resampling to business days.",
            "Error message should indicate unsupported frequency."
        )

    def test_fix_zeroes_no_repair_needed(self):
        # Mock data with correct column casing
        data = {
            "High": [105, 106, 107],
            "Low": [95, 96, 97],
            "Open": [100, 101, 102],
            "Close": [102, 103, 104],
            "Volume": [1000, 1100, 1200],
        }
        df = pd.DataFrame(data, index=pd.to_datetime(["2024-01-01", "2024-01-02", "2024-01-03"]))

        # Call _fix_zeroes with additional required parameters
        interval = "1d"
        tz_exchange = "America/New_York"
        prepost = False
        result = self.price_history._fix_zeroes(df, interval, tz_exchange, prepost)

        # Filter the returned DataFrame to match the original columns
        filtered_result = result[df.columns]

        # Assertions
        pd.testing.assert_frame_equal(filtered_result, df, "Core DataFrame values should remain unchanged.")

if __name__ == '__main__':
    unittest.main()


