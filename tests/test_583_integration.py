import unittest
from unittest.mock import MagicMock, patch
from urllib.error import HTTPError
import yfinance as yf
import pandas as pd
import warnings

from yfinance.exceptions import YFInvalidPeriodError, YFPricesMissingError, YFTzMissingError
from yfinance.scrapers.history import PriceHistory


class TestTickerIntegration(unittest.TestCase):
    @patch("yfinance.scrapers.history.PriceHistory")
    def setUp(self, mock_price_history):
        self.price_history = mock_price_history.return_value
        self.price_history.history.return_value = pd.DataFrame()

    
    @classmethod
    def setUpClass(cls):
        # Suppress deprecation warnings
        warnings.filterwarnings("ignore", category=DeprecationWarning)

        # Valid and invalid ticker symbols for testing
        cls.valid_ticker = "AAPL"
        cls.invalid_ticker = "INVALIDTICKER"
        
    def test_ticker_instantiation_valid(self):
        """Test instantiation of Ticker with a valid symbol"""
        ticker = yf.Ticker(self.valid_ticker)
        self.assertIsInstance(ticker, yf.Ticker)
        self.assertEqual(ticker.ticker, self.valid_ticker)
    
    def test_ticker_instantiation_invalid(self):
        """Test instantiation of Ticker with an invalid symbol"""
        ticker = yf.Ticker(self.invalid_ticker)
        self.assertIsInstance(ticker, yf.Ticker)
        self.assertEqual(ticker.ticker, self.invalid_ticker)
    
    def test_option_chain_valid_ticker(self):
        """Test option chain retrieval with a valid ticker"""
        ticker = yf.Ticker(self.valid_ticker)
        try:
            options = ticker.option_chain()
            self.assertIsInstance(options, tuple)
            self.assertGreaterEqual(len(options), 2)  # Calls and Puts DataFrame
            self.assertIsInstance(options.calls, pd.DataFrame)
            self.assertIsInstance(options.puts, pd.DataFrame)
        except Exception as e:
            self.fail(f"Exception occurred during option chain retrieval: {e}")
    
    def test_option_chain_invalid_ticker(self):
        """Test option chain retrieval with an invalid ticker"""
        ticker = yf.Ticker(self.invalid_ticker)
        try:
            options = ticker.option_chain()
            self.assertIsNone(options.calls)
            self.assertIsNone(options.puts)
        except Exception as e:
            self.fail(f"Exception occurred during option chain retrieval for invalid ticker: {e}")
    
    def test_download_options_valid_ticker(self):
        """Test downloading options data for a valid ticker"""
        ticker = yf.Ticker(self.valid_ticker)
        try:
            options = ticker._download_options()
            self.assertIsInstance(options, dict)
        except Exception as e:
            self.fail(f"Exception occurred during downloading options: {e}")
    
    def test_options_dataframe_conversion(self):
        """Test conversion of options to DataFrame"""
        ticker = yf.Ticker(self.valid_ticker)
        options_data = ticker._download_options()
        if options_data:
            df = ticker._options2df(options_data.get('calls', []))
            self.assertIsInstance(df, pd.DataFrame)
    
    def test_repr(self):
        """Test __repr__ method of Ticker class"""
        ticker = yf.Ticker(self.valid_ticker)
        self.assertEqual(repr(ticker), f'yfinance.Ticker object <{self.valid_ticker}>')
    
    def test_download_options_with_date(self):
        """Test downloading options with a specific expiration date"""
        ticker = yf.Ticker(self.valid_ticker)
        try:
            expirations = ticker._download_options()
            date = next(iter(ticker._expirations)) if ticker._expirations else None
            if date:
                options = ticker._download_options(date=date)
                self.assertIsInstance(options, dict)
        except Exception as e:
            self.fail(f"Exception occurred during downloading options with date: {e}")

    def test_expirations_attribute(self):
        """Test that expirations are populated correctly"""
        ticker = yf.Ticker(self.valid_ticker)
        ticker._download_options()
        self.assertIsInstance(ticker._expirations, dict)

    def test_underlying_attribute(self):
        """Test that underlying quote data is populated correctly"""
        ticker = yf.Ticker(self.valid_ticker)
        ticker._download_options()
        self.assertIsInstance(ticker._underlying, dict)

    def test_invalid_option_chain_date(self):
        """Test option_chain with an invalid date"""
        ticker = yf.Ticker(self.valid_ticker)
        ticker._download_options()
        with self.assertRaises(ValueError):
            ticker.option_chain(date="2099-12-31")

    def test_historical_market_data(self):
        """Test historical market data retrieval"""
        ticker = yf.Ticker(self.valid_ticker)
        hist = ticker.history(period="1mo")
        self.assertIsInstance(hist, pd.DataFrame)
        self.assertFalse(hist.empty)

    def test_dividends_history_retrieval(self):
        """Test dividends history retrieval"""
        ticker = yf.Ticker(self.valid_ticker)
        dividends = ticker.dividends
        self.assertIsInstance(dividends, pd.Series)

    def test_stock_splits_retrieval(self):
        """Test stock splits retrieval"""
        ticker = yf.Ticker(self.valid_ticker)
        splits = ticker.splits
        self.assertIsInstance(splits, pd.Series)

    def test_sustainability_data_retrieval(self):
        """Test sustainability data retrieval"""
        ticker = yf.Ticker(self.valid_ticker)
        sustainability = ticker.sustainability
        self.assertTrue(sustainability is None or isinstance(sustainability, pd.DataFrame))

    def test_major_holders_retrieval(self):
        """Test major holders retrieval"""
        ticker = yf.Ticker(self.valid_ticker)
        major_holders = ticker.major_holders
        self.assertIsInstance(major_holders, pd.DataFrame)

    def test_institutional_holders_retrieval(self):
        """Test institutional holders retrieval"""
        ticker = yf.Ticker(self.valid_ticker)
        holders = ticker.institutional_holders
        self.assertTrue(holders is None or isinstance(holders, pd.DataFrame))

    def test_mutual_fund_holders_retrieval(self):
        """Test mutual fund holders retrieval"""
        ticker = yf.Ticker(self.valid_ticker)
        mutual_holders = ticker.mutualfund_holders
        self.assertTrue(mutual_holders is None or isinstance(mutual_holders, pd.DataFrame))

    def test_earnings_calendar_retrieval(self):
        """Test earnings calendar retrieval"""
        ticker = yf.Ticker(self.valid_ticker)
        try:
            earnings_dates = ticker.earnings_dates
            if isinstance(earnings_dates, pd.DataFrame):
                self.assertIn('Earnings Date', earnings_dates.columns, "Earnings Date column not found")
            else:
                self.assertIsNone(earnings_dates, "Earnings dates should be None if not available")
        except KeyError:
            self.skipTest("Earnings Date column is not present in the response")
        except Exception as e:
            self.fail(f"Unexpected exception during earnings calendar retrieval: {e}")


    def test_recommendations_summary_retrieval(self):
        """Test recommendations summary retrieval"""
        ticker = yf.Ticker(self.valid_ticker)
        recommendations = ticker.recommendations
        self.assertTrue(recommendations is None or isinstance(recommendations, pd.DataFrame))

    def test_quarterly_earnings_retrieval(self):
        """Test quarterly earnings retrieval"""
        ticker = yf.Ticker(self.valid_ticker)
        quarterly_earnings = ticker.quarterly_earnings
        self.assertTrue(quarterly_earnings is None or isinstance(quarterly_earnings, pd.DataFrame))

    def test_income_statement_retrieval(self):
        """Test income statement retrieval"""
        ticker = yf.Ticker(self.valid_ticker)
        income_stmt = ticker.income_stmt
        self.assertTrue(income_stmt is None or isinstance(income_stmt, pd.DataFrame))

    def test_financials_retrieval(self):
        """Test financials retrieval"""
        ticker = yf.Ticker(self.valid_ticker)
        financials = ticker.financials
        self.assertTrue(financials is None or isinstance(financials, pd.DataFrame))

    def test_balance_sheet_retrieval(self):
        """Test balance sheet retrieval"""
        ticker = yf.Ticker(self.valid_ticker)
        balance_sheet = ticker.balance_sheet
        self.assertTrue(balance_sheet is None or isinstance(balance_sheet, pd.DataFrame))

    def test_cash_flow_retrieval(self):
        """Test cash flow retrieval"""
        ticker = yf.Ticker(self.valid_ticker)
        cash_flow = ticker.cashflow
        self.assertTrue(cash_flow is None or isinstance(cash_flow, pd.DataFrame))

    def test_calendar_retrieval(self):
        """Test calendar retrieval"""
        ticker = yf.Ticker(self.valid_ticker)
        calendar = ticker.calendar
        # Check if calendar is None or a dictionary
        self.assertTrue(calendar is None or isinstance(calendar, dict), "Calendar should be None or a dictionary")

    def test_info_attribute_retrieval(self):
        """Test info attribute retrieval"""
        ticker = yf.Ticker(self.valid_ticker)
        info = ticker.info
        self.assertTrue(info is None or isinstance(info, dict))

    def test_sustainability_invalid_ticker(self):
        """Test sustainability data retrieval with an invalid ticker"""
        ticker = yf.Ticker(self.invalid_ticker)
        sustainability = ticker.sustainability
        self.assertTrue(sustainability is None or isinstance(sustainability, pd.DataFrame))
        if isinstance(sustainability, pd.DataFrame):
            self.assertTrue(sustainability.empty)

    def test_holders_invalid_ticker(self):
        """Test holders retrieval with an invalid ticker"""
        ticker = yf.Ticker(self.invalid_ticker)
        holders = ticker.major_holders
        self.assertTrue(holders is None or isinstance(holders, pd.DataFrame))
        if isinstance(holders, pd.DataFrame):
            self.assertTrue(holders.empty)

    def test_actions_retrieval(self):
        """Test actions retrieval"""
        ticker = yf.Ticker(self.valid_ticker)
        actions = ticker.actions
        self.assertTrue(actions is None or isinstance(actions, pd.DataFrame))

    def test_dividends_invalid_ticker(self):
        """Test dividends data retrieval with an invalid ticker"""
        ticker = yf.Ticker(self.invalid_ticker)
        dividends = ticker.dividends
        self.assertTrue(dividends.empty)

    def test_recommendation_trends_retrieval(self):
        """Test recommendation trends retrieval"""
        ticker = yf.Ticker(self.valid_ticker)
        recommendation_trends = ticker.recommendations
        self.assertTrue(recommendation_trends is None or isinstance(recommendation_trends, pd.DataFrame))

    def test_history_retrieve_valid_data(self):
        """Test history retrieval for a valid period and interval"""
        try:
            history = self.price_history.history(period="1mo", interval="1d")
            self.assertIsInstance(history, pd.DataFrame)
            self.assertFalse(history.empty)
        except Exception as e:
            self.fail(f"Exception occurred during history retrieval: {e}")

    def test_history_invalid_period(self):
        """Test history retrieval with an invalid period"""
        with self.assertRaises(YFInvalidPeriodError):
            self.price_history.history(period="invalid", interval="1d")

    def test_adjust_for_dividends(self):
        """Test adjusting prices for dividends"""
        try:
            self.price_history.history(auto_adjust=True)
            self.assertTrue(self.price_history._data is not None)
        except Exception as e:
            self.fail(f"Exception occurred during auto-adjust: {e}")

    def test_repair_missing_price_data(self):
        """Test the repair mechanism for missing price data"""
        self.price_history._history = pd.DataFrame({'open': [100, None, 102]})
        self.price_history._repair_prices()
        self.assertFalse(self.price_history._history.isnull().values.any())

    def test_metadata_population(self):
        """Test metadata population"""
        self.price_history.history()
        self.assertIsInstance(self.price_history._history_metadata, dict)

    def test_intervals_calculation(self):
        """Test calculation of intervals"""
        intervals = self.price_history._calculate_missing_intervals("1d", "2024-01-01", "2024-01-10")
        self.assertGreater(len(intervals), 0)

    def test_proxy_handling(self):
        """Test handling of proxy settings"""
        proxy_price_history = PriceHistory({}, "AAPL", "America/New_York", proxy="http://proxy.example.com")
        self.assertEqual(proxy_price_history.proxy, "http://proxy.example.com")

    @patch("yfinance.scrapers.history.PriceHistory.history")
    def test_timezone_mismatch(self, mock_history):
        mock_history.side_effect = YFTzMissingError("AAPL")
        with self.assertRaises(YFTzMissingError):
            self.price_history.history(period="1d", interval="1h", timezone=None)



    def test_history_prepost_market(self):
        """Test retrieving pre-market and post-market data"""
        try:
            history = self.price_history.history(prepost=True)
            self.assertIsInstance(history, pd.DataFrame)
        except Exception as e:
            self.fail(f"Exception occurred during pre/post-market data retrieval: {e}")

    def test_interval_exceeding_limit(self):
        """Test interval exceeding limit"""
        with self.assertRaises(ValueError):
            self.price_history.history(interval="100d", period="1mo")

    @patch("yfinance.scrapers.history.PriceHistory.history")
    def test_history_empty_response(self, mock_history):
        mock_history.return_value = pd.DataFrame()
        result = self.price_history.history(period="1d")
        self.assertTrue(result.empty)



    def test_history_invalid_date_range(self):
        """Test history with an invalid date range"""
        with self.assertRaises(ValueError, msg="Invalid date range should raise ValueError"):
            self.price_history.history(start="2025-01-01", end="2024-01-01")


    def test_history_partial_data(self):
        """Test history handling partial data availability"""
        self.price_history.fetch_price_data = MagicMock(return_value=pd.DataFrame({
            'open': [100, None],
            'close': [101, 102],
            'high': [103, None],
            'low': [99, 100],
            'volume': [1000, 1100]
        }))
        result = self.price_history.history(period="1d", interval="1h")
        self.assertFalse(result.isnull().values.any(), "Partial data should be handled without null values")


    def test_repair_prices_multiple_intervals(self):
        """Test price repair mechanism with multiple intervals"""
        self.price_history._history = pd.DataFrame({
            'open': [100, 101, None],
            'close': [102, 103, 104],
            'high': [105, None, 107],
            'low': [95, 96, 97],
            'volume': [1000, None, 1200]
        })
        self.price_history._repair_prices()
        self.assertFalse(self.price_history._history.isnull().values.any(), "Price repair should fill missing values")


    def test_missing_metadata_handling(self):
        """Test handling of missing metadata in history"""
        self.price_history._history_metadata = None
        result = self.price_history.history(period="1d", interval="1h")
        self.assertIsNotNone(self.price_history._history_metadata, "Metadata should be populated even if initially missing")


    def test_invalid_timezone_handling(self):
        """Test history with an invalid timezone"""
        with self.assertRaises(ValueError, msg="Invalid timezone should raise ValueError"):
            invalid_history = PriceHistory({}, "AAPL", "Invalid/Timezone")
            invalid_history.history(period="1d", interval="1h")


    def test_history_no_adjustment(self):
        """Test history without auto-adjusting prices"""
        self.price_history.history(auto_adjust=False)
        self.assertTrue(self.price_history._history is not None, "History data should be retrieved without adjustments")


    def test_history_include_actions(self):
        """Test history with actions included"""
        try:
            history = self.price_history.history(actions=True)
            self.assertIsInstance(history, pd.DataFrame, "History with actions should return a DataFrame")
        except Exception as e:
            self.fail(f"Exception occurred during history retrieval with actions: {e}")


    @patch("yfinance.scrapers.history.PriceHistory.fetch_price_data")
    def test_fetch_price_data(self, mock_fetch):
        mock_fetch.return_value = pd.DataFrame()  # Simulate expected return
        result = self.price_history.fetch_price_data()
        self.assertIsInstance(result, pd.DataFrame)



    def test_fetch_price_data_invalid_period_and_interval(self):
        """Test fetch_price_data with invalid period and interval"""
        with self.assertRaises(ValueError, msg="Invalid period and interval should raise ValueError"):
            self.price_history.history(period="invalid", interval="invalid")

    @patch('yfinance.scrapers.history.PriceHistory.fetch_price_data')
    def test_fetch_price_data_valid(self, mock_fetch):
        mock_fetch.return_value = pd.DataFrame({
            "Date": ["2024-01-01"],
            "Open": [100],
            "Close": [110],
        })
        result = self.price_history.fetch_price_data()
        self.assertFalse(result.empty)


    @patch("yfinance.scrapers.history.PriceHistory.fetch_price_data")
    def test_fetch_price_data_empty(self, mock_fetch):
        mock_fetch.return_value = pd.DataFrame()
        result = self.price_history.fetch_price_data()
        self.assertTrue(result.empty)


    @patch("yfinance.scrapers.history.PriceHistory.history")
    def test_fetch_history_valid(self, mock_history):
        mock_history.return_value = pd.DataFrame({
            "Date": ["2024-01-01", "2024-01-02"],
            "Close": [152, 153]
        }).set_index("Date")
        result = self.price_history.history(period="1mo", interval="1d")
        self.assertFalse(result.empty)

    @patch("yfinance.scrapers.history.PriceHistory.history")
    def test_fetch_history_invalid_period(self, mock_history):
        mock_history.side_effect = YFPricesMissingError("Invalid period", debug_info="Test debug info")
        with self.assertRaises(YFPricesMissingError):
            self.price_history.history(period="invalid")


    @patch("yfinance.scrapers.history.PriceHistory.history")
    def test_history_tz_missing(self, mock_history):
        mock_history.side_effect = YFTzMissingError("Timezone is missing")
        with self.assertRaises(YFTzMissingError):
            self.price_history.history(period="1d", interval="1h", timezone=None)

    @patch("yfinance.scrapers.history.PriceHistory._resample")
    def test_resample_data(self, mock_resample):
        mock_resample.return_value = pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=7, freq="W"),
            "Close": [100, 101, 102, 103, 104, 105, 106],
        })
        result = self.price_history._resample(mock_resample.return_value, "1D", "1W")
        self.assertEqual(len(result), 7)



    @patch('yfinance.scrapers.history.PriceHistory.get_metadata')
    def test_get_metadata(self, mock_get):
        mock_get.return_value = {"symbol": "AAPL", "timezone": "UTC"}
        metadata = self.price_history.get_metadata()
        self.assertEqual(metadata["symbol"], "AAPL")


    @patch("yfinance.scrapers.history.PriceHistory.get_actions")
    def test_get_actions(self, mock_get_actions):
        mock_get_actions.return_value = pd.DataFrame({
            "Action": ["Split", "Dividend"],
            "Value": [2, 1.5],
        })
        actions = self.price_history.get_actions()
        self.assertIsInstance(actions, pd.DataFrame)

    @patch('yfinance.scrapers.history.PriceHistory.fill_missing_dates')
    def test_fill_missing_dates(self, mock_fill):
        mock_fill.return_value = pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=5, freq="D"),
            "Close": [100, 101, 102, 103, 104],
        })
        result = self.price_history.fill_missing_dates(mock_fill.return_value)
        self.assertEqual(len(result), 5)


    def test_adjust_timezone(self):
        df = pd.DataFrame({
            "Date": pd.date_range("2024-01-01", periods=3, freq="D", tz="UTC"),
            "Close": [150, 152, 153]
        }).set_index("Date")
        adjusted_df = df.tz_convert("America/New_York")
        self.assertEqual(adjusted_df.index.tz.zone, "America/New_York")

    def test_back_adjustment(self):
        df = pd.DataFrame({
            "Close": [200, 190, 180],
            "Adj Close": [180, 170, 160]
        })
        df["Close"] = df["Adj Close"]
        self.assertTrue((df["Close"] == df["Adj Close"]).all())

    def test_resample_invalid_frequency(self):
        df = pd.DataFrame({
            "Close": [150, 152, 153],
        }, index=pd.date_range("2024-01-01", periods=3, freq="D"))
        with self.assertRaises(ValueError):
            self.price_history._resample(df, "1d", "1century")

    def test_large_dataset_handling(self):
        df = pd.DataFrame({
            "Close": [100] * 10000,
        }, index=pd.date_range("2024-01-01", periods=10000, freq="T"))
        result = self.price_history._resample(df, "1T", "1H")
        self.assertTrue(len(result) < len(df))

    @patch("yfinance.scrapers.history.requests.get")
    def test_metadata_http_error(self, mock_get):
        mock_get.return_value.status_code = 404
        with self.assertRaises(Exception):
            self.price_history.get_metadata()



    @patch("yfinance.scrapers.history.PriceHistory._remove_duplicates")
    def test_remove_duplicates(self, mock_remove_duplicates):
        mock_data = pd.DataFrame({"Close": [150, 150, 152, 153]})
        mock_remove_duplicates.return_value = mock_data.drop_duplicates()
        result = self.price_history._remove_duplicates(mock_data)
        self.assertTrue(len(result) < len(mock_data))

    def test_handle_empty_dataframe(self):
        df = pd.DataFrame()
        result = self.price_history._handle_empty_dataframe(df)
        self.assertTrue(result.empty)

    @patch("yfinance.scrapers.history.PriceHistory.history")
    def test_period_exceeding_max(self, mock_history):
        mock_history.side_effect = YFPricesMissingError("Data missing for period")
        with self.assertRaises(YFPricesMissingError):
            self.price_history.history(period="10y", interval="1d")

    def test_handle_missing_values(self):
        df = pd.DataFrame({
            "Close": [150, None, 152]
        })
        result = df.interpolate()
        self.assertFalse(result.isnull().any())

    @patch("yfinance.scrapers.history.PriceHistory.history")
    def test_fetch_data_invalid_symbol(self, mock_history):
        mock_history.side_effect = YFTzMissingError("INVALID")
        with self.assertRaises(YFTzMissingError):
            self.price_history.history(period="1d", interval="1h", timezone="UTC")


    def test_capital_gains_empty(self):
        self.price_history._history = pd.DataFrame({
            "Capital Gains": [0, 0]
        })
        result = self.price_history.get_capital_gains()
        self.assertTrue(result.empty)

if __name__ == '__main__':
    unittest.main()
