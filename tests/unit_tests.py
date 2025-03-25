import unittest
from unittest.mock import patch, MagicMock, PropertyMock
import pandas as pd
import requests
from yfinance.base import TickerBase
from yfinance.cache import Cache
from yfinance.const import _BASE_URL_, _ROOT_URL_
from yfinance.data import YfData
from yfinance.exceptions import YFEarningsDateMissing
from yfinance.multi import MultiTicker
from yfinance.ticker import Ticker
from yfinance.tickers import Tickers
from yfinance.utils import is_isin, get_ticker_by_isin, get_yf_logger
from yfinance import utils


class TestYFinance(unittest.TestCase):

    # --- TickerBase Tests ---
    def setUp(self):
        self.ticker = 'AAPL'
        self.ticker_base = TickerBase(self.ticker)

    @patch('yfinance.utils.is_isin', return_value=False)
    def test_tickerbase_init(self, mock_is_isin):
        ticker = TickerBase('MSFT')
        self.assertEqual(ticker.ticker, 'MSFT')
        mock_is_isin.assert_called_once()

    @patch('yfinance.base.PriceHistory')
    def test_tickerbase_lazy_load_price_history(self, mock_price_history):
        price_history_instance = mock_price_history.return_value
        price_history_instance.history.return_value = pd.DataFrame()
        self.ticker_base._lazy_load_price_history()
        mock_price_history.assert_called_once()
        price_history_instance.history.assert_called_once()

    @patch('yfinance.base.TickerBase._fetch_ticker_tz', return_value='America/New_York')
    def test_tickerbase_get_ticker_tz(self, mock_fetch_tz):
        tz = self.ticker_base._get_ticker_tz(proxy=None, timeout=10)
        self.assertEqual(tz, 'America/New_York')
        mock_fetch_tz.assert_called_once()

    @patch('yfinance.base.TickerBase._fetch_ticker_tz')
    @patch('yfinance.base.PriceHistory.history', return_value=pd.DataFrame())
    def test_tickerbase_history(self, mock_history, mock_fetch_tz):
        mock_fetch_tz.return_value = 'America/New_York'
        result = self.ticker_base.history()
        self.assertIsInstance(result, pd.DataFrame)
        mock_fetch_tz.assert_called_once()
        mock_history.assert_called_once()

    @patch('yfinance.base.Quote')
    def test_tickerbase_get_recommendations(self, mock_quote):
        mock_quote.return_value.recommendations = pd.DataFrame({'period': ['2024Q1'], 'strongBuy': [10]})
        self.ticker_base._quote = mock_quote.return_value
        result = self.ticker_base.get_recommendations()
        self.assertIsInstance(result, pd.DataFrame)
        self.assertEqual(result['period'][0], '2024Q1')

    @patch('yfinance.base.Quote')
    def test_tickerbase_get_calendar(self, mock_quote):
        mock_quote.return_value.calendar = {'Event': 'Earnings Call'}
        self.ticker_base._quote = mock_quote.return_value
        result = self.ticker_base.get_calendar()
        self.assertEqual(result, {'Event': 'Earnings Call'})

    @patch('yfinance.base.Quote')
    def test_tickerbase_get_sec_filings(self, mock_quote):
        mock_quote.return_value.sec_filings = {'Filing': '10-Q'}
        self.ticker_base._quote = mock_quote.return_value
        result = self.ticker_base.get_sec_filings()
        self.assertEqual(result, {'Filing': '10-Q'})

    @patch('yfinance.base.Holders')
    def test_tickerbase_get_major_holders(self, mock_holders):
        mock_holders.return_value.major = pd.DataFrame({'Holder': ['BlackRock']})
        self.ticker_base._holders = mock_holders.return_value
        result = self.ticker_base.get_major_holders()
        self.assertIsInstance(result, pd.DataFrame)

    @patch('yfinance.base.Fundamentals')
    def test_tickerbase_get_income_stmt(self, mock_fundamentals):
        mock_fundamentals.return_value.financials.get_income_time_series.return_value = pd.DataFrame({'Year': ['2021']})
        self.ticker_base._fundamentals = mock_fundamentals.return_value
        result = self.ticker_base.get_income_stmt()
        self.assertIsInstance(result, pd.DataFrame)

    @patch('yfinance.base.TickerBase._lazy_load_price_history')
    def test_tickerbase_get_dividends(self, mock_price_history):
        mock_price_history.return_value.get_dividends.return_value = pd.Series([0.22, 0.23, 0.25])
        result = self.ticker_base.get_dividends()
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(len(result), 3)

    @patch('yfinance.base.TickerBase._lazy_load_price_history')
    def test_tickerbase_get_splits(self, mock_price_history):
        mock_price_history.return_value.get_splits.return_value = pd.Series([1, 2, 4])
        result = self.ticker_base.get_splits()
        self.assertIsInstance(result, pd.Series)
        self.assertEqual(result.iloc[0], 1)

    @patch('yfinance.base.Analysis')
    def test_tickerbase_get_eps_trend(self, mock_analysis):
        mock_analysis.return_value.eps_trend = pd.DataFrame({'0q': ['trend1']})
        self.ticker_base._analysis = mock_analysis.return_value
        result = self.ticker_base.get_eps_trend()
        self.assertIsInstance(result, pd.DataFrame)

    @patch('yfinance.base.TickerBase.get_news')
    def test_tickerbase_get_news(self, mock_get_news):
        mock_get_news.return_value = [{'title': 'Stock rises'}]
        result = self.ticker_base.get_news()
        self.assertIsInstance(result, list)
        self.assertEqual(result[0]['title'], 'Stock rises')

    @patch('yfinance.base.TickerBase.get_shares_full')
    def test_tickerbase_get_shares_full(self, mock_get_shares_full):
        mock_get_shares_full.return_value = pd.Series([1.5e9, 1.6e9], index=pd.to_datetime(['2021-01-01', '2022-01-01']))
        result = self.ticker_base.get_shares_full()
        self.assertIsInstance(result, pd.Series)

    @patch('yfinance.base.Fundamentals')
    def test_tickerbase_get_earnings(self, mock_fundamentals):
        mock_fundamentals.return_value.earnings = {'yearly': pd.DataFrame({'Year': ['2021'], 'Earnings': [1000]})}
        self.ticker_base._fundamentals = mock_fundamentals.return_value
        result = self.ticker_base.get_earnings()
        self.assertIsInstance(result, pd.DataFrame)

    @patch('yfinance.base.TickerBase.get_earnings_dates')
    def test_tickerbase_get_earnings_dates(self, mock_get_earnings_dates):
        mock_get_earnings_dates.return_value = pd.DataFrame({'Earnings Date': ['2021-10-20']})
        result = self.ticker_base.get_earnings_dates()
        self.assertIsInstance(result, pd.DataFrame)

    @patch('yfinance.base.TickerBase._fetch_ticker_tz')
    def test_tickerbase_fetch_ticker_tz(self, mock_fetch_tz):
        mock_fetch_tz.return_value = 'America/New_York'
        tz = self.ticker_base._fetch_ticker_tz(proxy=None, timeout=10)
        self.assertEqual(tz, 'America/New_York')
        mock_fetch_tz.assert_called_once()

    # --- Cache Tests ---
    def test_cache_instance(self):
        cache = Cache()
        self.assertIsInstance(cache, Cache)
        self.assertEqual(cache.lookup('ticker'), None)

    # --- Const Tests ---
    def test_const_values(self):
        self.assertIsInstance(_BASE_URL_, str)
        self.assertIsInstance(_ROOT_URL_, str)

    # --- Data Tests ---
    @patch('yfinance.data.requests.get')
    def test_yfdata_cache_get(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {"key": "value"}
        mock_get.return_value = mock_response

        yfdata = YfData()
        result = yfdata.cache_get("http://example.com")
        self.assertEqual(result.json(), {"key": "value"})

    @patch('yfinance.data.requests.get', side_effect=requests.exceptions.RequestException)
    def test_yfdata_cache_get_exception(self, mock_get):
        yfdata = YfData()
        with self.assertRaises(requests.exceptions.RequestException):
            yfdata.cache_get("http://example.com")

    # --- Exceptions Tests ---
    def test_yf_earnings_date_missing(self):
        with self.assertRaises(YFEarningsDateMissing):
            raise YFEarningsDateMissing("AAPL")

    # --- MultiTicker Tests ---
    @patch('yfinance.multi.Ticker')
    def test_multi_ticker_init(self, mock_ticker):
        tickers = MultiTicker('AAPL MSFT')
        self.assertIsInstance(tickers, MultiTicker)

    # --- Ticker Tests ---
    @patch('yfinance.data.YfData')
    def test_ticker_instance(self, mock_data):
        ticker = Ticker('AAPL')
        self.assertIsInstance(ticker, Ticker)

    # --- Tickers Tests ---
    @patch('yfinance.multi.Ticker')
    def test_tickers_instance(self, mock_ticker):
        tickers = Tickers('AAPL MSFT')
        self.assertIsInstance(tickers, Tickers)

    # --- Utils Tests ---
    def test_utils_is_isin(self):
        result = is_isin('US0378331005')
        self.assertTrue(result)

    @patch('yfinance.utils.requests.get')
    def test_utils_get_ticker_by_isin(self, mock_get):
        mock_response = MagicMock()
        mock_response.json.return_value = {"quoteResponse": {"result": [{"symbol": "AAPL"}]}}
        mock_get.return_value = mock_response

        result = get_ticker_by_isin('US0378331005')
        self.assertEqual(result, 'AAPL')

    @patch('yfinance.utils.requests.get', side_effect=requests.exceptions.RequestException)
    def test_utils_get_ticker_by_isin_exception(self, mock_get):
        with self.assertRaises(requests.exceptions.RequestException):
            get_ticker_by_isin('US0378331005')

    def test_utils_get_yf_logger(self):
        logger = get_yf_logger()
        self.assertIsNotNone(logger)


if __name__ == '__main__':
    unittest.main()
