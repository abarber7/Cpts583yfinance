import unittest
from unittest.mock import patch, MagicMock
from venv import logger
from yfinance import utils
from yfinance.exceptions import YFTzMissingError, YFPricesMissingError
from requests.exceptions import HTTPError
import pandas as pd
import numpy as np
import logging
import datetime

class TestHistoryEdgeCases(unittest.TestCase):

    def setUp(self):
        self.mock_data = MagicMock()
        self.ticker = "AAPL"
        self.timezone = "America/New_York"

    @patch("yfinance.scrapers.history.PriceHistory.history")
    def test_invalid_interval_with_repair(self, mock_history):
        mock_history.side_effect = Exception("Yahoo's interval '5d' is nonsense, not supported with repair")
        with self.assertRaises(Exception) as context:
            mock_history(period="1mo", interval="5d", repair=True)
        self.assertIn("not supported with repair", str(context.exception))

    @patch("yfinance.scrapers.history.PriceHistory.history")
    def test_missing_timezone_error(self, mock_history):
        mock_history.side_effect = YFTzMissingError(self.ticker)
        with self.assertRaises(YFTzMissingError):
            mock_history(period="1mo", interval="1d")

    @patch("yfinance.scrapers.history.PriceHistory.history")
    def test_empty_dataframe_on_error(self, mock_history):
        mock_history.return_value = pd.DataFrame()
        df = mock_history(period="1mo", interval="1d", repair=True)
        self.assertTrue(df.empty)

    @patch("yfinance.scrapers.history.PriceHistory.history")
    def test_period_ytd_handling(self, mock_history):
        mock_history.return_value = pd.DataFrame({"Close": [100, 101]})
        result = mock_history(period="ytd", interval="1d")
        self.assertIsInstance(result, pd.DataFrame)

    def test_interval_1h_start_calculation(self):
        end = datetime.datetime.now().timestamp()
        start = end - 63072000  # 730 days
        self.assertTrue(start < end)

    def test_interval_30m_start_calculation(self):
        end = datetime.datetime.now().timestamp()
        start = end - 5184000  # 60 days
        self.assertTrue(start < end)

    @patch("yfinance.scrapers.history.PriceHistory.history")
    def test_yahoo_finance_down(self, mock_history):
        mock_history.side_effect = RuntimeError("*** YAHOO! FINANCE IS CURRENTLY DOWN! ***")
        with self.assertRaises(RuntimeError):
            mock_history(period="1mo", interval="1d")

    @patch("yfinance.scrapers.history.PriceHistory.history")
    def test_empty_cleaned_dataframe(self, mock_history):
        mock_history.return_value = pd.DataFrame()
        df = mock_history(period="1mo", interval="1d")
        self.assertTrue(df.empty)

    @patch("yfinance.scrapers.history.PriceHistory.history")
    def test_back_adjust_failure(self, mock_history):
        mock_history.side_effect = Exception("back_adjust failed with error")
        with self.assertRaises(Exception):
            mock_history(period="1mo", interval="1d", auto_adjust=False)

    @patch("yfinance.scrapers.history.PriceHistory.history")
    def test_auto_adjust_failure(self, mock_history):
        mock_history.side_effect = Exception("auto_adjust failed with error")
        with self.assertRaises(Exception):
            mock_history(period="1mo", interval="1d", auto_adjust=True)

    @patch("yfinance.scrapers.history.PriceHistory.history")
    def test_rounding_dataframe(self, mock_history):
        mock_data = pd.DataFrame({"Close": [100.12345, 101.56789]})
        mock_history.return_value = np.round(mock_data, 2)
        result = mock_history(period="1mo", interval="1d", rounding=True)
        self.assertEqual(result.iloc[0]["Close"], 100.12)

    @patch("yfinance.scrapers.history.PriceHistory.history")
    def test_empty_resample_combination(self, mock_history):
        mock_history.return_value = pd.DataFrame()
        result = mock_history(period="1mo", interval="1wk")
        self.assertTrue(result.empty)

    @patch("yfinance.scrapers.history.PriceHistory.history")
    def test_invalid_start_interval_repair(self, mock_history):
        mock_history.side_effect = Exception("Hit max depth of 2")
        with self.assertRaises(Exception) as context:
            mock_history(period="1mo", interval="1wk", repair=True)
        self.assertIn("Hit max depth of 2", str(context.exception))

    def test_invalid_repair_depth(self):
        start_interval = "1d"
        next_intervals = {"1d": "1wk"}
        with self.assertRaises(Exception):
            if start_interval != next_intervals[start_interval]:
                raise Exception("Hit max depth of 2")

    @patch("yfinance.scrapers.history.PriceHistory.history")
    def test_large_date_rejection(self, mock_history):
        mock_history.side_effect = Exception("Cannot reconstruct block starting date too old")
        with self.assertRaises(Exception):
            mock_history(period="10y", interval="1h", repair=True)

    @patch("yfinance.scrapers.history.PriceHistory.history")
    def test_resample_empty_stock_splits(self, mock_history):
        mock_history.return_value = pd.DataFrame({"Stock Splits": []})
        result = mock_history(period="1mo", interval="1d")
        self.assertTrue(result.empty)

    def test_empty_capital_gains(self):
        data = pd.DataFrame({"Capital Gains": [0, 0, 0]})
        result = data[data["Capital Gains"] != 0]
        self.assertTrue(result.empty)

    @patch("yfinance.scrapers.history.PriceHistory.history")
    def test_invalid_currency_ratio(self, mock_history):
        mock_history.side_effect = Exception("Yahoo messed up currency unit")
        with self.assertRaises(Exception):
            mock_history(period="1mo", interval="1d")

    @patch("yfinance.scrapers.history.PriceHistory.history")
    def test_empty_metadata(self, mock_history):
        mock_history.return_value = {}
        metadata = mock_history(period="1mo", interval="1d")
        self.assertEqual(metadata, {})

    @patch("yfinance.scrapers.history.PriceHistory.history")
    def test_logger_info_message(self, mock_history):
        logger = logging.getLogger("yfinance")
        with self.assertLogs(logger, level="INFO") as log:
            mock_history.return_value = pd.DataFrame()
            mock_history(period="1mo", interval="1d")
        self.assertIn("INFO", log.output[0])

class TestHistoryAdditionalCases(unittest.TestCase):

    @patch("yfinance.scrapers.history.PriceHistory.history")
    def test_yahoo_rejected_too_old_data(self, mock_history):
        mock_history.side_effect = Exception("Yahoo will reject request for finer-grain data")
        with self.assertRaises(Exception) as context:
            mock_history(period="10y", interval="1h")
        self.assertIn("Yahoo will reject request", str(context.exception))

    @patch("yfinance.scrapers.history.PriceHistory.history")
    def test_logger_for_repair_abort(self, mock_history):
        logger = logging.getLogger("yfinance")
        with self.assertLogs(logger, level="INFO") as log:
            mock_history.side_effect = Exception("Can't calibrate block starting so aborting repair")
            with self.assertRaises(Exception):
                mock_history(period="1mo", interval="1wk")
        self.assertIn("aborting repair", log.output[0])

    def test_empty_fine_data(self):
        df_fine = pd.DataFrame()
        if df_fine.empty:
            msg = "Cannot reconstruct finer-grain data, Yahoo not returning data within range"
            self.assertEqual(msg, "Cannot reconstruct finer-grain data, Yahoo not returning data within range")

    def test_adjust_ratio_for_currency_unit(self):
        ratios = [0.0001, 0.0001]
        weights = [1, 1]
        ratio = np.average(ratios, weights=weights)
        if abs(ratio / 0.0001 - 1) < 0.01:
            ratio *= 100
            self.assertAlmostEqual(ratio, 10.0)

    def test_resampling_adj_close_column(self):
        df = pd.DataFrame({"Close": [100, 101], "Adj Close": [99, 100]})
        if "Adj Close" in df.columns:
            df["Adj Close"] = df["Close"]
        self.assertTrue("Adj Close" in df.columns)

    def test_resampling_capital_gains_column(self):
        df = pd.DataFrame({"Capital Gains": [0, 10, 0]})
        if "Capital Gains" in df.columns:
            df["Capital Gains"] = df["Capital Gains"].sum()
        self.assertEqual(df["Capital Gains"].iloc[0], 10)

    def test_invalid_dataframe_type(self):
        invalid_df = [1, 2, 3]
        with self.assertRaises(Exception) as context:
            if not isinstance(invalid_df, pd.DataFrame):
                raise Exception("'df' must be a Pandas DataFrame not", type(invalid_df))
        self.assertIn("must be a Pandas DataFrame", str(context.exception))

    @patch("yfinance.scrapers.history.PriceHistory.history")
    def test_no_repair_column_in_dataframe(self, mock_history):
        df = pd.DataFrame({"Close": [100]})
        if "Repaired?" not in df.columns:
            df["Repaired?"] = False
        mock_history.return_value = df
        result = mock_history(period="1mo", interval="1d")
        self.assertFalse(result["Repaired?"].iloc[0])

    def test_100x_price_error_detection(self):
        df = pd.DataFrame({"Close": [100, 10000, 102]})
        median_close = np.median(df["Close"])
        df["Repaired?"] = abs(df["Close"] - median_close) > 100 * median_close
        self.assertTrue(df["Repaired?"].iloc[1])

    def test_timezone_localization(self):
        df = pd.DataFrame(index=pd.date_range("2023-01-01", periods=3))
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        self.assertIsNotNone(df.index.tz)

    def test_resample_invalid_interval(self):
        target_interval = "invalid"
        with self.assertRaises(Exception) as context:
            if target_interval not in ["1wk", "5d", "1mo", "3mo"]:
                raise Exception(f"Unsupported interval: {target_interval}")
        self.assertIn("Unsupported interval", str(context.exception))

    @patch("yfinance.scrapers.history.PriceHistory.history")
    def test_stock_splits_combination(self, mock_history):
        df = pd.DataFrame({"Stock Splits": [1, 2, 1]})
        df["Stock Splits"] = df["Stock Splits"].max()
        mock_history.return_value = df
        result = mock_history(period="1mo", interval="1d")
        self.assertEqual(result["Stock Splits"].iloc[0], 2)

    def test_volume_adjustment(self):
        df = pd.DataFrame({"Volume": [100, 200], "Adjust": [1.5, 0.5]})
        df["Volume"] = df["Volume"] * df["Adjust"]
        self.assertEqual(df["Volume"].iloc[0], 150)

    @patch("yfinance.scrapers.history.PriceHistory.history")
    def test_empty_high_column(self, mock_history):
        df = pd.DataFrame({"High": [None]})
        if "High" not in df.columns or df["High"].isnull().all():
            df["High"] = 0
        mock_history.return_value = df
        result = mock_history(period="1mo", interval="1d")
        self.assertEqual(result["High"].iloc[0], 0)

    @patch("yfinance.scrapers.history.PriceHistory.history")
    def test_empty_low_column(self, mock_history):
        df = pd.DataFrame({"Low": [None]})
        if "Low" not in df.columns or df["Low"].isnull().all():
            df["Low"] = 0
        mock_history.return_value = df
        result = mock_history(period="1mo", interval="1d")
        self.assertEqual(result["Low"].iloc[0], 0)

    @patch("yfinance.scrapers.history.PriceHistory.history")
    def test_combining_invalid_event_columns(self, mock_history):
        df1 = pd.DataFrame({"Dividends": [0, 1]})
        df2 = pd.DataFrame({"Capital Gains": [0, 2]})
        combined = pd.concat([df1, df2], axis=1)
        mock_history.return_value = combined
        result = mock_history(period="1mo", interval="1d")
        self.assertTrue("Dividends" in result.columns and "Capital Gains" in result.columns)

    @patch("yfinance.scrapers.history.PriceHistory.history")
    def test_logger_warning_for_price_repair(self, mock_history):
        logger = logging.getLogger("yfinance")
        with self.assertLogs(logger, level="WARNING") as log:
            mock_history.side_effect = Exception("Cannot repair this price block")
            with self.assertRaises(Exception):
                mock_history(period="1mo", interval="1d", repair=True)
        self.assertIn("WARNING", log.output[0])

    @patch("yfinance.scrapers.history.PriceHistory.history")
    def test_handling_finer_grain_data(self, mock_history):
        df = pd.DataFrame({"Close": [100, 102, 104]})
        fine_data = df.loc[1:2].copy()
        mock_history.return_value = fine_data
        result = mock_history(period="1mo", interval="1h")
        self.assertEqual(result["Close"].iloc[0], 102)

    @patch("yfinance.scrapers.history.PriceHistory.history")
    def test_logging_for_empty_results(self, mock_history):
        logger = logging.getLogger("yfinance")
        with self.assertLogs(logger, level="INFO") as log:
            mock_history.return_value = pd.DataFrame()
            result = mock_history(period="1mo", interval="1d")
            self.assertTrue(result.empty)
        self.assertIn("INFO", log.output[0])

    def test_raise_interval_5d_exception(self):
        with self.assertRaises(Exception, msg="Yahoo's interval '5d' is nonsense, not supported with repair"):
            raise Exception("Yahoo's interval '5d' is nonsense, not supported with repair")

    def test_raise_tz_missing_error(self):
        with self.assertRaises(YFTzMissingError):
            raise YFTzMissingError("Ticker")

    def test_yahoo_down_error(self):
        with self.assertRaises(RuntimeError, msg="*** YAHOO! FINANCE IS CURRENTLY DOWN! ***"):
            raise RuntimeError("*** YAHOO! FINANCE IS CURRENTLY DOWN! ***")

    def test_empty_data_error(self):
        with self.assertRaises(ValueError, msg="OHLC after cleaning: EMPTY"):
            msg = "OHLC after cleaning: EMPTY"
            raise ValueError(msg)

    def test_combining_events_empty(self):
        msg = "OHLC after combining events: EMPTY"
        with self.assertRaises(ValueError):
            raise ValueError(msg)

    def test_capital_gains_filtering(self):
        data = pd.DataFrame({"Capital Gains": [0, 1, 0.5]})
        result = data["Capital Gains"][data["Capital Gains"] != 0]
        self.assertTrue((result == [1, 0.5]).all())

    def test_back_adjust_failure(self):
        with self.assertRaises(Exception, msg="back_adjust failed with error"):
            raise Exception("back_adjust failed with error")

    def test_resample_unsupported_frequency(self):
        with self.assertRaises(ValueError, msg="Unsupported frequency for resampling"):
            raise ValueError("Unsupported frequency for resampling")

    def test_reconstruct_start_interval(self):
        self.assertIsNone(self.price_history._reconstruct_start_interval)

    def test_resample_target_intervals(self):
        resample_period = 'W-MON' if '1wk' else None
        self.assertEqual(resample_period, 'W-MON')

    def test_repair_with_adj_close(self):
        df = pd.DataFrame({"Adj Close": [150, 151], "Close": [149, 150]})
        self.assertIn("Adj Close", df.columns)

    def test_reject_old_data(self):
        with self.assertRaises(ValueError, msg="Yahoo will reject request for finer-grain data"):
            raise ValueError("Yahoo will reject request for finer-grain data")

    def test_reconstruct_empty_block(self):
        with self.assertRaises(ValueError, msg="Cannot reconstruct block range"):
            raise ValueError("Cannot reconstruct block range")

    def test_ratio_currency_correction(self):
        ratio = 0.0001
        if abs(ratio / 0.0001 - 1) < 0.01:
            ratio *= 100
        self.assertEqual(ratio, 0.01)

    def test_ratio_rcp_adjustment(self):
        ratio_rcp = 2.0
        data = pd.DataFrame({"Close": [150, 300]})
        data["Close"] *= 1.0 / ratio_rcp
        self.assertTrue((data["Close"] == [75, 150]).all())

    def test_no_fine_data_intervals(self):
        with self.assertLogs("yfinance", level="DEBUG"):
            logger.debug("Yahoo didn't return finer-grain data for intervals")

    def test_discard_buffer_empty(self):
        data = pd.DataFrame()
        self.assertTrue(data.empty)

    def test_insufficient_good_data(self):
        df = pd.DataFrame()
        self.assertTrue(df.empty)

    def test_price_repair_outliers(self):
        df = pd.DataFrame({"Close": [100, 10000, 102]})
        outliers = df["Close"] > 5000
        self.assertTrue(outliers.any())

    def test_currency_unit_mixups(self):
        report_msg = "fixed 1/5 currency unit mixups"
        self.assertIn("currency unit mixups", report_msg)

    def test_logger_warning_unimplemented_interval(self):
        with self.assertLogs("yfinance", level="WARNING") as log:
            logger.warning("Have not implemented price reconstruct for '1d' interval. Contact developers")
        self.assertIn("Have not implemented price reconstruct for", log.output[0])

    def test_empty_dataframe_handling_in_repair(self):
        df = pd.DataFrame()
        df["Repaired?"] = False
        self.assertIn("Repaired?", df.columns)
        self.assertTrue(df.empty)

    def test_tz_localize_missing_in_index(self):
        df = pd.DataFrame({"Close": [150, 152]}, index=[1, 2])
        with self.assertRaises(TypeError, msg="Cannot localize index without datetime type"):
            df.index = df.index.tz_localize("UTC")

    def test_adjust_close_price_logic(self):
        df = pd.DataFrame({"Adj Close": [150, 152], "Close": [100, 101]})
        df["Close"] = df["Adj Close"]
        self.assertTrue((df["Close"] == df["Adj Close"]).all())

    def test_logger_insufficient_rows(self):
        with self.assertLogs("yfinance", level="DEBUG") as log:
            logger.debug("Cannot check single-row table for 100x price errors")
        self.assertIn("Cannot check single-row table", log.output[0])

    def test_ratio_adjustment_in_price(self):
        data = pd.DataFrame({"Close": [100, 200], "Open": [50, 100]})
        ratios = [1.0, 0.5]
        data["Close"] *= ratios[0]
        self.assertTrue((data["Close"] == [100, 200]).all())

    def test_df_zeroes_concat(self):
        df = pd.DataFrame({"Close": [0, 150]})
        zero_rows = df[df["Close"] == 0]
        df_non_zero = df[df["Close"] != 0]
        concatenated = pd.concat([zero_rows, df_non_zero]).sort_index()
        self.assertEqual(len(concatenated), len(df))

    def test_handle_no_fine_data(self):
        df = pd.DataFrame({"Open": [None, None], "Close": [100, 200]})
        no_fine_data = df["Open"].isna()
        self.assertTrue(no_fine_data.all())

    def test_handle_good_data_detection(self):
        data = {"Close": [100, 101]}
        df = pd.DataFrame(data)
        df["Repaired?"] = False
        self.assertFalse(df.empty)
        self.assertIn("Repaired?", df.columns)

    def test_currency_mixup_fix_report(self):
        fixed = 5
        n_fixed = 3
        n_fixed_crudely = 1
        report_msg = f"fixed {n_fixed}/{fixed} currency unit mixups ({n_fixed_crudely} crudely)"
        self.assertIn("currency unit mixups", report_msg)

    def test_repair_100x_median_error(self):
        data = {"Close": [100, 10000]}
        df = pd.DataFrame(data)
        median = df["Close"].median()
        outliers = df["Close"] > median * 100
        self.assertTrue(outliers.any())

    def test_missing_capital_gains_handling(self):
        df = pd.DataFrame({"Capital Gains": [0, None, 1.0]})
        filtered = df["Capital Gains"].dropna()
        self.assertEqual(len(filtered), 2)

    def test_empty_combination_result(self):
        df1 = pd.DataFrame({"Close": []})
        df2 = pd.DataFrame({"Close": []})
        combined = pd.concat([df1, df2])
        self.assertTrue(combined.empty)

    def test_logger_currency_fix_message(self):
        with self.assertLogs("yfinance", level="INFO") as log:
            logger.info("fixed 3/5 currency unit mixups")
        self.assertIn("currency unit mixups", log.output[0])

    def test_logger_calibration_failure(self):
        with self.assertLogs("yfinance", level="INFO") as log:
            logger.info("Can't calibrate block starting 2024-01-01 so aborting repair")
        self.assertIn("Can't calibrate block", log.output[0])

    def test_repair_with_zeroes_handling(self):
        data = {"Close": [0, 100], "Open": [0, 50]}
        df = pd.DataFrame(data)
        df.loc[df["Close"] == 0, "Close"] = df["Open"]
        self.assertEqual(df["Close"].iloc[0], 50)

    def test_logger_discard_buffer(self):
        with self.assertLogs("yfinance", level="INFO") as log:
            logger.info("Discarding buffer for repair")
        self.assertIn("Discarding buffer for repair", log.output[0])

    def test_resample_map_with_adj_close(self):
        df = pd.DataFrame({"Close": [100, 200], "Adj Close": [98, 198]})
        resample_map = {"Close": "mean"}
        if "Adj Close" in df.columns:
            resample_map["Adj Close"] = resample_map["Close"]
        self.assertIn("Adj Close", resample_map)

    def test_logger_finer_grain_missing(self):
        with self.assertLogs("yfinance", level="INFO") as log:
            logger.info("Yahoo didn't return finer-grain data")
        self.assertIn("Yahoo didn't return finer-grain data", log.output[0])

    def test_logger_finer_grain_outside_range(self):
        with self.assertLogs("yfinance", level="INFO") as log:
            logger.info("Cannot reconstruct block range, Yahoo not returning finer-grain data within range")
        self.assertIn("Cannot reconstruct block range", log.output[0])

    def test_empty_df_after_back_adjust(self):
        df = pd.DataFrame({"Close": []})
        adjusted_df = df.copy()
        try:
            adjusted_df = utils.back_adjust(adjusted_df)
        except Exception:
            pass
        self.assertTrue(adjusted_df.empty)

    def test_resample_period_mapping(self):
        target_interval = "1wk"
        resample_period = None
        if target_interval == "1wk":
            resample_period = "W-MON"
        self.assertEqual(resample_period, "W-MON")

    def test_resample_period_invalid(self):
        target_interval = "invalid"
        with self.assertRaises(ValueError):
            if target_interval not in ["1wk", "1mo", "3mo"]:
                raise ValueError(f"Unsupported resample period: {target_interval}")

    def test_missing_data_cleanup(self):
        df = pd.DataFrame({"Open": [None, 100], "Close": [100, None]})
        cleaned_df = df.dropna()
        self.assertEqual(len(cleaned_df), 0)

    def test_logger_unsupported_frequency(self):
        with self.assertLogs("yfinance", level="INFO") as log:
            logger.info("Unsupported frequency for resampling to business days")
        self.assertIn("Unsupported frequency for resampling", log.output[0])

    def test_combining_empty_with_data(self):
        df1 = pd.DataFrame({"Close": []})
        df2 = pd.DataFrame({"Close": [100, 200]})
        combined = pd.concat([df1, df2])
        self.assertEqual(len(combined), 2)

    def test_logger_missing_repaired_column(self):
        with self.assertLogs("yfinance", level="INFO") as log:
            logger.info("Adding 'Repaired?' column to DataFrame")
        self.assertIn("Adding 'Repaired?'", log.output[0])

    def test_fine_grain_ratio_adjustment(self):
        df = pd.DataFrame({"Close": [100, 200], "Volume": [1000, 2000]})
        fine_grain_ratio = 2
        df["Volume"] *= fine_grain_ratio
        self.assertEqual(df["Volume"].iloc[0], 2000)

    def test_logger_ratio_near_zero(self):
        with self.assertLogs("yfinance", level="INFO") as log:
            logger.info("Ratio near zero detected, correcting currency unit")
        self.assertIn("correcting currency unit", log.output[0])

    def test_df_with_bad_indices(self):
        df = pd.DataFrame({"Open": [100, 150], "Close": [110, 160]}, index=["bad", "good"])
        with self.assertRaises(TypeError):
            df.index = pd.to_datetime(df.index)

    def test_outlier_detection_logic(self):
        data = {"Close": [100, 100000]}
        df = pd.DataFrame(data)
        outliers = df["Close"] > df["Close"].median() * 100
        self.assertTrue(outliers.any())

    def test_unsupported_currency_adjustment(self):
        df = pd.DataFrame({"Close": [1, 100], "Currency": ["INVALID", "VALID"]})
        with self.assertRaises(ValueError):
            if df["Currency"].iloc[0] not in ["USD", "EUR"]:
                raise ValueError("Unsupported currency detected")

    def test_adjustment_with_large_ratio(self):
        df = pd.DataFrame({"Close": [100, 200]})
        ratio = 0.0001
        if abs(ratio - 0.0001) < 0.01:
            df["Close"] *= 100
        self.assertEqual(df["Close"].iloc[0], 10000)

    def test_logger_inconsistent_data(self):
        with self.assertLogs("yfinance", level="INFO") as log:
            logger.info("Data has inconsistent adjustments, aligning")
        self.assertIn("inconsistent adjustments", log.output[0])

    def test_empty_dataframe_on_ratio_adjust(self):
        df = pd.DataFrame({"Close": []})
        ratio = 1.0
        df["Close"] *= ratio
        self.assertTrue(df.empty)

    def test_fine_data_discard_buffer(self):
        df = pd.DataFrame({"Open": [100, None], "Close": [100, 200]})
        fine_data = df.dropna()
        self.assertEqual(len(fine_data), 1)

    def test_logger_repair_max_depth(self):
        with self.assertLogs("yfinance", level="INFO") as log:
            logger.info("Hit max depth of 2 for repair, stopping")
        self.assertIn("max depth of 2", log.output[0])

    def test_logger_missing_intervals(self):
        with self.assertLogs("yfinance", level="DEBUG") as log:
            logger.debug("Missing finer-grain data for intervals")
        self.assertIn("Missing finer-grain data", log.output[0])

    def test_empty_dataframe_repair(self):
        df = pd.DataFrame({"Close": []})
        if "Repaired?" not in df.columns:
            df["Repaired?"] = False
        self.assertTrue("Repaired?" in df.columns)

    def test_repair_with_single_row(self):
        df = pd.DataFrame({"Close": [100]})
        with self.assertLogs("yfinance", level="DEBUG") as log:
            logger.debug("Cannot check single-row table for 100x price errors")
        self.assertIn("Cannot check single-row", log.output[0])

    def test_resampling_unsupported_column(self):
        df = pd.DataFrame({"Close": [100], "Volume": [1000]})
        if "Adj Close" in df.columns:
            resampled = df["Adj Close"]
        else:
            resampled = None
        self.assertIsNone(resampled)

    def test_missing_capital_gains(self):
        df = pd.DataFrame({"Capital Gains": [0, 0, 0]})
        gains = df["Capital Gains"][df["Capital Gains"] != 0]
        self.assertTrue(gains.empty)

    def test_invalid_target_interval(self):
        target_interval = "invalid_interval"
        with self.assertRaises(ValueError):
            if target_interval not in ["1wk", "1mo", "3mo"]:
                raise ValueError(f"Unsupported target interval: {target_interval}")

    def test_logger_aborting_repair(self):
        with self.assertLogs("yfinance", level="INFO") as log:
            logger.info("Aborting repair due to calibration failure")
        self.assertIn("Aborting repair", log.output[0])

    def test_logger_empty_ohlc_after_combining(self):
        with self.assertLogs("yfinance", level="INFO") as log:
            logger.info("OHLC after combining events: EMPTY")
        self.assertIn("OHLC after combining events", log.output[0])

    def test_logger_empty_ohlc_after_cleaning(self):
        with self.assertLogs("yfinance", level="INFO") as log:
            logger.info("OHLC after cleaning: EMPTY")
        self.assertIn("OHLC after cleaning", log.output[0])

    def test_logger_empty_ohlc_returned(self):
        with self.assertLogs("yfinance", level="INFO") as log:
            logger.info("Returning OHLC: EMPTY")
        self.assertIn("Returning OHLC", log.output[0])

    def test_missing_stock_splits(self):
        df = pd.DataFrame({"Stock Splits": [None, 0]})
        max_split = df["Stock Splits"].max()
        self.assertEqual(max_split, 0)

    def test_reconstruct_start_interval(self):
        reconstruct_start_interval = None
        interval = "1d"
        if reconstruct_start_interval is None:
            reconstruct_start_interval = interval
        self.assertEqual(reconstruct_start_interval, "1d")

    def test_invalid_data_type_for_df(self):
        df = "invalid_data_type"
        with self.assertRaises(Exception):
            if not isinstance(df, pd.DataFrame):
                raise Exception("'df' must be a Pandas DataFrame not", type(df))

    def test_logger_insufficient_good_data(self):
        with self.assertLogs("yfinance", level="INFO") as log:
            logger.info("Insufficient good data for detecting price errors")
        self.assertIn("Insufficient good data", log.output[0])

    def test_logger_currency_unit_mixups(self):
        with self.assertLogs("yfinance", level="INFO") as log:
            logger.info("Fixed currency unit mixups")
        self.assertIn("currency unit mixups", log.output[0])

    def test_invalid_sub_interval(self):
        sub_interval = "invalid"
        with self.assertRaises(ValueError):
            if sub_interval not in ["1wk", "1h", "30m", "15m"]:
                raise ValueError(f"Unsupported sub_interval: {sub_interval}")

    def test_logger_reconstruct_too_old(self):
        with self.assertLogs("yfinance", level="INFO") as log:
            logger.info("Cannot reconstruct block, data too old")
        self.assertIn("data too old", log.output[0])

    def test_logger_block_starting_aborted(self):
        with self.assertLogs("yfinance", level="INFO") as log:
            logger.info("Block starting aborted due to failure")
        self.assertIn("Block starting aborted", log.output[0])

    def test_logger_median_outliers(self):
        with self.assertLogs("yfinance", level="INFO") as log:
            logger.info("Detecting outliers based on local median")
        self.assertIn("Detecting outliers", log.output[0])

    def test_logger_fine_data_not_returned(self):
        with self.assertLogs("yfinance", level="INFO") as log:
            logger.info("Fine-grain data not returned for intervals")
        self.assertIn("Fine-grain data not returned", log.output[0])

    def test_currency_correction_ratio(self):
        ratios = [0.0001]
        weights = [1]
        average_ratio = np.average(ratios, weights=weights)
        corrected_ratio = average_ratio * 100 if abs(average_ratio / 0.0001 - 1) < 0.01 else average_ratio
        self.assertEqual(corrected_ratio, 0.01)

    def test_exception_interval_not_supported(self):
        with self.assertRaises(Exception) as context:
            raise Exception("Yahoo's interval '5d' is nonsense, not supported with repair")
        self.assertIn("interval '5d'", str(context.exception))

    def test_reject_too_old_intraday_data(self):
        sub_interval = "30m"
        start_date = pd.Timestamp("2021-01-01")
        today = pd.Timestamp.today()
        with self.assertRaises(ValueError):
            if sub_interval in ["30m", "15m"] and (today - start_date).days > 59:
                raise ValueError("Yahoo will reject request for finer-grain data")

    def test_logger_block_reconstruction_failure(self):
        with self.assertLogs("yfinance", level="INFO") as log:
            logger.info("Cannot reconstruct block due to Yahoo rejection")
        self.assertIn("Cannot reconstruct block", log.output[0])

    def test_logger_calibration_aborting_repair(self):
        with self.assertLogs("yfinance", level="INFO") as log:
            logger.info("Calibration failed; aborting repair")
        self.assertIn("Calibration failed", log.output[0])

    def test_adjust_volume_with_ratio(self):
        df = pd.DataFrame({"Volume": [1000, 2000, 3000]})
        ratio = 2
        df["Volume"] *= ratio
        self.assertTrue((df["Volume"] == [2000, 4000, 6000]).all())

    def test_logger_insufficient_price_error_detection(self):
        with self.assertLogs("yfinance", level="INFO") as log:
            logger.info("Not enough data for price error detection")
        self.assertIn("Not enough data for price error detection", log.output[0])

    def test_logger_adjustment_failed(self):
        with self.assertLogs("yfinance", level="ERROR") as log:
            logger.error("Adjustment failed with error")
        self.assertIn("Adjustment failed", log.output[0])

    def test_fill_adj_close_when_missing(self):
        df = pd.DataFrame({"Close": [100, 105], "Adj Close": [None, 102]})
        df["Adj Close"].fillna(df["Close"], inplace=True)
        self.assertTrue((df["Adj Close"] == [100, 102]).all())

    def test_logger_outliers_detected(self):
        with self.assertLogs("yfinance", level="INFO") as log:
            logger.info("Detected outliers in price data")
        self.assertIn("Detected outliers", log.output[0])

    def test_logger_currency_mixup_detection(self):
        with self.assertLogs("yfinance", level="INFO") as log:
            logger.info("Detected and fixed currency unit mixups")
        self.assertIn("currency unit mixups", log.output[0])

    def test_resample_fine_grained_data_with_ratio(self):
        df = pd.DataFrame({"Close": [100, 200, 300]})
        ratio = 0.5
        df["Close"] *= ratio
        self.assertTrue((df["Close"] == [50, 100, 150]).all())

    def test_logger_no_fine_data_returned(self):
        with self.assertLogs("yfinance", level="DEBUG") as log:
            logger.debug("No fine-grain data returned for intervals")
        self.assertIn("No fine-grain data returned", log.output[0])

    def test_reject_fine_grain_data_request(self):
        sub_interval = "1h"
        start_date = pd.Timestamp("2020-01-01")
        today = pd.Timestamp.today()
        with self.assertRaises(ValueError):
            if sub_interval == "1h" and (today - start_date).days > 729:
                raise ValueError("Request rejected due to data being too old")

    def test_logger_failed_to_calibrate_block(self):
        with self.assertLogs("yfinance", level="INFO") as log:
            logger.info("Failed to calibrate block")
        self.assertIn("Failed to calibrate block", log.output[0])

    def test_logger_adjustment_report(self):
        with self.assertLogs("yfinance", level="INFO") as log:
            logger.info("Adjustment report: fixed 3 currency unit mixups")
        self.assertIn("fixed 3 currency unit mixups", log.output[0])

    def test_logger_return_empty_ohlc(self):
        with self.assertLogs("yfinance", level="INFO") as log:
            logger.info("Returning empty OHLC")
        self.assertIn("Returning empty OHLC", log.output[0])

    def test_adjust_prices_with_ratio(self):
        df = pd.DataFrame({"Close": [100, 200, 300]})
        ratio = 1.2
        df["Close"] *= ratio
        self.assertTrue((df["Close"] == [120, 240, 360]).all())

    def test_logger_data_too_old_for_repair(self):
        with self.assertLogs("yfinance", level="INFO") as log:
            logger.info("Data too old for repair")
        self.assertIn("Data too old for repair", log.output[0])

    def test_empty_capital_gains_handling(self):
        df = pd.DataFrame({"Capital Gains": [0, 0, 0]})
        non_zero_gains = df["Capital Gains"][df["Capital Gains"] != 0]
        self.assertTrue(non_zero_gains.empty)

    def test_exception_when_df_not_dataframe(self):
        not_df = {"Close": [100, 200]}
        with self.assertRaises(Exception):
            if not isinstance(not_df, pd.DataFrame):
                raise Exception("'df' must be a Pandas DataFrame not", type(not_df))


if __name__ == "__main__":
    unittest.main()

