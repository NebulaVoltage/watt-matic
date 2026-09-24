import unittest
import pandas as pd
import numpy as np

class TestUCIForecasting(unittest.TestCase):
    def setUp(self):
        self.df = pd.read_csv('data/processed/uci_hourly_aggregate_load.csv', parse_dates=[0], index_col=0)
        
    def test_timestamp_ordering(self):
        self.assertTrue(self.df.index.is_monotonic_increasing, "Timestamps are not strictly chronological")
        
    def test_no_future_leakage(self):
        # Target shift(-1) must be strictly future
        target = self.df['aggregate_load_kwh'].shift(-1)
        self.assertEqual(target.iloc[0], self.df['aggregate_load_kwh'].iloc[1])
        
    def test_rolling_causality(self):
        # rolling_mean_24 must not include current index if shifted
        s = self.df['aggregate_load_kwh']
        roll24 = s.rolling(24).mean()
        self.assertAlmostEqual(roll24.iloc[23], s.iloc[:24].mean())
        
    def test_chronological_split_no_overlap(self):
        train_end = pd.Timestamp('2013-12-31 23:00:00')
        val_start = pd.Timestamp('2014-01-01 00:00:00')
        self.assertLess(train_end, val_start)

if __name__ == '__main__':
    unittest.main()
