import unittest
import numpy as np
from metrics_evaluation import ModelMetrics

class TestModelMetrics(unittest.TestCase):
    def setUp(self):
        self.y_true = np.array([0, 1, 0, 1])
        self.y_pred = np.array([0, 1, 1, 0])
        self.y_score = np.array([0.1, 0.9, 0.6, 0.2])

    def test_report_metrics(self):
        '''Test calculation of metrics. This test checks for errors but does not assert values.'''
        metrics = ModelMetrics(self.y_true, self.y_pred, self.y_score)
        try:
            metrics.report_metrics()
        except Exception as e:
            self.fail(f"report_metrics raised an exception {e}")

        with self.assertRaises(TypeError):
            # TypeError expected when no input passed
            ModelMetrics()

if __name__ == '__main__':
    unittest.main()
