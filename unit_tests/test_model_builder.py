import unittest
from xgboost import XGBClassifier
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from model_builder import ModelComparer


class TestModelComparer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Create a synthetic binary classification dataset
        cls.X, cls.y = make_classification(n_samples=100, n_features=20, n_informative=2, n_redundant=10, random_state=42)

        cls.models_params = {
            XGBClassifier(use_label_encoder=False, eval_metric='logloss'): {
                'n_estimators': [200], 'max_depth': [5, 10], 'learning_rate': [0.01, 0.1], 'subsample': [0.8, 1]
            },
            RandomForestClassifier(): {
                'n_estimators': [200], 'max_depth': [5, 10]
            },
            LogisticRegression(): {
                'C': [0.1, 1]
            }
        }


    def test_compare_models(self):
        comparer = ModelComparer(self.models_params)
        best_model = comparer.compare_models(self.X, self.y)

        # Check if a best model is selected
        self.assertIsNotNone(best_model, "No best model was selected.")

        # Ensure the best model is an instance of a provided model class
        model_classes = [model.__class__ for model in self.models_params.keys()]
        self.assertIn(type(best_model), model_classes, "Best model is not an instance of the provided model classes.")

        # Check if best score is reasonable
        self.assertGreaterEqual(comparer.best_score, 0.5, "Best score is unexpectedly low.")

        # Ensure model_scores is populated
        self.assertGreater(len(comparer.model_scores), 0, "Model scores list is empty.")

        # Prediction length test
        predictions = comparer.predict(self.X)
        self.assertEqual(len(predictions), len(self.y), "Predictions length does not match the labels length.")

        with self.assertRaises(TypeError):
            # TypeError expected when no input passed
            ModelComparer()

if __name__ == '__main__':
    unittest.main()
