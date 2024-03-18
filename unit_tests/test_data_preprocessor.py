import unittest
import numpy as np
import pandas as pd
from data_preprocessing import DataPreprocessor

class TestDataPreprocessor(unittest.TestCase):
    def setUp(self):
        self.numeric_features = ['Age', 'Fare']
        self.categorical_features = ['Embarked', 'Sex']
        self.target_feature = 'Survived'

        self.train_df = pd.DataFrame({
            'Age': [22.0, np.nan, 25.0, 24.0],
            'Fare': [7.25, 71.2833, 8.05, 0],
            'Embarked': ['S', 'C', 'S', 'S'],
            'Sex': ['male', 'female', 'male', 'female'],
            'Survived': [0, 1, 1, 0]
        })

        self.test_df = pd.DataFrame({
            'Age': [30.0, np.nan, 27.0],
            'Fare': [15.0, 33.0, 20.0],
            'Embarked': ['C', 'S', 'Q'],
            'Sex': ['female', 'female', 'male']
        })

        self.preprocessor = DataPreprocessor(
            numeric_features=self.numeric_features,
            categorical_features=self.categorical_features,
            target_feature=self.target_feature
        )

    def test_preprocess(self):
        X_train_transformed, y_train, X_test_transformed = self.preprocessor.preprocess(self.train_df, self.test_df)

        # Check if y_train is correctly returned
        np.testing.assert_array_equal(y_train, self.train_df[self.target_feature].values)

        self.assertEqual(X_train_transformed.shape[0], 4)  # Number of rows in train
        self.assertEqual(X_test_transformed.shape[0], 3)  # Number of rows in test

        expected_feature_count = len(self.numeric_features) + len(np.unique(self.train_df['Embarked'])) + len(np.unique(self.train_df['Sex']))
        self.assertEqual(X_train_transformed.shape[1], expected_feature_count)
        self.assertEqual(X_test_transformed.shape[1], expected_feature_count)

        with self.assertRaises(TypeError):
            # TypeError expected when no input passed
            self.preprocessor.preprocess()

if __name__ == '__main__':
    unittest.main()
