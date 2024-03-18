import unittest
import pandas as pd
from feature_engineering import FeatureEngineer

class TestFeatureEngineer(unittest.TestCase):
    def test_extract_title(self):
        '''Test title extraction from names.'''
        df = pd.DataFrame({
            'Name': ['Doe, Mr. John', 'Smith, Mrs. Jane', 'Brown, Miss. Ann', 'White, Rev. Mark']
        })
        df_test = FeatureEngineer.extract_title(df)
        expected_titles = ['Mr', 'Mrs', 'Miss', 'Rev']
        self.assertTrue((df_test['Title'] == expected_titles).all())

        df = pd.DataFrame({
            'Name': ['Mr. John Doe']
        })
        df_test = FeatureEngineer.extract_title(df)
        expected_titles = ['Mr']

        # When format is incorrect, expect output to be wrong
        self.assertNotEqual(df_test['Title'].values[0], expected_titles[0])

        with self.assertRaises(TypeError):
            # TypeError expected when no input passed
            FeatureEngineer.extract_title()



    def test_add_family_size_and_alone(self):
        '''Test family size and alone feature addition.'''
        df = pd.DataFrame({
            'SibSp': [1, 0, 4],
            'Parch': [0, 2, 1]
        })
        df_test = FeatureEngineer.add_family_size_and_alone(df)
        self.assertTrue((df_test['FamilySize'] == [2, 3, 6]).all())
        self.assertTrue((df_test['IsAlone'] == [0, 0, 0]).all())

        with self.assertRaises(TypeError):
            # TypeError expected when no input passed
            FeatureEngineer.add_family_size_and_alone()

    def test_extract_cabin_level(self):
        '''Test cabin level extraction.'''
        df = pd.DataFrame({
            'Cabin': [None, 'C123', 'B45']
        })
        df_test = FeatureEngineer.extract_cabin_level(df)
        expected_levels = ['U', 'C', 'B']  # 'U' for unknown
        self.assertTrue((df_test['CabinLevel'] == expected_levels).all())

        with self.assertRaises(TypeError):
            # TypeError expected when no input passed
            FeatureEngineer.extract_cabin_level()

    def test_bin_fare(self):
        '''Test fare binning into quartiles.'''
        df = pd.DataFrame({
            'Fare': [10, 50, 100, 500]
        })
        df_test, _ = FeatureEngineer.bin_fare(df)
        # Verify that bins are correctly assigned
        self.assertIn('FareBin', df_test.columns)

        with self.assertRaises(TypeError):
            # TypeError expected when no input passed
            FeatureEngineer.bin_fare()

    def test_bin_age(self):
        '''Test age binning into categories.'''
        df = pd.DataFrame({
            'Age': [5, 17, 25, 55, 80]
        })
        df_test = FeatureEngineer.bin_age(df)
        self.assertIn('AgeBin', df_test.columns)

        with self.assertRaises(TypeError):
            # TypeError expected when no input passed
            FeatureEngineer.bin_age()

if __name__ == '__main__':
    unittest.main()
