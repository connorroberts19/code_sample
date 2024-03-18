import pandas as pd

class FeatureEngineer:
    def __init__(self):
        pass

    @staticmethod
    def extract_title(df):
        '''Extracts title from the Name column.'''
        df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        return df

    @staticmethod
    def add_family_size_and_alone(df):
        '''Adds FamilySize and Alone features.'''
        df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
        df['IsAlone'] = 0
        df.loc[df['FamilySize'] == 1, 'IsAlone'] = 1
        return df

    @staticmethod
    def extract_cabin_level(df):
        '''Extracts the first letter of the Cabin as the CabinLevel.'''
        df['CabinLevel'] = df['Cabin'].apply(lambda x: x[0] if pd.notnull(x) else 'U')  # 'U' for unknown
        return df

    @staticmethod
    def bin_fare(df, fare_bins=None):
        '''Bins Fare into quantiles for normalized comparison. If fare_bins is provided, use it directly.'''
        if fare_bins is None:
            # Apply qcut and capture the bins used
            _, fare_bins = pd.qcut(df['Fare'], 4, retbins=True, labels=False, duplicates='drop')
            df['FareBin'] = pd.cut(df['Fare'], bins=fare_bins, labels=False, include_lowest=True)
        else:
            # Use bins calculated froming training data
            df['FareBin'] = pd.cut(df['Fare'], bins=fare_bins, labels=False, include_lowest=True)
        return df, fare_bins

    @staticmethod
    def bin_age(df):
        '''Bins Age into categories for clearer age group analysis.'''
        df['AgeBin'] = pd.cut(df['Age'], bins=[0, 12, 20, 40, 60, 120], labels=False, right=False)
        return df

    def transform_features(self, train_df, test_df):
        '''Applies all the feature transformations to the training and testing dataset.'''
        train_df = self.extract_title(train_df)
        train_df = self.add_family_size_and_alone(train_df)
        train_df = self.extract_cabin_level(train_df)
        train_df, fare_bins = self.bin_fare(train_df)
        train_df = self.bin_age(train_df)

        # Apply transformations to test_df using parameters from train_df
        test_df = self.extract_title(test_df)
        test_df = self.add_family_size_and_alone(test_df)
        test_df = self.extract_cabin_level(test_df)
        test_df, _ = self.bin_fare(test_df, fare_bins=fare_bins)  # Use fare_bins from train_df
        test_df = self.bin_age(test_df)

        columns_to_drop = ['Name'] # Only the title will be relevant to our model
        train_df.drop(columns_to_drop, axis=1, inplace=True)

        test_df.drop(columns_to_drop, axis=1, inplace=True)

        return train_df, test_df
