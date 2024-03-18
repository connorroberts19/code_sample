from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import KNNImputer

class DataPreprocessor:
    def __init__(self, numeric_features, categorical_features, target_feature=None):
        self.numeric_features = numeric_features
        self.categorical_features = categorical_features
        self.target_feature = target_feature
        self.setup_preprocessor()

    def setup_preprocessor(self):
        '''Sets up preprocessing pipelines for both numerical and categorical data.'''
        numeric_transformer = Pipeline(steps=[
            ('imputer', KNNImputer(n_neighbors=5)), # KNN Imputer to handle unknown numerical data
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.numeric_features),
                ('cat', categorical_transformer, self.categorical_features)
            ])

    def preprocess(self, train_df, test_df):
        '''
        Fits the preprocessors to the training data and transforms both training and testing data.
        Returns the transformed training and testing datasets.
        '''
        # Separate features and target from training data
        X_train = train_df.drop(self.target_feature, axis=1)
        y_train = train_df[self.target_feature]

        # Fit and transform the training data
        X_train_transformed = self.preprocessor.fit_transform(X_train)

        #Just transform the test data
        X_test_transformed = self.preprocessor.transform(test_df)

        return X_train_transformed, y_train, X_test_transformed
