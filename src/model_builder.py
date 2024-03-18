from sklearn.model_selection import GridSearchCV

class ModelComparer:
    def __init__(self, models_params):
        '''
        models_params: A dictionary where keys are model instances and values are dictionaries
                       of parameter grids for GridSearchCV.
        '''
        self.models_params = models_params
        self.best_model = None
        self.best_score = -1
        self.best_params = None
        self.model_scores = []

    def compare_models(self, X_train, y_train):
        '''Compare models using GridSearchCV for parameter optimization and select the best model.'''
        for model, params in self.models_params.items():
            print(f"Training and tuning {model.__class__.__name__}...")
            grid_search = GridSearchCV(model, params, cv=5, scoring='accuracy')
            grid_search.fit(X_train, y_train)
            score = grid_search.best_score_
            print(f"{model.__class__.__name__} best score: {score}")
            self.model_scores.append((model.__class__.__name__, score, grid_search.best_params_))

            if score > self.best_score:
                self.best_score = score
                self.best_model = grid_search.best_estimator_
                self.best_params = grid_search.best_params_

        print(f"Best model: {self.best_model.__class__.__name__} with score {self.best_score} and params {self.best_params}")
        return self.best_model

    def predict(self, X_test):
        '''Make predictions using the best model.'''
        return self.best_model.predict(X_test)