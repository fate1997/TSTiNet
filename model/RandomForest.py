from sklearn.ensemble import RandomForestRegressor
import torch
from util import data_preprocessing, save_outputs
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
import sklearn.metrics as metrics
import os

"""Extract train_evluate data from excel"""
# data preprocessing
device = torch.device('cpu')
data_path = '../data'
X_train, X_val, X_test, y_train, y_val, y_test = data_preprocessing(data_path, device)
X_train, X_val, X_test, y_train, y_val, y_test = \
    X_train.numpy(), X_val.numpy(), X_test.numpy(), y_train.numpy(), y_val.numpy(), y_test.numpy()

X_train = np.concatenate([X_train, X_val], axis=0)
y_train = np.concatenate([y_train, y_val], axis=0)

"""
# tuning max_features, max_depth, min_samples_split, min_samples_leaf by RandomSearchCV

max_features = [20, 30, 40, 50, 60]
max_depth = [50, 75, 100, 125, 150]
min_samples_split = [2, 5, 10, 15, 20]
min_samples_leaf = [1, 2, 5, 10, 15]
n_estimators = [200]
random_grid = {'max_features': max_features,
               'n_estimators': n_estimators,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf}

rf = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=50,
                               scoring='neg_mean_squared_error')
rf_random.fit(X_train, y_train)
print(f'best model: {rf_random.best_estimator_}')
"""

# tuning n_estimators
rf_estimators = RandomForestRegressor(max_features=40, max_depth=100)
rf_random = RandomizedSearchCV(estimator=rf_estimators,
                               scoring='neg_mean_squared_error',
                               param_distributions={'n_estimators': np.linspace(start=10, stop=500, num=50, dtype=int)},
                               n_iter=50)
rf_random.fit(X_train, y_train)

y_pred = rf_random.predict(X_test)
y_pred_exp = np.exp(y_pred)
y_test_exp = np.exp(y_test)

print(f'test AARD = {metrics.mean_absolute_percentage_error(y_test_exp, y_pred_exp)}, '
      f'test R2 = {metrics.r2_score(y_test_exp, y_pred_exp)}')

print(f'best model: {rf_random.best_estimator_}')

test_results = {'viscosity_test': y_test_exp.tolist(),
                'viscosity_test_pred': y_pred_exp.tolist()}

train_results = {'viscosity_train': np.exp(y_train).tolist(),
                 'viscosity_train_pred': np.exp(rf_random.predict(X_train)).tolist()}

save_outputs([test_results, train_results],
             ['test_results', 'train_results'],
             suffix=os.path.basename(__file__).split(".")[0],
             save_path='../outputs/other results/')
