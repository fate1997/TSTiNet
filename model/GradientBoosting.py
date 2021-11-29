from sklearn.ensemble import GradientBoostingRegressor
import torch
from util import data_preprocessing, save_outputs
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
import sklearn.metrics as metrics
import os

"""Extract train_evaluate data from excel"""
# data preprocessing
device = torch.device('cpu')
data_path = '../data'
X_train, X_val, X_test, y_train, y_val, y_test = data_preprocessing(data_path, device)
X_train, X_val, X_test, y_train, y_val, y_test = \
    X_train.numpy(), X_val.numpy(), X_test.numpy(), y_train.numpy(), y_val.numpy(), y_test.numpy()

X_train = np.concatenate([X_train, X_val], axis=0)
y_train = np.concatenate([y_train, y_val], axis=0)

params = {
    'n_estimators': [400, 800, 1200, 1600, 2000],
    'max_depth': [4, 8, 10, 12, 14, 16],
    'subsample': [.25, .5, .75, 1],
    'learning_rate': [0.01, 0.03, 0.05, 0.07, 0.1]
}

gbr = GradientBoostingRegressor()
gbr_random = RandomizedSearchCV(estimator=gbr, param_distributions=params, n_iter=10,
                                scoring='neg_mean_squared_error')
gbr_random.fit(X_train, y_train)
y_pred = gbr_random.predict(X_test)


y_pred_exp = np.exp(y_pred)
y_test_exp = np.exp(y_test)

print(f'test AARD = {metrics.mean_absolute_percentage_error(y_test_exp, y_pred_exp)}, '
      f'test R2 = {metrics.r2_score(y_test_exp, y_pred_exp)}')

print(f'best model: {gbr_random.best_estimator_}')

test_results = {'viscosity_test': y_test_exp.tolist(),
                'viscosity_test_pred': y_pred_exp.tolist()}

train_results = {'viscosity_train': np.exp(y_train).tolist(),
                 'viscosity_train_pred': np.exp(gbr_random.predict(X_train)).tolist()}

save_outputs([test_results, train_results],
             ['test_results', 'train_results'],
             suffix=os.path.basename(__file__).split(".")[0],
             save_path='../outputs/other results/')
