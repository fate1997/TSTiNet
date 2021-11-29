from lightgbm import LGBMRegressor
import torch
from util import data_preprocessing, save_outputs
import os
import numpy as np
import sklearn.metrics as metrics
from sklearn.model_selection import RandomizedSearchCV

"""Extract data"""
# data preprocessing
device = torch.device('cpu')
data_path = '../data'
X_train, X_val, X_test, y_train, y_val, y_test = data_preprocessing(data_path, device)
X_train, X_val, X_test, y_train, y_val, y_test = \
    X_train.numpy(), X_val.numpy(), X_test.numpy(), y_train.numpy(), y_val.numpy(), y_test.numpy()

X_train = np.concatenate([X_train, X_val], axis=0)
y_train = np.concatenate([y_train, y_val], axis=0)

params = {
    'n_estimators': [1000, 2000, 3000, 4000],
    'max_depth': [8, 10, 12, 14],
    'subsample': [0.75, 0.78, 0.8, 0.82, 0.85],
    'learning_rate': [0.1, 0.05, 0.01],
    'colsample_bytree': [0.25, 0.30, 0.35, 0.40, 0.45],
    'subsample_freq': [2, 4, 6],
    'num_leaves': [5, 10, 15, 20, 25]
}

lgb = LGBMRegressor(objective='regression', metric='mse', verbose=0)

lgb_random = RandomizedSearchCV(estimator=lgb, param_distributions=params, verbose=100, n_iter=50,
                                scoring='neg_mean_squared_error', n_jobs=3, random_state=1)

lgb_random.fit(X_train, y_train)
y_pred = lgb_random.predict(X_test)

y_pred_exp = np.exp(y_pred)
y_test_exp = np.exp(y_test)

print(f'test AARD = {metrics.mean_absolute_percentage_error(y_test_exp, y_pred_exp)}, '
      f'test R2 = {metrics.r2_score(y_test_exp, y_pred_exp)}')

print(f'best model: {lgb_random.best_estimator_}')

test_results = {'viscosity_test': y_test_exp.tolist(),
                'viscosity_test_pred': y_pred_exp.tolist()}

train_results = {'viscosity_train': np.exp(y_train).tolist(),
                 'viscosity_train_pred': np.exp(lgb_random.predict(X_train)).tolist()}

save_outputs([test_results, train_results],
             ['test_results', 'train_results'],
             suffix=os.path.basename(__file__).split(".")[0],
             save_path='../outputs/other results/')