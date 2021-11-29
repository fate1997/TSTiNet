import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np


# extract data from csv files
def data_preprocessing(data_path, device='cpu'):
    # train_dataset
    train_validation_test = ['train', 'validation', 'test']
    X, y = {}, {}
    for dataset in train_validation_test:
        path = data_path + '/' + dataset + '.csv'
        df = pd.read_csv(path, encoding='gb18030')

        # extract mole fraction and molecular weight of hba and hbd
        x_hba = torch.tensor(df['XHBA'].values, dtype=torch.float32)
        mw_hba = torch.tensor(df['MwHBA'].values, dtype=torch.float32)

        x_hbd = torch.tensor(df['XHBD'].values, dtype=torch.float32)
        mw_hbd = torch.tensor(df['MwHBD'].values, dtype=torch.float32)

        # extract group information of hba and hbd
        X_hba = torch.tensor(df.iloc[:, 6:51].values, dtype=torch.float32)
        X_hbd = torch.tensor(df.iloc[:, 55:-10].values, dtype=torch.float32)

        # extract component interaction parameters, output (ln(viscosity)) and temperature
        G_ij = torch.tensor(df.iloc[:, -10:-5].values, dtype=torch.float32)
        y[dataset] = torch.tensor(df.iloc[:, -1].values, dtype=torch.float32).reshape(-1, )
        T = torch.tensor(df.iloc[:, -3].values, dtype=torch.float32)

        # concatenate input variables as matrix (num_features=100)
        X[dataset] = torch.cat([X_hba, X_hbd, x_hba.reshape((-1, 1)), x_hbd.reshape((-1, 1)), G_ij,
                                T.reshape((-1, 1)), mw_hba.reshape((-1, 1)), mw_hbd.reshape((-1, 1))], dim=1).to(device)

    # X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=2021)
    return X['train'].to(device), X['validation'].to(device), X['test'].to(device), \
           y['train'].to(device), y['validation'].to(device), y['test'].to(device)


class EarlyStopping:
    """Early stops the training if validation score doesn't improve after a given patience."""

    def __init__(self, patience=100, verbose=False, delta=0,
                 path='../outputs/model_parameters/', model_type='TSTiNet'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.model_type = model_type

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        # Saves model when validation score increase.
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path + self.model_type + '.pt')
        if self.model_type == 'TSTiNet-mixed':
            torch.save(model.net1.state_dict(), '../outputs/model_parameters/net1_TSTiNet.pt')
            torch.save(model.net1.state_dict(), '../outputs/model_parameters/net2_TSTiNet.pt')
            parameters = {'beta': model.beta.tolist(),
                          'inter_para_1': model.inter_para[:, 0].tolist(),
                          'inter_para_2': model.inter_para[:, 1].tolist(),
                          'inter_para_3': model.inter_para[:, 2].tolist(),
                          'inter_para_4': model.inter_para[:, 3].tolist(),
                          'inter_para_5': model.inter_para[:, 4].tolist()}
            parameters = pd.DataFrame(parameters, index=None)
            parameters.to_csv('../outputs/model_parameters/parameters_TSTiNet.csv')
        self.val_loss_min = val_loss


# save results to outputs folder
def save_outputs(dic_list, name_list, suffix, save_path='../outputs/'):
    for i, dic in enumerate(dic_list):
        df = pd.DataFrame(dic, index=None)
        path = save_path + name_list[i] + '_' + suffix + '.csv'
        df.to_csv(path)


# preview results
def plotting(x, y, xlabel, ylabel, loss_epoch=False):
    plt.figure(figsize=(8, 8))

    if not loss_epoch:
        if type(x) == torch.Tensor:
            x, y = x.cpu().detach(), y.cpu().detach()
        x_min, y_min = min(x), min(y)
        x_max, y_max = max(x) * 1.2, max(y) * 1.2
        x_max, y_max = max(x_max, y_max), max(x_max, y_max)
        plt.plot([x_min, x_max], [y_min, y_max])
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.scatter(x, y, marker='x')

    else:
        y_train, y_test = y['train_loss'], y['val_loss']
        x_min, y_min = min(x), min(min(y_train), min(y_test))
        x_max, y_max = max(x) * 1.2, max(max(y_train), max(y_test)) * 1.2
        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)
        plt.plot(x, y_train, marker='x', label='train_evluate loss')
        plt.plot(x, y_test, marker='o', label='val loss')

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()
