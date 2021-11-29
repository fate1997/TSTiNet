import torch
import torch.nn as nn
import numpy as np
import pandas as pd


"""Load Model"""
# network architecture of TSTiNet
class Viscosity(nn.Module):
    def __init__(self, net1, net2, net3):
        super(Viscosity, self).__init__()
        self.net1 = net1
        self.net2 = net2
        self.net3 = net3

    def forward(self, X):
        # equation parameters of hba
        X1 = X[:, :45]
        out1 = self.net1(X1)

        # equation parameters of hbd
        X2 = X[:, 45:90]
        out2 = self.net2(X2)

        # equation for ln(viscosity)
        x1, x2, g_ij, mw1, mw2, T = X[:, 90], X[:, 91], X[:, 92:97], X[:, 98], X[:, 99], X[:, 97]
        out = x1 * (out1[:, 0] + out1[:, 1] / T + out1[:, 2] / (T - self.E.mean(dim=0)) + out1[:, 3] * torch.log(
            x1 * mw1)) + \
              x2 * (out2[:, 0] + out2[:, 1] / T + out2[:, 2] / (T - self.E.mean(dim=0)) + out2[:, 3] * torch.log(
            x2 * mw2)) + \
              x1 * x2 * torch.mm(g_ij, (self.inter_para.mean(dim=0)).reshape(5, -1)).reshape(-1, )
        return out


def hidden_block(hidden_size, num_hidden_layers):
    hidden_list = []
    for i in range(num_hidden_layers):
        hidden_list.append(nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                         nn.GELU(),
                                         nn.BatchNorm1d(hidden_size)))
    return hidden_list


input_size = 45
hidden_size = 32
num_hidden_layers = 2
net1 = nn.Sequential(nn.BatchNorm1d(input_size, affine=False),
                     nn.Linear(input_size, hidden_size),
                     nn.GELU(),
                     nn.BatchNorm1d(hidden_size),
                     nn.Sequential(*hidden_block(hidden_size, num_hidden_layers)),
                     nn.Linear(hidden_size, 4))
net2 = nn.Sequential(nn.BatchNorm1d(input_size, affine=False),
                     nn.Linear(input_size, hidden_size),
                     nn.GELU(),
                     nn.BatchNorm1d(hidden_size),
                     nn.Sequential(*hidden_block(hidden_size, num_hidden_layers)),
                     nn.Linear(hidden_size, 4))
net3 = nn.Sequential(nn.BatchNorm1d(100, affine=False),
                     nn.Linear(100, hidden_size),
                     nn.GELU(),
                     nn.BatchNorm1d(hidden_size),
                     nn.Sequential(*hidden_block(hidden_size, num_hidden_layers)),
                     nn.Linear(hidden_size, 6))

model = Viscosity(net1, net2, net3)

# load model parameters
model_para_path = '../trained model/'
net1.load_state_dict(torch.load(model_para_path + 'net1_TSTiNet.pt'))
net2.load_state_dict(torch.load(model_para_path + 'net2_TSTiNet.pt'))
model.load_state_dict(torch.load(model_para_path + 'TSTiNet-mixed.pt'))

# load energy parameters
parameters = pd.read_csv('../trained model/parameters_TSTiNet.csv')
model.E = torch.as_tensor(parameters.iloc[:, 1], dtype=torch.float32)
model.inter_para = torch.as_tensor(parameters.iloc[:, -5:].to_numpy(), dtype=torch.float32)

# evaluation mode
net1.eval()
net2.eval()
net3.eval()
model.eval()


"""Load Data"""
data_input = pd.read_excel('input.xlsx')
x_hba = np.expand_dims(data_input.iloc[1:, 2].to_numpy(dtype=float), axis=-1)
mw_hba = np.expand_dims(data_input.iloc[1:, 3].to_numpy(dtype=float), axis=-1)

x_hbd = np.expand_dims(data_input.iloc[1:, 50].to_numpy(dtype=float), axis=-1)
mw_hbd = np.expand_dims(data_input.iloc[1:, 51].to_numpy(dtype=float), axis=-1)

X_hba = data_input.iloc[1:, 4:49].to_numpy(dtype=float)
X_hbd = data_input.iloc[1:, 52:97].to_numpy(dtype=float)

G_ij = data_input.iloc[1:, -6:-1].to_numpy(dtype=float)
temperature = np.expand_dims(data_input.iloc[1:, -1].to_numpy(dtype=float), axis=-1)

number = data_input.iloc[1:, 0].to_numpy(dtype=str)

X_input = np.concatenate([X_hba, X_hbd, x_hba, x_hbd, G_ij, temperature, mw_hba, mw_hbd], axis=-1)

X_input = torch.as_tensor(X_input, dtype=torch.float32)

viscosity = np.around(torch.exp(model(X_input)).detach().numpy(), 3)

results = np.stack([number, viscosity], axis=1)
results = pd.DataFrame(results, columns=['No.', 'viscosity/mPa.s']).set_index('No.')
print(results)