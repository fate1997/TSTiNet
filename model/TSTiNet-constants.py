from util import data_preprocessing, plotting, save_outputs, EarlyStopping
import torch
import torch.nn as nn
import sklearn.metrics as metrics
import os

file_path = os.path.basename(__file__).split(".")[0]
# data preprocessing
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data_path = '../data'
X_train, X_val, X_test, y_train, y_val, y_test = data_preprocessing(data_path, device)


# define network architecture
class Viscosity(nn.Module):
    def __init__(self, net1, net2):
        super(Viscosity, self).__init__()
        self.net1 = net1
        self.net2 = net2
        self.inter_para = nn.Parameter(torch.ones(5).requires_grad_())
        self.beta = nn.Parameter(torch.ones(1).requires_grad_())

    def forward(self, X, training=True):
        # equation parameters of hba
        X1 = X[:, :45]
        out1 = self.net1(X1)

        # equation parameters of hbd
        X2 = X[:, 45:90]
        out2 = self.net2(X2)

        # equation for ln(viscosity)
        x1, x2, g_ij, mw1, mw2, T = X[:, 90], X[:, 91], X[:, 92:97], X[:, 98], X[:, 99], X[:, 97]
        out = x1 * (out1[:, 0] + out1[:, 1] / T + out1[:, 2] / (T - self.beta) + out1[:, 3] * torch.log(
            x1 * mw1)) + \
              x2 * (out2[:, 0] + out2[:, 1] / T + out2[:, 2] / (T - self.beta) + out2[:, 3] * torch.log(
            x2 * mw2)) + \
              x1 * x2 * torch.mm(g_ij, self.inter_para.reshape(5, -1)).reshape(-1, )
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

model = Viscosity(net1, net2)
model.to(device)


# training
def init_weight(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
    elif isinstance(m, nn.Conv1d):
        nn.init.kaiming_uniform_(m.weight)
    elif isinstance(m, nn.BatchNorm1d) and m.affine:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


num_epochs = 100000
criterion = nn.SmoothL1Loss()
optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.0001, lr=0.001)
model.apply(init_weight)

early_stopping = EarlyStopping(patience=1000, model_type='TSTiNet-constants')

print(f'Training on {device}')
loss_trace = {'train_loss': [], 'val_loss': [], 'beta': [], 'inter_parameters': []}
epoch_trace = []
for epoch in range(num_epochs):
    # forward
    y_train, y_val = y_train.to(device), y_val.to(device)
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    model.eval()
    # cal AARD
    y_pred = model(X_val, training=False)
    ard = abs((torch.exp(y_pred) - torch.exp(y_val)) / (torch.exp(y_val)))
    loss_val = criterion(y_pred, y_val).item()
    mse = ((y_pred - y_val) ** 2).mean().item()

    # print information and loss-epoch
    if (epoch + 1) % 100 == 0:
        print(
            f'epoch {epoch + 1} / {num_epochs}, '
            f'loss = {loss.item():.4f}, '
            f'AARD_val = {ard.mean():.4f}, '
            f'max_ard = {max(ard):.4f}')
        loss_trace['train_loss'].append(loss.item())
        loss_trace['val_loss'].append(loss_val)
        loss_trace['beta'].append(model.beta.item())
        loss_trace['inter_parameters'].append(model.inter_para.tolist())
        epoch_trace.append(epoch)

    # early stopping
    early_stopping(mse, model)
    if early_stopping.early_stop:
        print('Early stopping!')
        break

model.load_state_dict(torch.load('../outputs/model_parameters/TSTiNet-constants.pt'))

# evaluate model on test set
viscosity_test_pred = torch.exp(model(X_test, training=False)).cpu().detach().numpy()
viscosity_test = torch.exp(y_test).cpu().detach().numpy()
ard_test = abs((viscosity_test - viscosity_test_pred) / viscosity_test)
r2_test = metrics.r2_score(viscosity_test, viscosity_test_pred)
print(f'Performance on Test set: aard = {ard_test.mean():.4f}, max ard = {max(ard_test):.4f}, r2 = {r2_test:.4f}')

# save outputs
test_results = {'viscosity_test': viscosity_test.tolist(),
                'viscosity_test_pred': viscosity_test_pred.tolist()}

viscosity_val = torch.exp(y_val)
viscosity_val_pred = torch.exp(y_pred)
val_results = {'viscosity_val': viscosity_val.tolist(),
               'viscosity_val_pred': viscosity_val_pred.tolist()}

viscosity_train = torch.exp(y_train)
viscosity_train_pred = torch.exp(outputs)
train_results = {'viscosity_train': torch.exp(y_train).tolist(),
                 'viscosity_train_pred': torch.exp(outputs).tolist()}

para_results = {'beta': model.beta.repeat(model.inter_para.shape).tolist(),
                'inter_para': model.inter_para.tolist()}

save_outputs([test_results, val_results, train_results, para_results, loss_trace],
             ['test_results',
              'val_results',
              'train_results',
              'para_results',
              'loss_results'], suffix=file_path)

# plotting
plotting(viscosity_val, viscosity_val_pred, 'viscosity_val', 'viscosity_val_pred')
plotting(viscosity_test, viscosity_test_pred, 'viscosity_test', 'viscosity_test_pred')
plotting(viscosity_train, viscosity_train_pred, 'viscosity_train', 'viscosity_train_pred')
plotting(epoch_trace, loss_trace, 'epoch', 'loss', loss_epoch=True)
