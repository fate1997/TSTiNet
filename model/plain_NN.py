from util import data_preprocessing, plotting, save_outputs, EarlyStopping
import torch
import torch.nn as nn
import sklearn.metrics as metrics
import os

file_name = os.path.basename(__file__).split(".")[0]
# data preprocessing
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
data_path = '../data'
X_train, X_val, X_test, y_train, y_val, y_test = data_preprocessing(data_path, device)


# define network architecture
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, num_hidden_layers=2):
        super(NeuralNetwork, self).__init__()

        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, 1)

        self.activate = nn.GELU()

        self.bn1 = nn.BatchNorm1d(input_size, affine=False)
        self.bn2 = nn.BatchNorm1d(hidden_size)

        hidden_list = []
        for i in range(num_hidden_layers):
            hidden_list.append(nn.Sequential(nn.Linear(hidden_size, hidden_size),
                                             nn.GELU(),
                                             nn.BatchNorm1d(hidden_size)))
        self.hidden = nn.Sequential(*hidden_list)

    def forward(self, X):
        out1 = self.bn1(X)
        out1 = self.l1(out1)
        out1 = self.activate(out1)
        out1 = self.bn2(out1)
        out1 = self.hidden(out1)
        out1 = self.l3(out1)
        return out1.reshape(-1, )


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
model = NeuralNetwork(input_size=100, hidden_size=32)
model.to(device)
criterion = nn.SmoothL1Loss()
optimizer = torch.optim.AdamW(model.parameters(), weight_decay=0.0001, lr=0.001)
model.apply(init_weight)

early_stopping = EarlyStopping(patience=1000, model_type='plain_NN')

print(f'Training on {device}')
loss_trace = {'train_loss': [], 'val_loss': []}
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
    y_pred = model(X_val)
    ard = abs((torch.exp(y_pred) - torch.exp(y_val)) / (torch.exp(y_val)))
    mse = ((y_pred - y_val) ** 2).mean().item()
    loss_val = criterion(y_pred, y_val).item()

    # print information and loss-epoch
    if (epoch + 1) % 100 == 0:
        print(
            f'epoch {epoch + 1} / {num_epochs}, '
            f'loss = {loss.item():.4f}, '
            f'AARD_val = {ard.mean():.4f}, '
            f'max_ard = {max(ard):.4f}')
        loss_trace['train_loss'].append(loss.item())
        loss_trace['val_loss'].append(loss_val)
        epoch_trace.append(epoch)

    # early stopping
    early_stopping(mse, model)
    if early_stopping.early_stop:
        print('Early stopping!')
        break

model.load_state_dict(torch.load('../outputs/model_parameters/plain_NN.pt'))

# evaluate model on test set
viscosity_test_pred = torch.exp(model(X_test)).cpu().detach().numpy()
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

save_outputs([test_results, val_results, train_results, loss_trace],
             ['test_results',
              'val_results',
              'train_results',
              'loss_results'], suffix=file_name)

# plotting
plotting(viscosity_val, viscosity_val_pred, 'viscosity_val', 'viscosity_val_pred')
plotting(viscosity_test, viscosity_test_pred, 'viscosity_test', 'viscosity_test_pred')
plotting(viscosity_train, viscosity_train_pred, 'viscosity_train', 'viscosity_train_pred')
plotting(epoch_trace, loss_trace, 'epoch', 'loss', loss_epoch=True)
