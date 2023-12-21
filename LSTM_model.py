import copy
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import pickle
from torch.utils.data import DataLoader, TensorDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Uncomment this to run on GPU
print('Using device:', device)

BATCH_SIZE = 16
EPOCHS = 1000
Volt_sample = np.arange(1.0, 0.0, -0.1)

def view_EIS(EIS_data):
    label = 'EIS'
    color = [cm.coolwarm(x) for x in range(len(EIS_data))]
    fig = plt.figure(figsize=(12, 8))
    for i in range(len(EIS_data)):
        plt.plot(EIS_data[i][:36], -EIS_data[i][36:], color=color[i])
    plt.savefig(f'figs/{label}/all in one.jpg')
    plt.close()

def view_LSV(EI_data, I_sample, fig_name):
    plt.figure()
    test_label = list(EI_data.keys())[0]
    volt, current = EI_data[test_label][0], EI_data[test_label][1]
    plt.plot(current, volt, 'b-', label='exp')
    plt.plot(I_sample, Volt_sample, 'ro', label='sample')
    plt.legend()
    plt.xlabel(r'${\rm I}/(\rm {A/cm^2})$')
    plt.ylabel(r'${\rm E/V}$')
    plt.text(0.6, 1.0, f'test_label: {test_label}', fontsize=12)
    plt.savefig(f'figs/{fig_name}.jpg', dpi=300)
    plt.close()

def create_dataset(dataset, len_target, lookback):
    # modified from https://machinelearningmastery.com/lstm-for-time-series-prediction-in-pytorch/
    """Transform a time series into a prediction dataset
    Args:
        dataset: A numpy array of time series, first dimension is the time steps
        lookback: Size of window for prediction
    """
    X, y = [], []
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback]
        target = dataset[i+1:i+lookback+1, -len_target:]
        X.append(feature)
        y.append(target)
    X = torch.tensor(X, device=device, dtype=torch.float32)
    y = torch.tensor(y, device=device, dtype=torch.float32)
    return X, y

def load_data(data_path, train_ratio=0.8, lookback=1):
    # load the list of current and voltage measured by experiments
    with open(f'{data_path}/E_I_array.pkl', 'rb') as f:
        EI_array = pickle.load(f)
    
    Impedance, Current = np.load(f'{data_path}/EIS_data.npy'), np.load(f'{data_path}/I_sample.npy')
    len_freq, len_current = Impedance.shape[1], Current.shape[1]
    len_dataset = len(Impedance)
    # subtract the Ohmic resistance
    Ohm = np.min(Impedance[:, :36], axis=1, keepdims=True)
    Impedance[:, :36] = Impedance[:, :36] - Ohm
    # split the dataset into train and test. Note that for sequence model, shuffle is not implemented 
    train_size = int(len_dataset*train_ratio)
    test_size = len_dataset - train_size
    # preprocess the impedance data
    Z_scaler = MinMaxScaler()
    Z_scaler.fit(Impedance[:train_size])
    Z_scaled = Z_scaler.transform(Impedance)
    # for the sequence mode, we use the impedance and LSV as input, and LSV only as target
    dataset = np.hstack((Z_scaled, Current))
    train_dataset, test_dataset = dataset[:train_size], dataset[train_size:]
    X_train, y_train = create_dataset(train_dataset, len_current, lookback=lookback)
    X_test, y_test = create_dataset(test_dataset, len_current, lookback=lookback)

    return X_train, y_train, X_test, y_test, EI_array[train_size+lookback:]

class LSV_LSTM(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_size=36):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden_size, num_layers=2, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_dim)
    def forward(self, x):
        x, _ = self.lstm(x)
        x = self.linear(x)
        return x
    
def visualize_loss(train_loss, eval_loss):
    plt.figure(figsize=(9, 6))
    plt.semilogy(train_loss[:,0], train_loss[:,1], 'k-', label='train')
    plt.semilogy(eval_loss[:,0], eval_loss[:,1], 'r-', label='evaluate')
    plt.legend()
    plt.xlabel('Iter')
    plt.ylabel('Loss')
    plt.savefig(f'figs/LSTM train loss.jpg', dpi=300)

def train_model(X_train, y_train, X_test, y_test, best_weights):
    # Define the model, loss function and optimizer. The code is modified from this tutorial
    # https://machinelearningmastery.com/building-a-regression-model-in-pytorch/
    input_dim, output_dim = X_train.shape[-1], y_train.shape[-1]
    model = LSV_LSTM(input_dim, output_dim).to(device)
    loss_fn = nn.MSELoss()  # mean square error
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    best_mse = np.inf   # init to infinity
    train_loss, eval_loss = [], []
    best_epoch = -1
    early_stop_thresh = 50
    loader = DataLoader(TensorDataset(X_train, y_train), shuffle=True, batch_size=BATCH_SIZE)
    for epoch in range(EPOCHS):
        model.train()
        loss_temp = []
        for X_batch, y_batch in loader:
            # forward pass
            y_pred = model(X_batch)
            loss = loss_fn(y_pred, y_batch)
            loss_temp.append(loss.item())
            # backward pass
            optimizer.zero_grad()
            loss.backward()
            # update weights
            optimizer.step()
        train_loss.append([epoch, np.mean(loss_temp)])
        # print(f"Epoch {epoch}: {train_loss[-1]}")
        # Validation
        if epoch % 10 != 0:
            continue
        # evaluate accuracy at end of each epoch
        model.eval()
        y_pred = model(X_test)
        mse = loss_fn(y_pred, y_test)
        eval_loss.append([epoch, mse.item()])
        if mse < best_mse:
            best_mse = mse
            best_epoch = epoch
            torch.save(model.state_dict(), f'./checkpoints/{best_weights}')
        elif epoch - best_epoch > early_stop_thresh:
            print(f"Early stopped training at epoch {epoch}")
            break  # terminate the training loop
    print(f"MSE: {best_mse:.2f}; best epoch: {best_epoch}")
    visualize_loss(np.array(train_loss), np.array(eval_loss))

def eval_model(best_weights, X_test, y_test, EI_data):
    # evaluated the model's performance
    test_size = 10
    rand_ind = np.random.randint(len(y_test), size=test_size)
    input_dim, output_dim = X_test.shape[-1], y_test.shape[-1]
    model = LSV_LSTM(input_dim, output_dim).to(device)
    model.load_state_dict(torch.load(f'./checkpoints/{best_weights}'))
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)[:, -1, :]
        for ind in rand_ind:
            test_label = list(EI_data[ind].keys())[0]
            view_LSV(EI_data[ind], y_pred[ind].cpu().numpy(), test_label+'_pred_LSTM')

if __name__=="__main__":
    # prepare the data
    X_train, y_train, X_test, y_test, EI_array_test = load_data('./data', lookback=4)

    # set the random seed to ensure reproducibility
    random_seed = 42
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    best_weights = 'best_chekpoints_LSTM.pt'
    train_model(X_train, y_train, X_test, y_test, best_weights)
    
    eval_model(best_weights, X_test, y_test, EI_array_test)