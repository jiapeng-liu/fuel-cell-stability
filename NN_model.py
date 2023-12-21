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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # Uncomment this to run on GPU
print('Using device:', device)

BATCH_SIZE = 32
EPOCHS = 3000
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

def load_data(data_path):
    # load the list of current and voltage measured by experiments
    with open(f'{data_path}/E_I_array.pkl', 'rb') as f:
        EI_array = pickle.load(f)
    
    X, y = np.load(f'{data_path}/EIS_data.npy'), np.load(f'{data_path}/I_sample.npy')
    # subtract the Ohmic resistance
    X_min = np.min(X[:, :36], axis=1, keepdims=True)
    X[:, :36] = X[:, :36] - X_min
    # view_EIS(X)
    # print(X.shape, y.shape)
    # train-test split for model evaluation
    X_train, X_test, y_train, y_test, _, EI_array_test = train_test_split(X, y, EI_array, train_size=0.8, random_state=42, shuffle=True)
    rand_ind = np.random.randint(len(y_test))
    view_LSV(EI_array_test[rand_ind], y_test[rand_ind], 'target_check')

    X_scaler = MinMaxScaler()
    X_scaler.fit(X_train)
    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)

    # Convert to 2D PyTorch tensors
    X_train = torch.tensor(X_train, device=device, dtype=torch.float32)
    y_train = torch.tensor(y_train, device=device, dtype=torch.float32)
    X_test = torch.tensor(X_test, device=device, dtype=torch.float32)
    y_test = torch.tensor(y_test, device=device, dtype=torch.float32)
    # check the shape consistency
    # print(X_train.shape, y_train.shape)
    return X_train, y_train, X_test, y_test, EI_array_test

class LSV_pred_NN(nn.Module):
    def __init__(self, input_dim, output_dim=10):
        super().__init__()
        self.hidden1 = nn.Linear(input_dim, 36)
        self.act1 = nn.ReLU()
        self.hidden2 = nn.Linear(36, 18)
        self.act2 = nn.ReLU()
        self.output = nn.Linear(18, output_dim)
 
    def forward(self, x):
        x = self.act1(self.hidden1(x))
        x = self.act2(self.hidden2(x))
        x = self.output(x)
        return x
    
def visualize_loss(train_loss, eval_loss):
    plt.figure(figsize=(12, 8))
    plt.semilogy(train_loss, 'k-', label='train')
    plt.semilogy(eval_loss, 'r--', label='evaluate')
    plt.legend()
    plt.xlabel('Iter')
    plt.ylabel('Loss')
    plt.savefig(f'figs/NN train loss.jpg', dpi=300)

def train_model(X_train, y_train, X_test, y_test, best_weights):
    # Define the model, loss function and optimizer. The code is modified from this tutorial
    # https://machinelearningmastery.com/building-a-regression-model-in-pytorch/
    input_dim = X_train.shape[1]
    model = LSV_pred_NN(input_dim).to(device)
    loss_fn = nn.MSELoss()  # mean square error
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    best_mse = np.inf   # init to infinity
    train_loss, eval_loss = [], []
    best_epoch = -1
    early_stop_thresh = 5
    batch_start = torch.arange(0, len(X_train), BATCH_SIZE)
    for epoch in range(EPOCHS):
        model.train()
        loss_temp = []
        with tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                # take a batch
                X_batch = X_train[start:start+BATCH_SIZE]
                y_batch = y_train[start:start+BATCH_SIZE]
                # forward pass
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                loss_temp.append(loss.item())
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # print progress
                bar.set_postfix(mse=float(loss))
        train_loss.append(np.mean(loss_temp))
        # print(f"Epoch {epoch}: {train_loss[-1]}")
        # evaluate accuracy at end of each epoch
        model.eval()
        y_pred = model(X_test)
        mse = loss_fn(y_pred, y_test)
        eval_loss.append(mse.item())
        if mse < best_mse:
            best_mse = mse
            best_epoch = epoch
            torch.save(model.state_dict(), f'./checkpoints/{best_weights}')
        elif epoch - best_epoch > early_stop_thresh:
            print(f"Early stopped training at epoch {epoch}")
            break  # terminate the training loop
    print(f"MSE: {best_mse:.2f}; best epoch: {best_epoch}")
    visualize_loss(train_loss, eval_loss)

def eval_model(best_weights, X_test, y_test, EI_data):
    # evaluated the model's performance
    test_size = 10
    rand_ind = np.random.randint(len(y_test), size=test_size)
    input_dim = X_test.shape[1]
    model = LSV_pred_NN(input_dim).to(device)
    model.load_state_dict(torch.load(f'./checkpoints/{best_weights}'))
    model.eval()
    with torch.no_grad():
        y_pred = model(X_test)
        for ind in rand_ind:
            test_label = list(EI_data[ind].keys())[0]
            view_LSV(EI_data[ind], y_pred[ind].cpu().numpy(), test_label+'_pred_NN')

if __name__=="__main__":
    # prepare the data
    X_train, y_train, X_test, y_test, EI_array_test = load_data('./data')

    # set the random seed to ensure reproducibility
    random_seed = 42
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    best_weights = 'best_chekpoints_NN.pt'
    train_model(X_train, y_train, X_test, y_test, best_weights)
    
    # randomly choose $test_size$ test sets to visualize the model's performance
    eval_model(best_weights, X_test, y_test, EI_array_test)