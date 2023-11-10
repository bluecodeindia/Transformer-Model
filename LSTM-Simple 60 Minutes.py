#!/usr/bin/env python
# coding: utf-8

# In[1]:


seed=197


# In[ ]:





# In[2]:


# importing required libraries
import torch.nn as nn
import torch
import torch.nn.functional as F
import math,copy,re
import warnings
import pandas as pd
import numpy as np
import random
import pickle
import torch.nn.init as init
from torchviz import make_dot
import plotly.graph_objects as go
import pickle
# import seaborn as sns
# import torchtext
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import display
import matplotlib.pyplot as plt
import os 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score

warnings.simplefilter("ignore")
print(torch.__version__)


# In[3]:


# Check if CUDA is available
if torch.cuda.is_available():
    # CUDA is available, so PyTorch can use a GPU
    device = torch.device("cuda")
    print("PyTorch is using GPU.")
else:
    # CUDA is not available, PyTorch will use CPU
    device = torch.device("cpu")
    print("PyTorch is using CPU.")

# Create a tensor and move it to the device
tensor = torch.randn(3, 4).to(device)

# Check the device of the tensor
print("Tensor device:", tensor.device)


# In[4]:


def reset_random_seeds(seed):
    os.environ['PYTHONHASHSEED']=str(seed)
#     tf.random.set_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# In[5]:


# pickle_file_path = '/media/terminator/Data/NatureData/Ten Minutes.pkl'
pickle_file_path = '/media/terminator/NatureData/Sixty Minutes.pkl'


# In[6]:


# Load data from the pickle file
with open(pickle_file_path, "rb") as f:
    loaded_data_dict = pickle.load(f)

# Retrieve data and labels
X_train = loaded_data_dict["X_train"]
y_train = loaded_data_dict["y_train"]
X_test = loaded_data_dict["X_test"]
y_test = loaded_data_dict["y_test"]


# In[7]:


X_train.shape,y_train.shape


# In[8]:


X_test.shape,y_test.shape


# In[9]:


features=13


# In[10]:


Valid,Test = X_test[:100000],X_test[100000:]
Valid_label,Test_label = y_test[:100000],y_test[100000:]


# In[11]:


def norm(out):
    out = out.view(-1,13)
    min_values = out.min(dim=0).values
    max_values = out.max(dim=0).values

    # Apply min-max normalization
    normalized_tensor = (out - min_values) / (max_values - min_values)

    out = normalized_tensor.view(-1,144,13)
    
    return out


# In[12]:


# X_train = norm(X_train)
# Valid = norm(Valid)
# Test = norm(Test)


# In[13]:


y_train.shape


# In[14]:


num_epochs = 500
# Define batch size
batch_size = 512
# Create DataLoader for training set
train_dataset = torch.utils.data.TensorDataset(X_train.float(), y_train.float())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
valid_dataset = torch.utils.data.TensorDataset(Valid.float(), Valid_label.float())
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_dataset = torch.utils.data.TensorDataset(Test.float(), Test_label.float())
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# In[15]:


print(X_train.dtype)
print(y_train.dtype)


# In[16]:


X_train.shape, y_train.shape


# In[17]:


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
        
        # Fully connected layer
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size*144, 1024),
            nn.ReLU(), 
            nn.Linear(1024, 3),
            nn.Softmax(dim=1)  
        )
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        
        # Index hidden state of last time step
        out = torch.reshape(out,(out.shape[0],-1))
        out = self.classifier(out)
        return out


# In[24]:


train = 0
avg = 'weighted'
seq_length = 144
reset_random_seeds(seed)
src_vocab_size = features
target_vocab_size = features
num_layers = 2
latent_dim = 8
hidden_size = 100
n_heads = 16
expansion_factor = 2
input_size =13 
output_size = 3

model = LSTMModel(input_size, hidden_size, num_layers, output_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define early stopping parameters
patience = 10
best_loss = 0.0
epochs_without_improvement = 0

if train:
    # Step 4: Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        latents = []
        LS = []

        # Iterate over mini-batches
        for i, (inputs, labels) in enumerate(train_loader):
            
            probability = 0.0
            coin_toss = random.random()  

            if coin_toss < probability:
                continue
            
            # Clear gradients
            optimizer.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            loss = loss.float()

            # Backpropagation
            loss.backward()
            optimizer.step()

            # Update running loss
            running_loss += loss.item()
            
            latents.append(outputs)
            LS.append(labels)
            

        # Calculate average loss per epoch
        epoch_loss = running_loss / len(latents)
        latents = torch.cat(latents, dim=0)
        LS = torch.cat(LS, dim=0)
        _, predicted = torch.max(latents.data, 1)
        _, lbl = torch.max(LS.data, 1)
        total = lbl.size(0)
        correct = (predicted == lbl).sum().item()

        accuracy = 100 * correct / total
        print(f'Train Accuracy: {accuracy:.2f}%')
        conf_matrix = confusion_matrix(lbl.cpu().detach().numpy(),predicted.cpu().detach().numpy())
        print('Confusion Matrix:')
        print(conf_matrix)
        
        if epoch%1==0 and epoch>=0:
            M,N = [],[]
            for j, (inp, lab) in enumerate(valid_loader):
                inp = inp.to(device)
                lab = lab.to(device)
                
                with torch.no_grad():
                    outp = model(inp)
                M.append(outp)
                N.append(lab)
                    
            M = torch.cat(M, dim=0)
            N = torch.cat(N, dim=0)
            _, pre = torch.max(M.data, 1)
            _, lb = torch.max(N.data, 1)
            total = lb.size(0)
            correct = (pre == lb).sum().item()

            vaccuracy = 100 * correct / total
            print('-----------------------------------------')
            print(f'Valid Accuracy: {vaccuracy:.2f}%')
            lb = lb.cpu().detach().numpy()
            pre = pre.cpu().detach().numpy()
            conf_matrix = confusion_matrix(lb,pre)
            precision = precision_score(lb, pre,average=avg)
            recall = recall_score(lb, pre,average=avg)
            f1 = f1_score(lb, pre,average=avg)
            print('Confusion Matrix:')
            print(conf_matrix)
            print()
            print("Precision:", precision)
            print("Recall:", recall)
            print("F1 Score:", f1)
            print('-----------------------------------------')
            # Log epoch progress (optional)
            print(f"Epoch: {epoch+1}, Loss: {round(epoch_loss,6)}")
            print(epochs_without_improvement, best_loss)
            print()
            print()
        
        # Check for early stopping and save the best weights
        # Check for early stopping and save the best weights
        th = (conf_matrix[0][0]> conf_matrix[0][1]+conf_matrix[0][2] and conf_matrix[1][1]>conf_matrix[1][0]+conf_matrix[1][2] and conf_matrix[2][2]>conf_matrix[2][0]+conf_matrix[2][1])
        if th and vaccuracy > best_loss:
            best_loss = vaccuracy
            epochs_without_improvement = 0
            # Save the model weights
            torch.save(model.state_dict(), 'LSTMClass60.pt')
            print('##############################################')
            print('saved', best_loss)
            print('##############################################')
            print()
            print()
            
            
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping triggered. Training stopped.")
                break

# Recover the best weights
model.load_state_dict(torch.load('LSTMClass60.pt'))


# In[19]:


model.load_state_dict(torch.load('LSTMClass60.pt'))


# In[23]:


epoch=0
M,N = [],[]
for j, (inp, lab) in enumerate(train_loader):
    inp = inp.to(device)
    lab = lab.to(device)

    with torch.no_grad():
        outp = model(inp)
    M.append(outp)
    N.append(lab)

M = torch.cat(M, dim=0)
N = torch.cat(N, dim=0)
_, pre = torch.max(M.data, 1)
_, lb = torch.max(N.data, 1)
total = lb.size(0)
correct = (pre == lb).sum().item()

vaccuracy = 100 * correct / total
print('-----------------------------------------')
print(f'Train Accuracy: {vaccuracy:.2f}%')
lb = lb.cpu().detach().numpy()
pre = pre.cpu().detach().numpy()
conf_matrix = confusion_matrix(lb,pre)
precision = precision_score(lb, pre,average=avg)
recall = recall_score(lb, pre,average=avg)
f1 = f1_score(lb, pre,average=avg)
auc = roc_auc_score(lb,M.cpu().detach().numpy(),multi_class='ovr')
print('Confusion Matrix:')
print(conf_matrix)
print()
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("AUC ",auc)
print(vaccuracy,",",precision,",",recall,",",f1,",",auc)
print('-----------------------------------------')
# Log epoch progress (optional)
print()
print()

M,N = [],[]
for j, (inp, lab) in enumerate(valid_loader):
    inp = inp.to(device)
    lab = lab.to(device)

    with torch.no_grad():
        outp = model(inp)
    M.append(outp)
    N.append(lab)

M = torch.cat(M, dim=0)
N = torch.cat(N, dim=0)
_, pre = torch.max(M.data, 1)
_, lb = torch.max(N.data, 1)
total = lb.size(0)
correct = (pre == lb).sum().item()

vaccuracy = 100 * correct / total
print('-----------------------------------------')
print(f'Valid Accuracy: {vaccuracy:.2f}%')
lb = lb.cpu().detach().numpy()
pre = pre.cpu().detach().numpy()
conf_matrix = confusion_matrix(lb,pre)
precision = precision_score(lb, pre,average=avg)
recall = recall_score(lb, pre,average=avg)
f1 = f1_score(lb, pre,average=avg)
auc = roc_auc_score(lb,M.cpu().detach().numpy(),multi_class='ovr')
print('Confusion Matrix:')
print(conf_matrix)
print()
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("AUC ",auc)
print(vaccuracy,",",precision,",",recall,",",f1,",",auc)
print('-----------------------------------------')
# Log epoch progress (optional)
print()
print()

M,N = [],[]
for j, (inp, lab) in enumerate(test_loader):
    inp = inp.to(device)
    lab = lab.to(device)

    with torch.no_grad():
        outp = model(inp)
    M.append(outp)
    N.append(lab)

M = torch.cat(M, dim=0)
N = torch.cat(N, dim=0)
_, pre = torch.max(M.data, 1)
_, lb = torch.max(N.data, 1)
total = lb.size(0)
correct = (pre == lb).sum().item()

vaccuracy = 100 * correct / total
print('-----------------------------------------')
print(f'Test Accuracy: {vaccuracy:.2f}%')
lb = lb.cpu().detach().numpy()
pre = pre.cpu().detach().numpy()
conf_matrix = confusion_matrix(lb,pre)
precision = precision_score(lb, pre,average=avg)
recall = recall_score(lb, pre,average=avg)
f1 = f1_score(lb, pre,average=avg)
auc = roc_auc_score(lb,M.cpu().detach().numpy(),multi_class='ovr')
print('Confusion Matrix:')
print(conf_matrix)
print()
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
print("AUC ",auc)
print(vaccuracy,",",precision,",",recall,",",f1,",",auc)
print('-----------------------------------------')
# Log epoch progress (optional)
print()
print()


# In[ ]:





# In[ ]:




