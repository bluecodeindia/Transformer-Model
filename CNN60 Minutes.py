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


# pickle_file_path = '/media/terminator/Data/NatureData/Auto Ten Minutes.pkl'
pickle_file_path = '/media/terminator/NatureData/Sixty Minutes.pkl'


# In[6]:


# Load data from the pickle file
with open(pickle_file_path, "rb") as f:
    loaded_data_dict = pickle.load(f)


# In[7]:


# Retrieve data and labels
X_train = loaded_data_dict["X_train"]
y_train = loaded_data_dict["y_train"]
X_test = loaded_data_dict["X_test"]
y_test = loaded_data_dict["y_test"]


# In[8]:


X_train.shape,y_train.shape


# In[9]:


features=13


# In[10]:


def reconstructData(packets):
    # Take the first packet
    first_packet = packets[0]
    
    # Concatenate the last index of every packet
    reconstructed_data = np.concatenate([packet[-1:] for packet in packets[1:]])
    
    # Combine the first packet with the concatenated data
    reconstructed_data = np.concatenate([first_packet, reconstructed_data])
    
    return reconstructed_data


# In[ ]:





# In[11]:


Valid,Test = X_test[:100000],X_test[100000:]
Valid_label,Test_label = y_test[:100000],y_test[100000:]


# In[12]:


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


# In[13]:


print(X_train.dtype)
print(y_train.dtype)


# In[14]:


X_train.shape, y_train.shape


# In[15]:


import torch
import torch.nn as nn

class TimeSeriesEncoder(nn.Module):
    def __init__(self, input_channels):
        super(TimeSeriesEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1),  # Reduce sequence length to 36 / 4 = 9
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=1)  # Reduce sequence length to 9 / 2 = 4.5 (rounded to 4)
        )
        self.classifier = nn.Sequential(
            nn.Linear(142, 1024),
            nn.ReLU(), 
            nn.Linear(1024, 3), 
            nn.Softmax(dim=1)  
        )
        
    
    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.encoder(x)
#         print(x.shape)
        x = torch.max(x,dim=1)[0]
        x = self.classifier(x)
#         x = x.permute(0,2,1)
        return x

# Create the autoencoder model
input_channels = 13  # Number of input features
# encoder = TimeSeriesEncoder(input_channels).to(device)




# In[20]:


train = 0
avg = 'weighted'
seq_length = 144
num_epochs = 5000
reset_random_seeds(seed)
src_vocab_size = features+5
target_vocab_size = features
num_layers = 2
latent_dim = 8
embed_dim = 128
n_heads = 8
expansion_factor = 2

Cf = TimeSeriesEncoder(input_channels).to(device)

criterion = nn.CrossEntropyLoss()
optimizer_cf = optim.Adam(Cf.parameters(), lr=1e-4)

# Define early stopping parameters
patience = 10
best_loss = 0
epochs_without_improvement = 0

if train:
    # Step 4: Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        latents = []
        PR, PL = [], []
        train_input,train_output = [], []

        # Iterate over mini-batches
        for i, (inputs, labels) in enumerate(train_loader):
            
            probability = 0.0
            coin_toss = random.random()  

            if coin_toss < probability:
                continue
            
            # Clear gradients
            optimizer_cf.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            
            prediction = Cf(inputs)
            
            
            # Calculate loss
            loss = criterion(prediction, labels)
            loss = loss.float()

            # Backpropagation
            loss.backward()
            
            optimizer_cf.step()
            
            # Update running loss
            running_loss += loss.item()
            
            
            PR.append(prediction)
            PL.append(labels)
            

        # Calculate average loss per epoch
        PR = torch.cat(PR, dim=0)
        PL = torch.cat(PL, dim=0)
        _, predicted = torch.max(PR.data, 1)
        _, lbl = torch.max(PL.data, 1)
        total = lbl.size(0)
        correct = (predicted == lbl).sum().item()
        lb = lbl.cpu().detach().numpy()
        pre = predicted.cpu().detach().numpy()
        accuracy = 100 * correct / total
        print(f'Train Accuracy: {accuracy:.2f}%')
        conf_matrix = confusion_matrix(lbl.cpu().detach().numpy(),predicted.cpu().detach().numpy())
        print('Confusion Matrix:')
        print(conf_matrix)
        precision = precision_score(lb, pre,average=avg)
        recall = recall_score(lb, pre,average=avg)
        f1 = f1_score(lb, pre,average=avg)
        auc = roc_auc_score(lb,PR.cpu().detach().numpy(),multi_class='ovr')
        print()
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)
        print("auc:", auc)
        print(epochs_without_improvement, best_loss)
        print('-----------------------------------------')
        print()
        print()
        # Calculate average loss per epoch
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch: {epoch+1}, Train Loss: {round(epoch_loss,6)}")
        
        if epoch%1==0:
            valid_latents = []
            VR,VL = [], []
            valid_loss = 0.0
            for j, (inp, lab) in enumerate(valid_loader):
                probability = 0.0
                coin_toss = random.random()  

                if coin_toss < probability:
                    continue
                inp = inp.to(device)
                lab = lab.to(device)
                
                with torch.no_grad():
                    # Forward pass
                    outp = Cf(inp)
                    
                VR.append(outp)
                VL.append(lab)
            
            VR = torch.cat(VR, dim=0)
            VL = torch.cat(VL, dim=0)
            _, vpre = torch.max(VR.data, 1)
            _, vlbl = torch.max(VL.data, 1)
            total = vlbl.size(0)
            correct = (vpre == vlbl).sum().item()

            vaccuracy = 100 * correct / total
            print('-----------------------------------------')
            print(f'Valid Accuracy: {vaccuracy:.2f}%')
            lb = vlbl.cpu().detach().numpy()
            pre = vpre.cpu().detach().numpy()
            conf_matrix = confusion_matrix(lb,pre)
            precision = precision_score(lb, pre,average=avg)
            recall = recall_score(lb, pre,average=avg)
            f1 = f1_score(lb, pre,average=avg)
            auc = roc_auc_score(lb,VR.cpu().detach().numpy(),multi_class='ovr')
            print('Confusion Matrix:')
            print(conf_matrix)
            print()
            print("Precision:", precision)
            print("Recall:", recall)
            print("F1 Score:", f1)
            print("auc:", auc)
            print(epochs_without_improvement, best_loss)
            print('-----------------------------------------')
            print()
            print()
            
            valid_latents = []
            VR,VL = [], []
            valid_loss = 0.0
            for j, (inp, lab) in enumerate(test_loader):
                probability = 0.0
                coin_toss = random.random()  

                if coin_toss < probability:
                    continue
                inp = inp.to(device)
                lab = lab.to(device)

                with torch.no_grad():
                    # Forward pass
                    outp = Cf(inp)

                VR.append(outp)
                VL.append(lab)

            VR = torch.cat(VR, dim=0)
            VL = torch.cat(VL, dim=0)
            _, vpre = torch.max(VR.data, 1)
            _, vlbl = torch.max(VL.data, 1)
            total = vlbl.size(0)
            correct = (vpre == vlbl).sum().item()

            taccuracy = 100 * correct / total
            print('-----------------------------------------')
            print(f'Test Accuracy: {taccuracy:.2f}%')
            lb = vlbl.cpu().detach().numpy()
            pre = vpre.cpu().detach().numpy()
            conf_matrix = confusion_matrix(lb,pre)
            precision = precision_score(lb, pre,average=avg)
            recall = recall_score(lb, pre,average=avg)
            f1 = f1_score(lb, pre,average=avg)
            auc = roc_auc_score(lb,VR.cpu().detach().numpy(),multi_class='ovr')
            print('Confusion Matrix:')
            print(conf_matrix)
            print()
            print("Precision:", precision)
            print("Recall:", recall)
            print("F1 Score:", f1)
            print("auc: ",auc)
            print(epochs_without_improvement, best_loss)
            print('-----------------------------------------')
            print()
            print()
                
                    
        # Check for early stopping and save the best weights
        th = (conf_matrix[0][0]> conf_matrix[0][1]+conf_matrix[0][2] and conf_matrix[1][1]>conf_matrix[1][0]+conf_matrix[1][2] and conf_matrix[2][2]>conf_matrix[2][0]+conf_matrix[2][1])
        if th and vaccuracy > best_loss:
            best_loss = vaccuracy
            epochs_without_improvement = 0
            # Save the model weights
            torch.save(Cf.state_dict(), 'cnn60.pt')
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
Cf.load_state_dict(torch.load('cnn60.pt'))


# In[17]:


Cf.load_state_dict(torch.load('cnn60.pt'))


# In[19]:


valid_latents = []
VR,VL = [], []
valid_loss = 0.0
for j, (inp, lab) in enumerate(train_loader):
    probability = 0.0
    coin_toss = random.random()  

    if coin_toss < probability:
        continue
    inp = inp.to(device)
    lab = lab.to(device)

    with torch.no_grad():
        # Forward pass
        outp = Cf(inp)

    VR.append(outp)
    VL.append(lab)

VR = torch.cat(VR, dim=0)
VL = torch.cat(VL, dim=0)
_, vpre = torch.max(VR.data, 1)
_, vlbl = torch.max(VL.data, 1)
total = vlbl.size(0)
correct = (vpre == vlbl).sum().item()

vaccuracy = 100 * correct / total
print('-----------------------------------------')
print(f'Train Accuracy: {vaccuracy:.2f}%')
lb = vlbl.cpu().detach().numpy()
pre = vpre.cpu().detach().numpy()
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
auc = roc_auc_score(lb,VR.cpu().detach().numpy(),multi_class='ovr')
print("AUC:", auc)
print(epochs_without_improvement, best_loss)
print(vaccuracy,",",precision,",",recall,",",f1,",",auc)
print('-----------------------------------------')
print()
print()

valid_latents = []
VR,VL = [], []
valid_loss = 0.0
for j, (inp, lab) in enumerate(valid_loader):
    probability = 0.0
    coin_toss = random.random()  

    if coin_toss < probability:
        continue
    inp = inp.to(device)
    lab = lab.to(device)

    with torch.no_grad():
        # Forward pass
        outp = Cf(inp)

    VR.append(outp)
    VL.append(lab)

VR = torch.cat(VR, dim=0)
VL = torch.cat(VL, dim=0)
_, vpre = torch.max(VR.data, 1)
_, vlbl = torch.max(VL.data, 1)
total = vlbl.size(0)
correct = (vpre == vlbl).sum().item()

vaccuracy = 100 * correct / total
print('-----------------------------------------')
print(f'Valid Accuracy: {vaccuracy:.2f}%')
lb = vlbl.cpu().detach().numpy()
pre = vpre.cpu().detach().numpy()
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
auc = roc_auc_score(lb,VR.cpu().detach().numpy(),multi_class='ovr')
print("AUC:", auc)
print(epochs_without_improvement, best_loss)
print(vaccuracy,",",precision,",",recall,",",f1,",",auc)
print('-----------------------------------------')
print()
print()

valid_latents = []
VR,VL = [], []
valid_loss = 0.0
for j, (inp, lab) in enumerate(test_loader):
    probability = 0.0
    coin_toss = random.random()  

    if coin_toss < probability:
        continue
    inp = inp.to(device)
    lab = lab.to(device)

    with torch.no_grad():
        # Forward pass
        outp = Cf(inp)

    VR.append(outp)
    VL.append(lab)

VR = torch.cat(VR, dim=0)
VL = torch.cat(VL, dim=0)
_, vpre = torch.max(VR.data, 1)
_, vlbl = torch.max(VL.data, 1)
total = vlbl.size(0)
correct = (vpre == vlbl).sum().item()

vaccuracy = 100 * correct / total
print('-----------------------------------------')
print(f'Test Accuracy: {vaccuracy:.2f}%')
lb = vlbl.cpu().detach().numpy()
pre = vpre.cpu().detach().numpy()
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
auc = roc_auc_score(lb,VR.cpu().detach().numpy(),multi_class='ovr')
print("AUC:", auc)
print(epochs_without_improvement, best_loss)
print(vaccuracy,",",precision,",",recall,",",f1,",",auc)
print('-----------------------------------------')
print()
print()


# In[ ]:


-----------------------------------------
Train Accuracy: 84.17%
Confusion Matrix:
[[5958 1531  287]
 [1026 6338  412]
 [ 350   86 7340]]

Precision: 0.8407203982260056
Recall: 0.8417352537722909
F1 Score: 0.8408823235023491
AUC: 0.9099564854241816
0 74.835
84.17352537722908 , 0.8407203982260056 , 0.8417352537722909 , 0.8408823235023491 , 0.9099564854241816
-----------------------------------------


-----------------------------------------
Valid Accuracy: 74.83%
Confusion Matrix:
[[71918 20968  3597]
 [  403  2196   149]
 [   41     7   721]]

Precision: 0.9627555702633718
Recall: 0.74835
F1 Score: 0.8286955641815008
AUC: 0.8676320952179916
0 74.835
74.835 , 0.9627555702633718 , 0.74835 , 0.8286955641815008 , 0.8676320952179916
-----------------------------------------


-----------------------------------------
Test Accuracy: 79.39%
Confusion Matrix:
[[126548  28397   4654]
 [   553   2790    168]
 [    42     11    919]]

Precision: 0.9709882402101777
Recall: 0.7938530734632684
F1 Score: 0.8636072286420343
AUC: 0.8821922382402924
0 74.835
79.38530734632684 , 0.9709882402101777 , 0.7938530734632684 , 0.8636072286420343 , 0.8821922382402924
-----------------------------------------


# In[ ]:




