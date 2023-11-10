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


pickle_file_path = '/media/terminator/NatureData/Thirty Minutes.pkl'


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


import torch.nn.init as init

class FeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(FeedForward, self).__init__()

        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

        # Initialize the weights and biases
#         for module in self.feed_forward.modules():
#             if isinstance(module, nn.Linear):
#                 init.xavier_uniform_(module.weight)
#                 init.constant_(module.bias, 0.0)

    def forward(self, x):
        x = x.float()
        out = self.feed_forward(x)

        return out


# In[11]:


import torch
import torch.nn as nn

class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, embed_dim):
        super(PositionalEmbedding, self).__init__()
        self.max_len = max_len
        self.emb = embed_dim

        if embed_dim % 2 == 0:
            self.embed_dim = embed_dim
        else:
            self.embed_dim = embed_dim + 1

    def get_angles(self, pos, i):
        angle_rates = 1 / torch.pow(10000.0, (2 * (i // 2)) / self.embed_dim)
        pos = pos.float()
        return pos * angle_rates

    def forward(self, x):
        # Create position indices
        pos = torch.arange(0, self.max_len).unsqueeze(-1)

        # Compute angles
        angles = self.get_angles(pos, torch.arange(self.embed_dim).float())
        # Apply sin to even indices in the array; 2i
        angles = angles.float()
        angles = torch.where(torch.remainder(torch.arange(self.embed_dim), 2) == 0, torch.sin(angles), torch.cos(angles))

        # Expand angles tensor to have the same shape as input tensor
        angles = angles.unsqueeze(0).repeat(x.shape[0], 1, 1)[:,:,:self.emb]

        # Cast input tensor to float
        x = x.float()
        x = x * math.sqrt(self.emb)
        angles = angles.float()
        angles = angles.to(x.device)
        

        # Add embeddings and input
        return x + angles


# In[12]:


# Create a test input tensor
batch_size = 4
seq_leng = 2048
embed_dim = 512
test_input = torch.randn(batch_size, seq_leng, embed_dim)*0

# Create an instance of the PositionalEmbedding module
max_len = seq_leng
pos_embedding = PositionalEmbedding(max_len, embed_dim)

# Pass the test input through the positional embedding module
output = pos_embedding(test_input)

# Print the output shape
print(output.shape)


# In[13]:


plt.pcolormesh(output[0].numpy().T, cmap='RdBu')
plt.ylabel('Depth')
plt.xlabel('Position')
plt.colorbar()
plt.show()


# In[14]:


def random_noise(samples, seq_length, dim):
    random_numbers = torch.randn(samples, seq_length, dim)
    random_numbers = torch.tensor(random_numbers)
    random_numbers = torch.reshape(random_numbers,(samples,seq_length,dim))
    
    return random_numbers
    


# In[15]:


class CustomMSELoss(nn.Module):
    def __init__(self, scale_factor=1.0):
        super(CustomMSELoss, self).__init__()
        self.scale_factor = scale_factor

    def forward(self, predicted, target):
        mse_loss = nn.MSELoss()
        mse = mse_loss(predicted, target)

        # Apply scaling factor to the MSE loss
        scaled_loss = self.scale_factor * mse

        return scaled_loss
    

class CustomBCELoss(nn.Module):
    def __init__(self):
        super(CustomBCELoss, self).__init__()

    def forward(self, inputs, target):
        # Apply sigmoid activation function to input tensor
        inputs = torch.sigmoid(inputs)

        # Calculate binary cross entropy loss
        loss = -target * torch.log(inputs) - (1 - target) * torch.log(1 - inputs)

        # Take the mean across the batch
        loss = torch.mean(loss)
        
        # Convert the predicted probabilities to binary predictions (0 or 1)
        binary_pred = (inputs >= 0.5).float()

        # Compare the binary predictions with the ground truth labels
        correct = (binary_pred == target).sum().item()

        # Calculate the accuracy
        accuracy = correct / len(target)

        return loss, accuracy


# In[16]:


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=512, n_heads=8):
        """
        Args:
            embed_dim: dimension of embeding vector output
            n_heads: number of self attention heads
        """
        super(MultiHeadAttention, self).__init__()

        self.embed_dim = embed_dim    #512 dim
        self.n_heads = n_heads   #8
        self.single_head_dim = int(self.embed_dim / self.n_heads)   #512/8 = 64  . each key,query, value will be of 64d
       
        #key,query and value matrixes    #64 x 64   
        self.query_matrix = nn.Linear(self.single_head_dim , self.single_head_dim ,bias=False)  # single key matrix for all 8 keys #512x512
        self.key_matrix = nn.Linear(self.single_head_dim  , self.single_head_dim, bias=False)
        self.value_matrix = nn.Linear(self.single_head_dim ,self.single_head_dim , bias=False)
        self.out = nn.Linear(self.n_heads*self.single_head_dim ,self.embed_dim) 
        
        # Apply weight initialization
#         for module in [self.query_matrix, self.key_matrix, self.value_matrix, self.out]:
#             init.kaiming_uniform_(module.weight, nonlinearity='relu')
#             init.constant_(module.bias, 0.0)

    def forward(self,key,query,value,mask=None):    #batch_size x sequence_length x embedding_dim    # 32 x 10 x 512
        
        """
        Args:
           key : key vector
           query : query vector
           value : value vector
           mask: mask for decoder
        
        Returns:
           output vector from multihead attention
        """
        batch_size = key.size(0)
        seq_length = key.size(1)
        
        # query dimension can change in decoder during inference. 
        # so we cant take general seq_length
        seq_length_query = query.size(1)
        
        # 32x10x512
        key = key.view(batch_size, seq_length, self.n_heads, self.single_head_dim)  #batch_size x sequence_length x n_heads x single_head_dim = (32x10x8x64)
        query = query.view(batch_size, seq_length_query, self.n_heads, self.single_head_dim) #(32x10x8x64)
        value = value.view(batch_size, seq_length, self.n_heads, self.single_head_dim) #(32x10x8x64)
       
        k = self.key_matrix(key)       # (32x10x8x64)
        q = self.query_matrix(query)   
        v = self.value_matrix(value)

        q = q.transpose(1,2)  # (batch_size, n_heads, seq_len, single_head_dim)    # (32 x 8 x 10 x 64)
        k = k.transpose(1,2)  # (batch_size, n_heads, seq_len, single_head_dim)
        v = v.transpose(1,2)  # (batch_size, n_heads, seq_len, single_head_dim)
       
        # computes attention
        # adjust key for matrix multiplication
        k_adjusted = k.transpose(-1,-2)  #(batch_size, n_heads, single_head_dim, seq_ken)  #(32 x 8 x 64 x 10)
        product = torch.matmul(q, k_adjusted)  #(32 x 8 x 10 x 64) x (32 x 8 x 64 x 10) = #(32x8x10x10)
      
        
        # fill those positions of product matrix as (-1e20) where mask positions are 0
        if mask is not None:
             product = product.masked_fill(mask == 0, float("-1e20"))

        #divising by square root of key dimension
        product = product / math.sqrt(self.single_head_dim) # / sqrt(64)

        #applying softmax
        scores = F.softmax(product, dim=-1)
        
 
        #mutiply with value matrix
        scores = torch.matmul(scores, v)  ##(32x8x 10x 10) x (32 x 8 x 10 x 64) = (32 x 8 x 10 x 64) 
#         print('Score: ', scores)
        #concatenated output
        concat = scores.transpose(1,2).contiguous().view(batch_size, seq_length_query, self.single_head_dim*self.n_heads)  # (32x8x10x64) -> (32x10x8x64)  -> (32,10,512)
        
        output = self.out(concat) #(32,10,512) -> (32,10,512)
       
        return output


# In[17]:


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, n_heads=8):
        super(TransformerBlock, self).__init__()
        

        self.attention = MultiHeadAttention(embed_dim, n_heads)
        
        self.norm1 = nn.LayerNorm(embed_dim) 
        self.norm2 = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, expansion_factor * embed_dim),
            nn.ReLU(),
            nn.Linear(expansion_factor * embed_dim, embed_dim),
        )
        



        
        self.dropout1 = nn.Dropout(0.1)
        self.dropout2 = nn.Dropout(0.1)
        

    def forward(self,key,query,value):
        
        attention_out = self.attention(key, query, value)
        
        attention_residual_out = attention_out + value
        norm1_out = self.norm1(attention_residual_out) 
        norm1_out = self.dropout1(norm1_out)
        

        feed_fwd_out = self.feed_forward(norm1_out)
        
        feed_fwd_residual_out = feed_fwd_out + norm1_out
        norm2_out = self.norm2(feed_fwd_residual_out)  
        norm2_out = self.dropout2(norm2_out)
        

        return norm2_out



class TransformerEncoder(nn.Module):
    """
    Args:
        seq_len : length of input sequence
        embed_dim: dimension of embedding
        num_layers: number of encoder layers
        expansion_factor: factor which determines number of linear layers in feed forward layer
        n_heads: number of heads in multihead attention
        
    Returns:
        out: output of the encoder
    """
    def __init__(self, seq_len, vocab_size, embed_dim, num_layers=2, expansion_factor=4, n_heads=8, latent_dim=8):
        super(TransformerEncoder, self).__init__()
        
        self.embedding_layer = FeedForward(vocab_size, 32, embed_dim)
        self.positional_encoder = PositionalEmbedding(seq_len, embed_dim)
        self.embed_dim = embed_dim

        self.dropout1 = nn.Dropout(0.1)
#         self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads)
#         self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.layers = nn.ModuleList([TransformerBlock(embed_dim, expansion_factor, n_heads) for i in range(num_layers)])
        self.latent_dim = latent_dim
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 1024),
            nn.ReLU(), 
            nn.Linear(1024, 3), 
            nn.Softmax(dim=1)  
        )

        
        # Initialize the weights and biases
#         for module in self.classifier.modules():
#             if isinstance(module, nn.Linear):
#                 init.kaiming_uniform_(module.weight)
#                 init.constant_(module.bias, 0.0)
    
    def forward(self, x):
        out = self.embedding_layer(x)
        out = self.positional_encoder(out)
        out = self.dropout1(out)
#         out = self.transformer_encoder(out)
       
        
        for layer in self.layers:
            out = layer(out,out,out)
        
#         print(out.shape)
#         out = out.view(out.size(0), -1)
#         print(out.shape)
        out = torch.max(out,dim=1)[0]
#         print(out.shape)
        out = self.classifier(out)
        return out  #32x10x512


# In[18]:


V,T = X_test[:100000],X_test[100000:]
I,J = y_test[:100000],y_test[100000:]


# In[19]:


(I[:,0]==1).sum(),(I[:,1]==1).sum(),(I[:,2]==1).sum()


# In[20]:


(J[:,0]==1).sum(),(J[:,1]==1).sum(),(J[:,2]==1).sum()


# In[21]:


num_epochs = 500
# Define batch size
batch_size = 256
# Create DataLoader for training set
train_dataset = torch.utils.data.TensorDataset(X_train.float(), y_train.float())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
valid_dataset = torch.utils.data.TensorDataset(V.float(), I.float())
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_dataset = torch.utils.data.TensorDataset(T.float(), J.float())
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# In[22]:


print(X_train.dtype)
print(y_train.dtype)


# In[23]:


X_train.shape, y_train.shape


# In[28]:


train = 0
avg = 'weighted'
seq_length = 144
reset_random_seeds(seed)
src_vocab_size = features
target_vocab_size = features
num_layers = 1
latent_dim = 8
embed_dim = 512
n_heads = 16
expansion_factor = 2

Encoder = TransformerEncoder(seq_length, src_vocab_size, embed_dim, num_layers=num_layers, expansion_factor=expansion_factor, n_heads=n_heads, latent_dim=latent_dim).to(device)

criterion = nn.CrossEntropyLoss()
optimizer_encoder = optim.Adam(Encoder.parameters(), lr=1e-5)

# Define early stopping parameters
patience = 500
best_loss = 0.0
epochs_without_improvement = 0
Encoder.load_state_dict(torch.load('EncoderClass301.pt'))
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
            optimizer_encoder.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = Encoder(inputs)
            
            # Calculate loss
            loss = criterion(outputs, labels)
            loss = loss.float()

            # Backpropagation
            loss.backward()
            optimizer_encoder.step()

            # Update running loss
            running_loss += loss.item()
            
            latents.append(outputs)
            LS.append(labels)
            

        # Calculate average loss per epoch
        epoch_loss = running_loss / len(train_loader)
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
        print(epochs_without_improvement, best_loss)
        
        if epoch%1==0:
            M,N = [],[]
            for j, (inp, lab) in enumerate(valid_loader):
                inp = inp.to(device)
                lab = lab.to(device)
                
                with torch.no_grad():
                    outp = Encoder(inp)
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
            print()
            print()
        
        # Check for early stopping and save the best weights
        th = (conf_matrix[0][0]> conf_matrix[0][1]+conf_matrix[0][2] and conf_matrix[1][1]>conf_matrix[1][0]+conf_matrix[1][2] and conf_matrix[2][2]>conf_matrix[2][0]+conf_matrix[2][1])
        if th and vaccuracy > best_loss:
            best_loss = vaccuracy
            epochs_without_improvement = 0
            # Save the model weights
            torch.save(Encoder.state_dict(), 'EncoderClass301.pt')
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
# Recover the best weights
Encoder.load_state_dict(torch.load('EncoderClass301.pt'))


# In[27]:


evalution = 1


if evalution:
    M,N = [],[]
    for j, (inp, lab) in enumerate(train_loader):
        inp = inp.to(device)
        lab = lab.to(device)

        with torch.no_grad():
            outp = Encoder(inp)
        M.append(outp)
        N.append(lab)

    M = torch.cat(M, dim=0)
    N = torch.cat(N, dim=0)
    _, pre = torch.max(M.data, 1)
    _, lb = torch.max(N.data, 1)
    total = lb.size(0)
    correct = (pre == lb).sum().item()

    accuracy = 100 * correct / total
    print('-----------------------------------------')
    print(f'Train Accuracy: {accuracy:.2f}%')
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
    print(accuracy,",",precision,",",recall,",",f1,",",auc)
    print('-----------------------------------------')

    M,N = [],[]
    for j, (inp, lab) in enumerate(valid_loader):
        inp = inp.to(device)
        lab = lab.to(device)

        with torch.no_grad():
            outp = Encoder(inp)
        M.append(outp)
        N.append(lab)

    M = torch.cat(M, dim=0)
    N = torch.cat(N, dim=0)
    _, pre = torch.max(M.data, 1)
    _, lb = torch.max(N.data, 1)
    total = lb.size(0)
    correct = (pre == lb).sum().item()

    accuracy = 100 * correct / total
    print('-----------------------------------------')
    print(f'Valid Accuracy: {accuracy:.2f}%')
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
    print(accuracy,",",precision,",",recall,",",f1,",",auc)
    print('-----------------------------------------')
    
    M,N = [],[]
    for j, (inp, lab) in enumerate(test_loader):
        inp = inp.to(device)
        lab = lab.to(device)

        with torch.no_grad():
            outp = Encoder(inp)
        M.append(outp)
        N.append(lab)

    M = torch.cat(M, dim=0)
    N = torch.cat(N, dim=0)
    _, pre = torch.max(M.data, 1)
    _, lb = torch.max(N.data, 1)
    total = lb.size(0)
    correct = (pre == lb).sum().item()

    accuracy = 100 * correct / total
    print('-----------------------------------------')
    print(f'Test Accuracy: {accuracy:.2f}%')
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
    print(accuracy,",",precision,",",recall,",",f1,",",auc)
    print('-----------------------------------------')


# In[ ]:




