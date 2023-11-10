#!/usr/bin/env python
# coding: utf-8

# In[1]:


seed=198


# In[ ]:





# In[7]:


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
from sklearn.preprocessing import OneHotEncoder
import os 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, accuracy_score

from torch.optim.lr_scheduler import StepLR

warnings.simplefilter("ignore")
print(torch.__version__)


# In[8]:


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


# In[49]:


def reset_random_seeds(seed):
    os.environ['PYTHONHASHSEED']=str(seed)
#     tf.random.set_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


# In[50]:


# pickle_file_path = '/media/terminator/Data/NatureData/Auto Ten Minutes.pkl'
pickle_file_path = '/media/terminator/NatureData/Auto Ten Minutes.pkl'


# In[51]:


# Load data from the pickle file
with open(pickle_file_path, "rb") as f:
    loaded_data_dict = pickle.load(f)

# Retrieve data and labels
X_train = loaded_data_dict["X_train"]
y_train = loaded_data_dict["y_train"]
X_test = loaded_data_dict["X_test"]
y_test = loaded_data_dict["y_test"]


# In[52]:


X_train.shape,y_train.shape


# In[53]:


features=13


# In[54]:


X_train


# In[55]:


def reconstructData(packets):
    # Take the first packet
    first_packet = packets[0]
    
    # Concatenate the last index of every packet
    reconstructed_data = np.concatenate([packet[-1:] for packet in packets[1:]])
    
    # Combine the first packet with the concatenated data
    reconstructed_data = np.concatenate([first_packet, reconstructed_data])
    
    return reconstructed_data


# In[56]:


def norm(out):
    out = out.view(-1,13)
    min_values = out.min(dim=0).values
    max_values = out.max(dim=0).values

    # Apply min-max normalization
    normalized_tensor = (out - min_values) / (max_values - min_values)

    out = normalized_tensor.view(-1,144,13)
    
    return out


# In[57]:


Valid,Test = X_test[:100000],X_test[100000:]
Valid_label,Test_label = y_test[:100000],y_test[100000:]

# X_train = norm(X_train)
# y_train = norm(y_train)
# Valid = norm(Valid)
# Valid_label = norm(Valid_label)
# Test = norm(Test)
# Test_label = norm(Test_label)

train_mov = y_train[:,-1,-1]
train_mov[np.where(train_mov==1)]=2
train_mov[np.where(train_mov==0.5)]=1

valid_mov = Valid_label[:,-1,-1]
valid_mov[np.where(valid_mov==1)]=2
valid_mov[np.where(valid_mov==0.5)]=1

test_mov = Test_label[:,-1,-1]
test_mov[np.where(test_mov==1)]=2
test_mov[np.where(test_mov==0.5)]=1

X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train)
Valid = torch.tensor(Valid)
Valid_label = torch.tensor(Valid_label)
Test = torch.tensor(Test)
Test_label = torch.tensor(Test_label)

encoder = OneHotEncoder(categories='auto', sparse=False)
train_mov = encoder.fit_transform(train_mov.view(-1,1))
valid_mov = encoder.fit_transform(valid_mov.view(-1,1))
test_mov = encoder.fit_transform(test_mov.view(-1,1))

train_mov = torch.tensor(train_mov)
valid_mov = torch.tensor(valid_mov)
test_mov = torch.tensor(test_mov)

# train_mov = y_train1
# valid_mov = y_test1[:100000]
# test_mov = y_test1[100000:]


# Define batch size
batch_size = 128
# Create DataLoader for training set
train_dataset = torch.utils.data.TensorDataset(X_train.float(), y_train.float(),train_mov.float())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
valid_dataset = torch.utils.data.TensorDataset(Valid.float(), Valid_label.float(),valid_mov.float())
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_dataset = torch.utils.data.TensorDataset(Test.float(), Test_label.float(),test_mov.float())
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# In[58]:


np.where(valid_mov[:,2]==1)[0].shape


# In[59]:


X_train.shape, y_train.shape


# In[60]:


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


# In[61]:


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


# In[62]:


plt.pcolormesh(output[0].numpy().T, cmap='RdBu')
plt.ylabel('Depth')
plt.xlabel('Position')
plt.colorbar()
plt.show()


# In[63]:


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


# In[64]:


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by num_heads"
        num_heads=n_heads
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
        for layer in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(layer.weight)
            if layer.bias is not None:
                nn.init.constant_(layer.bias, 0)
        
    def scaled_dot_product_attention(self, K, Q, V, mask=None):
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, -1e9)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        output = torch.matmul(attn_probs, V)
        return output
        
    def split_heads(self, x):
        batch_size, seq_length, d_model = x.size()
        return x.view(batch_size, seq_length, self.num_heads, self.d_k).transpose(1, 2)
        
    def combine_heads(self, x):
        batch_size, _, seq_length, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_length, self.d_model)
        
    def forward(self, K, Q, V, mask=None):
        Q = self.split_heads(self.W_q(Q))
        K = self.split_heads(self.W_k(K))
        V = self.split_heads(self.W_v(V))
        
        attn_output = self.scaled_dot_product_attention(Q, K, V, mask)
        output = self.W_o(self.combine_heads(attn_output))
        return output


# In[65]:


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


# In[66]:


class TransformerEncoder(nn.Module):
   
    def __init__(self, seq_len, vocab_size, embed_dim, num_layers=2, expansion_factor=4, n_heads=8, latent_dim=8):
        super(TransformerEncoder, self).__init__()
        

        self.dropout1 = nn.Dropout(0.1)

        self.layers = nn.ModuleList([TransformerBlock(embed_dim, expansion_factor, n_heads) for i in range(num_layers)])
        


    
    def forward(self, x):
        
        for layer in self.layers:
            x = layer(x,x,x)
        
        return x
    


# In[67]:


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
            nn.Linear(hidden_size, output_size), 
        )
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, hn = self.lstm(x, (h0.detach(), c0.detach()))
        
        # Index hidden state of last time step
        out = self.classifier(out)
        
        return out


# In[68]:


class LSTMModelE(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModelE, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        for name, param in self.lstm.named_parameters():
            if 'weight' in name:
                nn.init.xavier_uniform_(param)
        
        # Fully connected layer
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, output_size), 
        )
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, hn = self.lstm(x, (h0.detach(), c0.detach()))
        
        # Index hidden state of last time step
        out = self.classifier(out[:,-1,:])
        
        return out


# In[69]:



class Classifier(nn.Module):
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
        super(Classifier, self).__init__()
        
        self.embedding_layer = nn.Sequential(
            nn.Linear(8, embed_dim*2),
            nn.ReLU(),
            nn.Linear(embed_dim*2, embed_dim)
        )
        self.positional_encoder = PositionalEmbedding(seq_len, embed_dim)
        self.embed_dim = embed_dim
        self.lstm = LSTMModel(input_size=embed_dim, hidden_size=embed_dim*2, num_layers=1, output_size=embed_dim)
        self.dropout1 = nn.Dropout(0.1)

        self.layers = nn.ModuleList([TransformerBlock(embed_dim, expansion_factor, n_heads) for i in range(num_layers)])
        self.latent_dim = latent_dim
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim*1, 1024), 
            nn.ReLU(),
            nn.Linear(1024,3),
            nn.Softmax(dim=1)  
        )


    
    def forward(self, x):
        out = x
        out = self.embedding_layer(x)
#         out = self.positional_encoder(out)
#         out = self.dropout1(out)
       
#         out=x
        for layer in self.layers:
            out = layer(out,out,out)
#         out = x
#         out = self.lstm(out)
#         out = out.view(out.size(0), -1)
        out = self.classifier(out[:,-1,:])
        return out  


# In[ ]:





# In[70]:


class HierarchicalTransformerEncoder(nn.Module):
   
    def __init__(self, seq_len, vocab_size, embed_dim, num_layers=2, expansion_factor=4, n_heads=8, latent_dim=8):
        super(HierarchicalTransformerEncoder, self).__init__()
        
        self.embedding_layer =  nn.Sequential(
            nn.Linear(13, embed_dim*2),
            nn.ReLU(),
            nn.Linear(embed_dim*2, embed_dim)
        )
        self.positional_encoder = PositionalEmbedding(seq_len, embed_dim)
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.lstm = LSTMModel(embed_dim, embed_dim*2, 1, embed_dim)

        self.dropout1 = nn.Dropout(0.1)

        self.normal_transformer = TransformerEncoder(seq_length, src_vocab_size, embed_dim, num_layers=num_layers, expansion_factor=expansion_factor, n_heads=n_heads, latent_dim=latent_dim)
        self.lower_transformer = TransformerEncoder(6, src_vocab_size, embed_dim, num_layers=num_layers, expansion_factor=expansion_factor, n_heads=n_heads, latent_dim=latent_dim)
        self.higher_transformer = TransformerEncoder(24, src_vocab_size, embed_dim, num_layers=num_layers, expansion_factor=expansion_factor, n_heads=n_heads, latent_dim=latent_dim)
        self.latent_dim = latent_dim
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 1024), 
            nn.ReLU(),
            nn.Linear(1024,8),
        )
        

        

    def forward(self, x):
        out = self.embedding_layer(x)
        out = self.dropout1(out)
        out = self.positional_encoder(out)
        
        out = self.normal_transformer(out)
        
        segments = out.view(-1, 24, 6, out.size(2))
        
        lower_level_outputs = []
        for segment in segments:
            segment_encoded = self.lower_transformer(segment)
            
            lower_level_outputs.append(segment_encoded)
       
       
        aggregated_representation = torch.stack(lower_level_outputs, dim=0).mean(dim=2)
        
        out = self.higher_transformer(aggregated_representation)

        
#         out = out.reshape(out.shape[0],-1)
        out = self.classifier(out)
        
        return out 
    


# In[71]:


import torch.nn as nn
import torch

class AttentionUpsampling(nn.Module):
    def __init__(self, embed_dim):
        super(AttentionUpsampling, self).__init__()

    def upsampling_function(self, x, x0):
        # Assuming x is of shape (batch_size, num_channels, seq_len)
        # and x0 is of shape (batch_size, num_channels, seq_len)
        
        # Duplicate the single channel to create six channels
        
        upsampled_x = x.repeat(1, 6, 1)

        return upsampled_x

    def forward(self, x, x0):
        # Apply upsampling using the upsampling_function
        upsampled_x = self.upsampling_function(x, x0)
        return upsampled_x


# In[ ]:





# In[72]:


import torch.nn as nn
import torch

class HierarchicalTransformerDecoder(nn.Module):
    def __init__(self, seq_len, vocab_size, embed_dim, num_layers=2, expansion_factor=4, n_heads=8, latent_dim=8):
        super(HierarchicalTransformerDecoder, self).__init__()
        
        self.embedding_layer = nn.Sequential(
            nn.Linear(8, embed_dim*2),
            nn.ReLU(),
            nn.Linear(embed_dim*2, embed_dim)
        )
        self.positional_encoder = PositionalEmbedding(seq_len, embed_dim)
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.latent_dim = latent_dim

        self.dropout1 = nn.Dropout(0.1)
        self.lstm = LSTMModel(embed_dim, embed_dim*2, 1, embed_dim)
        self.normal_transformer = TransformerEncoder(seq_len, vocab_size, embed_dim, num_layers=num_layers, expansion_factor=expansion_factor, n_heads=n_heads, latent_dim=latent_dim)
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=3, stride=1, padding=1)
        self.attention_upsampling = AttentionUpsampling(embed_dim)  # Initialize the AttentionUpsampling module

        self.lower_transformer = TransformerEncoder(6, src_vocab_size, embed_dim, num_layers=num_layers, expansion_factor=expansion_factor, n_heads=n_heads, latent_dim=latent_dim)
        self.higher_transformer = TransformerEncoder(24, src_vocab_size, embed_dim, num_layers=num_layers, expansion_factor=expansion_factor, n_heads=n_heads, latent_dim=latent_dim)
        self.latent_dim = latent_dim
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 13)
        )
        
        self.lower_parallel = nn.ModuleList([nn.Conv1d(in_channels=1, out_channels=6, kernel_size=3, stride=1, padding=1) for _ in range(24)])  # Create 24 parallel convolution layers
        

        

    def forward(self, x):
        out = self.embedding_layer(x)
        out = self.dropout1(out)
        out = self.higher_transformer(out)
        out = out.view(-1, 24, 1, self.embed_dim)
        new = []

        for segment in out:
            # Use the attention-based upsampling module here
#             print(segment.shape)
            cn = self.conv1(segment)  # Use the same segment for queries and keys
#             print(cn.shape)
#             cn = self.lstm(cn)
            cn = self.lower_transformer(cn)
            cn = cn.view(144, self.embed_dim)
            new.append(cn)

        higher = torch.stack(new, dim=0)
        out = self.normal_transformer(higher)
        out = self.classifier(out)
        return out


# In[73]:


train = 0
avg = 'weighted'
seq_length = 144
num_epochs = 5000
reset_random_seeds(seed)
src_vocab_size = features
target_vocab_size = features
num_layers = 1
latent_dim = 8
embed_dim = 512
n_heads = 16
expansion_factor = 2

Encoder = HierarchicalTransformerEncoder(seq_length, src_vocab_size, embed_dim, num_layers=num_layers, expansion_factor=expansion_factor, n_heads=n_heads, latent_dim=latent_dim).to(device)
Decoder = HierarchicalTransformerDecoder(seq_length, src_vocab_size, embed_dim, num_layers=num_layers, expansion_factor=expansion_factor, n_heads=n_heads, latent_dim=latent_dim).to(device)

Cf = Classifier(24, vocab_size=latent_dim, embed_dim=128, num_layers=1, expansion_factor=2, n_heads=2, latent_dim=latent_dim).to(device)

criterion = nn.MSELoss()
criterion2 = nn.CrossEntropyLoss()
optimizer_encoder = optim.Adam(Encoder.parameters(), lr=1e-6)
optimizer_decoder = optim.Adam(Decoder.parameters(), lr=1e-6)
optimizer_cf = optim.Adam(Cf.parameters(), lr=1e-6)

# Define early stopping parameters
patience = 500
best_loss = 95.0
epochs_without_improvement = 0
vaccuracy = 0
loss1=0

# # Recover the best weights
Encoder.load_state_dict(torch.load('EncoderAuto1107.pt'))
Decoder.load_state_dict(torch.load('DecoderAuto1107.pt'))
Cf.load_state_dict(torch.load('CfAuto1107.pt'))

if train:
    # Step 4: Training loop
    for epoch in range(0,num_epochs):
        running_loss = 0.0
        running_loss2 = 0.0
        latents = []
        PR, PL = [], []
        train_input,train_output = [], []

        # Iterate over mini-batches
        for i, (inputs, labels,pre) in enumerate(train_loader):
            
            probability = 0.0
            coin_toss = random.random()  

            if coin_toss < probability:
                continue
            
            # Clear gradients
            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()
            optimizer_cf.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            pre = pre.to(device)
            
            # Forward pass
            latent = Encoder(inputs)
            
            outputs = Decoder(latent)
            
            
            # Calculate loss
            loss1 = criterion(outputs, labels)
            loss1 = loss1.float()
            

            # Backpropagation
            loss1.backward()
            
            optimizer_encoder.step()
            optimizer_decoder.step()
            
            optimizer_encoder.zero_grad()
            optimizer_cf.zero_grad()
            latent1 = Encoder(inputs)
            
            prediction = Cf(latent1)
            loss2 = criterion2(pre,prediction)
            loss2 = loss2.float()
            loss2.backward()
            optimizer_cf.step()
            optimizer_encoder.step()
            
            # Update running loss
            running_loss += loss1.item()
#             running_loss=0
            running_loss2 += loss2.item()
            
            
            PR.append(prediction)
            PL.append(pre)
            
            latents.append(latent)
            

        # Calculate average loss per epoch
        PR = torch.cat(PR, dim=0)
        PL = torch.cat(PL, dim=0)
        _, predicted = torch.max(PR.data, 1)
        _, lbl = torch.max(PL.data, 1)
        total = lbl.size(0)
        correct = (predicted == lbl).sum().item()

        accuracy = 100 * correct / total
        print(f'Train Accuracy: {accuracy:.2f}%')
        auc = roc_auc_score(lbl.cpu().detach().numpy(),PR.cpu().detach().numpy(),multi_class='ovr')
        print("AUC ",auc)
        
        conf_matrix2 = confusion_matrix(lbl.cpu().detach().numpy(),predicted.cpu().detach().numpy())
        print('Confusion Matrix:')
        print(conf_matrix2)
        # Calculate average loss per epoch
        epoch_loss = running_loss / len(train_loader)
        epoch_loss2 = running_loss2 / len(train_loader)
        print(f"Epoch: {epoch+1}, Train Loss: {round(epoch_loss,6)}, Train Loss2: {round(epoch_loss2,6)}")
        print(epochs_without_improvement, best_loss)
        
        LT = torch.cat(latents, dim=0)
        LT = LT.cpu().detach().numpy()
        labl = lbl.cpu().detach().numpy()
        
        if epoch%50==0:
            real_data_reshaped = np.reshape(LT, (LT.shape[0], -1))


            # Perform tsne on real and fake data
            perplexity = min(30, real_data_reshaped.shape[0] - 1)
            tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed)
            real_tsne = tsne.fit_transform(real_data_reshaped)
            label_names = ['No-Movement', 'Small Movement',  'Large Movement']
            # Plot the t-SNE visualization with labeled data
            plt.figure(figsize=(20, 15))
            for label in np.unique(labl):
                indices = np.where(labl == label)
                plt.scatter(real_tsne[indices, 0], real_tsne[indices, 1], label=label_names[int(label)], edgecolors='black')

            plt.legend()
            plt.show()
        
        th = (conf_matrix2[0][0]> conf_matrix2[0][1]+conf_matrix2[0][2] and conf_matrix2[1][1]>conf_matrix2[1][0]+conf_matrix2[1][2] and conf_matrix2[2][2]>conf_matrix2[2][0]+conf_matrix2[2][1])
        if epoch%1==0:
            valid_latents = []
            VR,VL = [], []
            valid_loss = 0.0
            for j, (inp, lab, ps) in enumerate(valid_loader):
                probability = 0.0
                coin_toss = random.random()  

                if coin_toss < probability:
                    continue
                inp = inp.to(device)
                lab = lab.to(device)
                ps = ps.to(device)
                
                with torch.no_grad():
                    # Forward pass
                    lt = Encoder(inp)
                    outp = Cf(lt)
                    
                VR.append(outp)
                VL.append(ps)
            
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
            print("AUC ",auc)
            print('-----------------------------------------')
            print()
            print()
            valid_latents = []
            VR,VL = [], []
            valid_loss = 0.0
#             for j, (inp, lab, ps) in enumerate(test_loader):
#                 probability = 0.0
#                 coin_toss = random.random()  

#                 if coin_toss < probability:
#                     continue
#                 inp = inp.to(device)
#                 lab = lab.to(device)
#                 ps = ps.to(device)
                
#                 with torch.no_grad():
#                     # Forward pass
#                     lt = Encoder(inp)
#                     outp = Cf(lt)
                    
#                 VR.append(outp)
#                 VL.append(ps)
            
#             VR = torch.cat(VR, dim=0)
#             VL = torch.cat(VL, dim=0)
#             _, vpre = torch.max(VR.data, 1)
#             _, vlbl = torch.max(VL.data, 1)
#             total = vlbl.size(0)
#             correct = (vpre == vlbl).sum().item()

#             taccuracy = 100 * correct / total
#             print('-----------------------------------------')
#             print(f'Test Accuracy: {taccuracy:.2f}%')
#             lb = vlbl.cpu().detach().numpy()
#             pre = vpre.cpu().detach().numpy()
#             conf_matrix1 = confusion_matrix(lb,pre)
#             precision = precision_score(lb, pre,average=avg)
#             recall = recall_score(lb, pre,average=avg)
#             f1 = f1_score(lb, pre,average=avg)
            
#             print('Confusion Matrix:')
#             print(conf_matrix1)
#             print()
#             print("Precision:", precision)
#             print("Recall:", recall)
#             print("F1 Score:", f1)
#             auc = roc_auc_score(lb,VR.cpu().detach().numpy(),multi_class='ovr')
#             print("AUC ",auc)
#             print('-----------------------------------------')
#             print()
#             print()
                    
                    
            # Check for early stopping and save the best weights
            th = (conf_matrix[0][0]> conf_matrix[0][1]+conf_matrix[0][2] and conf_matrix[1][1]>conf_matrix[1][0]+conf_matrix[1][2] and conf_matrix[2][2]>conf_matrix[2][0]+conf_matrix[2][1])
            if th and vaccuracy > best_loss:
                best_loss = vaccuracy
                epochs_without_improvement = 0
                # Save the model weights
                torch.save(Encoder.state_dict(), 'EncoderAuto1107.pt')
                torch.save(Decoder.state_dict(), 'DecoderAuto1107.pt')
                torch.save(Cf.state_dict(), 'CfAuto1107.pt')
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
Encoder.load_state_dict(torch.load('EncoderAuto1107.pt'))
Decoder.load_state_dict(torch.load('DecoderAuto1107.pt'))
Cf.load_state_dict(torch.load('CfAuto1107.pt'))


# In[74]:


print('Done')


# In[75]:


avg = 'weighted'
seq_length = 144
num_epochs = 5000
reset_random_seeds(seed)
src_vocab_size = features
target_vocab_size = features
num_layers = 1
latent_dim = 8
embed_dim = 512
n_heads = 16
expansion_factor = 2

Encoder = HierarchicalTransformerEncoder(seq_length, src_vocab_size, embed_dim, num_layers=num_layers, expansion_factor=expansion_factor, n_heads=n_heads, latent_dim=latent_dim).to(device)
Decoder = HierarchicalTransformerDecoder(seq_length, src_vocab_size, embed_dim, num_layers=num_layers, expansion_factor=expansion_factor, n_heads=n_heads, latent_dim=latent_dim).to(device)

Cf = Classifier(24, vocab_size=latent_dim, embed_dim=128, num_layers=1, expansion_factor=2, n_heads=2, latent_dim=latent_dim).to(device)


# In[76]:


Encoder.load_state_dict(torch.load('EncoderAuto1107.pt'))
Decoder.load_state_dict(torch.load('DecoderAuto1107.pt'))
Cf.load_state_dict(torch.load('CfAuto1107.pt'))


# In[42]:


epoch=0
valid_latents = []
VR,VL = [], []
valid_loss = 0.0
for j, (inp, lab, ps) in enumerate(train_loader):
    probability = 0.0
    coin_toss = random.random()  

    if coin_toss < probability:
        continue
    inp = inp.to(device)
    lab = lab.to(device)
    ps = ps.to(device)

    with torch.no_grad():
        # Forward pass
        lt = Encoder(inp)
        outp = Cf(lt)

    VR.append(outp)
    VL.append(ps)

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
print("AUC ",auc)
print(vaccuracy,",",precision,",",recall,",",f1,",",auc)
print('-----------------------------------------')
print()
print()



valid_latents = []
VR,VL = [], []
valid_loss = 0.0
for j, (inp, lab, ps) in enumerate(valid_loader):
    probability = 0.0
    coin_toss = random.random()  

    if coin_toss < probability:
        continue
    inp = inp.to(device)
    lab = lab.to(device)
    ps = ps.to(device)

    with torch.no_grad():
        # Forward pass
        lt = Encoder(inp)
        outp = Cf(lt)

    VR.append(outp)
    VL.append(ps)

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
print("AUC ",auc)
print(vaccuracy,",",precision,",",recall,",",f1,",",auc)
print('-----------------------------------------')
print()
print()


valid_latents = []
VR,VL = [], []
valid_loss = 0.0
for j, (inp, lab, ps) in enumerate(test_loader):
    probability = 0.0
    coin_toss = random.random()  

    if coin_toss < probability:
        continue
    inp = inp.to(device)
    lab = lab.to(device)
    ps = ps.to(device)

    with torch.no_grad():
        # Forward pass
        lt = Encoder(inp)
        outp = Cf(lt)

    VR.append(outp)
    VL.append(ps)

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
print("AUC ",auc)
print(vaccuracy,",",precision,",",recall,",",f1,",",auc)
print('-----------------------------------------')
# Log epoch progress (optional)
print()
print()


# In[77]:


epoch=100


# In[ ]:


bs = 1024
bsl = bs*20

for k in range(2,5):
    idx = k
    m=0
    vsam = []
    vsal = []
    vsamm = []
    vsall = []
    vsammr = []
    vsallr = []
    vsamr = []
    vsalr = []
    for j in range(epoch):
        print(j, end='')
        m= m+0.1
        Test = X_train[np.where(train_mov[:,0]==1)[0]][:bsl]
        TestLabel = train_mov[np.where(train_mov[:,0]==1)[0]][:bsl]
        mean = Test[:,:, idx].mean()
        std = Test[:,:, idx].std()

        R = mean + m + std * torch.randn(1)
        Test[:,:, idx] = Test[:,:, idx] + R

        test_dataset = torch.utils.data.TensorDataset(Test.float(), TestLabel.float())
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=False)

        predicted = []
        actual = []
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                lt = Encoder(inputs)
                test_outputs = Cf(lt)
                actual.append(labels)
                predicted.append(test_outputs)

        predicted = torch.cat(predicted, dim=0)
        actual = torch.cat(actual, dim=0)
        _, pre = torch.max(predicted.data, 1)
        _, lb = torch.max(actual.data, 1)
        total = lb.size(0)
        correct = (pre == lb).sum().item()

        vaccuracy = 100 * correct / total

        lb = lb.cpu().detach().numpy()
        pre = pre.cpu().detach().numpy()
        zero = np.where(pre==0)[0].shape[0]
        one  = np.where(pre==1)[0].shape[0]
        two  = np.where(pre==0)[0].shape[0]
        vsamm.append(m)
        vsall.append(one)

        Test = X_train[np.where(train_mov[:,1]==1)[0]][:bsl]
        TestLabel = train_mov[np.where(train_mov[:,1]==1)[0]][:bsl]
        mean = Test[:,:, idx].mean()
        std = Test[:,:, idx].std()

        R = mean + m + std * torch.randn(1)
        Test[:,:, idx] = Test[:,:, idx] - R

        test_dataset = torch.utils.data.TensorDataset(Test.float(), TestLabel.float())
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=False)

        predicted = []
        actual = []
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                lt = Encoder(inputs)
                test_outputs = Cf(lt)
                actual.append(labels)
                predicted.append(test_outputs)

        predicted = torch.cat(predicted, dim=0)
        actual = torch.cat(actual, dim=0)
        _, pre = torch.max(predicted.data, 1)
        _, lb = torch.max(actual.data, 1)
        total = lb.size(0)
        correct = (pre == lb).sum().item()

        vaccuracy = 100 * correct / total

        lb = lb.cpu().detach().numpy()
        pre = pre.cpu().detach().numpy()
        zero = np.where(pre==0)[0].shape[0]
        one = np.where(pre==1)[0].shape[0]
        two = np.where(pre==0)[0].shape[0]
        vsammr.append(m)
        vsallr.append(zero)

        Test = X_train[np.where(train_mov[:,1]==1)[0]][:bsl]
        TestLabel = train_mov[np.where(train_mov[:,1]==1)[0]][:bsl]
        mean = Test[:,:, idx].mean()
        std = Test[:,:, idx].std()

        R = mean+m + std * torch.randn(1)
        Test[:,:, idx] = Test[:,:, idx] + R

        test_dataset = torch.utils.data.TensorDataset(Test.float(), TestLabel.float())
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=False)

        predicted = []
        actual = []
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                lt = Encoder(inputs)
                test_outputs = Cf(lt)
                actual.append(labels)
                predicted.append(test_outputs)

        predicted = torch.cat(predicted, dim=0)
        actual = torch.cat(actual, dim=0)
        _, pre = torch.max(predicted.data, 1)
        _, lb = torch.max(actual.data, 1)
        total = lb.size(0)
        correct = (pre == lb).sum().item()

        vaccuracy = 100 * correct / total

        lb = lb.cpu().detach().numpy()
        pre = pre.cpu().detach().numpy()
        zero = np.where(pre==0)[0].shape[0]
        one = np.where(pre==1)[0].shape[0]
        two = np.where(pre==0)[0].shape[0]
        vsam.append(m)
        vsal.append(two)

        Test = X_train[np.where(train_mov[:,2]==1)[0]][:bsl]
        TestLabel = train_mov[np.where(train_mov[:,2]==1)[0]][:bsl]
        mean = Test[:,:, idx].mean()
        std = Test[:,:, idx].std()

        R = mean+m + std * torch.randn(1)
        Test[:,:, idx] = Test[:,:, idx]-R

        test_dataset = torch.utils.data.TensorDataset(Test.float(), TestLabel.float())
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=bs, shuffle=False)

        predicted = []
        actual = []
        for i, (inputs, labels) in enumerate(test_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                lt = Encoder(inputs)
                test_outputs = Cf(lt)
                actual.append(labels)
                predicted.append(test_outputs)

        predicted = torch.cat(predicted, dim=0)
        actual = torch.cat(actual, dim=0)
        _, pre = torch.max(predicted.data, 1)
        _, lb = torch.max(actual.data, 1)
        total = lb.size(0)
        correct = (pre == lb).sum().item()

        vaccuracy = 100 * correct / total

        lb = lb.cpu().detach().numpy()
        pre = pre.cpu().detach().numpy()
        zero = np.where(pre==0)[0].shape[0]
        one = np.where(pre==1)[0].shape[0]
        two = np.where(pre==0)[0].shape[0]
        vsamr.append(m)
        vsalr.append(one)

    print('done')
    print(idx)
    plt.plot(vsam,vsal)
    plt.plot(vsamr,vsalr)
    plt.show()
    plt.plot(vsamm,vsall)
    plt.plot(vsammr,vsallr)
    plt.show()
    for i in range(epoch):
        print(round(vsam[i],1),vsal[i], round(vsamr[i],1),vsalr[i])


# In[ ]:




