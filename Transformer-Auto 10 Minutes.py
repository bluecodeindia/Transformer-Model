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
from sklearn.preprocessing import OneHotEncoder
import os 
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

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


pickle_file_path = '/media/terminator/NatureData/Ten Minutes.pkl'


# In[6]:


# Load data from the pickle file
with open(pickle_file_path, "rb") as f:
    loaded_data_dict = pickle.load(f)


# In[7]:


# Retrieve data and labels
y_train1 = loaded_data_dict["y_train"]
y_test1 = loaded_data_dict["y_test"]


# In[8]:


# pickle_file_path = '/media/terminator/Data/NatureData/Auto Ten Minutes.pkl'
pickle_file_path = '/media/terminator/NatureData/Auto Ten Minutes.pkl'


# In[9]:


# Load data from the pickle file
with open(pickle_file_path, "rb") as f:
    loaded_data_dict = pickle.load(f)


# In[10]:


# Retrieve data and labels
X_train = loaded_data_dict["X_train"]
y_train = loaded_data_dict["y_train"]
X_test = loaded_data_dict["X_test"]
y_test = loaded_data_dict["y_test"]


# In[11]:


X_train.shape,y_train.shape


# In[12]:


features=13


# In[ ]:





# In[13]:


def reconstructData(packets):
    # Take the first packet
    first_packet = packets[0]
    
    # Concatenate the last index of every packet
    reconstructed_data = np.concatenate([packet[-1:] for packet in packets[1:]])
    
    # Combine the first packet with the concatenated data
    reconstructed_data = np.concatenate([first_packet, reconstructed_data])
    
    return reconstructed_data


# In[14]:


def norm(out):
    out = out.view(-1,13)
    min_values = out.min(dim=0).values
    max_values = out.max(dim=0).values

    # Apply min-max normalization
    normalized_tensor = (out - min_values) / (max_values - min_values)

    out = normalized_tensor.view(-1,144,13)
    
    return out


# In[15]:


Valid,Test = X_test[:100000],X_test[100000:]
Valid_label,Test_label = y_test[:100000],y_test[100000:]

X_train = norm(X_train)
y_train = norm(y_train)
Valid = norm(Valid)
Valid_label = norm(Valid_label)
Test = norm(Test)
Test_label = norm(Test_label)

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
batch_size = 512
# Create DataLoader for training set
train_dataset = torch.utils.data.TensorDataset(X_train.float(), y_train.float(),train_mov.float())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
valid_dataset = torch.utils.data.TensorDataset(Valid.float(), Valid_label.float(),valid_mov.float())
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_dataset = torch.utils.data.TensorDataset(Test.float(), Test_label.float(),test_mov.float())
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# In[16]:


X_train.shape, y_train.shape


# In[17]:


import torch.nn.init as init

class FeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, seq_len):
        super(FeedForward, self).__init__()

        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim*seq_len, seq_len*output_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, output_dim)
        )
        self.seq_len = seq_len
        self.output_dim = output_dim

    def forward(self, x):
        x = x.float()
        x = torch.reshape(x,(x.shape[0],-1))
        out = self.feed_forward(x)
        out = torch.reshape(out,(out.shape[0],self.seq_len,self.output_dim))

        return out


# In[18]:


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


# In[19]:


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


# In[20]:


plt.pcolormesh(output[0].numpy().T, cmap='RdBu')
plt.ylabel('Depth')
plt.xlabel('Position')
plt.colorbar()
plt.show()


# In[21]:


def random_noise(samples, seq_length, dim):
    random_numbers = torch.randn(samples, seq_length, dim)
    random_numbers = torch.tensor(random_numbers)
    random_numbers = torch.reshape(random_numbers,(samples,seq_length,dim))
    
    return random_numbers
    


# In[22]:


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


# In[23]:


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


# In[24]:


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
        
       
        
    def forward(self, x):
        # Initialize hidden state with zeros
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Initialize cell state
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # We need to detach as we are doing truncated backpropagation through time (BPTT)
        # If we don't, we'll backprop all the way to the start even after going through another batch
        out, _ = self.lstm(x, (h0.detach(), c0.detach()))
        
        
        return out


# In[25]:


import torch
import torch.nn as nn

class TimeSeriesEncoder(nn.Module):
    def __init__(self, input_channels,seq_len):
        super(TimeSeriesEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),  # Reduce sequence length to 36 / 4 = 9
            nn.Conv1d(128, seq_len, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)  # Reduce sequence length to 9 / 2 = 4.5 (rounded to 4)
        )
        
    
    def forward(self, x):
        x = x.permute(0,2,1)
#         print('enc1',x.shape)
        x = self.encoder(x)
#         print('enc2',x.shape)
#         x = x.permute(0,2,1)
        return x

# Create the autoencoder model
input_channels = 13  # Number of input features
# encoder = TimeSeriesEncoder(input_channels).to(device)

import torch
import torch.nn as nn

class TimeSeriesDecoder(nn.Module):
    def __init__(self, input_channels):
        super(TimeSeriesDecoder, self).__init__()
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=4),  # Increase sequence length to 40
            nn.ReLU(),
            nn.ConvTranspose1d(64, 128, kernel_size=6, stride=6),  # Increase sequence length to 240
            nn.ReLU(),
            nn.ConvTranspose1d(128, 128, kernel_size=3, stride=1, padding=1),  # Maintain sequence length at 240
            nn.ReLU(),
            nn.ConvTranspose1d(128, 144, kernel_size=3, stride=1, padding=1)  # Increase sequence length to 144
        )
        
        self.fc = nn.Linear(240,13)

        
    
    def forward(self, x):
        x = x.permute(0,2,1)
#         print('dec1',x.shape)
        x = self.decoder(x)
#         print('dec2',x.shape)
        x = self.fc(x)
#         x = x.permute(0,2,1)
        return x

# Create the autoencoder model
input_channelss = 10  # Number of input features
# decoder = TimeSeriesDecoder(input_channelss).to(device)


# In[26]:


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, n_heads=8):
        super(TransformerBlock, self).__init__()
        
        """
        Args:
           embed_dim: dimension of the embedding
           expansion_factor: fator ehich determines output dimension of linear layer
           n_heads: number of attention heads
        
        """
        self.attention = MultiHeadAttention(embed_dim, n_heads)
        
        self.norm1 = nn.LayerNorm(embed_dim) 
        self.norm2 = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim
        
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, expansion_factor * embed_dim),
            nn.ReLU(),
            nn.Linear(expansion_factor * embed_dim, embed_dim),
        )




        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        
#         # Initialize the weights and biases
#         for module in self.feed_forward.modules():
#             if isinstance(module, nn.Linear):
#                 init.kaiming_uniform_(module.weight, nonlinearity='relu')
#                 init.constant_(module.bias, 0.0)

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
        
        self.embedding_layer = FeedForward(18, 64, embed_dim, seq_len)
        self.positional_encoder = PositionalEmbedding(seq_len, embed_dim)
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.encodercnn = TimeSeriesEncoder(13,seq_len).to(device)

        self.dropout1 = nn.Dropout(0.2)
#         self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads)
#         self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.layers = nn.ModuleList([TransformerBlock(embed_dim, expansion_factor, n_heads) for i in range(num_layers)])
        self.latent_dim = latent_dim
        self.lstmmodel = LSTMModel(input_size=embed_dim, hidden_size=latent_dim, num_layers=1, output_size=latent_dim)
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim*seq_len, latent_dim*seq_len),
            nn.ReLU(),
#             nn.Linear(64, latent_dim), 
        )

        
#         Initialize the weights and biases
#         for module in self.classifier.modules():
#             if isinstance(module, nn.Linear):
#                 init.kaiming_uniform_(module.weight)
#                 init.constant_(module.bias, 0.0)
    
    def forward(self, x):
        out = self.encodercnn(x)
        out = self.embedding_layer(out)
        out = self.positional_encoder(out)
        out = self.dropout1(out)
#         out = self.transformer_encoder(out)
       
        
        for layer in self.layers:
            out = layer(out,out,out)
        
#         out = out[:, -1, :]
#         out = out.view(out.shape[0],1,-1)
        out = self.lstmmodel(out)
#         out = torch.reshape(out,(out.shape[0],-1))
#         out = self.classifier(out)
#         out = torch.reshape(out,(out.shape[0],self.seq_len,self.latent_dim))
        
        return out 


# In[27]:


class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, expansion_factor=4, n_heads=8):
        super(DecoderBlock, self).__init__()

        """
        Args:
           embed_dim: dimension of the embedding
           expansion_factor: fator ehich determines output dimension of linear layer
           n_heads: number of attention heads
        
        """
        self.attention = MultiHeadAttention(embed_dim, n_heads=8)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.2)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        self.norm1 = nn.LayerNorm(embed_dim) 
        self.norm2 = nn.LayerNorm(embed_dim)
        self.lstmmodel = LSTMModel(input_size=embed_dim, hidden_size=embed_dim, num_layers=2, output_size=13)
        self.transformer_block = TransformerBlock(embed_dim, expansion_factor, n_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, expansion_factor * embed_dim),
#             nn.BatchNorm1d(20),  # Apply batch normalization along seq_length
            nn.ReLU(),
            nn.Linear(expansion_factor * embed_dim, embed_dim),
#             nn.BatchNorm1d(20)  # Apply batch normalization along seq_length
        )
        
    def gate_mechanism(self, Q, z):
        # Get the input size from z
        input_size = z.size(-1)

        # Define the linear layers
        linear_q = nn.Linear(input_size, input_size).to(Q.device)
        linear_z = nn.Linear(input_size, input_size).to(Q.device)
        gate_transform = nn.Linear(input_size, input_size).to(Q.device)
        
#         for module in [linear_q, linear_z, gate_transform]:
#             init.xavier_uniform_(module.weight)
#             init.constant_(module.bias, 0.0)
       

        # Apply linear transformations to Q and z
        transformed_q = linear_q(Q)
        transformed_z = linear_z(z)

        # Compute the gating mechanism
        gate_input = transformed_q + transformed_z
        sigmoid = nn.Sigmoid()
        gate_output = sigmoid(gate_transform(gate_input))

        # Apply the gate to the output representation z
        output = gate_output * z

        return output
    
    def forward(self, x, enc_out, mask):
        
        """
        Args:
           key: key vector
           query: query vector
           value: value vector
           mask: mask to be given for multi head attention 
        Returns:
           out: output of transformer block
    
        """
        x = self.lstmmodel(x)#trg, enc_out
        #we need to pass mask mask only to fst attention
        attention = self.attention(x,x,x,mask=mask) #32x10x512
        x = self.dropout(self.norm(attention + x))
        
        out = self.transformer_block(enc_out, x, enc_out) #Key, Query, Value
        
        norm1_out = self.dropout1(self.norm1(out)) #32x10x512

        feed_fwd_out = self.feed_forward(norm1_out) #32x10x512 -> #32x10x2048 -> 32x10x512
        feed_fwd_residual_out = feed_fwd_out + norm1_out #32x10x512
        norm2_out = self.dropout2(self.norm2(feed_fwd_residual_out)) #32x10x512

        
        return norm2_out


class TransformerDecoder(nn.Module):
    def __init__(self, target_vocab_size, embed_dim, seq_len, num_layers=2, expansion_factor=4, n_heads=8, latent_dim=8):
        super(TransformerDecoder, self).__init__()
        """  
        Args:
           target_vocab_size: vocabulary size of taget
           embed_dim: dimension of embedding
           seq_len : length of input sequence
           num_layers: number of encoder layers
           expansion_factor: factor which determines number of linear layers in feed forward layer
           n_heads: number of heads in multihead attention
        
        """
        self.word_embedding = FeedForward(embed_dim, 512, embed_dim, seq_len)
        self.position_embedding = PositionalEmbedding(seq_len, embed_dim)
        self.embed_dim = embed_dim
        self.seq_len = seq_len
        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_dim, expansion_factor=4, n_heads=8) 
                for _ in range(num_layers)
            ]

        )
        self.lstmmodell = LSTMModel(input_size=latent_dim, hidden_size=embed_dim, num_layers=2, output_size=13)
        self.lstmmodel = LSTMModel(input_size=embed_dim, hidden_size=embed_dim, num_layers=2, output_size=13)
        self.fc_out = nn.Linear(embed_dim, target_vocab_size)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, features),
#             nn.Sigmoid()
        )
        self.query_encoded = nn.Sequential(
                          nn.Linear(embed_dim, 64),
                          nn.ReLU(),
                          nn.Linear(64, embed_dim),
        )
        self.enc_forward = nn.Sequential(
                          nn.Linear(embed_dim*seq_len, embed_dim*seq_len),
#                           nn.ReLU(),
#                           nn.Linear(64, embed_dim),
        )
        self.dropout = nn.Dropout(0.2)
        self.decodercnn = TimeSeriesDecoder(seq_len).to(device)
        self.Q_enc = TransformerEncoder(seq_length, 13+5, embed_dim, num_layers=num_layers, 
                                        expansion_factor=expansion_factor, n_heads=n_heads, latent_dim=embed_dim)
        
#         for module in self.feed_forward.modules():
#             if isinstance(module, nn.Linear):
#                 init.kaiming_uniform_(module.weight, nonlinearity='relu')
#                 init.constant_(module.bias, 0.0)
                
#         # Initialize the weights and biases
#         for module in self.query_encoded.modules():
#             if isinstance(module, nn.Linear):
#                 init.kaiming_uniform_(module.weight, nonlinearity='relu')
#                 init.constant_(module.bias, 0.0)
                
#         # Initialize the weights and biases
#         for module in self.enc_forward.modules():
#             if isinstance(module, nn.Linear):
#                 init.kaiming_uniform_(module.weight, nonlinearity='relu')
#                 init.constant_(module.bias, 0.0)    
                
    def make_trg_mask(self, trg):
        """
        Args:
            trg: target sequence
        Returns:
            trg_mask: target mask
        """
        batch_size, trg_len = trg.shape[0], trg.shape[1]
        # returns the lower triangular part of matrix filled with ones
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            batch_size, 1, trg_len, trg_len
        )
        return trg_mask
    
    def forward(self, enc_out):
        
#         enc_out = enc_out.repeat(1,144,1)
        random_numbers = torch.randn(enc_out.shape[0], 144, 13).to(device)
#         x = self.Q_enc(random_numbers)
        enc_out = self.lstmmodell(enc_out)
        enc_out = torch.reshape(enc_out,(enc_out.shape[0],-1))
        enc_out = self.enc_forward(enc_out)  
        enc_out = torch.reshape(enc_out,(enc_out.shape[0],self.seq_len, self.embed_dim))
        enc_out = self.position_embedding(enc_out)
#         x = self.query_encoded(enc_out)
        
#         x = self.word_embedding(x)  #32x10x512
#         x = self.position_embedding(x) #32x10x512
#         x = self.dropout(x)
#         enc_out = self.dropout(enc_out)
        
        mask = self.make_trg_mask(enc_out).to(device)
        mask = None
        for layer in self.layers:
            enc_out = layer(enc_out, enc_out, mask) #trg, enc_out
            mask = None

        out = self.lstmmodel(enc_out)
        out = self.decodercnn(out)
        
        

        return out


# In[ ]:





# In[28]:



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
        
        self.embedding_layer = FeedForward(vocab_size, 200, embed_dim,seq_len)
        self.positional_encoder = PositionalEmbedding(seq_len, embed_dim)
        self.embed_dim = embed_dim
        self.lstmmodel = LSTMModel(input_size=embed_dim, hidden_size=embed_dim, num_layers=2, output_size=13)
        self.dropout1 = nn.Dropout(0.2)

        self.layers = nn.ModuleList([TransformerBlock(embed_dim, expansion_factor, n_heads) for i in range(num_layers)])
        self.latent_dim = latent_dim
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim*seq_len, 1024),
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
#         out = self.dropout1(out)
       
#         out=x
        for layer in self.layers:
            out = layer(out,out,out)
#         out = x
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out  


# In[29]:


train = 1
avg = 'weighted'
seq_length = 10
num_epochs = 5000
reset_random_seeds(seed)
src_vocab_size = features
target_vocab_size = features
num_layers = 2
latent_dim = 16
embed_dim = 128
n_heads = 16
expansion_factor = 2

Encoder = TransformerEncoder(seq_length, src_vocab_size, embed_dim, num_layers=num_layers, expansion_factor=expansion_factor, n_heads=n_heads, latent_dim=latent_dim).to(device)
Decoder = TransformerDecoder(target_vocab_size, embed_dim, seq_length, num_layers=num_layers, expansion_factor=expansion_factor, n_heads=n_heads, latent_dim=latent_dim).to(device)
Cf = Classifier(seq_length, vocab_size=latent_dim, embed_dim=128, num_layers=2, expansion_factor=expansion_factor, n_heads=4, latent_dim=latent_dim).to(device)

criterion = nn.MSELoss()
criterion2 = nn.CrossEntropyLoss()
optimizer_encoder = optim.Adam(Encoder.parameters(), lr=0.001)
optimizer_decoder = optim.Adam(Decoder.parameters(), lr=0.001)
optimizer_cf = optim.Adam(Cf.parameters(), lr=0.00001)

# Define early stopping parameters
patience = 500
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
            latent = Encoder(inputs)
            
            prediction = Cf(latent)
            loss2 = criterion2(pre,prediction)
            loss2 = loss2.float()
            loss2.backward()
            optimizer_cf.step()
            optimizer_encoder.step()
            
            # Update running loss
            running_loss += loss1.item()
            
            
            PR.append(prediction)
            PL.append(pre)
            

        # Calculate average loss per epoch
        PR = torch.cat(PR, dim=0)
        PL = torch.cat(PL, dim=0)
        _, predicted = torch.max(PR.data, 1)
        _, lbl = torch.max(PL.data, 1)
        total = lbl.size(0)
        correct = (predicted == lbl).sum().item()

        accuracy = 100 * correct / total
        print(f'Train Accuracy: {accuracy:.2f}%')
        conf_matrix = confusion_matrix(lbl.cpu().detach().numpy(),predicted.cpu().detach().numpy())
        print('Confusion Matrix:')
        print(conf_matrix)
        # Calculate average loss per epoch
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch: {epoch+1}, Train Loss: {round(epoch_loss,6)}")
        print(epochs_without_improvement, best_loss)
        
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
            print('-----------------------------------------')
            print()
            print()
            
                
                    
        # Check for early stopping and save the best weights
        th = (conf_matrix[0][0]> conf_matrix[0][1]+conf_matrix[0][2] and conf_matrix[1][1]>conf_matrix[1][0]+conf_matrix[1][2] and conf_matrix[2][2]>conf_matrix[2][0]+conf_matrix[2][1])
        if th and vaccuracy > best_loss:
            best_loss = vaccuracy
            epochs_without_improvement = 0
            # Save the model weights
            torch.save(Encoder.state_dict(), 'EncoderAuto103.pt')
            torch.save(Decoder.state_dict(), 'DecoderAuto103.pt')
            torch.save(Cf.state_dict(), 'CfAuto103.pt')
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
Encoder.load_state_dict(torch.load('EncoderAuto103.pt'))
Decoder.load_state_dict(torch.load('DecoderAuto103.pt'))
Cf.load_state_dict(torch.load('CfAuto103.pt'))


# In[30]:


Encoder.load_state_dict(torch.load('EncoderAuto103.pt'))
Decoder.load_state_dict(torch.load('DecoderAuto103.pt'))
Cf.load_state_dict(torch.load('CfAuto103.pt'))


# In[31]:


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

accuracy = 100 * correct / total
print('-----------------------------------------')
print(f'Train Accuracy: {accuracy:.2f}%')
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

accuracy = 100 * correct / total
print('-----------------------------------------')
print(f'Valid Accuracy: {accuracy:.2f}%')
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

accuracy = 100 * correct / total
print('-----------------------------------------')
print(f'Test Accuracy: {accuracy:.2f}%')
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
print('-----------------------------------------')
print()
print()


# In[ ]:





# In[32]:


# Iterate over mini-batches
latents = []
train_label = []
for i, (inputs, labels, lbl) in enumerate(train_loader):

   inputs = inputs.to(device)
   lbl = lbl.to(device)
   
   with torch.no_grad():

       # Forward pass
       latent = Encoder(inputs)


   latents.append(latent)
   train_label.append(lbl)


# In[33]:


latents = torch.cat(latents,dim=0)


# In[34]:


train_label = torch.cat(train_label,dim=0)


# In[35]:


latents = latents.cpu().detach().numpy()


# In[36]:


rs = reconstructData(latents)


# In[37]:


torch.argmax(train_label,dim=1)


# In[38]:


labl = torch.argmax(train_label,dim=1).cpu().detach().numpy()


# In[39]:


rs.shape


# In[40]:


real_data_reshaped = np.reshape(latents, (latents.shape[0], -1))


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


# In[ ]:





# In[ ]:





# In[ ]:





# In[41]:


10,
-----------------------------------------
Train Accuracy: 94.98%
Confusion Matrix:
[[7403  294   79]
 [ 662 7013  101]
 [  23   11 7742]]

Precision: 0.9503023460076053
Recall: 0.9498456790123457
F1 Score: 0.9496397390466299
-----------------------------------------


-----------------------------------------
Valid Accuracy: 92.14%
Confusion Matrix:
[[88919  6312  1251]
 [  252  2452    45]
 [    3     1   765]]

Precision: 0.9726056889160571
Recall: 0.92136
F1 Score: 0.9400572251031517
-----------------------------------------


-----------------------------------------
Test Accuracy: 93.48%
Confusion Matrix:
[[149316   8568   1715]
 [   372   3101     38]
 [     4      1    967]]

Precision: 0.9780270406977736
Recall: 0.9348008922368084
F1 Score: 0.9510021230532714
-----------------------------------------


# In[ ]:


10
-----------------------------------------
Train Accuracy: 90.10%
Confusion Matrix:
[[6975  681  120]
 [1066 6647   63]
 [ 290   90 7396]]

Precision: 0.9030507963315313
Recall: 0.900977366255144
F1 Score: 0.901456042212968
-----------------------------------------


-----------------------------------------
Valid Accuracy: 87.52%
Confusion Matrix:
[[84476 10397  1609]
 [  401  2321    27]
 [   34     9   726]]

Precision: 0.9672541674713002
Recall: 0.87523
F1 Score: 0.9104587768348801
-----------------------------------------


-----------------------------------------
Test Accuracy: 90.02%
Confusion Matrix:
[[143859  13688   2052]
 [   572   2914     25]
 [    33     12    927]]

Precision: 0.9741859160792065
Recall: 0.9001596762594313
F1 Score: 0.9293505617243973
-----------------------------------------


# In[ ]:


30
-----------------------------------------
Train Accuracy: 77.02%
Confusion Matrix:
[[6218 1322  236]
 [2902 4447  427]
 [ 380   93 7303]]

Precision: 0.7766374669845691
Recall: 0.7702331961591221
F1 Score: 0.7666091087770827
-----------------------------------------


-----------------------------------------
Valid Accuracy: 77.66%
Confusion Matrix:
[[75383 18028  3071]
 [ 1038  1557   154]
 [   40     8   721]]

Precision: 0.9548069444753483
Recall: 0.77661
F1 Score: 0.8472816087075055
-----------------------------------------


-----------------------------------------
Test Accuracy: 82.93%
Confusion Matrix:
[[133302  22253   4044]
 [  1469   1855    187]
 [    50     10    912]]

Precision: 0.9644155848830128
Recall: 0.8292743871966456
F1 Score: 0.8854225708440681
-----------------------------------------


# In[ ]:


20
-----------------------------------------
Train Accuracy: 73.17%
Confusion Matrix:
[[6054 1367  355]
 [3137 4091  548]
 [ 547  304 6925]]

Precision: 0.7387765417471319
Recall: 0.7317386831275721
F1 Score: 0.7277661498020973
-----------------------------------------


-----------------------------------------
Valid Accuracy: 75.45%
Confusion Matrix:
[[73334 18497  4651]
 [ 1111  1437   201]
 [   60    33   676]]

Precision: 0.9525746648086995
Recall: 0.75447
F1 Score: 0.8327254507433948
-----------------------------------------


-----------------------------------------
Test Accuracy: 80.96%
Confusion Matrix:
[[130236  23369   5994]
 [  1544   1745    222]
 [    63     42    867]]

Precision: 0.9630319949885767
Recall: 0.8096439585085506
F1 Score: 0.8731971798321934
-----------------------------------------


# In[ ]:


x = torch.randn((10,24,32))


# In[ ]:


x.shape


# In[ ]:


conv1 = nn.Conv1d(in_channels=1, out_channels=6, kernel_size=3, stride=1, padding=1)


# In[ ]:


z = x[0].view(24,1,32)


# In[ ]:


z.shape


# In[ ]:


a=conv1(z)


# In[ ]:


a.shape


# In[ ]:


a


# In[ ]:




