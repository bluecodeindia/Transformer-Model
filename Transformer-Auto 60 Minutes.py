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


# pickle_file_path = '/media/terminator/Data/NatureData/Auto Ten Minutes.pkl'
pickle_file_path = '/media/terminator/NatureData/Auto Sixty Minutes.pkl'


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


# In[11]:


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


# In[12]:


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


# In[13]:


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


# In[14]:


plt.pcolormesh(output[0].numpy().T, cmap='RdBu')
plt.ylabel('Depth')
plt.xlabel('Position')
plt.colorbar()
plt.show()


# In[15]:


def random_noise(samples, seq_length, dim):
    random_numbers = torch.randn(samples, seq_length, dim)
    random_numbers = torch.tensor(random_numbers)
    random_numbers = torch.reshape(random_numbers,(samples,seq_length,dim))
    
    return random_numbers
    


# In[16]:


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


# In[17]:


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


# In[18]:


import torch
import torch.nn as nn

class TimeSeriesEncoder(nn.Module):
    def __init__(self, input_channels,seq_len):
        super(TimeSeriesEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=4, stride=4),  # Reduce sequence length to 36 / 4 = 9
            nn.Conv1d(32, seq_len, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2)  # Reduce sequence length to 9 / 2 = 4.5 (rounded to 4)
        )
        
    
    def forward(self, x):
        x = x.permute(0,2,1)
        x = self.encoder(x)
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
            nn.ConvTranspose1d(input_channels, 32, kernel_size=3, stride=3),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 13, kernel_size=4, stride=4),
#             nn.Sigmoid()  # Sigmoid activation for reconstructing the input
        )
        
        self.fc = nn.Linear(156,144)

        
    
    def forward(self, x):
#         x = x.permute(0,2,1)
        x = self.decoder(x)
        x = self.fc(x)
        x = x.permute(0,2,1)
        return x

# Create the autoencoder model
input_channelss = 10  # Number of input features
# decoder = TimeSeriesDecoder(input_channelss).to(device)


# In[19]:


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
        
        self.embedding_layer = FeedForward(vocab_size, 32, embed_dim)
        self.positional_encoder = PositionalEmbedding(seq_len, embed_dim)
        self.embed_dim = embed_dim
        self.encodercnn = TimeSeriesEncoder(13,seq_len).to(device)

        self.dropout1 = nn.Dropout(0.2)
#         self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads)
#         self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)

        self.layers = nn.ModuleList([TransformerBlock(embed_dim, expansion_factor, n_heads) for i in range(num_layers)])
        self.latent_dim = latent_dim
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 64),
            nn.ReLU(), 
            nn.Linear(64, latent_dim), 
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
        out = self.classifier(out)
        
        return out 


# In[20]:


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
        self.transformer_block = TransformerBlock(embed_dim, expansion_factor, n_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, expansion_factor * embed_dim),
#             nn.BatchNorm1d(20),  # Apply batch normalization along seq_length
            nn.LeakyReLU(),
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
        x = self.gate_mechanism(x, enc_out)#trg, enc_out
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
        self.word_embedding = FeedForward(embed_dim, 512, embed_dim)
        self.position_embedding = PositionalEmbedding(seq_len, embed_dim)

        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_dim, expansion_factor=4, n_heads=8) 
                for _ in range(num_layers)
            ]

        )
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
                          nn.Linear(latent_dim, 64),
                          nn.ReLU(),
                          nn.Linear(64, embed_dim),
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
        x = self.Q_enc(random_numbers)
        
        enc_out = self.enc_forward(enc_out)  
#         x = self.query_encoded(enc_out)
        
        x = self.word_embedding(x)  #32x10x512
        x = self.position_embedding(x) #32x10x512
        x = self.dropout(x)
#         enc_out = self.dropout(enc_out)
        
        mask = self.make_trg_mask(enc_out).to(device)
#         mask = None
        for layer in self.layers:
            x = layer(x, enc_out, mask) #trg, enc_out
            mask = None

        out = self.feed_forward(x)
        out = self.decodercnn(out)
        
        out = out.reshape(-1, 13)
        min_values = out.min(dim=0).values
        max_values = out.max(dim=0).values

        # Apply min-max normalization
        normalized_tensor = (out - min_values) / (max_values - min_values)
        
        out = normalized_tensor.view(-1,144,13)
        

        return out


# In[21]:


Valid,Test = X_test[:100000],X_test[100000:]
Valid_label,Test_label = y_test[:100000],y_test[100000:]


# In[22]:


def norm(out):
    out = out.view(-1,13)
    min_values = out.min(dim=0).values
    max_values = out.max(dim=0).values

    # Apply min-max normalization
    normalized_tensor = (out - min_values) / (max_values - min_values)

    out = normalized_tensor.view(-1,144,13)
    
    return out


# In[23]:


X_train = norm(X_train)
y_train = norm(y_train)
Valid = norm(Valid)
Valid_label = norm(Valid_label)
Test = norm(Test)
Test_label = norm(Test_label)


# In[24]:


train_mov = y_train[:,-1,-1]
train_mov[np.where(train_mov==1)]=2
train_mov[np.where(train_mov==0.5)]=1

valid_mov = Valid_label[:,-1,-1]
valid_mov[np.where(valid_mov==1)]=2
valid_mov[np.where(valid_mov==0.5)]=1

test_mov = Test_label[:,-1,-1]
test_mov[np.where(test_mov==1)]=2
test_mov[np.where(test_mov==0.5)]=1


# In[25]:


np.where(test_mov==1)[0].shape


# In[26]:


X_train = torch.tensor(X_train)
y_train = torch.tensor(y_train)
Valid = torch.tensor(Valid)
Valid_label = torch.tensor(Valid_label)
Test = torch.tensor(Test)
Test_label = torch.tensor(Test_label)


# In[27]:


from sklearn.preprocessing import OneHotEncoder


# In[28]:


encoder = OneHotEncoder(categories='auto', sparse=False)


# In[29]:


train_mov = encoder.fit_transform(train_mov.view(-1,1))
valid_mov = encoder.fit_transform(valid_mov.view(-1,1))
test_mov = encoder.fit_transform(test_mov.view(-1,1))


# In[30]:


train_mov = torch.tensor(train_mov)
valid_mov = torch.tensor(valid_mov)
test_mov = torch.tensor(test_mov)


# In[31]:


train_mov


# In[32]:



# Define batch size
batch_size = 512
# Create DataLoader for training set
train_dataset = torch.utils.data.TensorDataset(X_train.float(), y_train.float(),train_mov.float())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
valid_dataset = torch.utils.data.TensorDataset(Valid.float(), Valid_label.float(),valid_mov.float())
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_dataset = torch.utils.data.TensorDataset(Test.float(), Test_label.float(),test_mov.float())
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


# In[33]:


print(X_train.dtype)
print(y_train.dtype)


# In[34]:


X_train.shape, y_train.shape


# In[35]:


def dataPlot(train_input, train_output, valid_input, valid_output, latent, valid_latent):


    Input = torch.cat(train_input,dim=0)
    Output = torch.cat(train_output,dim=0)
    Input = Input.cpu().detach().numpy()
    Output = Output.cpu().detach().numpy()
    reconstruct_original = reconstructData(Input)
    reconstruct_predictd = reconstructData(Output)

    fig, axs = plt.subplots(4, 3, figsize=(12, 12))
    I = ['Tem', 'Hum', 'Pressure', 'Rain', 'Light', 'Ax', 'Ay', 'Az', 'Wx', 'Wy', 'Wz', 'Moisture', 'Count', 'D1', 'D2', 'D3']

    for i in range(12):
        row = i // 3
        col = i % 3

        axs[row, col].plot(reconstruct_original[:, i], label='Original')
        axs[row, col].plot(reconstruct_predictd[:, i], label='Predicted')
        axs[row, col].set_xlabel('Time')
        axs[row, col].set_ylabel('Value')
        axs[row, col].set_title(I[i])

    plt.tight_layout()
    plt.show()

    print('#################################### Valid #################################################')


    Input = torch.cat(valid_input,dim=0)
    Output = torch.cat(valid_output,dim=0)
    Input = Input.cpu().detach().numpy()
    Output = Output.cpu().detach().numpy()
    reconstruct_original = reconstructData(Input)
    reconstruct_predictd = reconstructData(Output)

    fig, axs = plt.subplots(4, 3, figsize=(12, 12))
    I = ['Tem', 'Hum', 'Pressure', 'Rain', 'Light', 'Ax', 'Ay', 'Az', 'Wx', 'Wy', 'Wz', 'Moisture', 'Count', 'D1', 'D2', 'D3']

    for i in range(12):
        row = i // 3
        col = i % 3

        axs[row, col].plot(reconstruct_original[:, i], label='Original')
        axs[row, col].plot(reconstruct_predictd[:, i], label='Predicted')
        axs[row, col].set_xlabel('Time')
        axs[row, col].set_ylabel('Value')
        axs[row, col].set_title(I[i])

    plt.tight_layout()
    plt.show()
    
    print('#################################### Valid #################################################')
#     train_latent = torch.cat(latent,dim=0)
#     train_latent = train_latent.cpu().detach().numpy()
#     valid_latent = torch.cat(valid_latent,dim=0)
#     valid_latent = valid_latent.cpu().detach().numpy()
#     real_data_reshaped = np.reshape(train_latent, (train_latent.shape[0], -1))
#     valid_data_reshaped = np.reshape(valid_latent, (valid_latent.shape[0], -1))


#     # Perform tsne on real and fake data
#     perplexity = min(30, real_data_reshaped.shape[0] - 1)
#     tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed)
#     real_tsne = tsne.fit_transform(real_data_reshaped)
#     valid_tsne = tsne.fit_transform(valid_data_reshaped)
#     plt.figure(figsize=(20, 15))
#     plt.scatter(real_tsne[:, 0], real_tsne[:, 1], c='red', edgecolors='black')
#     plt.scatter(valid_tsne[:, 0], valid_tsne[:, 1], c='blue', edgecolors='black',alpha=0.5)
#     plt.show()
        
      
      


# In[ ]:





# In[36]:



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
        
        self.embedding_layer = FeedForward(vocab_size, 16, embed_dim)
        self.positional_encoder = PositionalEmbedding(seq_len, embed_dim)
        self.embed_dim = embed_dim

        self.dropout1 = nn.Dropout(0.2)

        self.layers = nn.ModuleList([TransformerBlock(embed_dim, expansion_factor, n_heads) for i in range(num_layers)])
        self.latent_dim = latent_dim
        self.classifier = nn.Sequential(
            nn.Linear(embed_dim*seq_len, 10),
            nn.ReLU(), 
            nn.Linear(10, 3), 
            nn.Softmax(dim=1)  
        )

        
        # Initialize the weights and biases
#         for module in self.classifier.modules():
#             if isinstance(module, nn.Linear):
#                 init.kaiming_uniform_(module.weight)
#                 init.constant_(module.bias, 0.0)
    
    def forward(self, x):
        out = self.embedding_layer(x)
#         out = self.positional_encoder(out)
#         out = self.dropout1(out)
       
#         out=x
        for layer in self.layers:
            out = layer(out,out,out)
#         out = x
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out  


# In[ ]:


train = 1
avg = 'weighted'
seq_length = 10
num_epochs = 5000
reset_random_seeds(seed)
src_vocab_size = features+5
target_vocab_size = features
num_layers = 2
latent_dim = 8
embed_dim = 128
n_heads = 8
expansion_factor = 2

Encoder = TransformerEncoder(seq_length, src_vocab_size, embed_dim, num_layers=num_layers, expansion_factor=expansion_factor, n_heads=n_heads, latent_dim=latent_dim).to(device)
Decoder = TransformerDecoder(target_vocab_size, embed_dim, seq_length, num_layers=num_layers, expansion_factor=expansion_factor, n_heads=n_heads, latent_dim=latent_dim).to(device)
Cf = Classifier(seq_length, vocab_size=8, embed_dim=16, num_layers=2, expansion_factor=expansion_factor, n_heads=4, latent_dim=latent_dim).to(device)

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
            torch.save(Encoder.state_dict(), 'EncoderAuto601.pt')
            torch.save(Decoder.state_dict(), 'DecoderAuto601.pt')
            torch.save(Cf.state_dict(), 'CfAuto601.pt')
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
Encoder.load_state_dict(torch.load('EncoderAuto601.pt'))
Decoder.load_state_dict(torch.load('DecoderAuto601.pt'))
Cf.load_state_dict(torch.load('CfAuto601.pt'))


# In[ ]:


Encoder.load_state_dict(torch.load('EncoderAuto601.pt'))
Decoder.load_state_dict(torch.load('DecoderAuto601.pt'))
Cf.load_state_dict(torch.load('CfAuto601.pt'))


# In[ ]:


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





# In[ ]:


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


# In[ ]:


latents = torch.cat(latents,dim=0)


# In[ ]:


train_label = torch.cat(train_label,dim=0)


# In[ ]:


latents = latents.cpu().detach().numpy()


# In[ ]:


rs = reconstructData(latents)


# In[ ]:


torch.argmax(train_label,dim=1)


# In[ ]:


labl = torch.argmax(train_label,dim=1).cpu().detach().numpy()


# In[ ]:


rs.shape


# In[ ]:


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


10
-----------------------------------------
Train Accuracy: 91.42%
Confusion Matrix:
[[7221  441  114]
 [1086 6615   75]
 [ 277    9 7490]]

Precision: 0.9176364563124342
Recall: 0.9141803840877915
F1 Score: 0.9144926030904262
-----------------------------------------


-----------------------------------------
Valid Accuracy: 89.79%
Confusion Matrix:
[[86733  7689  2060]
 [  407  2309    33]
 [   25     0   744]]

Precision: 0.9684036429458313
Recall: 0.89786
F1 Score: 0.9244647845578191
-----------------------------------------


-----------------------------------------
Test Accuracy: 91.72%
Confusion Matrix:
[[146676  10251   2672]
 [   599   2881     31]
 [    35      1    936]]

Precision: 0.9747097993539637
Recall: 0.917181653075901
F1 Score: 0.9395252061957898
-----------------------------------------


# In[ ]:





# In[ ]:





# In[ ]:




