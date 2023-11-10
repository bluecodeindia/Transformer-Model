#!/usr/bin/env python
# coding: utf-8

# In[1]:


seed=197


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


import pickle

pickle_file_path = "data.pickle"

# Load data from the pickle file
with open(pickle_file_path, "rb") as f:
    loaded_data_dict = pickle.load(f)

# Retrieve data and labels
TrainData = loaded_data_dict["TrainData"]
TrainLabel1 = loaded_data_dict["TrainLabel1"]
# TrainLabel2 = loaded_data_dict["TrainLabel2"]
# TrainLabel3 = loaded_data_dict["TrainLabel3"]
TestData = loaded_data_dict["TestData"]
TestLabel1 = loaded_data_dict["TestLabel1"]
# TestLabel2 = loaded_data_dict["TestLabel2"]
# TestLabel3 = loaded_data_dict["TestLabel3"]


# In[ ]:





# In[6]:


# Convert to torch tensors
X_train = torch.tensor(TrainData)
X_test = torch.tensor(TestData)
y_train = torch.tensor(TrainLabel1)
y_test = torch.tensor(TestLabel1)


# In[7]:


X_train.shape


# In[ ]:





# In[ ]:





# In[ ]:





# In[8]:


features=13


# In[9]:


class Embedding(nn.Module):
    def __init__(self, vocab_size, embed_dim):
        """
        Args:
            vocab_size: size of vocabulary
            embed_dim: dimension of embeddings
        """
        super(Embedding, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
    def forward(self, x):
        """
        Args:
            x: input vector
        Returns:
            out: embedding vector
        """
        x= x.long()
        out = self.embed(x)
        return out


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
        for module in self.feed_forward.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                init.constant_(module.bias, 0.0)

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
        for module in [self.query_matrix, self.key_matrix, self.value_matrix, self.out]:
            init.kaiming_uniform_(module.weight, nonlinearity='relu')
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
            nn.Linear(embed_dim, expansion_factor + embed_dim),
            nn.BatchNorm1d(seq_length),  # Apply batch normalization along seq_length
            nn.LeakyReLU(),
            nn.Linear(expansion_factor + embed_dim, embed_dim),
            nn.BatchNorm1d(seq_length)  # Apply batch normalization along seq_length
        )




        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)
        
#         # Initialize the weights and biases
        for module in self.feed_forward.modules():
            if isinstance(module, nn.Linear):
                init.kaiming_uniform_(module.weight, nonlinearity='relu')
                init.constant_(module.bias, 0.0)

    def forward(self,key,query,value):
        
        """
        Args:
           key: key vector
           query: query vector
           value: value vector
           norm2_out: output of transformer block
        
        """
        attention_out = self.attention(key,query,value)  #32x10x512
        attention_residual_out = attention_out + value  #32x10x512
        norm1_out = self.dropout1(self.norm1(attention_residual_out)) #32x10x512

        feed_fwd_out = self.feed_forward(norm1_out) #32x10x512 -> #32x10x2048 -> 32x10x512
        feed_fwd_residual_out = feed_fwd_out + norm1_out #32x10x512
        norm2_out = self.dropout2(self.norm2(feed_fwd_residual_out)) #32x10x512

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

        self.dropout1 = nn.Dropout(0.2)

        self.layers = nn.ModuleList([TransformerBlock(embed_dim, expansion_factor, n_heads) for i in range(num_layers)])
        self.latent_dim = latent_dim
        self.latent_encoded = nn.Sequential(
                          nn.Linear(embed_dim, 64),
                          nn.LeakyReLU(),
                          nn.Linear(64, latent_dim),
        )
        
        # Initialize the weights and biases
        for module in self.latent_encoded.modules():
            if isinstance(module, nn.Linear):
                init.kaiming_uniform_(module.weight)
                init.constant_(module.bias, 0.0)
    
    def forward(self, x):
        out = self.embedding_layer(x)
        out = self.positional_encoder(out)
        out = self.dropout1(out)
       
        
        for layer in self.layers:
            out = layer(out,out,out)
        if self.latent_dim>0:
            out = self.latent_encoded(out)
        return out  #32x10x512


# In[18]:


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
            nn.Linear(embed_dim, expansion_factor + embed_dim),
            nn.BatchNorm1d(seq_length),  # Apply batch normalization along seq_length
            nn.LeakyReLU(),
            nn.Linear(expansion_factor + embed_dim, embed_dim),
            nn.BatchNorm1d(seq_length)  # Apply batch normalization along seq_length
        )
        
    def gate_mechanism(self, Q, z):
        # Get the input size from z
        input_size = z.size(-1)

        # Define the linear layers
        linear_q = nn.Linear(input_size, input_size).to(Q.device)
        linear_z = nn.Linear(input_size, input_size).to(Q.device)
        gate_transform = nn.Linear(input_size, input_size).to(Q.device)
        
        # Apply weight initialization
        for module in [linear_q, linear_z, gate_transform]:
            init.xavier_uniform_(module.weight)
            init.constant_(module.bias, 0.0)

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
            nn.LeakyReLU(),
            nn.Linear(256, 512),
            nn.LeakyReLU(),
            nn.Linear(512, features)
        )
        self.query_encoded = nn.Sequential(
                          nn.Linear(embed_dim, 64),
                          nn.LeakyReLU(),
                          nn.Linear(64, embed_dim),
        )
        self.enc_forward = nn.Sequential(
                          nn.Linear(latent_dim, 64),
                          nn.LeakyReLU(),
                          nn.Linear(64, embed_dim),
        )
        self.dropout = nn.Dropout(0.2)
        self.Q_enc = TransformerEncoder(seq_length, 13, embed_dim, num_layers=num_layers, 
                                        expansion_factor=expansion_factor, n_heads=n_heads, latent_dim=0)
        
        # Initialize the weights and biases
        for module in self.feed_forward.modules():
            if isinstance(module, nn.Linear):
                init.kaiming_uniform_(module.weight, nonlinearity='relu')
                init.constant_(module.bias, 0.0)
                
        # Initialize the weights and biases
        for module in self.query_encoded.modules():
            if isinstance(module, nn.Linear):
                init.kaiming_uniform_(module.weight, nonlinearity='relu')
                init.constant_(module.bias, 0.0)
                
        # Initialize the weights and biases
        for module in self.enc_forward.modules():
            if isinstance(module, nn.Linear):
                init.kaiming_uniform_(module.weight, nonlinearity='relu')
                init.constant_(module.bias, 0.0)
                
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
        

        random_numbers = torch.randn(enc_out.shape[0], seq_length, 13).to(device)
        x = self.Q_enc(random_numbers)
        
        enc_out = self.enc_forward(enc_out)  
#         x = self.query_encoded(enc_out)
        
#         x = self.word_embedding(x)  #32x10x512
        x = self.position_embedding(x) #32x10x512
        x = self.dropout(x)
        enc_out = self.position_embedding(enc_out) #32x10x512
        enc_out = self.dropout(enc_out)
        
        mask = self.make_trg_mask(enc_out).to(device)
        for layer in self.layers:
            x = layer(x, enc_out, mask) #trg, enc_out
            mask = None

        out = self.feed_forward(x)

        return out


# In[19]:


num_epochs = 500
# Define batch size
batch_size = 256
# Create DataLoader for training set
train_dataset = torch.utils.data.TensorDataset(X_train.float(), y_train.float())
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)


# In[20]:


print(X_train.dtype)
print(y_train.dtype)


# In[21]:


X_train.shape


# In[ ]:


train = 1
seq_length = 144
reset_random_seeds(seed)
src_vocab_size = features
target_vocab_size = features
num_layers = 2
latent_dim = 8
embed_dim = 512
n_heads = 16
expansion_factor = 2

Encoder = TransformerEncoder(seq_length, src_vocab_size, embed_dim, num_layers=num_layers, expansion_factor=expansion_factor, n_heads=n_heads, latent_dim=latent_dim).to(device)
Decoder = TransformerDecoder(target_vocab_size, embed_dim, seq_length, num_layers=num_layers, expansion_factor=expansion_factor, n_heads=n_heads, latent_dim=latent_dim).to(device)

criterion = CustomMSELoss(scale_factor=10.0)
optimizer_encoder = optim.Adam(Encoder.parameters(), lr=0.01)
optimizer_decoder = optim.Adam(Decoder.parameters(), lr=0.01)

# Define early stopping parameters
patience = 10
best_loss = float('inf')
epochs_without_improvement = 0

if train:
    # Step 4: Training loop
    for epoch in range(num_epochs):
        running_loss = 0.0
        latents = []
        LS = []

        # Iterate over mini-batches
        for i, (inputs, labels) in enumerate(train_loader):
            
            probability = 0.95
            coin_toss = random.random()  

            if coin_toss < probability:
                continue
            
            # Clear gradients
            optimizer_encoder.zero_grad()
            optimizer_decoder.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            

            # Forward pass
            latent = Encoder(inputs)
            
            outputs = Decoder(latent)
           

            # Calculate loss
            loss = criterion(outputs, labels)
            loss = loss.float()

            # Backpropagation
            loss.backward()
            optimizer_decoder.step()
            optimizer_encoder.step()

            # Update running loss
            running_loss += loss.item()
            
            latents.append(latent)
            LS.append(labels[:,-1,-1])

        # Calculate average loss per epoch
        epoch_loss = running_loss / len(train_loader)

        # Log epoch progress (optional)
        print(f"Epoch: {epoch+1}, Loss: {round(epoch_loss,6)}")
        
        # Check for early stopping and save the best weights
        if round(epoch_loss,6) < best_loss:
            best_loss = round(epoch_loss,6)
            epochs_without_improvement = 0
            # Save the model weights
            torch.save(Encoder.state_dict(), 'Encoder.pt')
            torch.save(Decoder.state_dict(), 'Decoder.pt')
            latents = torch.cat(latents, dim=0)
            latents = latents.cpu().detach().numpy()
            L = torch.cat(LS, dim=0)
            L = L.cpu().detach().numpy()
            L[np.where(L==1)]=2
            L[np.where(L==0.5)]=1
            with open('latent_data.pkl', 'wb') as f:
                pickle.dump(latents, f)
            with open('label_data.pkl', 'wb') as f:
                pickle.dump(L, f)
            print('Saved')
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                print("Early stopping triggered. Training stopped.")
                break

# Recover the best weights
Encoder.load_state_dict(torch.load('Encoder.pt'))
Decoder.load_state_dict(torch.load('Decoder.pt'))


# In[ ]:


batch_size = 32  # Define your desired batch size

# Move the model to the device (if not done already)


# Set the model to evaluation mode
Encoder.eval()
Decoder.eval()

# Split X_train into batches
num_samples = X_train.shape[0]
num_batches = (num_samples + batch_size - 1) // batch_size  # Round up the division

predictions = []  # List to store the predictions
latents = []

# Iterate over batches
for i in range(num_batches):
    # Get the start and end indices of the current batch
    start_idx = i * batch_size
    end_idx = min((i + 1) * batch_size, num_samples)  # Adjust for the last batch

    # Extract the current batch
    inputs = X_train[start_idx:end_idx]

    # Move the batch to the device
    inputs = inputs.to(device)

    # Perform forward pass
    with torch.no_grad():
        latent = Encoder(inputs)
            
        outputs = Decoder(latent)

    # Append the predictions to the list
    predictions.append(outputs)
    latents.append(latent)

# Concatenate the predictions from all batches
predictions = torch.cat(predictions, dim=0)
latents = torch.cat(latents, dim=0)


# In[ ]:


reconstruct_original = reconstructData(X_train.cpu())
reconstruct_predictd = reconstructData(predictions.cpu().detach().numpy())


# In[ ]:





# In[ ]:


fig, axs = plt.subplots(4, 3, figsize=(12, 12))
I = ['Tem', 'Hum', 'Pressure', 'Rain', 'Light', 'Ax', 'Ay', 'Az', 'Wx', 'Wy', 'Wz', 'Moisture', 'Count']

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


# In[ ]:


# plt.plot(reconstruct_original[:,5], label='Original')
# plt.plot(reconstruct_predictd[:,5], label='Predicted')
# plt.xlabel('Time')
# plt.ylabel('Value')
# plt.title('Original vs Reconstructed Data')
# plt.legend()
# plt.show()


# In[ ]:


t =0.1
reconstruct_predictd[:,5] = np.where((reconstruct_predictd[:,5] >= -1*t) & (reconstruct_predictd[:,5] <= t), 0, reconstruct_predictd[:,5])
reconstruct_predictd[:,6] = np.where((reconstruct_predictd[:,6] >= -1*t) & (reconstruct_predictd[:,6] <= t), 0, reconstruct_predictd[:,6])
reconstruct_predictd[:,7] = np.where((reconstruct_predictd[:,7] >= -1*t) & (reconstruct_predictd[:,7] <= t), 0, reconstruct_predictd[:,7])


# In[ ]:


plt.plot(np.cumsum(reconstruct_original[:,5]), label='Original')
plt.plot(np.cumsum(reconstruct_predictd[:,5]), label='Predicted')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('Original vs Reconstructed Data')
plt.legend()
plt.show()


# In[ ]:


latents.shape


# In[ ]:


with open ('latent_data.pkl','rb') as f:
    latents = pickle.load(f)


# In[ ]:


latents.shape


# In[ ]:


real_data = latents

real_data_np = real_data.cpu().detach().numpy()

# Reshape the data for t-SNE visualization
real_data_reshaped = np.reshape(real_data_np, (real_data_np.shape[0], -1))


# Perform tsne on real and fake data
perplexity = min(30, real_data_reshaped.shape[0] - 1)
tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed)
real_tsne = tsne.fit_transform(real_data_reshaped)
plt.scatter(real_tsne[:, 0], real_tsne[:, 1], edgecolors='black')


# In[ ]:



retrain = 1
if retrain:

    real_data = latents

    real_data_np = real_data.cpu().detach().numpy()

    # Reshape the data for t-SNE visualization
    real_data_reshaped = np.reshape(real_data_np, (real_data_np.shape[0], -1))


    # Perform tsne on real and fake data
    perplexity = min(30, real_data_reshaped.shape[0] - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=seed)
    real_tsne = tsne.fit_transform(real_data_reshaped)
    lbl = np.zeros(X_train.shape[0])
    lbl[np.where((X_train[:, -1, -1] >= -0.4) & (X_train[:, -1, -1] <= 0))]=1
    lbl[np.where((X_train[:, -1, -1] >= 0) & (X_train[:, -1, -1] <= 0.9))]=2
    lbl[np.where(X_train[:, -1, -1] ==1)]=3
    lbl[np.where((real_tsne[:, 0] >= -40) & (real_tsne[:, 0] <= 10) & (real_tsne[:, 1] >= 90))]=3
    lbl[np.where((real_tsne[:, 0] >= 10) & (real_tsne[:, 0] <= 80) & (real_tsne[:, 1] >= 70))]=2
    A = np.where(lbl==3)[0]
    selected_indexes = np.random.choice(A, size=100, replace=False)
    lbl[selected_indexes]=1
    A = np.where(lbl==2)[0]
    selected_indexes = np.random.choice(A, size=100, replace=False)
    lbl[selected_indexes]=1

    # Store real_tsne and lbl in a dictionary
    data_dict = {'real_tsne': real_tsne, 'lbl': lbl}

    # Save the data_dict using pickle
#     with open('tsnedata.pkl', 'wb') as f:
#         pickle.dump(data_dict, f)
else:
    # Load the data from the pickle file
    with open('tsnedata.pkl', 'rb') as f:
        data_dict = pickle.load(f)

    # Retrieve real_tsne and lbl from the data_dict
    real_tsne = data_dict['real_tsne']
    lbl = data_dict['lbl']

label_names = ['No-Movement', 'Small Movement', 'Moderate Movement', 'Large Movement']
# Plot the t-SNE visualization with labeled data
plt.figure(figsize=(20, 15))
for label in np.unique(lbl):
    indices = np.where(lbl == label)
    plt.scatter(real_tsne[indices, 0], real_tsne[indices, 1], label=label_names[int(label)], edgecolors='black')


plt.title('Real Data Visualization with Labels')
plt.legend()
plt.show()


# In[ ]:


np.where(lbl==0)[0].shape


# In[ ]:


import plotly.graph_objects as go

label_names = ['No-Movement', 'Small Movement', 'Moderate Movement', 'Large Movement']
tsne2 = 0

if tsne2:
    
    # Perform t-SNE on real and fake data
    perplexity = min(30, real_data_reshaped.shape[0] - 1)
    # tsne2 = TSNE(n_components=3, perplexity=perplexity, random_state=seed)
    # real_tsne2 = tsne2.fit_transform(real_data_reshaped)

    # Store real_tsne and lbl in a dictionary
    data_dict2 = {'real_tsne': real_tsne2, 'lbl': lbl}

    # Save the data_dict using pickle
    with open('tsnedata2.pkl', 'wb') as f:
        pickle.dump(data_dict2, f)
        
else:
    # Load the data from the pickle file
    with open('tsnedata2.pkl', 'rb') as f:
        data_dict = pickle.load(f)

    # Retrieve real_tsne and lbl from the data_dict
    real_tsne2 = data_dict['real_tsne']
    lbl = data_dict['lbl']
    

# Create the trace for each label
data = []
for label in np.unique(lbl):
    indices = np.where(lbl == label)
    trace = go.Scatter3d(
        x=real_tsne2[indices, 0].flatten(),
        y=real_tsne2[indices, 1].flatten(),
        z=real_tsne2[indices, 2].flatten(),
        mode='markers',
        name=label_names[int(label)],
        marker=dict(
            size=4,
            colorscale='Spectral',
            opacity=1,
            line=dict(
                width=0.5,
                color='black'
            )
        )
    )
    data.append(trace)

# Create the layout for the 3D plot
layout = go.Layout(
    title='Real Data Visualization with Labels (3D)',
    scene=dict(
        xaxis=dict(title='Dimension 1'),
        yaxis=dict(title='Dimension 2'),
        zaxis=dict(title='Dimension 3')
    ),
    showlegend=True,
    width=1000,  # Set width of the plot
    height=800  # Set height of the plot
)

# Create the figure and plot the data
fig = go.Figure(data=data, layout=layout)
fig.show()


# In[ ]:




