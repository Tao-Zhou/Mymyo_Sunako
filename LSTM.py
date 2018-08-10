import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.functional as F
import torch.optim as optim

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class LstmClassifier(nn.Module):
    def _init_(self,input_size, hidden_size, num_layers,output_size):
        super(LstmClassifier,self).__init__()
        self.input_size=16
        self.hidden_size= 64
        self.num_layers = 3
        self.output_size= 8
        self.lstm = nn.LSTM(input_size,hidden_size,num_layers,batch_first=True)
        self.out = nn.Linear(64,8)
    def init_hidden(self,batch_size):
        return(autograd.Variable(torch.randn(1, batch_size, self.hidden_size)),
						autograd.Variable(torch.randn(1, batch_size, self.hidden_size)))
    def forward(self,x):
        r_out,(h_n,h_c) = self.lstm(x,None)
        out = self.out(r_out[:,-1,:])#(batch,time_step,input_size)
        return out
