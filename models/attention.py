from math import gamma
import torch
import torch.nn as nn

class memory_attn(nn.Module):
    def __init__(self, key_size, memory_size, hidden_size):
        super(memory_attn, self).__init__()
        self.w_q = nn.Linear(key_size, hidden_size)
        self.w_m = nn.Linear(memory_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1, bias=False)
        self.softmax = nn.Softmax(-1)
        self.tanh = nn.Tanh()


    def forward(self, x, memory):
        # x : [batch, hidden], context: [batch, seq_len, hidden]
        h_x = torch.unsqueeze(self.w_q(x), 1)
        h_c = self.w_m(memory)
        score = torch.squeeze(self.v(self.tanh(h_x + h_c)), 2)
        score = self.softmax(score)
        c = torch.sum(memory * torch.unsqueeze(score, -1), 1)
        return c, score


class global_attn(nn.Module):
    def __init__(self, hidden_size, activation=None) -> None:
        super().__init__()
        self.linear_in = nn.Linear(hidden_size, hidden_size)
        self.linear_out = nn.Linear(2 * hidden_size, hidden_size)
        self.softmax = nn.Softmax(-1)
        self.tanh = nn.Tanh()
        self.activation = activation

    def forward(self, x, context, mask=None):
        gamma_h = self.linear_in(x).unsqueeze(2)
        if self.activation == 'tanh':
            gamma_h = self.tanh(gamma_h)
        
        weights = torch.bmm(context, gamma_h).squeeze(2)
        weights = self.softmax(weights)
        c_t = torch.bmm(weights.unsqueeze(1), context).squeeze(1)
        output = self.tanh(self.linear_out(torch.cat([c_t, x], 1)))
        return output, weights


class mask_attn(nn.Module):
    def __init__(self, hidden_size, activation=None) -> None:
        super().__init__()
        self.linear_in = nn.Linear(hidden_size, hidden_size)
        self.linear_out = nn.Linear(2 * hidden_size, hidden_size)
        self.softmax = nn.Softmax(-1)
        self.tanh = nn.Tanh()
        self.activation = activation

    def forward(self, x, context, mask):
        gamma_h = self.linear_in(x).unsqueeze(2)
        if self.activation == 'tanh':
            gamma_h = self.tanh(gamma_h)
        
        weights = torch.bmm(context, gamma_h).squeeze(2)
        weights = self.softmax(weights) * mask.float()
        c_t = torch.bmm(weights.unsqueeze(1),context).squeeze(2)
        output = self.tanh(self.linear_out(torch.cat([c_t, x], 1)))
        return output, weights
