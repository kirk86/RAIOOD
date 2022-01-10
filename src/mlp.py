import torch
#from .layers.sparsify import Sparsify1D_kactive
nn = torch.nn
F = torch.nn.functional


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.0):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 2*hidden_dim)
        # self.fc3 = nn.Linear(2*hidden_dim, output_dim)
        self.fc3 = nn.Linear((2*hidden_dim)//2, output_dim)
#         self.top_k = Sparsify1D_kactive(k=3)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout(x)
        x = F.leaky_relu(self.fc2(x))
        # x = F.gelu(self.fc1(x))
        # x = self.top_k(self.fc2(x))
        x = F.max_pool1d(x.unsqueeze(dim=0), kernel_size=2).squeeze()
        # x = F.glu(x)
        logits = self.dropout(self.fc3(x))
        return logits
