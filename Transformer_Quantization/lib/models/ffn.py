from torch import nn

class FFN(nn.Module):
    def __init__(self, embed_dim, source_dim, m):
        super(FFN, self).__init__()
        self.fc1 = nn.Linear(embed_dim, source_dim)
        self.norm = nn.LayerNorm(source_dim)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(source_dim*m, source_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.norm(x)
        x = self.relu(x)
        bs = x.shape[0]
        x = x.view(bs, -1)
        x = self.fc2(x)
        out = self.sigmoid(x)

        return out