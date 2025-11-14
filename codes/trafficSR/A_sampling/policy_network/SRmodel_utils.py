import torch.nn as nn
class labelEmbedding(nn.Module):
    def __init__(self, action_space_dim, embedding_dim, input_dim, out_dim):
        super(labelEmbedding, self).__init__()
        self.embedding = nn.Embedding(action_space_dim, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim * input_dim, 256)
        self.relu1 = nn.ReLU()
        self.linear2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.linear3 = nn.Linear(128, out_dim)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        out = self.embedding(x)
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        out = self.relu1(out)
        out = self.linear2(out)
        out = self.relu2(out)
        out = self.linear3(out)
        out = self.relu3(out)
        return out
    
class newEmbedding(nn.Module):
    def __init__(self, action_space_dim, embedding_dim, input_dim, out_dim):
        super(newEmbedding, self).__init__()
        self.embedding = nn.Embedding(action_space_dim, embedding_dim)
        self.linear1 = nn.Linear(embedding_dim * input_dim, out_dim)

    def forward(self, x):
        out = self.embedding(x)
        out = out.view(out.size(0), -1)
        out = self.linear1(out)
        return out