import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_class):
        super().__init__()
        self.embed = nn.EmbeddingBag(vocab_size, embed_dim)
        self.hidden = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
        )
        self.fc = nn.Linear(embed_dim, num_class)

    def forward(self, text, offsets, labels=None, train=True):
        if train:
            loss = self.train_step(text, labels, offsets)
            return loss
        else:
            out = self.predict(text, offsets)
            return out

    def train_step(self, text, labels, offsets):
        out = self.predict(text, offsets)
        loss = self.loss(out, labels)
        return loss
    
    def predict(self, text, offsets):
        embed = self.embed(text, offsets)
        latent = self.hidden(embed)
        out = self.fc(latent)
        return out

    def loss(self, model_out, label):
        loss = F.cross_entropy(model_out, label)
        return loss