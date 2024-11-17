import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

imdb = torch.load("IMDB.pkl")
tokenizer = get_tokenizer("basic_english")
def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)
breakpoint()
vocab = build_vocab_from_iterator(yield_tokens(imdb), specials=["<unk>"])
vocab.set_default_index(vocab["<unk>"])

def collater(batch):
    labels, texts, offsets = [], [], [0]
    for label, text in batch:
        labels.append(label-1)
        text_tensor = torch.tensor(vocab(tokenizer(text)), dtype=torch.long)
        texts.append(text_tensor)
        offsets.append(text_tensor.shape[0])
    labels = torch.tensor(labels, dtype=torch.long)
    texts = torch.cat(texts)
    offsets = torch.tensor(offsets[:-1]).cumsum(dim=0)

    return labels, texts, offsets
dl = DataLoader(imdb, batch_size=8, shuffle=True, collate_fn=collater)

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

m = Model(len(vocab), 64, 10)

optim = torch.optim.SGD(m.parameters(), lr=0.1)
for e in range(3):
    for i, (label, text, offset) in enumerate(dl):
        loss = m(text, offset, label)
        optim.zero_grad()
        loss.backward()
        optim.step()
        print(f"\r{i} / {len(dl)} | loss = {loss:.3f}", end="")
        if i % 2000 == 0:
            print()

review1 = "It is good and fantastic"
review2 = "It is bad and terrible"
review1 = torch.tensor(vocab(tokenizer(review1)))
review2 = torch.tensor(vocab(tokenizer(review2)))
result1 = m.predict(review1, offsets=torch.tensor([0]))
result2 = m.predict(review2, offsets=torch.tensor([0]))
breakpoint()