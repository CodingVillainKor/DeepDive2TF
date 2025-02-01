import torch
from torch.utils.data import DataLoader
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator

from glob import glob
import os.path as osp

class IMDB:
    def __init__(self, mode="train"):
        data_paths = glob(f"aclImdb/{mode}/neg/*.txt") + glob(f"aclImdb/{mode}/pos/*.txt")
        self.data = [self.get_sample(p) for p in data_paths]
        self.idx = 0

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    @staticmethod
    def get_sample(path):
        with open(path, "r", encoding="utf-8") as fr:
            txt = fr.read()
        basename = osp.basename(path)[:-4]
        label = int(basename.split("_")[-1])
        return label, txt

# imdb = torch.load("IMDB.pkl")
imdb = IMDB("train")
tokenizer = get_tokenizer("basic_english")
def yield_tokens(data_iter):
    for _, text in data_iter:
        yield tokenizer(text)

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

