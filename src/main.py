import torch
import torch.nn.functional as F

from model import Model
from imdb import dl, vocab, tokenizer

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