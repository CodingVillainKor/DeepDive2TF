import torch
import torch.nn.functional as F

from transformer import Transformer
from data import dl, tokenizer

m = Transformer(
    input_dim=tokenizer.vocab_size,
    output_dim=tokenizer.vocab_size,
    n_heads=8,
    n_layers=6,
    hidden_dim=512,
    dropout=0.1,
)

optim = torch.optim.SGD(m.parameters(), lr=0.1)
for e in range(3):
    for i, (question, answer) in enumerate(dl):
        loss = m(question, answer)
        optim.zero_grad()
        loss.backward()
        optim.step()
        print(f"\r{i} / {len(dl)} | loss = {loss:.3f}", end="")
        if i % 2000 == 0:
            print()

review1 = "It is good and fantastic"
review2 = "It is bad and terrible"
review1 = torch.tensor(tokenizer.encode(review1))
review2 = torch.tensor(tokenizer.encode(review2))
result1 = m.predict(review1, offsets=torch.tensor([0]))
result2 = m.predict(review2, offsets=torch.tensor([0]))
breakpoint()
