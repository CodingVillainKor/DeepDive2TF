import torch
import torch.nn.functional as F

from transformer import Transformer
from data import dl, tokenizer

m = Transformer(
    input_dim=tokenizer.vocab_size,
    output_dim=tokenizer.vocab_size,
    n_heads=8,
    n_layers=8,
    hidden_dim=512,
    dropout=0.0,
)

optim = torch.optim.Adam(m.parameters(), lr=0.0001)
for e in range(100):
    for i, (question, answer) in enumerate(dl):
        loss = m(question, answer)
        optim.zero_grad()
        loss.backward()
        optim.step()
        print(f"\r{e} / {100} | loss = {loss:.3f}", end="")

# save model
torch.save(m.state_dict(), "model.pth")

# inference with training dataset sample
m.eval()
sample = next(iter(dl))[0]
out = m.infer(sample)
for o in out:
    text = tokenizer.decode(o.tolist(), skip_special_tokens=True)
    print(text)
    print("-----")
    break