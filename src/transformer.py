import torch
import torch.nn as nn
import torch.nn.functional as F


class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, n_heads, n_layers, hidden_dim, dropout):
        super().__init__()
        self.embedding_q = nn.Embedding(input_dim, hidden_dim)
        self.embedding_a = nn.Embedding(output_dim + 1, hidden_dim)
        self.pe = PositionalEncoding(hidden_dim)

        self.encoder_layers = nn.ModuleList()
        self.decoder_layers = nn.ModuleList()
        for i in range(n_layers):
            self.encoder_layers.append(
                EncoderLayer(hidden_dim, n_heads, hidden_dim, dropout)
            )
            self.decoder_layers.append(
                DecoderLayer(hidden_dim, n_heads, hidden_dim, dropout)
            )

        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, output_dim),
        )

    def forward(self, src, tgt):
        src = self.embedding_q(src)
        decoder_input = tgt[:, :-1]
        true = tgt[:, 1:]
        tgt = self.embedding_a(decoder_input)
        breakpoint()
        src = self.pe(src)
        tgt = self.pe(tgt)
        for layer in self.encoder_layers:
            src = layer(src)

        for layer in self.decoder_layers:
            tgt = layer(tgt, src)

        out = self.fc(tgt)

        out = out.transpose(1, 2)
        loss = F.cross_entropy(out, true, ignore_index=0)
        return loss

    @torch.no_grad()
    def infer(self, src, tgt=None):
        src = self.embedding_q(src)
        src = self.pe(src)
        for layer in self.encoder_layers:
            src = layer(src)

        if tgt is None:
            tgt = torch.zeros(src.shape[0], 1, dtype=torch.long).to(src.device)

        for i in range(44):
            tgt_emb = self.embedding_a(tgt)
            tgt_emb = self.pe(tgt_emb)

            for layer in self.decoder_layers:
                tgt_emb = layer(tgt_emb, src)

            out = self.fc(tgt_emb)
            out = out[:, -1:, :]
            out = torch.argmax(out, dim=-1)
            tgt = torch.cat([tgt, out], dim=1)
        out = tgt[:, 1:]

        return out


class PositionalEncoding(nn.Module):
    def __init__(self, input_dim, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, input_dim)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, input_dim, 2).float()
            * -(torch.log(torch.tensor(10000.0)) / input_dim)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        pe = self.pe[:, : x.size(1)].to(x.device)
        x = x + pe
        return x


class MHAttn(nn.Module):
    def __init__(self, input_dim, n_heads):
        super().__init__()
        self.w_q = nn.Linear(input_dim, input_dim)
        self.w_k = nn.Linear(input_dim, input_dim)
        self.w_v = nn.Linear(input_dim, input_dim)
        self.w_o = nn.Linear(input_dim, input_dim)
        self.n_heads = n_heads

    def forward(self, q, k, v, causal=False):
        q = self.w_q(q) # B, L, D
        k = self.w_k(k)
        v = self.w_v(v)
        q = q.view(
            q.shape[0], q.shape[1], self.n_heads, q.shape[-1] // self.n_heads
        ).transpose(1, 2) # B, 8, L, D // 8
        k = k.view(
            k.shape[0], k.shape[1], self.n_heads, k.shape[-1] // self.n_heads
        ).transpose(1, 2)
        v = v.view(
            v.shape[0], v.shape[1], self.n_heads, v.shape[-1] // self.n_heads
        ).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) / (q.shape[-1] ** 0.5)
        if causal:
            mask = torch.triu(torch.ones(attn.shape[-2], attn.shape[-1]), 1).to(
                attn.device
            )
            attn = attn.masked_fill(mask == 1, -1e9)
        attn = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn, v) # B, 8, Lq, D // 8

        out = out.transpose(1, 2).contiguous()
        out = out.view(out.shape[0], -1, out.shape[-1] * self.n_heads) # B, Lq, D
        out = self.w_o(out)

        return out


class FFN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):
        return self.fc2(torch.relu(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, input_dim, n_heads, hidden_dim, dropout):
        super().__init__()
        self.self_attn = MHAttn(input_dim, n_heads)
        self.ffn = FFN(input_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)

    def forward(self, x):
        latent = self.self_attn(x, x, x)
        x = self.norm1(latent + x)

        latent = self.ffn(x)
        x = self.norm2(latent + x)

        return x


class DecoderLayer(nn.Module):
    def __init__(self, input_dim, n_heads, hidden_dim, dropout):
        super().__init__()
        self.self_attn = MHAttn(input_dim, n_heads)
        self.cross_attn = MHAttn(input_dim, n_heads)
        self.ffn = FFN(input_dim, hidden_dim)

        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.norm3 = nn.LayerNorm(input_dim)

    def forward(self, x, enc_out):
        latent = self.self_attn(x, x, x, causal=True)
        x = self.norm1(latent + x)

        latent = self.cross_attn(x, enc_out, enc_out)
        x = self.norm2(latent + x)

        latent = self.ffn(x)
        x = self.norm3(latent + x)

        return x


if __name__ == "__main__":
    # Hyperparameters
    input_dim = 1000
    output_dim = 1000
    n_heads = 8
    n_layers = 6
    hidden_dim = 512
    dropout = 0.1

    # Create a Transformer model
    model = Transformer(input_dim, output_dim, n_heads, n_layers, hidden_dim, dropout)

    # Dummy input data
    src = torch.randint(0, input_dim, (1, 10))  # Batch size 32, sequence length 10
    tgt = torch.randint(0, output_dim, (1, 10))  # Batch size 32, sequence length 10

    # Forward pass
    output = model(src, tgt)
    print("Output shape:", output.shape)

    # Inference
    inferred_output = model.infer(src)
    print("Inferred output shape:", inferred_output)
