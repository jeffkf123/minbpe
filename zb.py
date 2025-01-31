import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
# =====================
# 1. Simple Tokenizer
# =====================
class CharTokenizer:
    def __init__(self, text):
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch:i for i,ch in enumerate(self.chars)}
        self.itos = {i:ch for i,ch in enumerate(self.chars)}
    
    def encode(self, text):
        return [self.stoi[ch] for ch in text]
    
    def decode(self, tokens):
        return ''.join([self.itos[i] for i in tokens])

# =====================
# 2. Dataset Loader
# =====================
class ShakespeareDataset(Dataset):
    def __init__(self, text, tokenizer, seq_len=128):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        tokens = tokenizer.encode(text)
        self.data = torch.tensor(tokens, dtype=torch.long)
        
    def __len__(self):
        return len(self.data) - self.seq_len - 1
        
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.seq_len]
        y = self.data[idx+1:idx+self.seq_len+1]
        return x, y

# =====================
# 3. Model Architecture 
# =====================
# (Same as previous but with small tweaks)
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.Wq = nn.Linear(d_model, d_model)
        self.Wk = nn.Linear(d_model, d_model)
        self.Wv = nn.Linear(d_model, d_model)
        self.Wo = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.shape

        Q = self.Wq(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        K = self.Wk(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        V = self.Wv(x).view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)

        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.Wo(output)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(d_model, num_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attn_out = self.attention(self.norm1(x), mask)
        x = x + self.dropout(attn_out)
        ffn_out = self.ffn(self.norm2(x))
        x = x + self.dropout(ffn_out)
        return x

class SimpleLLM(nn.Module):
    def __init__(self, vocab_size, d_model=256, num_heads=8, num_layers=6, d_ff=1024, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Embedding(1024, d_model)  # Max context of 1024
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        batch_size, seq_len = x.shape
        device = x.device
        
        mask = torch.tril(torch.ones(seq_len, seq_len)).unsqueeze(0).unsqueeze(0).to(device)
        
        tok_emb = self.token_embedding(x)
        pos_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        pos_emb = self.pos_embedding(pos_ids)
        x = tok_emb + pos_emb

        for layer in self.layers:
            x = layer(x, mask)
        
        return self.output(self.norm(x))

# =====================
# 4. Training Setup
# =====================
def train():
    # Load Shakespeare data
    with open('pg100.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Initialize tokenizer
    tokenizer = CharTokenizer(text)
    print(f"Vocab size: {tokenizer.vocab_size}")
    
    # Create dataset
    dataset = ShakespeareDataset(text, tokenizer, seq_len=128)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)
    
    # Initialize model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SimpleLLM(tokenizer.vocab_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    
    # Training loop
    for epoch in range(10):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            
            logits = model(x)
            loss = F.cross_entropy(logits.view(-1, tokenizer.vocab_size), y.view(-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
        
        print(f"Epoch {epoch+1} | Avg Loss: {total_loss/len(dataloader):.4f}")
        
        # Generate sample text
        model.eval()
        prompt = "ROMEO:"
        tokens = tokenizer.encode(prompt)
        x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
        
        for _ in range(100):
            logits = model(x[:, -128:])
            next_token = logits.argmax(-1)[:, -1]
            x = torch.cat([x, next_token.unsqueeze(0)], dim=1)
        
        print("Generated:", tokenizer.decode(x[0].cpu().numpy()))
        print("="*50)

    # Save model
    torch.save(model.state_dict(), 'shakespeare_llm.pt')

if __name__ == '__main__':
    train()