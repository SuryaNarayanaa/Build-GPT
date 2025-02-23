import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr

# Load checkpoint
checkpoint = torch.load('thy_sonnet.pt', map_location='cuda')
hyper = checkpoint['hyperparameters']
BATCH_SIZE = hyper['BATCH_SIZE']
VOCAB_SIZE = hyper['VOCAB_SIZE']
CONTEXT_LEN = hyper['CONTEXT_LEN']
EMBEDDING_DIM = hyper['EMBEDDING_DIM']
NUM_HEADS = hyper['NUM_HEADS']
NUM_BLOCKS = hyper['NUM_BLOCKS']
MAX_ITERS = hyper['MAX_ITERS']
EVAL_ITERS = hyper['EVAL_ITERS']
LEARNING_RATE = hyper['LEARNING_RATE']
DROPOUT = hyper['DROPOUT']
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
vocabulary = checkpoint['vocabulary']
# A simple decode: list indices to string using vocabulary list.
decode = lambda l: ''.join([vocabulary[i] for i in l])

# Define an encode function: convert each character to its index.
# This assumes that every character in the prompt exists in the vocabulary.
encode = lambda s: [vocabulary.index(ch) for ch in s if ch in vocabulary]

# Model definitions (Head, MultiHeadAttention, FeedForward, Block, NanoGPT)

class Head(nn.Module):
    def __init__(self, HEAD_SIZE):
        super().__init__()
        self.key = nn.Linear(EMBEDDING_DIM, HEAD_SIZE, bias=False)
        self.query = nn.Linear(EMBEDDING_DIM, HEAD_SIZE, bias=False)
        self.value = nn.Linear(EMBEDDING_DIM, HEAD_SIZE, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones((CONTEXT_LEN, CONTEXT_LEN))))
        self.dropout = nn.Dropout(DROPOUT)
    
    def forward(self, x):
        B, T, C = x.shape 
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * C**-.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = torch.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v 
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, NUM_HEAD, HEAD_SIZE, EMBEDDING_DIM=EMBEDDING_DIM):
        super().__init__()
        self.heads = nn.ModuleList([Head(HEAD_SIZE=HEAD_SIZE) for _ in range(NUM_HEAD)])
        self.proj = nn.Linear(EMBEDDING_DIM, EMBEDDING_DIM)
        self.dropout = nn.Dropout(DROPOUT)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out

class FeedForward(nn.Module):
    def __init__(self, EMBEDDING_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(EMBEDDING_DIM, 4 * EMBEDDING_DIM),
            nn.ReLU(),
            nn.Linear(4 * EMBEDDING_DIM, EMBEDDING_DIM),
            nn.Dropout(DROPOUT)
        )
    
    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, EMBEDDING_DIM, NUM_HEAD):
        super().__init__()
        self.sa_heads = MultiHeadAttention(NUM_HEAD, EMBEDDING_DIM // NUM_HEAD)
        self.ffwd = FeedForward(EMBEDDING_DIM)
        self.ln1 = nn.LayerNorm(EMBEDDING_DIM)
        self.ln2 = nn.LayerNorm(EMBEDDING_DIM)
    
    def forward(self, x):
        x = x + self.sa_heads(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class NanoGPT(nn.Module):
    def __init__(self, VOCAB_SIZE, CONTEXT_LEN, EMBEDDING_DIM):
        super().__init__()
        self.token_embedding = nn.Embedding(num_embeddings=VOCAB_SIZE, embedding_dim=EMBEDDING_DIM)
        self.positional_embedding = nn.Embedding(num_embeddings=CONTEXT_LEN, embedding_dim=EMBEDDING_DIM)
        self.blocks = nn.Sequential(*[Block(EMBEDDING_DIM, NUM_HEADS) for _ in range(NUM_BLOCKS)])
        self.ln = nn.LayerNorm(EMBEDDING_DIM)
        self.lm_head = nn.Linear(EMBEDDING_DIM, VOCAB_SIZE)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.positional_embedding(torch.arange(T, device=DEVICE))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.lm_head(x)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, start_tok, max_new_tokens):
        idx = start_tok.view(1, -1)
        for _ in range(max_new_tokens):
            idx_cond = idx if idx.size(1) <= CONTEXT_LEN else idx[:, -CONTEXT_LEN:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return decode(idx[0].tolist())

# Instantiate the model, load trained state, and prepare for eval
model = NanoGPT(VOCAB_SIZE, CONTEXT_LEN, EMBEDDING_DIM)
model.load_state_dict(checkpoint['model_state_dict'])
model.to(DEVICE)
model.eval()

# Gradio interface function
def generate_text(prompt, max_new_tokens):
    # Encode the prompt. If a character is not found, it is skipped.
    encoded = encode(prompt)
    if not encoded:
        return "Error: prompt contains characters not in the model vocabulary."
    context = torch.tensor(encoded, dtype=torch.long, device=DEVICE).unsqueeze(0)
    generated = model.generate(context, max_new_tokens)
    return generated

# Create a Gradio interface
iface = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.components.Textbox(lines=2, label="Enter prompt"),
        gr.components.Slider(minimum=10, maximum=500, step=10, value=200, label="Max New Tokens")
    ],
    outputs=gr.components.Textbox(label="Generated text"),
    title="NanoGPT Text Generation",
    description="Enter a text prompt and generate text using the NanoGPT model."
)

if __name__ == "__main__":
    iface.launch()
