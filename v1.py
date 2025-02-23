
import torch 
import torch.nn as nn
from torch.nn import functional as F
# Removed unused import from zmq
torch.manual_seed(1337)
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()
vocabulary = list(sorted(set(text)))

ctoi = { c:i for i,c in enumerate(vocabulary)}
itoc = { i:c for c,i in ctoi.items()}
encode = lambda x : [ctoi[c] for c in x]# noqa: E731
decode = lambda x : "".join([itoc[c] for c in x])  # noqa: E731
# "hello"== decode(encode("hello"))

data = torch.tensor(encode(text), dtype=torch.long)
# data.shape , data.dtype
# data[:1000]

BATCH_SIZE =32
VOCAB_SIZE = len(vocabulary)
BLOCKSIZE = 8
N_EMBED = 32
MAX_ITERS = 3000
LEARNING_RATE = 1e-2
EVAL_ITERS =300

device = "cuda" if torch.cuda.is_available() else "cpu"

n = int(.9*len(data))
train_data = data[:n]
val_data = data[n:]
# len(train_data) /len(data), len(val_data)/len(data)


def get_batch(split):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - BLOCKSIZE, (BATCH_SIZE,))
    x =  torch.stack( [data[i:i+BLOCKSIZE] for i in ix] )
    y =  torch.stack( [data[i+1:i+BLOCKSIZE+1] for i in ix] )

    return x,y




class BiGramLanuageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(VOCAB_SIZE,N_EMBED)
        self.position_embedding_table = nn.Embedding(BLOCKSIZE, N_EMBED)
        self.lm_head = nn.Linear(N_EMBED,VOCAB_SIZE)
    
    def forward(self,idx, targets = None):
        B,T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B,T,C)
        pos_emv = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T,c)
        emb = pos_emv + tok_emb # (B,T,C)
        logits = self.lm_head(emb)



        
        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets)
        return logits, loss 

    def generate(self, idx ,max_new_tokens):

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -BLOCKSIZE:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1) # (B, C)

            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx




        
    

x,y = get_batch("train")
model = BiGramLanuageModel()
model.to(device=device)
# predictions , loss = model(x,y)
# loss.item() , -log(1/65)
# print(decode(model.generate(idx = torch.zeros((1, 1), dtype=torch.long), max_new_tokens=100)[0].tolist()))

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

for steps in range(MAX_ITERS):
    x,y = get_batch("train")
    x, y = x.to(device), y.to(device)
    logits, loss = model(x,y)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    if steps % 100 == 0:
        print(loss.item())

print(decode(model.generate(idx = torch.zeros((1, 1), dtype=torch.long,device= device), max_new_tokens=100)[0].tolist()))
