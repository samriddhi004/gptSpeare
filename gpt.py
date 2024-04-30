import torch
import torch.nn as nn
from torch.nn import functional as F
print(torch.cuda.is_available())
#hyperparameters
batch_size = 64 #independent sequences in parallel
block_size = 256 #max context length for prediction
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if  torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384 # look into multiheadattention
n_head = 6
n_layer = 6
dropout = 0.2
torch.manual_seed(1337)

with open('input.txt','r',encoding='utf-8') as f:
    text = f.read()
    
#all the unique characters that occur in our text
chars = sorted(list(set(text)))
vocab_size = len(chars)
#create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[ch] for ch in s]
decode = lambda l: ''.join([itos[i] for i in l])

#train and test splits
data = torch.tensor(encode(text),dtype = torch.long)
n = int(0.9* len(data)) #90% training data , 10% test data
train_data = data[:n]
val_data = data[n:]

#loading data
def get_batch(split):
    #generate small batch of data of inputs x and targets y
    data = train_data if split =='train' else val_data
    ix = torch.randint(len(data)-block_size,(batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x,y = x.to(device),y.to(device)
    return x,y

@torch.no_grad() # to avoid gradients we only want to inference not train so as to save memory usage
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train','val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X,Y = get_batch(split)
            logits,loss = model(X,Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


class Head(nn.Module):
 #one head of self-attention"""""
    def __init__(self,head_size):
        super().__init__()
        self.key   = nn.Linear(n_embd,head_size,bias=False)
        self.query = nn.Linear(n_embd,head_size,bias=False)
        self.value = nn.Linear(n_embd,head_size,bias=False)
        self.register_buffer('tril',torch.tril(torch.ones(block_size,block_size)))
        # tril = torch.tril(torch.ones(T,T))
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        # input of size (batch, time-step, channels)
        # output of size (batch, time-step, head size)
        B,T,C = x.shape
        k = self.key(x)  #(B,T,C)
        q = self.query(x)  #(B,T,C)
        #compute attention scores/AFFINITIES
        wei = q @ k.transpose(-2,-1) * k.shape[-1]**-0.5 # (B,T,C) @(B,C,T) --> (B.T.T)
        wei = wei.masked_fill(self.tril[:T,:T]==0,float('-inf')) #(B.T.T)
        # wei = wei.masked_fill(tril==0,float('-inf')) # so as to ONLY let the past tokens talk to the present token
        wei = F.softmax(wei,dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)  #(B,T,C)
        #weighted aggregation of values
        out = wei @ v # (B,T,T) @ (B,T,C) ---> (B,T,C)
        return out

class MultiHeadAttention(nn.Module):
   # multiple heads of self-attention in parallel
    def __init__(self,num_heads,head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads,n_embd)
        #PROJECTION IS THE LINEAR TRANSFORM OF OUTCOME OF THIS OUT LAYER
        self.dropout = nn.Dropout(dropout)
    def forward(self,x):
        out = torch.cat([h(x) for h in self.heads],dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    #to think on data individually
    #simple linear layer followed by non-linearity
    def __init__(self,n_embd):
        super().__init__()
        self.net = nn.Sequential (
            nn.Linear(n_embd,4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd,n_embd),
            nn.Dropout(dropout),
        )
    def forward(self,x):
        return self.net(x)
  
class Block(nn.Module):
    # TRANSFORMER BLOCK
    # communication followed by computation
    def __init__(self,n_embd,n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head,head_size) #communication
        self.ffwd = FeedForward(n_embd)# computation
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self,x):  #RESNET
        x = x + self.sa(self.ln1(x))   #attention(interaction)
        x = x + self.ffwd(self.ln2(x)) #computation
        return x 

#super simple bigram model
class GPTLanguageModel(nn.Module):
    def __init__(self):  #defines constructor method for class
        super().__init__()            #calls constructor of parent class=>nn.Module
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size,n_embd) #creates an embedding table for tokens, each token to a vector mapped in an embedding space
        self.position_embedding_table = nn.Embedding(block_size,n_embd)
        # self.blocks = nn.Sequential(
        #    Block(n_embed, n_head=4),
        #    Block(n_embed,n_head=4),
        #    Block(n_embed,n_head=4),
        #    nn.LayerNormal(n_embed),
        # )
        # self.blocks = nn.Sequential(*[Block(n_embd,n_head=n_head) for _ in range(n_layer)])
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        # self.sa_heads = MultiHeadAttention(4,n_embd//4) # 4 heads of 8-d self attention
        # self.ffwd = FeedForward(n_embd)
        self.lm_head = nn.Linear(n_embd,vocab_size)
        # self.sa_head = Head(n_embd)
        # better init
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
      #ensure that the neural network starts with reasonable initial parameter values
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    def forward(self,idx,targets=None):  #forward pass of nn,
      #idx is prolly a tensor containing indices representing tokens in a sequence
      #targets is another tensor containing indices+1 representing next token in a sequence wrt to idx
      #they both are (B,T) tensors of integers
        B,T = idx.shape
        token_emb = self.token_embedding_table(idx) #(b,t,n_embed)
        pos_emb = self.position_embedding_table(torch.arange(T,device=device)) # [integers from 0 to T-1 get embedded through table to create->]T,C matrix
        x = token_emb + pos_emb
        # x = self.sa_heads(x)
        # x = self.ffwd(x)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) # (b,t,vocab_size)
        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T,C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets)
        return logits,loss

    def generate(self,idx,max_new_tokens):
        #idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            idx_cond = idx[:,-block_size:]

            #getting predictions
            logits,loss = self(idx_cond) 
            # we don't need loss; tso lhere's no ground truth for what the next word should be!
            # self is instance of bigramlanguagemodel and it invokes forward() method
            #focus only on the last time step
            logits = logits[:,-1,:] # now it's (B,C)  /logits is prediction here
            #softmax to get probabilities
            probs = F.softmax(logits,dim=-1) #accounts for rows
            #sample from distribution
            idx_next = torch.multinomial(probs,num_samples=1) # (B,1)
            # each token has a list of probs from softmax(equal to total vocab) with prob of each vocab wrt which is more likely to come next after the TOKEN
            # multinomial samples an index which has max likelihook ????????
            #append sampled index to the running sequence
            idx = torch.cat((idx,idx_next),dim=1) #(B,T+1)
        return idx


model = GPTLanguageModel()
m = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in m.parameters())/1e6, 'M parameters')
#create a pytorch optimizer
optimizer = torch.optim.AdamW(model.parameters(),lr = learning_rate)

for iter in range(max_iters):
   #every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0  or iter == max_iters - 1:
        losses = estimate_loss()
        # print(f'Step {iter}: train loss {losses['train']:.4f},val loss{losses['val']:.4f}')
        print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        #sample a batch of data
    xb,yb = get_batch('train')
    #evaluate loss
    logits,loss = model(xb,yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

#generate from the model
context = torch.zeros((1,1),dtype=torch.long,device=device)
print(decode(m.generate(context,max_new_tokens=500)[0].tolist()))
open('generateMore.txt', 'w').write(decode(m.generate(context, max_new_tokens=10000)[0].tolist()))

 