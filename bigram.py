import torch
import torch.nn as nn
from torch.nn import functional as F

#hyperparameters
batch_size = 32 #independent sequences in parallel
block_size = 8 #max context length for prediction
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if  torch.cuda.is_available() else 'cpu'
eval_iters = 200

torch.manual_seed(1337)

with open('input.txt','r',encoding='utf-8') as f:
    text = f.read()

#all the unique characters that occur in our text
chars = sorted(list(set(text)))
vocab_size = len(chars)
#create a mapping from characters to integers
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[ch] for ch in s]
decode = lambda l: ''.join(itos[i] for i in l)

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

#super simple bigram model
class BigramLanguageModel(nn.Module):
  def __init__(self,vocab_size):  #defines constructor method for class
    super().__init__()            #calls constructor of parent class=>nn.Module
    self.token_embedding_table = nn.Embedding(vocab_size,vocab_size) #creates an embedding table for tokens, each token to a vector mapped in an embedding space

  def forward(self,idx,targets=None):  #forward pass of nn,
    #idx is prolly a tensor containing indices representing tokens in a sequence
    #targets is another tensor containing indices+1 representing next token in a sequence wrt to idx
    #they both are (B,T) tensors of integers
    logits = self.token_embedding_table(idx) #(b,t,c)
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
      #getting predictions
      logits,loss = self(idx) 
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


model = BigramLanguageModel(vocab_size)
m = model.to(device)

#create a pytorch optimizer
optimizer = torch.optim.AdamW(model.parameters(),lr = learning_rate)

for iter in range(max_iters):
   #every once in a while evaluate the loss on train and val sets
    if(iter%eval_interval ==0):
        losses = estimate_loss()
        # print(f'Step {iter}: train loss {losses['train']:.4f},val loss{losses['val']:.4f}')
        print(f'Step {iter}: train loss {losses["train"]:.4f}, val loss {losses["val"]:.4f}')
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

print(torch.cuda.is_available())

 