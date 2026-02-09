import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam, AdamW
from torch.utils.data import TensorDataset, DataLoader

import lightning as L

# 定义超参数
batch_size = 32
block_size = 8
max_iters = 3000
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200

torch.manual_seed(1008)

# cd /Users/apple/Pytorch-tutorial
# curl -O https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
# 把tiny Shakespeare数据集里的数据read到一个item里，方便后续处理
with open('input.txt', 'r', encoding='utf-8') as f:
  text = f.read()

# 从文本中构建字符级词表（character vocabulary）
chars = sorted(list(set(text)))
# set(text) <-- 取出字符串里出现过的所有不同字符（去重）
# list(set(text)) <-- 把集合变成 list
# sorted(list(set(text))) <-- 把字符按字典序排序
vocab_size = len(chars)

# 建立字符 ↔ 数字 ID 的双向映射
stoi = { ch:i for i,ch in enumerate(chars)}
# enumerate <-- 给每个字符分配一个整数编号
itos = { i:ch for i,ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # encode：把字符串 → 数字序列
decode = lambda l: ''.join([itos[i] for i in l])  # decode：把数字序列 → 字符串

# encode整个text数据集，用tensor储存
data = torch.tensor(encode(text), dtype=torch.long) # 64-bit 整数（int64）

# 把数据集分成训练集和验证集
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
  data = train_data if split == 'train' else val_data
  ix = torch.randint(len(data) - block_size, (batch_size,)) # 随机选 batch_size 个起点索引，用来从长序列里切训练样本
  # len(data) - block_size  <--  这是最大允许的起点位置。
  # torch.randint(high, size)
    # 生成 shape = size 的 tensor，
    # 每个元素是：0 ≤ x < high
  x = torch.stack([data[i:i+block_size] for i in ix])
  y = torch.stack([data[i+1:i+block_size+1] for i in ix])
  x, y = x.to(device), y.to(device) # 把 batch 的张量移动到指定计算设备上（CPU / GPU）
  return x, y

@torch.no_grad() # 下面的代码不需要计算梯度
def estimate_loss():
    out = {}
    model.eval() # 把模型切换到评估模式（不启用 dropout 等训练时特有的机制）
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split)
            logits, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # 切回训练模式
    return out

class BigramLanguageModel(nn.Module):
  def __init__(self, vocab_size):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    # nn.Embedding(num_embeddings, embedding_dim)
      # num_embeddings <-- 词表大小（多少个 token）
      # embedding_dim <-- 每个 token 用多少维向量表示
      # Embedding = 选矩阵的一行  Linear = 矩阵乘法

  def forward(self, idx, targets=None):

    # idx and targets are both (B, T)
    logits = self.token_embedding_table(idx) # (B, T, C)

    if targets is None:
      loss = None

    else:
      B, T, C = logits.shape
      relogits = logits.reshape(B*T, C)
      B1, T1 = targets.shape
      retargets = targets.reshape(B1*T1, )
      loss = F.cross_entropy(relogits, retargets)
    # logits的shape
     # 分类任务：(N, C)
      # N = 样本数
      # C = 类别数
     # 语言模型：（B, T, V) <-- V是vocab size，在bigram里V等于C
      # 要先reshape成(B*T, V)
    # targets的shape
     # 分类任务：(N, )
      # N = 样本数
     # 语言模型：（B, T)
      # 要先reshape成(B*T, )

    return logits, loss

  def generate(self, idx, max_new_tokens):  # 自回归文本生成
    # idx is (B, T)-array of indices in the current text
    # max_new_tokens: 要新生成多少个 token
    for _ in range(max_new_tokens):   # 循环 max_new_tokens 次，每次生成一个 token。
      # 预测
      logits, loss = self(idx) # 等价于logits, loss = self.forward(idx)
      # 只关注最后一次
      logits = logits[:, -1, :] # 变成(B, C)
      # apply softmax
      probs = F.softmax(logits, dim=-1) # (B, C)
      # sample from distribution
      idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
      # append samples index to the running sequence
      idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
    return idx

model = BigramLanguageModel(vocab_size) # BigramLanguageModel.__init__(self, vocab_size)
m = model.to(device) # 把模型的参数移动到指定计算设备上（CPU / GPU），m是model的别名，指向同一个对象
# logits, loss = m(xb, yb) # out = m.__call__(xb, yb)
                           # nn.Module.__call__() 内部会自动调用class的 forward() 方法
                           # m(xb, yb) ≡ m.forward(xb, yb)

# Train the Model
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

for step in range(max_iters):
  
  if step % eval_iters == 0:
    losses = estimate_loss()
    print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

  # Sample a batch of data
  xb, yb = get_batch('train')

  # evaluate the loss
  logits, loss = m(xb, yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()

print(decode(m.generate(idx = torch.zeros((1,1), dtype=torch.long), max_new_tokens=500)[0].tolist()))
