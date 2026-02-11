import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Adam, AdamW
from torch.utils.data import TensorDataset, DataLoader

import lightning as L

# 定义超参数
batch_size = 64
block_size = 256
max_iters = 5000
learning_rate = 3e-4
if torch.cuda.is_available():
  device = 'cuda'
elif torch.backends.mps.is_available():
  device = 'mps'
else:
  device = 'cpu'
eval_iters = 300
n_embd = 384 # 每个 token 用多少维向量表示
n_head = 8 # 多头注意力机制中 head 的数量
n_layer = 6 # Transformer block 的数量
dropout = 0.2 # dropout 的概率

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

class Head(nn.Module):
  """ one head of self-attention """

  def __init__(self, head_size):
    super().__init__()
    self.key = nn.Linear(n_embd, head_size, bias=False)
    self.query = nn.Linear(n_embd, head_size, bias=False)
    self.value = nn.Linear(n_embd, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # tril = lower triangular matrix
    # register_buffer <-- 把一个 tensor 注册为模型的 buffer
    # 这个 tensor 不会被视为模型的参数（不会被优化器更新），但会随着模型一起保存和加载。
    # 这里用来存储一个下三角矩阵，表示在自注意力计算中，哪些位置可以看到哪些位置
    # （即只能看到当前和之前的位置，不能看到未来的位置）。

    self.dropout = nn.Dropout(dropout) # dropout 层，随机丢弃一些神经元，防止过拟合

  def forward(self, x):
    B,T,C = x.shape
    k = self.key(x)   # (B,T,C) -> (B,T,head_size)
    q = self.query(x) # (B,T,C) -> (B,T,head_size)
    # compute attention scores ("affinities")
    wei = q @ k.transpose(-2,-1) * C**-0.5 # (B,T,head_size) @ (B,head_size,T) -> (B,T,T)
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B,T,T) 通过 tril 矩阵把未来位置的权重设置为 -inf，确保模型只能关注当前和之前的位置
    wei = F.softmax(wei, dim=-1) # (B,T,T) 对每一行进行 softmax，得到注意力权重
     # 这里的 wei 是一个注意力权重矩阵，表示每个位置对其他位置的关注程度。每一行的元素之和为 1，表示该位置的注意力分布。
     # 例如，wei[i,j] 表示位置 i 对位置 j 的关注程度。由于使用了 tril 矩阵，wei[i,j] 只有在 j <= i 时才可能非零，表示位置 i 只能关注位置 j（即当前和之前的位置）。
     # 这个注意力权重矩阵是自注意力机制的核心，决定了模型在处理序列数据时如何聚合不同位置的信息。
     # 注意力权重矩阵的形状是 (B,T,T)，其中 B 是批量大小，T 是序列长度。每个位置 i 的注意力权重分布在该位置的行上，表示位置 i 对所有位置 j 的关注程度。
    wei = self.dropout(wei) # dropout 作用在注意力权重上，随机丢弃一些注意力连接，防止过拟合
    v = self.value(x) # (B,T,C) -> (B,T,head_size)
    out = wei @ v # (B,T,T) @ (B,T,head_size) -> (B,T,head_size)
    return out
  
class MultiHeadAttention(nn.Module):
  """ multiple heads of self-attention in parallel """

  def __init__(self, num_heads, head_size):
    super().__init__()
    # 创建 num_heads 个 Head 模块，并把它们放在一个 ModuleList 中，方便在 forward 中迭代调用
    # ModuleList 是一个特殊的容器，用来存储 nn.Module 对象的
    self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
    self.proj = nn.Linear(n_embd, n_embd) # 最后把拼接后的输出映射回 n_embd 维，保持输入输出维度一致
    self.dropout = nn.Dropout(dropout) # dropout 层，随机丢弃一些神经元，防止过拟合

  def forward(self, x):
    # 把每个 head 的输出拼接在一起，得到最终的输出。每个 head 的输出维度是 head_size，拼接后总维度是 num_heads * head_size。
    # torch.cat 是 PyTorch 中的一个函数，用于在指定维度上连接多个张量。这里我们在最后一个维度上连接每个 head 的输出。
    out = torch.cat([h(x) for h in self.heads], dim=-1)
    out = self.proj(out) # (B,T,num_heads*head_size) -> (B,T,n_embd)
     # 这里的 out 是多头自注意力机制的输出，经过线性变换后保持与输入维度一致，方便后续的残差连接和前馈网络处理。
     # 多头自注意力机制通过并行计算多个 head 的注意力，可以让模型在不同的子空间中学习不同的表示，从而增强模型的表达能力。
     # 最后通过线性变换把拼接后的输出映射回 n_embd 维，保持输入输出维度一致，方便后续的残差连接和前馈网络处理。
     # 这个 MultiHeadAttention 模块现在包含了多个 Head，并且在 forward 中把它们的输出拼接在一起，形成了一个强大的自注意力机制。
     # 在 Transformer 模型中，MultiHeadAttention 是核心组件之一，负责捕捉序列中不同位置之间的依赖关系和上下文信息。
     # 通过多个 head 的并行计算，模型可以同时关注序列中的不同位置，从而更好地理解和生成文本数据。
    return out
  
class FeedForward(nn.Module):
  """ a simple linear layer followed by a non-linearity """

  def __init__(self, n_embd):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(n_embd, 4*n_embd), # 把 embedding 的维度扩展到 4 倍，增加模型的表达能力
      nn.ReLU(), # 非线性激活函数，增加模型的非线性表达能力
      nn.Linear(4*n_embd, n_embd), # 再把维度缩回 n_embd，保持输入输出维度一致
      nn.Dropout(dropout) # dropout 层，随机丢弃一些神经元，防止过拟合
    )

  def forward(self, x):
    return self.net(x)
  
class Block(nn.Module): # 一个 Block = 多头自注意力（通信） + 前馈网络（计算） + 残差连接
  """ Transformer block: communication followed by computation """

  def __init__(self, n_embd, n_head):
    super().__init__()
    head_size = n_embd // n_head # 每个 head 的维度
    self.sa = MultiHeadAttention(n_head, head_size) # 4 个 head，每个 head 的维度是 n_embd // 4，这样拼接后总维度还是 n_embd
    self.ffwd = FeedForward(n_embd) # (B, T, C) -> (B, T, C) 前馈网络保持输入输出维度一致
    self.ln1 = nn.LayerNorm(n_embd) # LayerNorm 是一种归一化方法，帮助模型更快收敛和更稳定训练
    self.ln2 = nn.LayerNorm(n_embd) # 每个 Block 有两个

  def forward(self, x):
    x = self.sa(self.ln1(x)) + x # 先经过多头自注意力模块，得到新的表示，然后加上原输入 x，形成残差连接。这样可以帮助信息流在网络中更好地传播，缓解梯度消失问题。
    x = self.ffwd(self.ln2(x)) + x # 再经过前馈网络，得到新的表示，再加上输入 x，形成另一个残差连接。这样每个 Block 都有两个残差连接，进一步增强信息流和梯度流。
    return x

# 这个 Block 现在包含了多头自注意力机制和前馈网络，并且在每个模块后都有残差连接。
# 这是 Transformer 模型的基本构建块，多个 Block 堆叠在一起可以构成一个深层的 Transformer 模型，具有强大的表达能力。


class BigramLanguageModel(nn.Module):
  def __init__(self):
    super().__init__()
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    # nn.Embedding(num_embeddings, embedding_dim)
      # num_embeddings <-- 词表大小（多少个 token）
      # embedding_dim <-- 每个 token 用多少维向量表示
      # Embedding = 选矩阵的一行  Linear = 矩阵乘法

    # 因为模型不再让 embedding 直接当 logits，而是：
      # 用低维 embedding 表示 token →
      # 经过网络处理 →
      # 最后再映射回 vocab_size。
    # n_embd = hidden dimension
    self.position_embedding_table = nn.Embedding(block_size, n_embd) 
    # 位置编码表，block_size 是最大上下文长度（模型能看到的最长文本长度），n_embd 是 embedding 的维度
    self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)]) # 4 个 Block 堆叠在一起，增加模型的深度和表达能力
    self.ln_f = nn.LayerNorm(n_embd) # 最后的 LayerNorm，保持输入输出维度一致
    self.lm_head = nn.Linear(n_embd, vocab_size) # 最后把 n_embd 维的表示映射回 vocab_size 维，得到每个 token 的预测概率分布

  def forward(self, idx, targets=None):
    B, T = idx.shape

    # idx and targets are both (B, T)
    tok_embd = self.token_embedding_table(idx) # (B, T, C)
    pos_embd = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
    x = tok_embd + pos_embd # (B, T, C) 位置编码和 token embedding 相加，得到最终的输入表示
    x = self.blocks(x) # (B, T, C) 经过多个 Block 的处理，得到新的表示
    x = self.ln_f(x) # (B, T, C) 最后的 LayerNorm，保持输入输出维度一致
    logits = self.lm_head(x) # (B, T, vocab_size)

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
    idx = idx.to(device)
    for _ in range(max_new_tokens):   # 循环 max_new_tokens 次，每次生成一个 token。
      # 每次循环，模型都会根据当前文本 idx 预测下一个 token 的概率分布，然后从中采样一个 token，追加到 idx 中，形成新的文本输入。
      idx_cond = idx[:, -block_size:] # (B, block_size) 取当前文本的最后 block_size 个 token 作为输入，确保输入长度不超过模型的上下文窗口。
      # 预测
      logits, loss = self(idx_cond) # 等价于logits, loss = self.forward(idx_cond)
      # 只关注最后一次
      logits = logits[:, -1, :] # 变成(B, C)
      # apply softmax
      probs = F.softmax(logits, dim=-1) # (B, C)
      # sample from distribution
      idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
      # append samples index to the running sequence
      idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
    return idx

model = BigramLanguageModel() # BigramLanguageModel.__init__(self, vocab_size)
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

print(decode(m.generate(idx = torch.zeros((1,1), dtype=torch.long, device=device), max_new_tokens=500)[0].tolist()))
