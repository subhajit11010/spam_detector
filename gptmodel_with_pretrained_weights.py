# import tiktoken
import torch
# from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
# import pandas as pd
# import numpy as np
# GPT_CONFIG_124M = {
#  "vocab_size": 50257,
#  "context_length": 256,
#  "emb_dim": 768,
#  "n_heads": 12,
#  "n_layers": 12,
#  "drop_rate": 0.1,
#  "qkv_bias": False
# }
# class GPTDatasetV1(Dataset):
#     def __init__(self, txt, tokenizer, max_length, stride):
#         self.tokenizer = tokenizer
#         self.input_ids = []
#         self.target_ids = []

#         token_ids = tokenizer.encode(txt) #tokenization and conversion to token ids using BPE
#         for i in range(0, len(token_ids) - max_length, stride):
#             input_chunk = token_ids[i:i+max_length]
#             target_chunk = token_ids[i+1:i+max_length+1]
#             self.input_ids.append(torch.tensor(input_chunk))
#             self.target_ids.append(torch.tensor(target_chunk))

#     def __len__(self):
#         return len(self.input_ids)
#     def __getitem__(self, idx):
#         return self.input_ids[idx], self.target_ids[idx]

# def create_dataloader_v1(txt, batch_size=4, max_length=256,
#                          stride=128, shuffle=True, drop_last=True,
#                          num_workers=0):

#     # Initialize the tokenizer
#     tokenizer = tiktoken.get_encoding("gpt2")

#     # Create dataset
#     dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)

#     # Create dataloader
#     dataloader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=shuffle,
#         drop_last=drop_last,
#         num_workers=num_workers
#     )

#     return dataloader

class LayerNorm(nn.Module):        # xi(cap) = ((xi - mew) / (sigma + epsilon)) * gamma + beta
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))
        # print(self.scale)
        # print(self.shift)
    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        var = x.var(dim=-1, keepdim = True, unbiased = False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        return self.scale * norm_x + self.shift

class GELU(nn.Module):      # GELU(x) ≈ 0.5 ⋅ x ⋅ (1 + tanh[√((2/π)) ⋅ (x + 0.044715 ⋅ x^3])
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * torch.pow(x,3))))

class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(nn.Linear(cfg['emb_dim'], 4 * cfg["emb_dim"]), GELU(), nn.Linear(4 * cfg["emb_dim"], cfg["emb_dim"]))
    def forward(self, x):
        return self.layers(x)

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out % num_heads == 0), \
            "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads # Reduce the projection dim to match desired output dim

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)  # Linear layer to combine head outputs
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length),
                       diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys = self.W_key(x) # Shape: (b, num_tokens, d_out)
        queries = self.W_query(x)
        values = self.W_value(x)

        # We implicitly split the matrix by adding a `num_heads` dimension
        # Unroll last dim: (b, num_tokens, d_out) -> (b, num_tokens, num_heads, head_dim)
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Compute scaled dot-product attention (aka self-attention) with a causal mask
        attn_scores = queries @ keys.transpose(2, 3)  # Dot product for each head

        # Original mask truncated to the number of tokens and converted to boolean
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        # Use the mask to fill attention scores
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Shape: (b, num_tokens, num_heads, head_dim)
        context_vec = (attn_weights @ values).transpose(1, 2)

        # Combine heads, where self.d_out = self.num_heads * self.head_dim
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        context_vec = self.out_proj(context_vec) # optional projection

        return context_vec
    
class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiHeadAttention(d_in=cfg['emb_dim'], d_out=cfg['emb_dim'], context_length=cfg['context_length'], dropout=cfg['drop_rate'], num_heads=cfg['n_heads'], qkv_bias=cfg["qkv_bias"])
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg['emb_dim'])
        self.norm2 = LayerNorm(cfg['emb_dim'])
        self.drop_resid = nn.Dropout(cfg['drop_rate'])
    def forward(self, x):
        # first sub-block
        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_resid(x)
        x = x + shortcut

        # second sub-block
        shortcut = x
        x = self.norm2(x)
        x = self.ff(x)
        x = self.drop_resid(x)
        x = x + shortcut
        return x

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.pos_emb = nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        self.drop_emb = nn.Dropout(cfg['drop_rate'])
        self.trf_blocks = nn.Sequential(*[TransformerBlock(cfg) for _ in range(cfg["n_layers"])])
        self.final_norm = LayerNorm(cfg['emb_dim'])
        self.out_head = nn.Linear(
            cfg["emb_dim"], cfg["vocab_size"], bias=False
        )
    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits

# def generate_text_simple(model, idx, max_new_tokens, context_size):
#     for _ in range(max_new_tokens):
#         idx_cond = idx[:, -context_size:]
#         with torch.no_grad():
#             logits = model(idx_cond)
#             # Focus only on the last time step
#         # (batch, n_tokens, vocab_size) becomes (batch, vocab_size)
#         logits = logits[:, -1, :]

#         # Apply softmax to get probabilities
#         probas = torch.softmax(logits, dim=-1)  # (batch, vocab_size)

#         # Get the idx of the vocab entry with the highest probability value
#         idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)

#         # Append sampled index to the running sequence
#         idx = torch.cat((idx, idx_next), dim=1)  # (batch, n_tokens+1)

#     return idx

# def generate(model, idx, max_new_tokens, context_size, temperature=0.0, top_k=None, eos_id=None):
#   for _ in range(max_new_tokens):
#     idx_cond = idx[:, -context_size:]
#     with torch.no_grad():
#       logits = model(idx_cond)
#     logits = logits[:,-1,:]
#     if top_k is not None:
#       top_logits, _ = torch.topk(logits, top_k)
#       min_val = top_logits[:,-1]
#       logits = torch.where(logits < min_val,
#                            torch.tensor(float('-inf')).to(logits.device),
#                            logits)
#     if temperature > 0.0:
#       logits = logits / temperature
#       probs = torch.softmax(logits, dim = -1)
#       idx_next = torch.multinomial(probs, num_samples = 1)
#     else:
#       idx_next = torch.argmax(logits, dim = -1, keepdim = True)
#     if idx_next == eos_id:
#       break
#     idx = torch.cat((idx, idx_next), dim = 1)
#   return idx

# from gpt_download import download_and_load_gpt2
# # settings, params = download_and_load_gpt2(
# #  model_size="124M", models_dir="gpt2"
# # )
model_configs = {
 "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
 "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
 "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
 "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}
# model_name = "gpt2-small (124M)"
# NEW_CONFIG = GPT_CONFIG_124M.copy()
# NEW_CONFIG.update(model_configs[model_name])
# NEW_CONFIG.update({"context_length" : 1024})
# NEW_CONFIG.update({"qkv_bias": True})

# gpt = GPTModel(NEW_CONFIG)
# gpt.eval()
# def assign(left, right):
#  if left.shape != right.shape:
#   raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
#  return torch.nn.Parameter(torch.tensor(right))

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# def load_weights_into_gpt(gpt, params):
#   gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe'])
#   gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])
#   for b in range(len(params["blocks"])):
#     q_w, k_w, v_w = np.split((params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
#     gpt.trf_blocks[b].att.W_query.weight = assign(gpt.trf_blocks[b].att.W_query.weight, q_w.T)
#     gpt.trf_blocks[b].att.W_key.weight = assign(gpt.trf_blocks[b].att.W_key.weight, k_w.T)
#     gpt.trf_blocks[b].att.W_value.weight = assign(gpt.trf_blocks[b].att.W_value.weight, v_w.T)

#     q_b, k_b, v_b = np.split((params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
#     gpt.trf_blocks[b].att.W_query.bias = assign(gpt.trf_blocks[b].att.W_query.bias, q_b)
#     gpt.trf_blocks[b].att.W_key.bias = assign(gpt.trf_blocks[b].att.W_key.bias, k_b)
#     gpt.trf_blocks[b].att.W_value.bias = assign(gpt.trf_blocks[b].att.W_value.bias, v_b)

#     gpt.trf_blocks[b].att.out_proj.weight = assign(gpt.trf_blocks[b].att.out_proj.weight, params["blocks"][b]["attn"]["c_proj"]["w"].T)
#     gpt.trf_blocks[b].att.out_proj.bias = assign(gpt.trf_blocks[b].att.out_proj.bias, params["blocks"][b]["attn"]["c_proj"]["b"])

#     gpt.trf_blocks[b].ff.layers[0].weight = assign(gpt.trf_blocks[b].ff.layers[0].weight, params["blocks"][b]["mlp"]["c_fc"]["w"].T)
#     gpt.trf_blocks[b].ff.layers[0].bias = assign(gpt.trf_blocks[b].ff.layers[0].bias, params["blocks"][b]["mlp"]["c_fc"]["b"])
#     gpt.trf_blocks[b].ff.layers[2].weight = assign(gpt.trf_blocks[b].ff.layers[2].weight, params["blocks"][b]["mlp"]["c_proj"]["w"].T)
#     gpt.trf_blocks[b].ff.layers[2].bias = assign(gpt.trf_blocks[b].ff.layers[2].bias, params["blocks"][b]["mlp"]["c_proj"]["b"])

#     gpt.trf_blocks[b].norm1.scale = assign(gpt.trf_blocks[b].norm1.scale, params["blocks"][b]["ln_1"]["g"])
#     gpt.trf_blocks[b].norm1.shift = assign(gpt.trf_blocks[b].norm1.shift, params["blocks"][b]["ln_1"]["b"])
#     gpt.trf_blocks[b].norm2.scale = assign(gpt.trf_blocks[b].norm2.scale, params["blocks"][b]["ln_2"]["g"])
#     gpt.trf_blocks[b].norm2.shift = assign(gpt.trf_blocks[b].norm2.shift, params["blocks"][b]["ln_2"]["b"])

#   gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
#   gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
#   gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"])

# # load_weights_into_gpt(gpt, params)
# gpt.to(device)
# text = "Do you know"
# tokenizer = tiktoken.get_encoding("gpt2")
# encoded = tokenizer.encode(text)
# encoded_tensor = torch.tensor(encoded).unsqueeze(0)
# output = generate_text_simple(model=gpt, idx=encoded_tensor, max_new_tokens=5, context_size=NEW_CONFIG["context_length"])
# flat = output.squeeze(0)
# decoded = tokenizer.decode(flat.tolist())
# # print(decoded)
# #################################################   Fine tuning   ##############################################################

