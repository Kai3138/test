
"""
GPT model:
- the initial stem consists of a combination of token encoding and a positional encoding
- the meat of it is a uniform sequence of Transformer blocks
    - each Transformer is a sequential combination of a 1-hidden-layer MLP block and a self-attention block
    - all blocks feed into a central residual pathway similar to resnets
- the final decoder is a linear projection into a vanilla Softmax classifier
"""

import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F

from .mlp import GatedMLP, Mlp
from .mamba_simple import Mamba
from .LPE import LPE_1

from timm.models.layers import DropPath, to_2tuple, trunc_normal_

class CrossCondGPT2(nn.Module):
    """  Danceba Pipeline  """
    def __init__(self, config):
        super().__init__()
        self.gpt_base = CrossCondGPTBase(config.base)
        self.gpt_head = CrossCondGPTHead(config.head)
        self.block_size = config.block_size

    def get_block_size(self):
        return self.block_size

    def sample(self, xs, cond, shift=None):
        print("do sample!!!")
        block_size = self.get_block_size() - 1
        if shift is not None:
            block_shift = min(shift, block_size)
        else:
            block_shift = block_size
        x_up, x_down = xs
        for k in range(cond.size(1)):
            x_cond_up = x_up if x_up.size(1) <= block_size else x_up[:, -(block_shift+(k-block_size-1)%(block_size-block_shift+1)):]
            x_cond_down = x_down if x_down.size(1) <= block_size else x_down[:, -(block_shift+(k-block_size-1)%(block_size-block_shift+1)):]  # crop context if needed

            cond_input = cond[:, :k+1] if k < block_size else cond[:, k-(block_shift+(k-block_size-1)%(block_size-block_shift+1))+1:k+1]
            logits, _ = self.forward((x_cond_up, x_cond_down), cond_input)
            logit_up, logit_down = logits
            logit_up = logit_up[:, -1, :]
            logit_down = logit_down[:, -1, :]

            probs_up = F.softmax(logit_up, dim=-1)
            probs_down = F.softmax(logit_down, dim=-1)

            _, ix_up = torch.topk(probs_up, k=1, dim=-1)
            _, ix_down = torch.topk(probs_down, k=1, dim=-1)

            # append to the sequence and continue
            x_up = torch.cat((x_up, ix_up), dim=1)
            x_down = torch.cat((x_down, ix_down), dim=1)

        return ([x_up], [x_down])

    def forward(self, idxs, cond, targets=None): # cond: music condition ("music_seq[:, config.ds_rate//music_relative_rate:]")
        idx_up, idx_down = idxs
        
        targets_up, targets_down = None, None
        if targets is not None:
            targets_up, targets_down = targets
        # print("L98:",cond.shape)
        feat = self.gpt_base(idx_up, idx_down, cond)
        logits_up, logits_down, loss_up, loss_down = self.gpt_head(feat, targets)
        # logits_down, loss_down = self.down_half_gpt(feat, targets_down)
        
        if loss_up is not None and loss_down is not None:
            loss = loss_up + loss_down
        else:
            loss = None

        return (logits_up, logits_down), loss


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class CausalCrossConditionalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("mask", torch.tril(torch.ones(config.block_size, config.block_size))
                                     .view(1, 1, config.block_size, config.block_size))
        # self.mask = se
        self.n_head = config.n_head

    def forward(self, x, layer_past=None):
        B, T, C = x.size()  # T = 3*t (music up down)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        t = T // 3
        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.mask[:,:,:t,:t].repeat(1, 1, 3, 3) == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y

class Block_Base(nn.Module):
    """ an Temporal-Gated Causal Attention (TGCA) block """

    def __init__(self, config):
        super().__init__()
        self.norm1 = RMSNorm(config.n_embd)
        self.norm2 = RMSNorm(config.n_embd)
        self.in_proj = nn.Linear(config.n_embd, config.n_embd)
        self.act_proj = nn.Linear(config.n_embd, config.n_embd)
        self.act = nn.SiLU()
        self.sigmoid =  nn.Sigmoid()
        self.out_proj = nn.Linear(config.n_embd, config.n_embd)

        self.attn = CausalCrossConditionalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )


    def forward(self, x):
        shortcut = x
        x = self.norm1(x)
        act_res = self.act(self.act_proj(x)) 
        # SiLU # {'fid_k': (12.918347226610166-6.601375950773499e-07j), 'fid_g': (14.377040083933046-2.347936684761542e-08j), 'div_k': 8.241817106803259, 'div_g': 7.440401058930617} 0.2896913823938913
        # act_res = self.sigmoid(self.act_proj(x)) 
        # Sigmoid # {'fid_k': (22.10337116047633-4.1560030123329423e-07j), 'fid_g': 12.051192377432386, 'div_k': 7.663574515092067, 'div_g': 6.547586472982015} 0.2630817338399422
        x = self.in_proj(x)
        x = self.act(x)
        x = self.attn(x)
        x = self.out_proj(x * act_res)
        x = shortcut + x

        x = x + self.mlp(self.norm2(x))
        return x


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class SMR(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, linear = False):
        super(SMR, self).__init__()
        self.conv = nn.Conv1d(in_features, out_features, kernel_size, stride=1)
        self.use_linear = linear
        if linear:
            self.linear = nn.Linear(in_features, out_features)
        self.pad = (kernel_size- 1, 0)
    def forward(self, x):
        # Input shape: (B, H, L)
        # Output shape: (B, H, L)
        if self.use_linear:
            factor = self.linear(self.conv(F.pad(x, self.pad, mode='constant',value=0.0)).transpose(1, 2)).transpose(1, 2)
        else:
            factor = self.conv(F.pad(x, self.pad, mode='constant', value=0.0))
        return torch.sigmoid(factor) * x

class Block_Head(nn.Module):
    """ an Parallel Mamba block """

    def __init__(self, config):
        super().__init__()
        self.ln1 = RMSNorm(config.n_embd)
        self.ln2 = RMSNorm(config.n_embd)
        self.ln3 = RMSNorm(config.n_embd)
        self.ln4 = RMSNorm(config.n_embd)
        self.ln5 = RMSNorm(config.n_embd)
        self.ln6 = RMSNorm(config.n_embd)
        self.mamba_1 = Mamba(
            d_model=768,
            d_state=128,
            d_conv=4,
            expand=4,
        )
        self.mamba_2 = Mamba(
            d_model=768,
            d_state=128,
            d_conv=4,
            expand=4,
        )
        self.mamba_3 = Mamba(
            d_model=768,
            d_state=128,
            d_conv=4,
            expand=4,
        )
        self.gate_mlp_1 = GatedMLP(
            in_features=config.n_embd,
            hidden_features=768,
            out_features=config.n_embd,
        )
        self.gate_mlp_2 = GatedMLP(
            in_features=config.n_embd,
            hidden_features=768,
            out_features=config.n_embd,
        )
        self.gate_mlp_3 = GatedMLP(
            in_features=config.n_embd,
            hidden_features=768,
            out_features=config.n_embd,
        )

    def forward(self, x):

        t = x.size(1) // 3
        music = x[:, :t, :]
        up = x[:, t:2*t, :]
        down = x[:, 2*t:, :]

        music = music + self.mamba_1(self.ln1(music))
        music = music + self.gate_mlp_1(self.ln2(music))

        up = up + self.mamba_2(self.ln3(up))
        up = up + self.gate_mlp_2(self.ln4(up))

        down = down + self.mamba_3(self.ln5(down))
        down = down + self.gate_mlp_3(self.ln6(down))

        x = torch.cat([music, up, down], dim=1)

        return x

class CrossCondGPTBase(nn.Module):
    """  the Global Beat Attention via Temporal Gating in Sec 3.3 """

    def __init__(self, config):
        super().__init__()

        self.tok_emb_up = nn.Embedding(config.vocab_size_up, config.n_embd  )
        self.tok_emb_down = nn.Embedding(config.vocab_size_down, config.n_embd)
        """  Phase-Based Rhythm Feature Extraction in Sec 3.2  """
        self.pos_emb = LPE_1(config)
        self.position_scale = nn.Parameter(torch.tensor(1e-6))
        self.cond_emb = nn.Linear(config.n_music, config.n_embd)
        self.drop = nn.Dropout(config.embd_pdrop)
        self.blocks = nn.Sequential(*[Block_Base(config) for _ in range(config.n_layer)])
        self.block_size = config.block_size
        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # module.weight.data.uniform_(math.sqrt(6.0/sum(module.weight.size())))
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, idx_up, idx_down, cond):
        b, t = idx_up.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        b, t = idx_down.size()
        assert t <= self.block_size, "Cannot forward, model block size is exhausted."
        token_embeddings_up = self.tok_emb_up(idx_up) # each index maps to a (learnable) vector
        token_embeddings_down = self.tok_emb_down(idx_down) # each index maps to a (learnable) vector
        token_embeddings = torch.cat([self.cond_emb(cond), token_embeddings_up, token_embeddings_down ], dim=1)
        position_embeddings = self.pos_emb(cond)
        pos_size = token_embeddings.shape[1]
        position_embeddings = position_embeddings[:, :pos_size, :]
        position_embeddings = self.position_scale * position_embeddings
        x = self.drop(token_embeddings + position_embeddings)
        x = self.blocks(x)

        return x

class CrossCondGPTHead(nn.Module):
    """  the Mamba-Based Parallel Motion Modeling in Sec 3.4 """

    def __init__(self, config):
        super().__init__()

        self.blocks = nn.Sequential(*[Block_Head(config) for _ in range(config.n_layer)])
        self.block_base = Block_Base(config)
        # decoder head
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.RMS_f = RMSNorm(config.n_embd)
        self.block_size = config.block_size
        self.head_up = nn.Linear(config.n_embd, config.vocab_size_up, bias=False)
        self.head_down = nn.Linear(config.n_embd, config.vocab_size_down, bias=False)
        self.apply(self._init_weights)

    def get_block_size(self):
        return self.block_size

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (torch.nn.Linear, )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def forward(self, x, targets=None):

        x = self.blocks(x)
        x = self.block_base(x)
        x = self.RMS_f(x)
        N, T, C = x.size()
        t = T//3
        logits_up = self.head_up(x[:, t:t*2, :])
        logits_down = self.head_down(x[:, t*2:t*3, :]) # down half 

        loss_up, loss_down = None, None

        if targets is not None:
            targets_up, targets_down = targets

            loss_up = F.cross_entropy(logits_up.view(-1, logits_up.size(-1)), targets_up.view(-1))
            loss_down = F.cross_entropy(logits_down.view(-1, logits_down.size(-1)), targets_down.view(-1))
            

        return logits_up, logits_down, loss_up, loss_down
