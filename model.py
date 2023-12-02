

import torch
import numpy as np  
import torch.nn.functional as F
from einops import repeat
from torch import nn
from x_transformers.autoregressive_wrapper import (
    exists,
    top_a,
    top_k,
    top_p
)

from x_transformers.x_transformers import (
    AbsolutePositionalEmbedding,
    AttentionLayers,
    Decoder,
    TokenEmbedding,
    always,
    default,
    exists
)

ENTMAX_ALPHA = 1.3
entmax = 1.5
DEVICE = torch.device("cuda")
class MusicTransformerWrapper(nn.Module):
    def __init__(self,
                 *,
                 max_seq_len,
                 attn_layers,
                 emb_dim=None,
                 max_beat=None,
                 max_mem_len=0.0,
                 shift_mem_down=0,
                 emb_dropout=0.0,
                 num_memory_tokens=None,
                 tie_embedding=False,
                 use_abs_pos_emb=True,
                 l2norm_embed=False
    ):
        
        super().__init__()
        assert isinstance(
            attn_layers, AttentionLayers
        ), "attention layers must be one of Encoder or Decoder"

        dim = attn_layers.dim
        self.emb_dim = 512 
        self.max_beat = max_beat
        self.max_seq_len = max_seq_len
        self.max_mem_len = max_mem_len
        self.shift_mem_down = shift_mem_down
        n_tokens = [3, 2048, 20, 129, 128, 33, 2, 5]

        if max_beat is not None:
            beat_dim = 1
            n_tokens[beat_dim] = max_beat + 1

        self.l2norm_embed = l2norm_embed
        self.token_emb = nn.ModuleList([TokenEmbedding(self.emb_dim, n) for n in n_tokens])
        
        self.pos_emb = (
            AbsolutePositionalEmbedding(
                self.emb_dim, max_seq_len, l2norm_embed=l2norm_embed
            )
            if (use_abs_pos_emb and not attn_layers.has_pos_emb)
            else always(0)
        )

        self.emb_dropout = nn.Dropout(emb_dropout)

        self.project_emb = (
            nn.Linear(self.emb_dim, dim) if self.emb_dim != dim else nn.Identity()
        )
        self.attn_layers = attn_layers
        self.norm = nn.LayerNorm(dim)

        self.init_()

        self.to_logits = (
            nn.ModuleList([nn.Linear(dim, n) for n in n_tokens])
            if not tie_embedding
            else [lambda t: t @ emb.weight.t() for emb in self.token_emb]
        )

        num_memory_tokens = default(num_memory_tokens, 0)
        self.num_memory_tokens = num_memory_tokens
        if num_memory_tokens > 0:
            self.memory_tokens = nn.Parameter(
                torch.randn(num_memory_tokens, dim)
            )
    
    def init_(self):
        if self.l2norm_embed:
            for emb in self.token_emb:
                nn.init.normal_(emb.emb.weight, std=1e-5)
            nn.init.normal_(self.pos_emb.emb.weight, std=1e-5)
            return

        for emb in self.token_emb:
            nn.init.kaiming_normal_(emb.emb.weight)
    
    def forward(self,
                x, 
                return_embeddings=False,
                mask=None,
                return_mems=False,
                return_attn=False,
                mems=None,
                **kwargs
    ):
        b, _, _ = x.shape
        x = x.to(DEVICE)

        num_mem = self.num_memory_tokens
        
        n_tokens = [3, 2048, 13, 129, 128, 33, 2, 5]
    
        token_emb = nn.ModuleList([TokenEmbedding(512, n) for n in n_tokens])
        token_emb = [emb.to(DEVICE) for emb in token_emb]
       
        x = sum(
            emb(x[..., i]) for i, emb in enumerate(token_emb)
        ) + self.pos_emb(x)

        x = self.emb_dropout(x)

        x = self.project_emb(x)
        if num_mem > 0:
            mem = repeat(self.memory_tokens, "n d -> b n d", b=b)
            x = torch.cat((mem, x), dim=1)
            
            if exists(mask):
                mask = F.pad(mask, (num_mem, 0), value=True)
        if self.shift_mem_down and exists(mems):
            mems_l, mems_r = (
                mems[: self.shift_mem_down],
                mems[self.shift_mem_down :],
            )
            mems = [*mems_r, *mems_l]

        x, intermediates = self.attn_layers(
            x, mask=mask, mems=mems, return_hiddens=True, **kwargs
        )
        x = self.norm(x)
        mem, x = x[:, :num_mem], x[:, num_mem:]

        out = (
            [to_logit(x) for to_logit in self.to_logits]
            if not return_embeddings
            else x
        )

        if return_mems:
            hiddens = intermediates.hiddens
            new_mems = (
                list(
                    map(
                        lambda pair: torch.cat(pair, dim=-2),
                        zip(mems, hiddens),
                    )
                )
                if exists(mems)
                else hiddens
            )
            new_mems = list(
                map(
                    lambda t: t[..., -self.max_mem_len :, :].detach(), new_mems
                )
            )
            return out, new_mems

        if return_attn:
            attn_maps = list(
                map(
                    lambda t: t.post_softmax_attn,
                    intermediates.attn_intermediates,
                )
            )
            return out, attn_maps

        return out
    
def sample(logits, 
           kind, 
           threshold, 
           temperature, 
           min_p_pow, 
           min_p_ratio
):
    
    """Sample from the logits with a specific sampling strategy."""
    if kind == "top_k":
        probs = F.softmax(top_k(logits, thres=threshold) / temperature, dim=-1)
    elif kind == "top_p":
        probs = F.softmax(top_p(logits, thres=threshold) / temperature, dim=-1)
    elif kind == "top_a":
        probs = F.softmax(
            top_a(logits, min_p_pow=min_p_pow, min_p_ratio=min_p_ratio)
            / temperature,
            dim=-1,
        )
    elif kind == "entmax":
        probs = entmax(logits / temperature, alpha=ENTMAX_ALPHA, dim=-1)
    else:
        raise ValueError(f"Unknown sampling strategy: {kind}")

    return torch.multinomial(probs, 1)


class MusicAutoregressiveWrapper(nn.Module):
    # @profile
    def __init__(self, 
                 net, 
                 ignore_index=-100, 
                 pad_value=0
    ):
        
        super().__init__()
        self.pad_value = pad_value
        self.ignore_index = ignore_index

        self.net = net
        self.max_seq_len = net.max_seq_len

        self.sos_type_code = 0
        self.note_type_code = 1 
        self.eos_type_code = 2 
        

        self.dimensions = {'type': 0, 'beat': 1, 'position': 2, 'pitch': 3, 'velocity':4, 'duration': 5, 'instrument': 6, 'section': 7} 
        assert self.dimensions["type"] == 0
    
    def forward(self, 
                x, 
                return_list=False, 
                **kwargs
    ):
        
        xi = x[:, :-1]
        xo = x[:, 1:]
        xo = xo.long()
        

        mask = kwargs.get("mask", None)
        if mask is not None and mask.shape[1] == x.shape[1]:
            mask = mask[:, :-1]
            kwargs["mask"] = mask
        
        out = self.net(xi, **kwargs)

        losses = [
            F.cross_entropy(    
                out[i].transpose(1, 2),
                xo[..., i],
                ignore_index=self.ignore_index,
            )
            for i in range(len(out))
        ]
        loss = sum(losses)

        if return_list:
            return loss, losses
        return loss


class MusicXTransformer(nn.Module):
    def __init__(self, 
                 *, 
                 dim, 
                 **kwargs
    ):
        
        super().__init__()
        assert "dim" not in kwargs, "dimension must be set with `dim` keyword"
        transformer_kwargs = {
            "max_seq_len": kwargs.pop("max_seq_len"),
            "max_beat": kwargs.pop("max_beat"),
            "emb_dropout": kwargs.pop("emb_dropout", 0),
            "use_abs_pos_emb": kwargs.pop("use_abs_pos_emb", True),
        }
        self.decoder = MusicTransformerWrapper(
            attn_layers=Decoder(dim=dim, attn_one_kv_head = True, ff_no_bias = True, **kwargs),
            **transformer_kwargs,
        )
        self.decoder = MusicAutoregressiveWrapper(
            self.decoder, 
        )

    @torch.no_grad()
    def generate(self, seq_in, seq_len, **kwargs):
        return self.decoder.generate(seq_in, seq_len, **kwargs)
    
    def forward(self, seq, mask=None, **kwargs):
        return self.decoder(seq, mask=mask, **kwargs)
 