import math
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from transformers.modeling_outputs import CausalLMOutput
from torch.backends.cuda import sdp_kernel, SDPBackend
from contextlib import nullcontext

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

def _bool_causal_mask(t: int, device) -> torch.Tensor:
    return torch.ones((t, t), dtype=torch.bool, device=device).triu(1)

class VanillaTransformer(nn.Module):
    """
    A vanilla Transformer model for Causal Language Modeling.
    Uses a boolean causal attn mask (src_mask) and a key padding mask.
    """
    def __init__(self, vocab_size: int, d_model: int, nhead: int, num_layers: int, 
                 dim_feedforward: int, dropout: float, gradient_checkpointing: bool = False,
                 use_flash_attention: bool = False):
        super().__init__()
        self.d_model = d_model
        self.gradient_checkpointing = gradient_checkpointing
        self.use_flash_attention = use_flash_attention
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
            for _ in range(num_layers)
        ])
        
        self.fc_out = nn.Linear(d_model, vocab_size)
        self._init_weights()

    def _init_weights(self) -> None:
        initrange = 0.1
        self.embedding.weight.data.uniform_(-initrange, initrange)
        self.fc_out.bias.data.zero_()
        self.fc_out.weight.data.uniform_(-initrange, initrange)

    def forward(self, input_ids, attention_mask=None, labels=None):
        seq_len = input_ids.size(1)
        device = input_ids.device

        embedded = self.embedding(input_ids) * math.sqrt(self.d_model)
        output = self.pos_encoder(embedded.transpose(0, 1)).transpose(0, 1)

        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        causal_mask = _bool_causal_mask(seq_len, device)

        ctx = sdp_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]) if self.use_flash_attention else nullcontext()
        with ctx:
            for layer in self.transformer_layers:
                if self.gradient_checkpointing and self.training:
                    def layer_f(t):
                        return layer(t, src_mask=causal_mask, src_key_padding_mask=key_padding_mask)
                    output = checkpoint(layer_f, output, use_reentrant=False)
                else:
                    output = layer(output, src_mask=causal_mask, src_key_padding_mask=key_padding_mask)
        
        logits = self.fc_out(output)

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return CausalLMOutput(loss=loss, logits=logits)