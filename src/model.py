import torch
from transformers import AutoTokenizer
from .vanilla_transformer import VanillaTransformer
import os

def get_model_and_tokenizer(config: dict):

    is_main_process = os.environ.get('LOCAL_RANK', '0') == '0'

    if is_main_process:
        print(f"Loading tokenizer: {config['tokenizer_name']}")
    tokenizer = AutoTokenizer.from_pretrained(config['tokenizer_name'])
    if tokenizer.pad_token is None:
        if is_main_process:
            print("Adding padding token to tokenizer.")
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    if 'sequence_length' in config:
        tokenizer.model_max_length = int(config['sequence_length'])
    
    vocab_size = len(tokenizer)

    if config['model_type'] == 'vanilla_transformer':
        if is_main_process:
            print("Instantiating Vanilla Transformer model.")
        model_cfg = config['model_config']
        
        gradient_checkpointing = config.get('gradient_checkpointing', False)
        use_flash_attention = config.get('use_flash_attention', False)

        if is_main_process:
            print(f"Gradient checkpointing: {'Enabled' if gradient_checkpointing else 'Disabled'}")
            print(f"Flash Attention: {'Enabled' if use_flash_attention else 'Disabled'}")
        
        model = VanillaTransformer(
            vocab_size=vocab_size,
            d_model=model_cfg['d_model'],
            nhead=model_cfg['nhead'],
            num_layers=model_cfg['num_layers'],
            dim_feedforward=model_cfg['dim_feedforward'],
            dropout=model_cfg['dropout'],
            gradient_checkpointing=gradient_checkpointing,
            use_flash_attention=use_flash_attention
        )
        
    return model, tokenizer