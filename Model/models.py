# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# GLIDE: https://github.com/openai/glide-text2im
# MAE: https://github.com/facebookresearch/mae/blob/main/models_mae.py
# --------------------------------------------------------

import torch
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import PatchEmbed, Mlp
import torch.nn.functional as F
from typing import Optional

# Import temporal embedders
from temporal_embedder import SimpleTemporalPooler, TemporalFeatureExtractor


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class SD3Attention(nn.Module):
    """
    SD3-style attention that combines self-attention with cross-attention to conditioning.
    This replaces the timm Attention module seamlessly.
    """
    def __init__(self, hidden_size, num_heads, qkv_bias=True, dropout=0.0, use_conditioning=False, **kwargs):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Self-attention projections (like timm Attention)
        self.to_q = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.to_k = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        self.to_v = nn.Linear(hidden_size, hidden_size, bias=qkv_bias)
        
        self.use_conditioning = use_conditioning
        self.norm_q = None
        self.norm_k = None
        self.norm_added_q = None
        self.norm_added_k = None

        # Flag to control context-only mode
        self.context_pre_only = False
        
        if use_conditioning:
            # Cross-attention projections for conditioning (SD3 style)
            self.add_q_proj = nn.Linear(hidden_size, hidden_size, bias=True)
            self.add_k_proj = nn.Linear(hidden_size, hidden_size, bias=True)
            self.add_v_proj = nn.Linear(hidden_size, hidden_size, bias=True)
            
            # Optional: additional output projection for conditioning (SD3 style)
            self.to_add_out = nn.Linear(hidden_size, hidden_size, bias=True)
        
        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(hidden_size, hidden_size, bias=True),
            nn.Dropout(dropout)  # No dropout for compatibility
        )
        
    def forward(self, x, encoder_hidden_states=None):
        """
        Forward pass of SD3-style attention.
        Args:
            x: (batch_size, seq_len, hidden_size) - main input
            encoder_hidden_states: (batch_size, cond_len, hidden_size) - conditioning input
        Returns:
            If encoder_hidden_states is provided: (output, cond_output)
            Otherwise: output
        """
        residual = x
        batch_size = x.shape[0]

        # Self-attention projections
        query = self.to_q(x)
        key = self.to_k(x)
        value = self.to_v(x)

        # Reshape for multi-head attention
        query = query.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Optional normalization (SD3 style)
        if self.norm_q is not None:
            query = self.norm_q(query)
        if self.norm_k is not None:
            key = self.norm_k(key)

        # Cross-attention with conditioning (SD3 style)
        if encoder_hidden_states is not None and self.use_conditioning:
            # Project conditioning through separate linear layers
            encoder_hidden_states_query_proj = self.add_q_proj(encoder_hidden_states)
            encoder_hidden_states_key_proj = self.add_k_proj(encoder_hidden_states)
            encoder_hidden_states_value_proj = self.add_v_proj(encoder_hidden_states)

            # Reshape conditioning projections
            encoder_hidden_states_query_proj = encoder_hidden_states_query_proj.view(
                batch_size, -1, self.num_heads, self.head_dim
            ).transpose(1, 2)
            encoder_hidden_states_key_proj = encoder_hidden_states_key_proj.view(
                batch_size, -1, self.num_heads, self.head_dim
            ).transpose(1, 2)
            encoder_hidden_states_value_proj = encoder_hidden_states_value_proj.view(
                batch_size, -1, self.num_heads, self.head_dim
            ).transpose(1, 2)

            # Optional normalization for conditioning (SD3 style)
            if self.norm_added_q is not None:
                encoder_hidden_states_query_proj = self.norm_added_q(encoder_hidden_states_query_proj)
            if self.norm_added_k is not None:
                encoder_hidden_states_key_proj = self.norm_added_k(encoder_hidden_states_key_proj)

            # Concatenate self-attention and cross-attention (SD3 style)
            query = torch.cat([query, encoder_hidden_states_query_proj], dim=2)
            key = torch.cat([key, encoder_hidden_states_key_proj], dim=2)
            value = torch.cat([value, encoder_hidden_states_value_proj], dim=2)

        # Scaled dot-product attention
        hidden_states = F.scaled_dot_product_attention(
            query, key, value, dropout_p=0.0, is_causal=False
        )
        hidden_states = hidden_states.transpose(1, 2).reshape(
            batch_size, -1, self.num_heads * self.head_dim
        )
        hidden_states = hidden_states.to(query.dtype)

        # Split attention outputs if conditioning was provided (SD3 style)
        if encoder_hidden_states is not None and self.use_conditioning:
            # Split the attention outputs
            hidden_states, encoder_hidden_states_output = (
                hidden_states[:, :residual.shape[1]],
                hidden_states[:, residual.shape[1]:],
            )
            if not self.context_pre_only:
                encoder_hidden_states_output = self.to_add_out(encoder_hidden_states_output)

        # Output projection
        hidden_states = self.to_out(hidden_states)

        if encoder_hidden_states is not None:
            return hidden_states, encoder_hidden_states_output
        else:
            return hidden_states


class FP32SiLU(nn.Module):
    """SiLU activation function with FP32 precision for numerical stability."""
    def forward(self, x):
        return nn.functional.silu(x.float()).to(dtype=x.dtype)


class PixArtAlphaTextProjection(nn.Module):
    """
    Projects caption embeddings. Also handles dropout for classifier-free guidance.

    Adapted from https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
    """

    def __init__(self, in_features, hidden_size, out_features=None, act_fn="gelu_tanh"):
        super().__init__()
        if out_features is None:
            out_features = hidden_size
        self.linear_1 = nn.Linear(in_features=in_features, out_features=hidden_size, bias=True)
        if act_fn == "gelu_tanh":
            self.act_1 = nn.GELU(approximate="tanh")
        elif act_fn == "silu":
            self.act_1 = nn.SiLU()
        elif act_fn == "silu_fp32":
            self.act_1 = FP32SiLU()
        else:
            raise ValueError(f"Unknown activation function: {act_fn}")
        self.linear_2 = nn.Linear(in_features=hidden_size, out_features=out_features, bias=True)

    def forward(self, caption):
        hidden_states = self.linear_1(caption)
        hidden_states = self.act_1(hidden_states)
        hidden_states = self.linear_2(hidden_states)
        return hidden_states


#################################################################################
#               Embedding Layers for Timesteps and Class Labels                 #
#################################################################################

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels)
        return embeddings

class TextEmbedder(nn.Module):
    """
    Embeds text into vector representations using CLIP ViT-L/14 and OpenCLIP ViT-bigG/14 (like SD3).
    Compatible with LabelEmbedder interface for classifier-free guidance.
    """
    def __init__(self, hidden_size, dropout_prob,
                 sd3_model_path="stabilityai/stable-diffusion-3-medium-diffusers"): # SD3 model path (HF Hub id; auto-downloads on first use)
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout_prob = dropout_prob
        
        # Import CLIP components
        try:
            from transformers import CLIPTextModelWithProjection, CLIPTokenizer
        except ImportError:
            raise ImportError("transformers library is required for TextEmbedder")
        
        # Load text encoders from SD3 model subfolders (like the training script)
        try:
            self.clip_encoder_1 = CLIPTextModelWithProjection.from_pretrained(
                sd3_model_path, subfolder="text_encoder"
            )  # First CLIP encoder
        except Exception as e:
            raise RuntimeError(f"Failed to load first CLIP encoder from SD3 model '{sd3_model_path}/text_encoder': {str(e)}")
            
        try:
            self.clip_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
                sd3_model_path, subfolder="text_encoder_2"
            )  # Second CLIP encoder
        except Exception as e:
            raise RuntimeError(f"Failed to load second CLIP encoder from SD3 model '{sd3_model_path}/text_encoder_2': {str(e)}")
            
        # Load two separate tokenizers (like SD3)
        try:
            self.clip_tokenizer_1 = CLIPTokenizer.from_pretrained(
                sd3_model_path, subfolder="tokenizer"
            )  # First tokenizer
        except Exception as e:
            raise RuntimeError(f"Failed to load first CLIP tokenizer from SD3 model '{sd3_model_path}/tokenizer': {str(e)}")
            
        try:
            self.clip_tokenizer_2 = CLIPTokenizer.from_pretrained(
                sd3_model_path, subfolder="tokenizer_2"
            )  # Second tokenizer
        except Exception as e:
            raise RuntimeError(f"Failed to load second CLIP tokenizer from SD3 model '{sd3_model_path}/tokenizer_2': {str(e)}")
        
        # Freeze the CLIP encoders
        for param in self.clip_encoder_1.parameters():
            param.requires_grad = False
        for param in self.clip_encoder_2.parameters():
            param.requires_grad = False
        
        # Get CLIP embedding dimensions
        clip_hidden_size_1 = self.clip_encoder_1.config.hidden_size
        clip_hidden_size_2 = self.clip_encoder_2.config.hidden_size
        
        # Concatenate pooled embeddings first, then use single projection (like SD3)
        combined_input_size = clip_hidden_size_1 + clip_hidden_size_2
        self.projection = PixArtAlphaTextProjection(
            in_features=combined_input_size,  # pooled_projection_dim
            hidden_size=hidden_size,          # embedding_dim
            out_features=hidden_size,         # embedding_dim
            act_fn="silu"
        )
        
        # For classifier-free guidance: empty text embedding
        self.empty_embedding = nn.Parameter(torch.randn(1, hidden_size) * 0.02)
        
    def tokenize_text(self, text):
        """
        Tokenize text using both CLIP tokenizers (like SD3).
        """
        if isinstance(text, str):
            text = [text]
        
        # Tokenize with first tokenizer
        text_inputs_1 = self.clip_tokenizer_1(
            text,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        
        # Tokenize with second tokenizer
        text_inputs_2 = self.clip_tokenizer_2(
            text,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_tensors="pt",
        )
        
        return text_inputs_1.input_ids, text_inputs_2.input_ids
    
    def encode_text(self, text_input_ids_1, text_input_ids_2):
        """
        Encode tokenized text using two CLIP encoders and get pooled embeddings.
        """
        with torch.no_grad():
            # Move to same device as encoders
            device = next(self.clip_encoder_1.parameters()).device
            text_input_ids_1 = text_input_ids_1.to(device)
            text_input_ids_2 = text_input_ids_2.to(device)
            
            # Encode with first CLIP encoder using first tokenizer
            outputs_1 = self.clip_encoder_1(text_input_ids_1, output_hidden_states=True)
            pooled_embeds_1 = outputs_1[0]  # First element of the tuple
            
            # Encode with second CLIP encoder using second tokenizer
            outputs_2 = self.clip_encoder_2(text_input_ids_2, output_hidden_states=True)
            pooled_embeds_2 = outputs_2[0]  # First element of the tuple
            
            # Concatenate pooled embeddings along the last dimension
            combined_pooled_embeds = torch.cat([pooled_embeds_1, pooled_embeds_2], dim=-1)
            
            # Project using single PixArtAlphaTextProjection (like SD3)
            pooled_text_embeddings = self.projection(combined_pooled_embeds.float())

            text_embeddings = self.projection(torch.cat([outputs_1[1], outputs_2[1]], dim = 2).float())
            # print(text_embeddings.shape, combined_pooled_embeds.shape)
            return text_embeddings
    
    def token_drop(self, text_input_ids_1, text_input_ids_2, force_drop_ids=None):
        """
        Drops text to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(text_input_ids_1.shape[0], device=text_input_ids_1.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        
        return drop_ids
    
    def forward(self, text, train, force_drop_ids=None):
        """
        Forward pass of TextEmbedder.
        Args:
            text: string or list of strings
            train: whether in training mode
            force_drop_ids: optional tensor to force certain items to be dropped
        Returns:
            embeddings: (batch_size, hidden_size)
        """
        use_dropout = self.dropout_prob > 0
        
        # Tokenize text with both tokenizers
        text_input_ids_1, text_input_ids_2 = self.tokenize_text(text)
        
        # Apply dropout for classifier-free guidance
        if (train and use_dropout) or (force_drop_ids is not None):
            drop_ids = self.token_drop(text_input_ids_1, text_input_ids_2, force_drop_ids)
        else:
            drop_ids = torch.zeros(text_input_ids_1.shape[0], dtype=torch.bool, device=text_input_ids_1.device)
        # Encode text
        text_embeddings = self.encode_text(text_input_ids_1, text_input_ids_2)  # [B, 77, 384]
        
        drop_ids = drop_ids.to(text_embeddings.device)  # [B]
        if drop_ids.any():
            B, T, D = text_embeddings.shape  # T=77, D=384
            device = text_embeddings.device
            # 如果你已有 self.empty_embedding: [D]，就复用；否则用零向量
            empty_token = getattr(self, "empty_embedding", torch.zeros(D, device=device, dtype=text_embeddings.dtype))
            # 扩成整条空序列 [B, T, D]
            empty_seq = empty_token.view(1, 1, D).expand(B, T, D)
            # 把 [B] 的 mask 扩成 [B, T, D]
            drop_mask = drop_ids.view(B, 1, 1).expand(B, T, D)
        
            text_embeddings = torch.where(drop_mask, empty_seq, text_embeddings)
        # # Encode text
        # text_embeddings = self.encode_text(text_input_ids_1, text_input_ids_2) # [bz, 77, 384]
        
        # drop_ids = drop_ids.to(text_embeddings.device)
        # # Replace dropped embeddings with empty embedding
        # if drop_ids.any():
        #     device = text_embeddings.device
        #     empty_emb = self.empty_embedding.to(device)
        #     text_embeddings = torch.where(
        #         drop_ids.expand_as(text_embeddings),
        #         empty_emb.expand(text_embeddings.shape[0], -1),
        #         text_embeddings
        #     )
        # print(f"shape of text_embeddings is {text_embeddings.shape}")
        return text_embeddings


class time_series_embedder(nn.Module):
    #to be completed in the future
    pass


#################################################################################
#                                 Core Model                                #
#################################################################################

class TransformerBlock(nn.Module):
    """
    A Model block with adaptive layer norm zero (adaLN-Zero) conditioning.
    Now with SD3-style separate adaLN modulation for conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, use_conditioning=False, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = SD3Attention(hidden_size, num_heads=num_heads, qkv_bias=True, use_conditioning=use_conditioning, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        
        # adaLN modulation for main input (c1: timestep + label)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )
        
        self.use_conditioning = use_conditioning
        
        if use_conditioning:
            # Separate adaLN modulation for conditioning (c1: timestep + label)
            self.adaLN_modulation_cond = nn.Sequential(
                nn.SiLU(),
                nn.Linear(hidden_size, 6 * hidden_size, bias=True)
            )
            
            # LayerNorm for conditioning input (c2)
            self.norm_cond = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            
            # Separate MLP for conditioning tokens
            self.mlp_cond = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
            
            # LayerNorm for conditioning after attention
            self.norm_cond_attn = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

    def forward(self, x, c1, c2):
        # adaLN modulation for main input
        # print(f"c1.shape is:{c1.shape}")
        # print(f"c2.shape is:{c2.shape}")
        # print(f"x.shape is:{x.shape}")
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c1).chunk(6, dim=1)
        
        # adaLN modulation for conditioning (only compute when use_conditioning is True and c2 is provided)
        if c2 is not None and self.use_conditioning:
            shift_msa_cond, scale_msa_cond, gate_msa_cond, shift_mlp_cond, scale_mlp_cond, gate_mlp_cond = self.adaLN_modulation_cond(c1).chunk(6, dim=1)
            
            # Apply LayerNorm to conditioning input first
            c2_norm = self.norm_cond(c2)  # (B, L, D)
            
            # Apply adaLN modulation to normalized conditioning input
            # c2_norm is (B, L, D), apply modulation along the last dimension
            c2_modulated = modulate(c2_norm, shift_msa_cond, scale_msa_cond)
            # print("c2_modulated", c2_modulated.shape)
            # Use SD3-style attention with modulated conditioning
            # c2_modulated is already (B, L, D) - pass directly to attention
            attn_output, cond_output = self.attn(
                modulate(self.norm1(x), shift_msa, scale_msa), 
                encoder_hidden_states=c2_modulated  # (B, L, D) for cross-attention
            )
            
            # Apply gate to conditioning output and process with MLP
            if cond_output is not None:
                # cond_output is (B, L, D), apply gate
                cond_output = gate_msa_cond.unsqueeze(1) * cond_output
                
                # Apply LayerNorm and MLP to conditioning (similar to main tokens)
                cond_output = cond_output + gate_mlp_cond.unsqueeze(1) * self.mlp_cond(
                    modulate(self.norm_cond_attn(cond_output), shift_mlp_cond, scale_mlp_cond)
                )
        else:
            # Fallback to self-attention only
            attn_output = self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
            cond_output = None
        
        # Main output processing
        x = x + gate_msa.unsqueeze(1) * attn_output
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        # print(f"x.shape after block is:{x.shape}")
        # print(f"cond_output.shape is:{cond_output.shape}")
        # Return both main output and conditioning output
        return x, cond_output


class FinalLayer(nn.Module):
    """
    The final layer of Model.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

    
# class PatchEmbed1D(nn.Module):
#     """Turn (B, C, T) into (B, N, D)"""
#     def __init__(self, input_len, patch_size, in_channels, embed_dim, bias=True):
#         super().__init__()
#         stride = patch_size // 2
#         self.proj = nn.Conv1d(in_channels,
#                               embed_dim,
#                               kernel_size=patch_size,
#                               stride=stride,
#                               bias=bias)
#         print("input length:", input_len, "patch_size:", patch_size)
#         self.patch_size = (patch_size,)          # int
#         self.num_patches = (input_len - patch_size) // stride + 1

#     def forward(self, x):                     # x: (B, C, T)
#         x = self.proj(x)                      # (B, D, N)
#         x = x.transpose(1, 2)                 # (B, N, D)
#         return x    


#################################################################################
#                   ResNet1D Encoder for Physiological Signals                 #
#################################################################################

class ResBlock1D(nn.Module):
    """
    Standard ResNet block for 1D time series data.
    Supports stride for downsampling.
    """
    def __init__(self, in_channels, out_channels, kernel_size=7, stride=1, dropout=0.0):
        super().__init__()
        padding = kernel_size // 2
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size,
                               stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()
    
    def forward(self, x):
        residual = self.shortcut(x)
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        out += residual
        out = self.relu(out)
        
        return out


class ResNet1DEncoder(nn.Module):
    """
    1D ResNet encoder for physiological signals.
    
    Converts (B, C_in, T_in) -> (B, N_patches, D_embed)
    
    Architecture:
        Stem -> Layer1 -> Layer2 -> Layer3 -> Layer4 -> AdaptivePool -> Projection
        
    Args:
        in_channels: Number of input channels
        input_len: Length of input sequence
        embed_dim: Output embedding dimension
        num_patches: Target number of output patches
        stem_channels: Number of channels after stem (default: 64)
        layer_channels: List of channels for each layer (default: [128, 256, 512, 512])
        blocks_per_layer: List of blocks per layer (default: [2, 2, 2, 2])
        kernel_size: Kernel size for conv layers (default: 7)
        dropout: Dropout rate (default: 0.0)
    
    Example:
        # For forecasting: encode medication signals (21 channels, 3000 timesteps)
        encoder = ResNet1DEncoder(in_channels=21, input_len=3000, embed_dim=1152, num_patches=150)
        x = torch.randn(4, 21, 3000)
        out = encoder(x)  # (4, 150, 1152)
        
        # For cross-channel: encode observed channels (7 channels, 3000 timesteps)
        encoder = ResNet1DEncoder(in_channels=7, input_len=3000, embed_dim=1152, num_patches=150)
        
        # For temporal boundary: encode boundary segments (28 channels, 2000 timesteps)
        encoder = ResNet1DEncoder(in_channels=28, input_len=2000, embed_dim=1152, num_patches=100)
    """
    def __init__(
        self,
        in_channels,
        input_len,
        embed_dim,
        num_patches,
        stem_channels=64,
        layer_channels=None,
        blocks_per_layer=None,
        kernel_size=7,
        dropout=0.0
    ):
        super().__init__()
        self.in_channels = in_channels
        self.input_len = input_len
        self.embed_dim = embed_dim
        self.num_patches = num_patches
        
        if layer_channels is None:
            layer_channels = [128, 256, 512, 512]
        if blocks_per_layer is None:
            blocks_per_layer = [2, 2, 2, 2]
        
        # Stem: Initial convolution with stride for downsampling
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, stem_channels, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(stem_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        )
        
        # Calculate stem output length
        stem_out_len = input_len // 2  # stride=2 in conv
        stem_out_len = (stem_out_len - 1) // 2 + 1  # maxpool with stride=2
        
        # ResNet layers
        self.layers = nn.ModuleList()
        in_ch = stem_channels
        
        for i, (out_ch, n_blocks) in enumerate(zip(layer_channels, blocks_per_layer)):
            # First block in each layer has stride=2 for downsampling (except first layer)
            stride = 2 if i > 0 else 1
            layer = self._make_layer(in_ch, out_ch, n_blocks, kernel_size, stride, dropout)
            self.layers.append(layer)
            in_ch = out_ch
        
        # Adaptive pooling to get exactly num_patches
        self.adaptive_pool = nn.AdaptiveAvgPool1d(num_patches)
        
        # Final projection to embed_dim
        self.proj = nn.Linear(layer_channels[-1], embed_dim)
        
        # For compatibility with PatchEmbed1D interface
        self.patch_size = (input_len // num_patches,)
        
        print(f"[ResNet1DEncoder] in_channels={in_channels}, input_len={input_len}, "
              f"embed_dim={embed_dim}, num_patches={num_patches}")
        print(f"[ResNet1DEncoder] Architecture: Stem({in_channels}->{stem_channels}) -> "
              f"Layers{layer_channels} -> AdaptivePool({num_patches}) -> Proj({embed_dim})")
    
    def _make_layer(self, in_channels, out_channels, n_blocks, kernel_size, stride, dropout):
        """Create a layer with n_blocks ResBlocks"""
        layers = []
        # First block with stride
        layers.append(ResBlock1D(in_channels, out_channels, kernel_size, stride, dropout))
        # Remaining blocks
        for _ in range(1, n_blocks):
            layers.append(ResBlock1D(out_channels, out_channels, kernel_size, 1, dropout))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Args:
            x: (B, C, T) input tensor
        Returns:
            out: (B, N, D) output tensor where N=num_patches, D=embed_dim
        """
        # Stem
        x = self.stem(x)  # (B, stem_channels, T')
        
        # ResNet layers
        for layer in self.layers:
            x = layer(x)  # (B, channels, T'')
        
        # Adaptive pooling to target num_patches
        x = self.adaptive_pool(x)  # (B, channels, num_patches)
        
        # Transpose and project
        x = x.transpose(1, 2)  # (B, num_patches, channels)
        x = self.proj(x)  # (B, num_patches, embed_dim)
        
        return x


class PatchEmbed1D(nn.Module):
    """Turn (B, C, T) into (B, N, D)"""
    def __init__(self, input_len, patch_size, in_channels, embed_dim, bias=True):
        super().__init__()
        self.proj = nn.Conv1d(in_channels,
                              embed_dim,
                              kernel_size=patch_size,
                              stride=patch_size,
                              bias=bias)
        print("input length:", input_len, "patch_size:", patch_size)
        self.patch_size = (patch_size,)          # int
        self.num_patches = input_len // patch_size

    def forward(self, x):                     # x: (B, C, T)
        x = self.proj(x)                      # (B, D, N)
        x = x.transpose(1, 2)                 # (B, N, D)
        return x

class Model(nn.Module):
    """
    Diffusion model with a Transformer backbone.
    
    Args:
        use_resnet_y_embedder: If True, use ResNet1DEncoder for y_embedder (default: False)
        use_temporal_y_embedder: If True, use SimpleTemporalPooler for y_embedder (default: False)
        use_temporal_c2_embedder: If True, use TemporalFeatureExtractor for c2_embedder (default: False)
        y_encoder_config: Dict with config for y_embedder:
            - 'input_len': input sequence length
            - 'in_channels': number of input channels
            - 'num_patches': target number of patches (default: computed from input_len/patch_size)
            - 'pool_type': for SimpleTemporalPooler, one of ['mean', 'max', 'attention']
        c2_encoder_config: Dict with config for c2_embedder (similar to y_encoder_config)
    """
    def __init__(
        self,
        input_size=32,
        patch_size=2,
        in_channels=4,
        input_channels=None,
        hidden_size=1152,
        depth=28,
        num_heads=16,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=1000,
        learn_sigma=True,
        use_conditioning=False,  # Control whether to use c2 conditioning
        use_resnet_y_embedder=False,  # NEW: Use ResNet for y_embedder
        use_temporal_y_embedder=False,  # NEW: Use SimpleTemporalPooler for y_embedder
        use_temporal_c2_embedder=False,  # NEW: Use TemporalFeatureExtractor for c2_embedder
        use_label_y_embedder=False,  # NEW: Use LabelEmbedder for class-conditional generation
        y_encoder_config=None,  # NEW: Config for y_embedder
        c2_encoder_config=None,  # NEW: Config for c2_embedder
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.input_channels = input_channels or in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.use_conditioning = use_conditioning
        self.use_resnet_y_embedder = use_resnet_y_embedder
        self.use_temporal_y_embedder = use_temporal_y_embedder
        self.use_temporal_c2_embedder = use_temporal_c2_embedder
        self.use_label_y_embedder = use_label_y_embedder
        
        # x_embedder: always use PatchEmbed1D for the main input (what we're generating)
        # Use input_size parameter instead of hardcoded 1000 to support different tasks
        self.x_embedder = PatchEmbed1D(input_len=input_size,
                               patch_size=patch_size,
                               in_channels=self.input_channels,
                               embed_dim=hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        
        # y_embedder: choose between TextEmbedder, SimpleTemporalPooler, ResNet, PatchEmbed, or LabelEmbedder
        self.use_text_y_embedder = (not use_temporal_y_embedder
                                     and not use_resnet_y_embedder
                                     and not use_label_y_embedder
                                     and y_encoder_config is None)
        if self.use_text_y_embedder:
            print(f"[Model] Using TextEmbedder for y_embedder (text conditioning)")
            self.y_embedder = TextEmbedder(hidden_size, class_dropout_prob)
            self.y_is_pooled = False

        if y_encoder_config is None:
            y_encoder_config = {'input_len': 3000, 'in_channels': 21}

        y_input_len = y_encoder_config.get('input_len', 3000)
        y_in_channels = y_encoder_config.get('in_channels', 21)
        y_num_patches = y_encoder_config.get('num_patches', y_input_len // patch_size)
        y_pool_type = y_encoder_config.get('pool_type', 'mean')

        if self.use_text_y_embedder:
            pass  # already created above
        elif use_label_y_embedder:
            print(f"[Model] Using LabelEmbedder for y_embedder (class-conditional, num_classes={num_classes})")
            self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
            self.y_is_pooled = True  # Outputs (B, D)
        elif use_temporal_y_embedder:
            print(f"[Model] Using SimpleTemporalPooler for y_embedder (pool={y_pool_type})")
            self.y_embedder = SimpleTemporalPooler(
                in_channels=y_in_channels,
                hidden_size=hidden_size,
                series_length=y_input_len,
                pool_type=y_pool_type,
                num_conv_layers=3
            )
            self.y_is_pooled = True  # Flag: y_embedder outputs (B, D) instead of (B, N, D)
        elif use_resnet_y_embedder:
            print(f"[Model] Using ResNet1DEncoder for y_embedder")
            self.y_embedder = ResNet1DEncoder(
                in_channels=y_in_channels,
                input_len=y_input_len,
                embed_dim=hidden_size,
                num_patches=y_num_patches,
                stem_channels=64,
                layer_channels=[128, 256, 512, 512],
                blocks_per_layer=[2, 2, 2, 2],
                kernel_size=7,
                dropout=0.0
            )
            self.y_is_pooled = False  # Outputs (B, N, D)
        else:
            print(f"[Model] Using PatchEmbed1D for y_embedder")
            self.y_embedder = PatchEmbed1D(
                input_len=y_input_len,
                patch_size=patch_size,
                in_channels=y_in_channels,
                embed_dim=hidden_size
            )
            self.y_is_pooled = False  # Outputs (B, N, D)
        
        # c2_embedder: choose between TemporalFeatureExtractor or PatchEmbed1D
        if c2_encoder_config is None:
            c2_encoder_config = {'input_len': 2000, 'in_channels': 7}
        
        c2_input_len = c2_encoder_config.get('input_len', 2000)
        c2_in_channels = c2_encoder_config.get('in_channels', 7)
        c2_patch_size = c2_encoder_config.get('patch_size', patch_size)
        c2_depth = c2_encoder_config.get('depth', 2)
        
        if use_temporal_c2_embedder:
            print(f"[Model] Using TemporalFeatureExtractor for c2_embedder (depth={c2_depth})")
            self.c2_embedder = TemporalFeatureExtractor(
                in_channels=c2_in_channels,
                hidden_size=hidden_size,
                context_length=c2_input_len,
                patch_size=c2_patch_size,
                depth=c2_depth
            )
        else:
            print(f"[Model] Using PatchEmbed1D for c2_embedder")
            self.c2_embedder = PatchEmbed1D(
                input_len=c2_input_len,
                patch_size=patch_size,
                in_channels=c2_in_channels,
                embed_dim=hidden_size
            )
        
        num_patches = self.x_embedder.num_patches
        # Will use fixed sin-cos embedding:
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio, use_conditioning=use_conditioning) for _ in range(depth)
        ])
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize transformer layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        # pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        pos_embed = get_1d_sincos_pos_embed_from_grid(self.pos_embed.shape[-1], np.arange(self.x_embedder.num_patches))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        # Initialize y_embedder based on its type
        if isinstance(self.y_embedder, ResNet1DEncoder):
            # ResNet is already initialized via _basic_init above
            # Optionally, we could add custom initialization here
            pass
        elif hasattr(self.y_embedder, 'embedding_table'):
            # LabelEmbedder
            nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        elif hasattr(self.y_embedder, 'projection'):
            # TextEmbedder with PixArtAlphaTextProjection
            nn.init.normal_(self.y_embedder.projection.linear_1.weight, std=0.02)
            nn.init.normal_(self.y_embedder.projection.linear_2.weight, std=0.02)
            nn.init.normal_(self.y_embedder.empty_embedding, std=0.02)
        elif hasattr(self.y_embedder, 'proj'):
            # PatchEmbed1D
            w = self.y_embedder.proj.weight.data
            nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            if self.y_embedder.proj.bias is not None:
                nn.init.constant_(self.y_embedder.proj.bias, 0)

        # Initialize timestep embedding MLP:
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in Model blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
            # Also zero-out the conditioning adaLN modulation if it exists
            if block.use_conditioning:
                nn.init.constant_(block.adaLN_modulation_cond[-1].weight, 0)
                nn.init.constant_(block.adaLN_modulation_cond[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

#     def unpatchify(self, x):
#         """
#         x: (N, T, patch_size**2 * C)
#         imgs: (N, H, W, C)
#         """
#         c = self.out_channels
#         p = self.x_embedder.patch_size[0]
#         h = w = int(x.shape[1] ** 0.5)
#         assert h * w == x.shape[1]

#         x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
#         x = torch.einsum('nhwpqc->nchpwq', x)
#         imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
#         return imgs
    def unpatchify(self, x):
        """
        1-D 逆 patchify
        Args:
            x: (B, N, p * C_out)  —— Transformer token 输出
        Returns:
            sig: (B, C_out, T)    —— 复原到时间序列
        """
        B, N, _ = x.shape
        p = self.patch_size            # 单 patch 长度（int）
        C = self.out_channels
        assert _ == p * C, "token dim 不符: 应为 p*C_out"

        # Step 1: (B, N, p, C)
        x = x.view(B, N, p, C)

        # Step 2: (B, C, N, p)
        x = x.permute(0, 3, 1, 2).contiguous()

        # Step 3: (B, C, N*p) = (B, C, T)
        sig = x.view(B, C, N * p)
        return sig

    def forward(self, x, t, y, c2=None, lowres_cond_img=None, **kwargs):
        """
        Forward pass of Model.
        x: (N, C, T) tensor of time series inputs
        t: (N,) tensor of diffusion timesteps
        y: conditioning signal (varies by embedder type)
        c2: context signal (optional, for cross-attention)
        """
        if lowres_cond_img is not None:
            x = torch.cat([x, lowres_cond_img], dim=1)
        if x.shape[1] != self.input_channels:
            raise ValueError(
                f"Model forward expected {self.input_channels} input channels after conditioning, "
                f"got {x.shape[1]}."
            )
        x = self.x_embedder(x) + self.pos_embed  # (N, T, D), where T = input_len / patch_size
        
        t = self.t_embedder(t)                   # (N, D)
        if y is None:
            y_cond = torch.zeros_like(t)
        elif self.use_text_y_embedder:
            y = self.y_embedder(y, self.training) # (N, 77, D) from TextEmbedder
            y_cond = y.mean(dim=1)                # (N, D) pooled for adaLN
            if c2 is None:
                c2 = y                            # use text sequence as cross-attention
        elif self.use_label_y_embedder:
            y_cond = self.y_embedder(y, self.training)  # (N, D) from LabelEmbedder
        else:
            y = self.y_embedder(y)               # (N, D) if pooled, else (N, L, D)
            if self.y_is_pooled:
                y_cond = y  # (N, D)
            else:
                y_cond = y.mean(dim=1)  # (N, D)
        
        c1 = t + y_cond  # (N, D) - timestep + conditioning for adaLN
        
        # Handle c2 conditioning (skip if already set from text embedding)
        if c2 is not None and not self.use_text_y_embedder:
            c2 = self.c2_embedder(c2)  # (N, L, D) - sequence for cross-attention
        elif c2 is not None:
            pass  # c2 already set from text embedding, already (N, 77, D)
        # print(c2.shape)
        # print(x.shape, c1.shape, c2.shape)
        # Process through blocks with conditioning flow
        for i, block in enumerate(self.blocks):
            x, cond_output = block(x, c1, c2)    # (N, T, D) - pass both conditionings
            
            # Update c2 for next block using the conditioning output from current block
            c2 = cond_output

        x = self.final_layer(x, c1)                # (N, T, patch_size ** 2 * out_channels)
        x = self.unpatchify(x)                   # (N, out_channels, H, W)
        
        if self.learn_sigma:
            x, _ = x.chunk(2, dim=1)
        return x

    def forward_with_cfg(self, x, t, y, cfg_scale, c2=None, lowres_cond_img=None):
        """
        Forward pass of Model, but also batches the unconditional forward pass for classifier-free guidance.
        """
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        lowres_combined = None
        if lowres_cond_img is not None:
            lowres_half = lowres_cond_img[: len(lowres_cond_img) // 2]
            lowres_combined = torch.cat([lowres_half, lowres_half], dim=0)
        model_out = self.forward(combined, t, y, c2=c2, lowres_cond_img=lowres_combined)
        eps, rest = model_out[:, :self.in_channels], model_out[:, self.in_channels:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)


#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   Model Configs                                  #
#################################################################################

def Model_XL_2(**kwargs):
    return Model(depth=28, hidden_size=1152, patch_size=2, num_heads=16, **kwargs)

def Model_XL_4(**kwargs):
    return Model(depth=28, hidden_size=1152, patch_size=4, num_heads=16, **kwargs)

def Model_XL_8(**kwargs):
    return Model(depth=28, hidden_size=1152, patch_size=8, num_heads=16, **kwargs)

def Model_L_2(**kwargs):
    return Model(depth=24, hidden_size=1024, patch_size=2, num_heads=16, **kwargs)

def Model_L_4(**kwargs):
    return Model(depth=24, hidden_size=1024, patch_size=4, num_heads=16, **kwargs)

def Model_L_8(**kwargs):
    return Model(depth=24, hidden_size=1024, patch_size=8, num_heads=16, **kwargs)

def Model_B_2(**kwargs):
    return Model(depth=12, hidden_size=768, patch_size=2, num_heads=12, **kwargs)

def Model_B_4(**kwargs):
    return Model(depth=12, hidden_size=768, patch_size=4, num_heads=12, **kwargs)

def Model_B_8(**kwargs):
    return Model(depth=12, hidden_size=768, patch_size=8, num_heads=12, **kwargs)

def Model_S_2(**kwargs):
    return Model(depth=12, hidden_size=384, patch_size=2, num_heads=6, **kwargs)

def Model_S_4(**kwargs):
    return Model(depth=12, hidden_size=384, patch_size=4, num_heads=6, **kwargs)

def Model_S_8(**kwargs):
    return Model(depth=12, hidden_size=384, patch_size=8, num_heads=6, **kwargs)

# def Model_S_1d(**kwargs):
#     return Model(depth=12, hidden_size=384, patch_size=40, num_heads=6, use_conditioning=False, **kwargs)
def Model_B_1d(**kwargs):
    return Model(depth=12, hidden_size=768, num_heads=12, use_conditioning=True, **kwargs)

def Model_S_1d(**kwargs):
    return Model(depth=6, hidden_size=384, num_heads=6, use_conditioning=True, **kwargs)

def Model_SS_1d(**kwargs):
    return Model(depth=2, hidden_size=128, patch_size=5, num_heads=4, use_conditioning=True, **kwargs)

MODELS = {
    'XL/2': Model_XL_2,  'XL/4': Model_XL_4,  'XL/8': Model_XL_8,
    'L/2':  Model_L_2,   'L/4':  Model_L_4,   'L/8':  Model_L_8,
    'B/2':  Model_B_2,   'B/4':  Model_B_4,   'B/8':  Model_B_8,
    'S/2':  Model_S_2,   'S/4':  Model_S_4,   'S/8':  Model_S_8,
    'S/1d': Model_S_1d,  'SS/1d': Model_SS_1d, 'B/1d': Model_B_1d,
}
