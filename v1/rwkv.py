import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass, field

@dataclass
class ModelConfig:
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 12
    block_size: int = 1024
    n_intermediate: int = 768 * 4
    layer_norm_epsilon: float = 1e-5
    
    # Make these optional to support both token and waveform models
    vocab_size: int | None = None # For token-based models
    input_dim: int | None = None  # For waveform/continuous input
    output_dim: int | None = None # For waveform/continuous output

    def __post_init__(self):
        if self.vocab_size is not None and (self.input_dim is not None or self.output_dim is not None):
            print("Warning: vocab_size is defined along with input_dim/output_dim. Ensure model handles this.")
        if self.vocab_size is None and (self.input_dim is None or self.output_dim is None):
            # This case is problematic if neither is set, but RWKVModel will likely fail later
            # For now, we allow it, but specific model use will dictate if it's an error
            print("Warning: Neither vocab_size nor input_dim/output_dim are set. Model might not initialize correctly.")

# Note: RWKVTimeMixing, RWKVChannelMixing, and RWKVBlock remain unchanged
# as their internal logic operates on n_embd dimensions and is independent
# of the model's specific input/output nature (tokens vs. continuous values)
# as long as the input to the first block and output from the last block
# are handled correctly by the main RWKVModel class.

class RWKVTimeMixing(nn.Module):
    def __init__(self, config: ModelConfig, layer_id: int):
        super().__init__()
        self.config = config
        self.layer_id = layer_id 
        self.n_embd = config.n_embd
        self.n_head = config.n_head
        self.head_size = self.n_embd // self.n_head
        
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"

        with torch.no_grad():
            ratio_0_to_1 = torch.arange(self.n_head).float() / (self.n_head - 1 if self.n_head > 1 else 1)
            tmp = torch.zeros(self.n_embd)
            for i in range(self.n_head):
                tmp[i * self.head_size:(i + 1) * self.head_size] = ratio_0_to_1[i]
            
            self.time_mix_k = nn.Parameter(tmp.clone())
            self.time_mix_v = nn.Parameter(tmp.clone())
            self.time_mix_r = nn.Parameter(tmp.clone())
            
            self.time_decay = nn.Parameter(-5 + 8 * (torch.arange(self.n_embd).float() / (self.n_embd - 1 if self.n_embd > 1 else 1.0)) ** 0.7)
            self.time_first = nn.Parameter(torch.ones(self.n_embd) * -3.0) 

        self.key = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.value = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.receptance = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.output = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, state):
        B, T, C = x.size()
        # Removed 'pass' placeholder from original, state handling is below
        k = self.key(x)    
        v = self.value(x)  
        r_in = self.receptance(x) 
        r = torch.sigmoid(r_in) 
        
        w = torch.exp(self.time_decay)
        u = self.time_first
        
        output_wkv = torch.zeros_like(k)
        
        current_aa, current_bb, current_pp = state if state is not None else (
            torch.zeros(B, C, device=x.device, dtype=x.dtype),
            torch.zeros(B, C, device=x.device, dtype=x.dtype),
            torch.full((B, C), -1e38, device=x.device, dtype=x.dtype) 
        )

        for t_step in range(T):
            kt = k[:, t_step, :] 
            vt = v[:, t_step, :] 
            ww = u + kt 
            p = torch.maximum(current_pp, ww) 
            e1 = torch.exp(current_pp - p) 
            e2 = torch.exp(ww - p) 
            wkv_t_step = (e1 * current_aa + e2 * vt) / (e1 * current_bb + e2) 
            output_wkv[:, t_step, :] = wkv_t_step
            ww = current_pp - w 
            p = torch.maximum(ww, kt) 
            e1 = torch.exp(ww - p) 
            e2 = torch.exp(kt - p) 
            current_aa = e1 * current_aa + e2 * vt 
            current_bb = e1 * current_bb + e2      
            current_pp = p                         
            
        rwkv_out = r * output_wkv
        new_wkv_state = (current_aa, current_bb, current_pp)
        return self.output(rwkv_out), new_wkv_state


class RWKVChannelMixing(nn.Module):
    def __init__(self, config: ModelConfig, layer_id: int): 
        super().__init__()
        self.config = config
        self.layer_id = layer_id 
        self.n_embd = config.n_embd
        
        with torch.no_grad():
            ratio_0_to_1 = torch.arange(self.n_embd).float() / (self.n_embd - 1 if self.n_embd > 1 else 1)
            self.time_mix_k = nn.Parameter(ratio_0_to_1.clone())
            self.time_mix_r = nn.Parameter(ratio_0_to_1.clone())

        self.key = nn.Linear(self.n_embd, self.config.n_intermediate, bias=False)
        self.receptance = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.value = nn.Linear(self.config.n_intermediate, self.n_embd, bias=False)

    def forward(self, x, prev_x_cm_state): 
        B, T, C = x.size()
        if prev_x_cm_state is None:
            prev_x_cm_state = torch.zeros(B, C, device=x.device, dtype=x.dtype)
        prev_x_cm_state_expanded = prev_x_cm_state.unsqueeze(1)
        xk_mixed = x * self.time_mix_k + prev_x_cm_state_expanded * (1 - self.time_mix_k)
        xr_mixed = x * self.time_mix_r + prev_x_cm_state_expanded * (1 - self.time_mix_r)
        k_val = self.key(xk_mixed) # Renamed k to k_val
        k_val = torch.square(torch.relu(k_val)) 
        r_val = self.receptance(xr_mixed) # Renamed r to r_val
        r_val = torch.sigmoid(r_val)
        vk = self.value(k_val)
        if T > 0:
            new_prev_x_cm_state = x[:, -1, :].clone()
        else: 
            new_prev_x_cm_state = prev_x_cm_state.clone()
        return r_val * vk, new_prev_x_cm_state


class RWKVBlock(nn.Module):
    def __init__(self, config: ModelConfig, layer_id: int):
        super().__init__()
        self.config = config
        self.layer_id = layer_id
        self.ln1 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.ln2 = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)
        self.att = RWKVTimeMixing(config, layer_id)
        self.ffn = RWKVChannelMixing(config, layer_id)

    def forward(self, x, time_mix_state, channel_mix_state):
        x_norm_att = self.ln1(x)
        att_output, new_time_mix_state = self.att(x_norm_att, time_mix_state)
        x = x + att_output
        x_norm_ffn = self.ln2(x)
        ffn_output, new_channel_mix_state = self.ffn(x_norm_ffn, channel_mix_state)
        x = x + ffn_output
        return x, new_time_mix_state, new_channel_mix_state


class RWKVModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        if config.vocab_size is not None:
            self.wte = nn.Embedding(config.vocab_size, config.n_embd)
            self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
            self.is_token_model = True
        elif config.input_dim is not None and config.output_dim is not None:
            self.input_proj = nn.Linear(config.input_dim, config.n_embd)
            self.lm_head = nn.Linear(config.n_embd, config.output_dim, bias=False)
            self.is_token_model = False
        else:
            raise ValueError("ModelConfig must define either vocab_size (for token model) or input_dim and output_dim (for waveform model)")

        self.blocks = nn.ModuleList([RWKVBlock(config, i) for i in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd, eps=config.layer_norm_epsilon)

    def forward(self, inputs, states=None): # Renamed idx/input_wave to generic 'inputs'
        if self.is_token_model:
            # inputs are token indices (B, T)
            B, T = inputs.size()
            x = self.wte(inputs) # (B, T, n_embd)
        else:
            # inputs are waveform values (B, T, input_dim) or (B,T) if input_dim=1
            if inputs.ndim == 2 and self.config.input_dim == 1:
                inputs = inputs.unsqueeze(-1)
            B, T, _ = inputs.size()
            x = self.input_proj(inputs.float()) # (B, T, n_embd)

        new_states_out = []
        param_dtype = next(self.parameters()).dtype

        if states is None:
            states = []
            for _ in range(self.config.n_layer):
                initial_wkv_aa = torch.zeros(B, self.config.n_embd, device=inputs.device, dtype=param_dtype)
                initial_wkv_bb = torch.zeros(B, self.config.n_embd, device=inputs.device, dtype=param_dtype)
                initial_wkv_pp = torch.full((B, self.config.n_embd), -1e38, device=inputs.device, dtype=param_dtype)
                wkv_state = (initial_wkv_aa, initial_wkv_bb, initial_wkv_pp)
                cm_state = torch.zeros(B, self.config.n_embd, device=inputs.device, dtype=param_dtype)
                states.append((wkv_state, cm_state))

        for i, block in enumerate(self.blocks):
            layer_wkv_state, layer_cm_state = states[i]
            x, next_layer_wkv_state, next_layer_cm_state = block(x, layer_wkv_state, layer_cm_state)
            new_states_out.append((next_layer_wkv_state, next_layer_cm_state))

        x = self.ln_f(x) 
        outputs = self.lm_head(x) # (B, T, vocab_size) or (B, T, output_dim)

        return outputs, new_states_out
