import torch
from typing import Optional

class RoPE_vllm(torch.nn.Module):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        is_neox_style: bool,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype
        # TODO(mgoin): disabled for now due to failures
        # Flashinfer only supports head_size=64, 128, 256, 512.
        # https://github.com/flashinfer-ai/flashinfer/blob/ebfd655efe830048dba5d582aaa61d61d1cf9a87/include/flashinfer/utils.cuh#L174-L202
        # self.use_flashinfer = (self.enabled()
        #                        and dtype in (torch.float16, torch.bfloat16)
        #                        and current_platform.is_cuda()
        #                        and has_flashinfer()
        #                        and self.head_size in [64, 128, 256, 512])
        self.use_flashinfer = False

        cache = self._compute_cos_sin_cache()
        if not self.use_flashinfer:
            cache = cache.to(dtype)
        self.cos_sin_cache: torch.Tensor
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def _compute_inv_freq(self, base: float) -> torch.Tensor:
        """Compute the inverse frequency."""
        # NOTE(woosuk): To exactly match the HF implementation, we need to
        # use CPU to compute the cache and then move it to GPU. However, we
        # create the cache on GPU for faster initialization. This may cause
        # a slight numerical difference between the HF implementation and ours.
        inv_freq = 1.0 / (base**(torch.arange(
            0, self.rotary_dim, 2, dtype=torch.float,device='cuda:0') / self.rotary_dim))
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        """Compute the cos and sin cache."""
        inv_freq = self._compute_inv_freq(self.base)
        t = torch.arange(self.max_position_embeddings, dtype=torch.float, device='cuda:0')

        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    def _match_cos_sin_cache_dtype(self, query: torch.Tensor) -> None:
        # __setattr__ in nn.Module (called by `self.cos_sin_cache = ...`)
        # is expensive, so avoid calling it if possible
        if self.cos_sin_cache.device != query.device or \
            self.cos_sin_cache.dtype != query.dtype:
            self.cos_sin_cache = self.cos_sin_cache.to(query.device,
                                                       dtype=query.dtype)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """A PyTorch-native implementation of forward()."""
        positions = positions.flatten()
        num_tokens = positions.shape[0]
        cos_sin = self.cos_sin_cache.index_select(0, positions)
        cos, sin = cos_sin.chunk(2, dim=-1)

        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)
        query_rot = query[..., :self.rotary_dim]
        query_pass = query[..., self.rotary_dim:]
        query_rot = apply_rotary_emb_torch(query_rot, cos, sin,
                                           self.is_neox_style)
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        # key may be None in some cases, e.g. cross-layer KV sharing
        if key is not None:
            key_shape = key.shape
            key = key.view(num_tokens, -1, self.head_size)
            key_rot = key[..., :self.rotary_dim]
            key_pass = key[..., self.rotary_dim:]
            key_rot = apply_rotary_emb_torch(key_rot, cos, sin,
                                             self.is_neox_style)
            key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        return query, key

def apply_rotary_emb_torch(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
        ) -> torch.Tensor:
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    if is_neox_style:
        return torch.cat((o1, o2), dim=-1)
    else:
        return torch.stack((o1, o2), dim=-1).flatten(-2)

class RoPE_opt(torch.nn.Module):
    def __init__(self, dim: int, max_seq_len: int, base: float = 10000.0, device: str = 'cuda:0', dtype: torch.dtype = torch.float):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base

        inv_freqs = 1.0 / (base ** (torch.arange(0, self.dim, 2, dtype=dtype, device=device) / self.dim))
        pos = torch.arange(0, self.max_seq_len, 1, dtype=torch.int, device=device)
        freqs = torch.outer(pos,inv_freqs)

        emb = torch.cat((freqs, freqs), dim=-1)

        self.register_buffer("cos_cached", emb.cos())
        self.register_buffer("sin_cached", emb.sin())

    def forward(self, x: torch.tensor, pos: torch.tensor):
        # x.shape [B, S, D]
        # pos.shape [B, S]
        
        # pos_f [B*S]
        pos_f = torch.flatten(pos,start_dim=0)
        # cos.shape, sin.shape [B, S, D//2]
        cos = self.cos_cached[pos_f, :self.dim//2].reshape(*pos.shape, self.dim//2)
        sin = self.sin_cached[pos_f, :self.dim//2].reshape(*pos.shape, self.dim//2)

        x_0 = x[..., :self.dim//2]
        x_1 = x[..., self.dim//2:]

        x_out = torch.cat(
            (x_0 * cos - x_1 * sin,
            x_0 * sin + x_1 * cos), dim=-1)
        
        return x_out

def main():
    max_seq_len = 8192 
    dim = 1024
    assert(dim % 2 == 0)

    dtype = torch.float
    device = 'cuda:0'

    torch.manual_seed(0)
    x = torch.rand((2,3,dim),dtype=dtype,device=device)
    pos = torch.tensor(
        [[0, 1, 2],
        [0, 1, 2]]
    ,dtype=torch.int, device=device)

    rope = RoPE_opt(dim,max_seq_len,device=device)
    x_rope = rope(x, pos)

    rope_vllm = RoPE_vllm(head_size=dim,
        rotary_dim=dim,
        max_position_embeddings=max_seq_len,
        base=10000.0,
        is_neox_style=True,
        dtype=torch.float,)
    x_rope_vllm = rope_vllm(pos,x)

    diff = torch.abs(x_rope - x_rope_vllm[0])
    if torch.any(diff>0.01):
        print(f"difference between rope and rope_vllm:{max(diff)}")
        exit(1)
    else:
        print("pass")

if __name__ == "__main__":
    main()