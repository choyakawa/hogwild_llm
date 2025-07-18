from typing import Tuple

import torch
import math
import numpy as np


torch.ops.load_library("libhogatt.so")


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim=2):
    """
    Applies rotary position embedding to the input tensor.
    Only the first `cos.shape[-1]` dimensions are rotated.
    """
    # unsqueeze_dim=2 to broadcast over Hq dimension in queries of shape (F, W, Hq, S, E)
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    rotary_dim = cos.shape[-1]
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
    x_rot_emb = (x_rot * cos) + (rotate_half(x_rot) * sin)
    return torch.cat((x_rot_emb, x_pass), dim=-1).to(x.dtype)


def hogwild_sdpa_ref(queries: torch.Tensor, locations: torch.Tensor, keys: list[torch.Tensor], values: list[torch.Tensor],
                     scale: float) -> torch.Tensor:
    qk = []
    vals = []
    for f in range(len(keys)):
        # GQA replication
        key = keys[f].repeat_interleave(queries.size(-3) // keys[f].size(-3), -3)
        val = values[f].repeat_interleave(queries.size(-3) // values[f].size(-3), -3)
        # key and query position -> mask
        kp = torch.arange(0, key.size(-2), dtype=torch.int32, device=key.device)
        qp = locations[f]
        kp = kp[None, None, None, :]
        qp = qp[:, None, :, None]
        mask = kp > qp
        att = queries[f].to(torch.float32) @ key.transpose(-1, -2).to(torch.float32)
        att.masked_fill_(mask, float("-inf"))
        qk.append(att)
        vals.append(val)

    qk = torch.concat(qk, dim=-1)
    vals = torch.concat(vals, dim=-2).to(torch.float32)
    att = torch.softmax(scale * qk, dim=-1)
    result = att @ vals
    return result


def hogwild_sdpa(queries: torch.Tensor, locations: torch.Tensor, keys: list[torch.Tensor], values: list[torch.Tensor],
                 scale: float, fragment_lengths=None) -> torch.Tensor:
    out = torch.empty((queries.size(1), queries.size(2), queries.size(3), values[0].size(3)), dtype=torch.float32, device=queries.device)
    if fragment_lengths is None:
        fragment_lengths = torch.tensor([k.size(2) for k in keys], dtype=torch.int32, device=queries.device)
    keys = [k.to(torch.float32).squeeze(0).contiguous() for k in keys]
    values = [v.to(torch.float32).squeeze(0).contiguous() for v in values]
    torch.ops.libhogatt.hogwild_sdpa(out, scale, locations.to(torch.int32), queries.to(torch.float32).contiguous(), fragment_lengths, keys, values)
    return out.to(queries.dtype)


@torch.no_grad()
def test_custom_kernel(F: int, W: int, Hq: int, Hkv: int, E: int, RotaryE: int, Ev: int, S: int, # noqa
                       frags: list[int], scale: float = None):
    # TODO make input distributions more interesting
    torch.random.manual_seed(42)
    if scale is None:
        scale = 1.0 / math.sqrt(E)
    queries = torch.rand((F, W, Hq, S, E))
    keys = [torch.rand((Hkv, f, E)) for f in frags]
    values = [torch.rand((Hkv, f, Ev)) for f in frags]
    frags_tensor = torch.tensor(frags, dtype=torch.int32)
    locations = torch.arange(0, S)[None, None, :] + (frags_tensor[:, None, None] - S)
    locations = torch.tile(locations, (1, W, 1)).to(torch.int32)

    cos = torch.rand((F, W, S, RotaryE))
    sin = torch.rand((F, W, S, RotaryE))
    rotated_queries = apply_rotary_pos_emb(queries, cos, sin)

    expected = hogwild_sdpa_ref(rotated_queries, locations, keys, values, scale=scale)

    keys_cuda = [k[None, ...].to("cuda") for k in keys]
    values_cuda = [v[None, ...].to("cuda") for v in values]
    actual = hogwild_sdpa(rotated_queries.to("cuda"), locations.to("cuda"), keys_cuda, values_cuda, scale=scale)

    print("Expected result:")
    print(expected)
    print("\nDifference (Expected - Actual):")
    print(expected.cpu() - actual.cpu())

    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-5, atol=1e-5)
    print("\nTest passed!")


test_custom_kernel(F=1, W=1, Hq=32, Hkv=2, E=128, RotaryE=64, Ev=128, S=1, frags=[100])
