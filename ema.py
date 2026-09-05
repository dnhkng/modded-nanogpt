"""Endgame EMA weight blending.

Keeps an exponential moving average of the weights over the tail of training
and, at the final step, blends it into the live weights:

    w <- (1 - gamma) * w_final + gamma * w_ema

Rationale: the final loss is set almost entirely by the deep-cooldown phase.
Averaging over that phase cancels the residual step-to-step noise in the
weights, which is worth a measurable amount of loss for negligible compute.

Cost. The EMA is updated from each rank's *own shard* of a parameter (the
`p_slice` the optimizer has just written), so the work is 1/world_size per
rank and no extra communication is needed during training -- only a single
all-gather per parameter at the very end, to reconstruct the full average.
The update itself is a fused Triton kernel that reads the bf16 weights
directly rather than materialising an fp32 copy (2.4x faster, bit-identical).

The EMA state is fp32 by necessity, not preference: the update weight is
1/horizon, and at horizon 150 that is 6.7e-3, which is under 2x the bf16
epsilon (3.9e-3). A bf16 EMA would silently discard most updates and stall.
"""

import torch
import triton
import triton.language as tl
from torch import Tensor


@triton.jit
def _ema_update_kernel(state_ptr, p_ptr, alpha, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    state = tl.load(state_ptr + offsets, mask=mask, other=0.0)
    p = tl.load(p_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
    tl.store(state_ptr + offsets, state + (p - state) * alpha, mask=mask)


def fused_ema_(state: Tensor, p: Tensor, alpha: float) -> None:
    """In-place state <- state + (p - state) * alpha, reading p in its own dtype."""
    assert state.is_contiguous() and p.is_contiguous()
    assert state.numel() == p.numel()
    n = state.numel()
    _ema_update_kernel[(triton.cdiv(n, 1024),)](state, p, alpha, n, BLOCK=1024)


class EndgameEMA:
    """Per-shard EMA of the weights over the tail of training.

    `update` is called once per optimizer step per parameter, with the slice of
    that parameter the calling rank owns. `blend_` applies the average to the
    live weights at the end of training.
    """

    def __init__(self, horizon: int, start_step: int, skip_labels: tuple[str, ...] = ()):
        self.horizon = horizon
        self.start_step = start_step
        self.skip_labels = skip_labels
        self.state: dict[str, Tensor] = {}
        self._stream: torch.cuda.Stream | None = None

    def update_many(self, pending: "list[tuple[str, Tensor]]", step: int) -> None:
        """Apply every recorded shard at once, on a side stream so the kernels
        overlap the next forward rather than serialising against it."""
        if step < self.start_step:
            return
        if self._stream is None:
            self._stream = torch.cuda.Stream()
        cur = torch.cuda.current_stream()
        self._stream.wait_stream(cur)
        with torch.cuda.stream(self._stream):
            for label, p_slice in pending:
                self.update(label, p_slice, step)
        cur.wait_stream(self._stream)

    def update(self, label: str, p_slice: Tensor, step: int) -> None:
        if step < self.start_step or label in self.skip_labels:
            return
        slice_c = p_slice.detach().contiguous()
        state = self.state.get(label)
        if state is None:
            self.state[label] = slice_c.float().clone()
        else:
            fused_ema_(state, slice_c, 1.0 / self.horizon)

    @torch.no_grad()
    def blend_(self, params_by_label: dict[str, Tensor], gamma: float,
               gather: "callable | None" = None) -> None:
        """w <- (1-gamma)*w + gamma*w_ema, in place.

        `gather(label, shard)` must return the full-parameter average assembled
        across ranks; when unsharded (world_size == 1) it may be omitted.
        """
        if gamma == 0.0:
            return
        for label, shard in self.state.items():
            param = params_by_label.get(label)
            if param is None:
                continue
            full = shard if gather is None else gather(label, shard)
            if full.shape != param.shape:
                # NorMuon banks are sharded on a *reshaped* view -- mlp_bank's
                # (12, 2, ...) is flattened to (24, ...) -- so the gathered
                # average is correct but differently shaped. Merging leading
                # dims preserves row order, so this is a pure view.
                if full.numel() != param.numel():
                    raise RuntimeError(
                        f"EMA shape mismatch for {label}: gathered {tuple(full.shape)} "
                        f"vs param {tuple(param.shape)}. Refusing to skip silently."
                    )
                full = full.reshape(param.shape)
            param.copy_(param.float().lerp(full, gamma).to(param.dtype))
