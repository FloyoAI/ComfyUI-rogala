"""
SmartAttentionDispatcher
========================
Automatically detects GPU architecture, OS, and installed libraries,
then patches the model to use the optimal attention kernel.
Category : rogala/Optimization
Version  : 3.0.0-rc17
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import sys
import logging
import importlib.metadata
from importlib.metadata import PackageNotFoundError

import torch
import comfy.ldm.modules.attention as comfy_attn
from comfy.ldm.modules.attention import attention_pytorch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# -- Identity --
_CATEGORY  = "rogala/Optimization"
_NODE_NAME = "SmartAttentionDispatcher"
_VERSION   = "3.0.0-rc17"

log = logging.getLogger("SAD")

# -- GPU SM thresholds --
_SM_MIN       = 75     # Turing minimum
_SM_AMPERE    = 80     # Ampere DC (A100)
_SM_AMPERE_C  = 86     # Ampere consumer (RTX 30xx)
_SM_ADA       = 89     # Ada Lovelace (RTX 40xx)
_SM_HOPPER    = 90     # Hopper (H100)
_SM_BLACKWELL_DC = 100 # Blackwell DC (B100, B200) — SA3 supported
_SM_BLACKWELL = 120    # Blackwell consumer (RTX 50xx)

# -- Version requirements --
_TORCH_MIN  = (2, 8, 0)   # SA3 requires >= 2.8.0
_CUDA_MIN   = (12, 6)
_CUDA_SA3   = (12, 8)

# -- SA2 full function names (sageattention package) --
_FN_AUTO        = "sageattn"                          # reserved (future use)
_FN_FP16_CUDA   = "sageattn_qk_int8_pv_fp16_cuda"
_FN_FP16_TRITON = "sageattn_qk_int8_pv_fp16_triton"
_FN_FP8_CUDA    = "sageattn_qk_int8_pv_fp8_cuda"
_FN_FP8_SM90    = "sageattn_qk_int8_pv_fp8_cuda_sm90"  # reserved (future use)

# -- SA2 kernel short names (internal keys) --
_K_NONE     = "none"
_K_FP16     = "fp16_cuda"
_K_FP8      = "fp8_cuda"
_K_FP8PP    = "fp8pp_cuda"
_K_TRITON   = "triton"

# -- SA2 pv_accum_dtype values --
_PV_FP32        = "fp32"
_PV_FP32_FP32   = "fp32+fp32"
_PV_FP32_FP16   = "fp32+fp16"   # SA2++ — faster on Ada/Hopper/Blackwell

# -- SA3 (sageattn3 package) --
_FN_SA3         = "sageattn3_blackwell"
_SA3_VALID_HD   = {64, 128, 256}  # SA3 supports these head_dim values
_SA3_PBM        = True            # per_block_mean variant — reserved for future

# -- Tensor layouts --
_NHD = "NHD"
_HND = "HND"

# -- UI mode strings (dropdown values) --
_MODE_SDPA = "Default (SDPA)"
_MODE_SA2  = "SageAttention2"
_MODE_SA3  = "SageAttention3"
_MODES     = [_MODE_SDPA, _MODE_SA2, _MODE_SA3]

# -- SA2 kernel dropdown values --
_SA2_DISABLE = "disable"
_SA2_AUTO    = "auto"
_SA2_FP16    = "fp16"
_SA2_FP8     = "fp8"
_SA2_FP8PP   = "fp8++"
_SA2_TRITON  = "triton"
_SA2_MODES   = [_SA2_DISABLE, _SA2_AUTO, _SA2_FP16, _SA2_FP8, _SA2_FP8PP, _SA2_TRITON]

# -- SA3 kernel dropdown values --
_SA3_DISABLE  = "disable"
_SA3_STANDARD = "standard"
_SA3_PBM_MODE = "per_block_mean"
_SA3_MODES    = [_SA3_DISABLE, _SA3_STANDARD, _SA3_PBM_MODE]

# -- Internal effective mode strings --
_EFF_SDPA    = "sdpa"
_EFF_SA2     = "sa2"
_EFF_SA3     = "sa3"
_EFF_DYNAMIC = "dynamic"

# -- Global patch state --
# Captured at import time — before any patch is applied
_ORIGINAL_OPTIMIZED   = comfy_attn.optimized_attention
_GLOBAL_PATCH_ACTIVE  = False
_ORIGINAL_MODULE_ATTN = {}  # tracks original functions in third-party modules

# -- Environment cache --
_CACHED_ENV = None

# -- Misc --
_MIN_WIDTH       = 420
_SEEN_SHAPES_CAP = 64

# ---------------------------------------------------------------------------
# Description
# ---------------------------------------------------------------------------

_DESCRIPTION = """\
## Smart Attention Dispatcher
Patches the model to use SageAttention kernels instead of PyTorch SDPA.
Detects GPU architecture, installed libraries, and selects the correct kernel automatically.

---

### Inputs
| Pin | Default | Description |
|---|---|---|
| `model` | — | Any ComfyUI model (Flux, SD3.5, SDXL, Qwen, ErnieImage, Z-Image, …). |
| `sdpa_kernel` | False | Force PyTorch SDPA. Overrides all SA2/SA3 settings. Use to compare output or disable SA entirely. |
| `sa2_kernel` | disable | SA2 kernel selection — see table below. |
| `combine` | False | Dynamic mode: SA2 on boundary steps (first/last), SA3 on middle steps. Requires both sa2_kernel ≠ disable and sa3_kernel ≠ disable. |
| `sa3_kernel` | disable | SA3 kernel selection — see table below. |

**sa2_kernel options:**
| Value | Description |
|---|---|
| `disable` | SA2 off. |
| `auto` | Best kernel for the detected GPU: fp8 on Ada/Hopper/Blackwell, fp16 on Turing/Ampere. |
| `fp16` | `sageattn_qk_int8_pv_fp16_cuda` — fp32 accumulator. Turing / Ampere. Blocked on Blackwell (SM120+). |
| `fp8` | `sageattn_qk_int8_pv_fp8_cuda` — fp32+fp32 accumulator. Ada and newer. Bit-exact with SDPA on tested models. |
| `fp8++` | `sageattn_qk_int8_pv_fp8_cuda` — fp32+fp16 accumulator. Slightly faster but results differ from SDPA. |
| `triton` | `sageattn_qk_int8_pv_fp16_triton` — Triton fallback, all GPUs. |

**sa3_kernel options:**
| Value | Description |
|---|---|
| `disable` | SA3 off. |
| `standard` | `sageattn3_blackwell` — FP4, Blackwell only (SM≥100, CUDA≥12.8, Python≥3.10). |
| `per_block_mean` | Same kernel with `per_block_mean=True` — slightly different numerics, marginally faster in some cases. |

---

### Outputs
| Pin | Type | Description |
|---|---|---|
| `model` | MODEL | Patched model with attention override applied. |

---

### Mode display (node status panel)
The node shows the active mode after each run:

| Display | Meaning |
|---|---|
| `SDPA` | PyTorch SDPA — baseline, no SA active. |
| `SA2` | SageAttention2 active on all steps. |
| `SA3` | SageAttention3 active on all steps. |
| `SA2-SA3-SA2` | Dynamic: SA2 on first/last step, SA3 on middle steps. |
| `SDPA-SA3-SDPA` | Dynamic: SDPA on first/last step, SA3 on middle steps. |
| `SA3 (not installed) >>> SA2` | Fallback — reason shown in parentheses. |

---

### Tested models
| Model | SA2 | SA3 | Notes |
|---|---|---|---|
| Flux.1 / Flux.2 / Flux.2 Klein | ✅ | ✅ | — |
| SD3.5 | ✅ | ✅ | Cross-attention layers auto-fallback to SDPA. |
| Z-Image (Lumina2) | ✅ | ✅ | — |
| SDXL | ✅ | — | SA3 not tested on UNet. No speed gain observed. |
| ErnieImage | ✅ | ✅ | Requires global sys.modules patch (applied automatically). |
| Qwen-Image / Qwen-Edit | ✅ | ⚠️ | SA3 numerically unstable at long sequences (seq > 7000). |
| LTX / Wan / HunyuanVideo | — | — | Not yet tested. |

---

### Performance notes
- On **RTX 50xx (Blackwell) with PyTorch 2.8+ / CUDA 13.0** — no measurable speed gain over baseline SDPA.
  PyTorch SDPA is already optimized for SM120. SA kernels add overhead without benefit.
- On **RTX 30xx / 40xx** — SA2 gives real speed improvement, especially at long sequences (Flux, SD3.5, Qwen).
- SA3 on Qwen (seq 7000–14000) produces unpredictable results due to FP4 quantization error accumulation.

---

### Compatibility notes
- Do **not** use `--use-sage-attention` launch flag together with this node.
  ComfyUI patches attention before the node loads — the node will restore the wrong baseline on SDPA mode.
- `--fast` flag (fp16_accumulation, fp8_matrix_mult, etc.) is safe to use alongside this node.
- Attention masks (inpainting, outpainting) automatically fall back to SDPA — SA2/SA3 do not support arbitrary masks.
"""

# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

def _parse_cuda_ver(s: str) -> tuple:
    """'12.8' → (12, 8). Returns (0, 0) on failure."""
    try:
        a, b = s.split(".")[:2]
        return (int(a), int(b))
    except Exception:
        return (0, 0)


def _parse_torch_ver(s: str) -> tuple:
    """'2.9.0+cu128' or '2.9.0.dev20250101' → (2, 9, 0). Returns (0, 0, 0) on failure."""
    try:
        base = s.split("+")[0].split(".dev")[0]
        parts = base.split(".")
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    except Exception:
        return (0, 0, 0)


def _parse_triton_ver(s: str) -> tuple:
    """'3.3.1.post21' → (3, 3, 1). Returns (0, 0, 0) on failure."""
    try:
        parts = s.split(".post")[0].split(".")
        return (int(parts[0]), int(parts[1]), int(parts[2]))
    except Exception:
        return (0, 0, 0)


def _detect_triton() -> tuple:
    """
    Returns (pkg_name, version_str).
    Windows: 'triton-windows'. Linux: 'triton'. Empty strings if not found.
    """
    for pkg in ("triton-windows", "triton"):
        try:
            return (pkg, importlib.metadata.version(pkg))
        except PackageNotFoundError:
            continue
    return ("", "")


def _probe_sa2() -> dict:
    """Returns {fn_name: bool} for each SA2 kernel in sageattention package."""
    kernels = {_FN_FP16_CUDA: False, _FN_FP8_CUDA: False, _FN_FP16_TRITON: False}
    try:
        import sageattention as _sa
        for k in kernels:
            kernels[k] = hasattr(_sa, k)
    except ImportError:
        pass
    return kernels


def _probe_sa3() -> bool:
    """Returns True if sageattn3 package has sageattn3_blackwell."""
    try:
        import sageattn3 as _sa3
        return hasattr(_sa3, _FN_SA3)
    except ImportError:
        return False


class Environment:
    """
    All pre-flight detection results.
    Built once per session via Environment.detect(), cached in _CACHED_ENV.

    Attributes
    ----------
    ok          : bool   — passed all minimum requirements
    fail_reason : str    — human-readable reason if ok=False
    sm          : int    — GPU compute capability as integer (e.g. 120)
    """

    def __init__(self):
        self.os_name        = ""
        self.python_ver     = (0, 0, 0)
        self.torch_ver      = (0, 0, 0)
        self.cuda_ver       = (0, 0)
        self.cuda_str       = ""
        self.sm             = 0
        self.sm_str         = ""
        self.gpu_name       = ""
        self.triton_pkg     = ""
        self.triton_ver_str = ""
        self.triton_ver     = (0, 0, 0)
        self.sa2_kernels    = {}
        self.sa3_available  = False
        self.cuda_available = False
        self.ok             = False
        self.fail_reason    = ""

    @classmethod
    def detect(cls) -> "Environment":
        env = cls()
        env.os_name    = "windows" if sys.platform == "win32" else "linux"
        vi             = sys.version_info
        env.python_ver = (vi.major, vi.minor, vi.micro)
        env.torch_ver  = _parse_torch_ver(torch.__version__)

        env.cuda_available = torch.cuda.is_available()
        if not env.cuda_available:
            env.fail_reason = "CUDA not available"
            return env

        env.cuda_str = torch.version.cuda or ""
        env.cuda_ver = _parse_cuda_ver(env.cuda_str)
        cap          = torch.cuda.get_device_capability()
        env.sm       = cap[0] * 10 + cap[1]
        env.sm_str   = f"{cap[0]}.{cap[1]}"
        env.gpu_name = torch.cuda.get_device_name()

        env.triton_pkg, env.triton_ver_str = _detect_triton()
        env.triton_ver    = _parse_triton_ver(env.triton_ver_str)
        env.sa2_kernels   = _probe_sa2()
        env.sa3_available = _probe_sa3()

        if env.torch_ver < _TORCH_MIN:
            env.fail_reason = f"PyTorch {torch.__version__} < {'.'.join(str(x) for x in _TORCH_MIN)}"
            return env
        if env.cuda_ver < _CUDA_MIN:
            env.fail_reason = f"CUDA {env.cuda_str} < {'.'.join(str(x) for x in _CUDA_MIN)}"
            return env
        if env.sm < _SM_MIN:
            env.fail_reason = f"SM {env.sm_str} < {_SM_MIN // 10}.{_SM_MIN % 10} — GPU not supported"
            return env

        env.ok = True
        return env

    # -- SA2 capability --
    def has_sa2_fp16(self) -> bool:
        return self.ok and self.sa2_kernels.get(_FN_FP16_CUDA, False)

    def has_sa2_fp8(self) -> bool:
        return self.ok and self.sa2_kernels.get(_FN_FP8_CUDA, False)

    def has_sa2_triton(self) -> bool:
        return self.ok and self.sa2_kernels.get(_FN_FP16_TRITON, False)

    def has_any_sa2(self) -> bool:
        return self.best_sa2_kernel() != _K_NONE

    def has_sa2_fp16_safe(self) -> bool:
        """fp16_cuda kernel is SM<=86 only — blocked on Blackwell (SM120+)."""
        return self.has_sa2_fp16() and self.sm < _SM_BLACKWELL

    def best_sa2_kernel(self) -> str:
        """
        Returns the most stable available SA2 kernel short name for this GPU.
        SM < 89  → fp16_cuda (Turing / Ampere)
        SM >= 89 → fp8_cuda  (Ada / Hopper / Blackwell) — stable fp32+fp32 accum
        Fallback → triton
        Note: fp8pp (SA2++, fp32+fp16 accum) is available via explicit dropdown only.
        Note: fp16_cuda is blocked on Blackwell (SM120+) — kernel requires SM<=86.
        """
        if self.sm >= _SM_ADA:
            if self.has_sa2_fp8():        return _K_FP8
            if self.has_sa2_fp16_safe():  return _K_FP16
            if self.has_sa2_triton():     return _K_TRITON
        else:
            if self.has_sa2_fp16():       return _K_FP16
            if self.has_sa2_triton():     return _K_TRITON
        return _K_NONE

    # -- SA3 capability --
    def has_sa3(self) -> bool:
        return (
            self.ok
            and self.python_ver >= (3, 10)   # SA3 requires Python >= 3.10
            and self.sm >= _SM_BLACKWELL_DC  # includes B100/B200 (SM100+)
            and self.cuda_ver >= _CUDA_SA3
            and self.sa3_available
        )

    # -- GPU label --
    def gpu_tier(self) -> str:
        sm = self.sm
        if sm >= _SM_BLACKWELL: return "Blackwell"
        if sm >= 100:           return "Blackwell DC"
        if sm >= _SM_HOPPER:    return "Hopper"
        if sm >= _SM_ADA:       return "Ada"
        if sm >= _SM_AMPERE_C:  return "Ampere"
        if sm >= _SM_AMPERE:    return "Ampere DC"
        if sm >= _SM_MIN:       return "Turing"
        return "Unknown"

    # -- Status string for node display --
    def format_status(self, req_mode: str, effective: str,
                      hd, heads, arch: str) -> str:
        """Build the status string displayed on the node after execution."""
        sa2_ok      = self.has_any_sa2()
        sa3_ok      = self.has_sa3()
        sa3_py_ok   = self.python_ver >= (3, 13)
        sa3_hw_ok   = self.sm >= _SM_BLACKWELL_DC and self.cuda_ver >= _CUDA_SA3
        sa3_pkg_ok  = self.sa3_available
        sa2_arch_ok = True                   # SA2 supported on all archs incl. unet
        sa3_arch_ok = arch != "unet"         # SA3 untested on unet
        hd_ok       = hd is None or hd in _SA3_VALID_HD
        sa2_mark    = self.best_sa2_kernel() if sa2_ok else "--"
        sa3_mark    = "OK" if sa3_ok else "--"
        hd_str      = str(hd)    if hd    is not None else "?"
        h_str       = str(heads) if heads is not None else "?"
        tier        = self.gpu_tier() if self.ok else "N/A"

        # Determine fallback reason
        reason = ""
        if effective not in (_EFF_DYNAMIC, _EFF_SA3) and "SA3" in req_mode:
            if not sa3_py_ok:      reason = "(Python<3.10)"
            elif not sa3_hw_ok:    reason = "(GPU/CUDA unsupported)"
            elif not sa3_pkg_ok:   reason = "(not installed)"
            elif not hd_ok:        reason = f"(hd={hd_str} not supported)"
            elif not sa3_arch_ok:  reason = "(arch not supported)"

        if effective == _EFF_SDPA and "SA2" in req_mode and not reason:
            if not sa2_ok:         reason = "(not installed)"
            # sa2_arch_ok is always True — no arch block for SA2

        # Format Mode label
        if effective == _EFF_SDPA:
            if req_mode == "SDPA":
                mode_label = "SDPA"
            else:
                mode_label = f"{req_mode} {reason} >>> SDPA".replace("  ", " ").strip()
        elif effective == _EFF_SA2:
            if req_mode == "SA2":
                mode_label = "SA2"
            else:
                mode_label = f"{req_mode} {reason} >>> SA2".replace("  ", " ").strip()
        elif effective == _EFF_DYNAMIC:
            mode_label = req_mode
        elif effective == _EFF_SA3:
            mode_label = "SA3"
        else:
            mode_label = effective.upper()

        py_str = ".".join(str(x) for x in self.python_ver)
        lines = [
            f"Mode: {mode_label}",
            f"GPU:  {self.gpu_name}  SM {self.sm_str}  {tier}",
            f"PyTorch: {torch.__version__}  |  Python: {py_str}",
            f"SA2: {sa2_mark}  SA3: {sa3_mark}"
            + (f"  |  Triton: {self.triton_ver_str}" if self.triton_ver_str else ""),
            f"arch: {arch}  |  head_dim: {hd_str}  |  heads: {h_str}",
        ]
        if self.fail_reason:
            lines.append(f"Note: {self.fail_reason}")
        return "\n".join(lines)


def _get_env() -> Environment:
    """Return cached Environment, detecting on first call only."""
    global _CACHED_ENV
    if _CACHED_ENV is None:
        _CACHED_ENV = Environment.detect()
    return _CACHED_ENV


# ---------------------------------------------------------------------------
# Model detection
# ---------------------------------------------------------------------------

def _detect_model_params(model_patcher) -> dict:
    """
    Detect head_dim, num_heads, and architecture from the diffusion model.

    Returns
    -------
    dict with keys: head_dim, num_heads, arch (dit | unet | mmdit | unknown)

    Detection order
    ---------------
    1. Architecture type from model attributes.
    2. Top-level model attributes (head_dim, dim_head, etc.).
    3. named_parameters() scan for first to_q.weight.
    4. Flux-specific: double_blocks[0].img_attn.
    """
    result = {"head_dim": None, "num_heads": None, "arch": "unknown"}
    try:
        dm = model_patcher.get_model_object("diffusion_model")

        # Architecture
        if hasattr(dm, "double_blocks") or hasattr(dm, "transformer_blocks"):
            result["arch"] = "dit"
        elif hasattr(dm, "joint_blocks") or hasattr(dm, "context_embedder"):
            result["arch"] = "mmdit"
        elif hasattr(dm, "input_blocks") or hasattr(dm, "down_blocks"):
            result["arch"] = "unet"
        elif hasattr(dm, "n_heads") and hasattr(dm, "dim"):
            # NextDiT (Z-Image / Lumina2) — no block-list attributes
            n_heads = int(dm.n_heads)
            if n_heads > 0:
                result["arch"]      = "dit"
                result["num_heads"] = n_heads
                result["head_dim"]  = int(dm.dim) // n_heads
                return result

        # Top-level attributes
        for attr in ("head_dim", "dim_head", "d_head"):
            if hasattr(dm, attr):
                result["head_dim"] = getattr(dm, attr)
                break
        for attr in ("num_heads", "n_heads", "heads", "num_attention_heads"):
            if hasattr(dm, attr):
                result["num_heads"] = getattr(dm, attr)
                break

        if result["head_dim"] is not None:
            return result

        # Scan named_parameters for to_q.weight
        # dit models use hd=128 typically; mmdit/unet/unknown use hd=64
        _hd_order = (128, 64, 96, 160, 256) if result["arch"] == "dit" \
                    else (64, 128, 96, 160, 256)
        for name, param in dm.named_parameters():
            if ".to_q.weight" not in name and ".attn1.to_q.weight" not in name:
                continue
            out_dim = param.shape[0]
            for hd in _hd_order:
                if out_dim % hd == 0:
                    result["head_dim"]  = hd
                    result["num_heads"] = out_dim // hd
                    break
            break

        # Flux-specific: double_blocks[0].img_attn
        if result["head_dim"] is None and hasattr(dm, "double_blocks"):
            try:
                blk = dm.double_blocks[0]
                for candidate in ("img_attn", "attn"):
                    attn = getattr(blk, candidate, None)
                    if attn is None:
                        continue
                    for attr in ("head_dim", "dim_head"):
                        if hasattr(attn, attr):
                            result["head_dim"] = getattr(attn, attr)
                            break
                    for attr in ("num_heads", "heads"):
                        if hasattr(attn, attr):
                            result["num_heads"] = getattr(attn, attr)
                            break
                    if result["head_dim"] is not None:
                        break
            except Exception:
                pass

    except Exception as exc:
        log.warning(f"[SAD] model param detection failed: {exc}")

    return result


# ---------------------------------------------------------------------------
# Operating modes
# ---------------------------------------------------------------------------

# ── Shared layout helpers ────────────────────────────────────────────────────

def _prepare_qkv(q, k, v, heads: int, skip_reshape: bool):
    """
    Normalize q/k/v to (B, S, H, D) — NHD format — for SA2 kernels.
    Returns (q, k, v, b, s, h, d).

    skip_reshape=False : input (B, S, H*D) → view to (B, S, H, D)
                         Supports GQA/MQA — k/v may have fewer heads than q.
    skip_reshape=True  : input (B, H, S, D) → transpose to (B, S, H, D)
    """
    # Guard: empty tensor (e.g. SD3.5 cross-attn with zero context length)
    if q.numel() == 0:
        raise ValueError(f"[SAD] zero-element tensor q.shape={list(q.shape)}")
    if skip_reshape:
        b, h, s, d = q.shape
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        if not q.is_contiguous(): q = q.contiguous()
        if not k.is_contiguous(): k = k.contiguous()
        if not v.is_contiguous(): v = v.contiguous()
        return q, k, v, b, s, h, d
    else:
        b, s, hd_total = q.shape
        h = heads
        if hd_total % h != 0:
            raise ValueError(
                f"[SAD] SA2: hd_total={hd_total} not divisible by heads={h} "
                f"q.shape={list(q.shape)} — cross-attention not supported"
            )
        d = hd_total // h
        # GQA/MQA: k and v may have fewer heads than q
        h_k = k.shape[2] // d if k.shape[2] % d == 0 else h
        h_v = v.shape[2] // d if v.shape[2] % d == 0 else h
        return (
            q.reshape(b, s, h, d),
            k.reshape(b, k.shape[1], h_k, d),
            v.reshape(b, v.shape[1], h_v, d),
            b, s, h, d
        )


def _restore_output(out, b: int, s: int, h: int, d: int,
                    skip_reshape: bool, skip_output_reshape: bool):
    """
    Restore SA2 kernel output (B, S, H, D) to the format ComfyUI expects.

    skip_output_reshape=True  → return (B, H, S, D)
    skip_output_reshape=False → return (B, S, H*D)  [default]
    """
    if skip_output_reshape:
        return out.transpose(1, 2).contiguous()
    return out.reshape(b, s, h * d)


def _get_head_dim(q, heads: int, skip_reshape: bool) -> int:
    """Extract head_dim from tensor at runtime."""
    return q.shape[3] if skip_reshape else q.shape[2] // heads


def _fp16_if_fp32(q, k, v):
    """Cast fp32 → fp16. SA kernels require fp16 or bf16."""
    if q.dtype == torch.float32:
        return q.half(), k.half(), v.half()
    return q, k, v


# ── SDPA ─────────────────────────────────────────────────────────────────────

def _call_sdpa(q, k, v, heads, mask=None,
               skip_reshape=False, skip_output_reshape=False, **kwargs):
    """PyTorch SDPA — always safe, always available."""
    return attention_pytorch(q, k, v, heads, mask=mask,
                             skip_reshape=skip_reshape,
                             skip_output_reshape=skip_output_reshape, **kwargs)


# ── SA2 ──────────────────────────────────────────────────────────────────────

# SA2 kernel function cache — imported once at first use
_SA2_KERNEL_CACHE: dict = {}


def _get_sa2_kernel(fn_name: str):
    """Return cached SA2 kernel function by full name, importing on first access."""
    if fn_name in _SA2_KERNEL_CACHE:
        return _SA2_KERNEL_CACHE[fn_name]
    try:
        import sageattention as _sa
        fn = getattr(_sa, fn_name)
        _SA2_KERNEL_CACHE[fn_name] = fn
        return fn
    except (ImportError, AttributeError) as e:
        raise ImportError(f"[SAD] SA2 kernel {fn_name} not available: {e}")


def _call_sa2(fn_name: str, pv_accum_dtype,
              q, k, v, heads, skip_reshape, skip_output_reshape, **kwargs):
    """
    Shared SA2 pipeline: prepare → cast → kernel → restore.
    All SA2 kernels use NHD layout.
    """
    fn       = _get_sa2_kernel(fn_name)
    in_dtype = v.dtype
    q, k, v  = _fp16_if_fp32(q, k, v)
    q, k, v, b, s, h, d = _prepare_qkv(q, k, v, heads, skip_reshape)
    kw = {"is_causal": False, "attn_mask": None, "tensor_layout": _NHD}
    if pv_accum_dtype is not None:
        kw["pv_accum_dtype"] = pv_accum_dtype
    out = fn(q, k, v, **kw).to(in_dtype)
    return _restore_output(out, b, s, h, d, skip_reshape, skip_output_reshape)


def _call_sa2_fp16(q, k, v, heads, mask=None,
                   skip_reshape=False, skip_output_reshape=False, **kwargs):
    """SA2 fp16 CUDA — SM 75–86 (Turing / Ampere)."""
    return _call_sa2(_FN_FP16_CUDA, _PV_FP32, q, k, v, heads,
                     skip_reshape, skip_output_reshape, **kwargs)


def _call_sa2_fp8(q, k, v, heads, mask=None,
                  skip_reshape=False, skip_output_reshape=False, **kwargs):
    """SA2 fp8 CUDA — SM 89+. Uses fp32+fp32 accumulator (standard)."""
    return _call_sa2(_FN_FP8_CUDA, _PV_FP32_FP32, q, k, v, heads,
                     skip_reshape, skip_output_reshape, **kwargs)


def _call_sa2_triton(q, k, v, heads, mask=None,
                     skip_reshape=False, skip_output_reshape=False, **kwargs):
    """SA2 fp16 Triton — fallback when CUDA kernels unavailable."""
    return _call_sa2(_FN_FP16_TRITON, None, q, k, v, heads,
                     skip_reshape, skip_output_reshape, **kwargs)


def _call_sa2_fp8pp(q, k, v, heads, mask=None,
                    skip_reshape=False, skip_output_reshape=False, **kwargs):
    """SA2++ fp8 CUDA — SM 89+ (Ada / Hopper / Blackwell). Uses fp32+fp16 accumulator."""
    return _call_sa2(_FN_FP8_CUDA, _PV_FP32_FP16, q, k, v, heads,
                     skip_reshape, skip_output_reshape, **kwargs)


# SA2 kernel short name → call function
_SA2_FN = {
    _K_FP16  : _call_sa2_fp16,
    _K_FP8   : _call_sa2_fp8,
    _K_FP8PP : _call_sa2_fp8pp,
    _K_TRITON: _call_sa2_triton,
}

# SA2 UI dropdown value → internal kernel short name
_SA2_MODE_TO_KERNEL = {
    _SA2_FP16   : _K_FP16,
    _SA2_FP8    : _K_FP8,
    _SA2_FP8PP  : _K_FP8PP,
    _SA2_TRITON : _K_TRITON,
}

# SA3 kernel cache — imported once at first use
_SA3_FN = None


def _get_sa3_kernel():
    """Return cached sageattn3_blackwell function, importing on first access."""
    global _SA3_FN
    if _SA3_FN is None:
        try:
            from sageattn3 import sageattn3_blackwell
            _SA3_FN = sageattn3_blackwell
        except ImportError as e:
            raise RuntimeError(
                f"[SAD] sageattn3 import failed: {e}. "
                "Install sageattn3 or switch to SA2/SDPA mode."
            ) from e
    return _SA3_FN


# ── SA3 ──────────────────────────────────────────────────────────────────────

def _call_sa3(q, k, v, heads, mask=None,
              skip_reshape=False, skip_output_reshape=False,
              use_pbm: bool = False, **kwargs):
    """
    SA3 — public API: sageattn3_blackwell.
    FP4, Blackwell only (SM >= 120, CUDA >= 12.8), Python >= 3.10.

    sageattn3_blackwell expects HND format: (B, H, S, D).

    skip_reshape=True  : input (B, H, S, D) — use directly
    skip_reshape=False : input (B, S, H*D) → reshape to (B, H, S, D)

    Output: (B, S, H*D) unless skip_output_reshape=True → (B, H, S, D)
    """
    fn       = _get_sa3_kernel()
    in_dtype = v.dtype
    if in_dtype == torch.float32:
        q, k, v = q.half(), k.half(), v.half()

    if skip_reshape:
        b, h, s, d = q.shape
        # CUDA FP4 kernel does not support strided tensors
        if not q.is_contiguous(): q = q.contiguous()
        if not k.is_contiguous(): k = k.contiguous()
        if not v.is_contiguous(): v = v.contiguous()
    else:
        b, s, hd_total = q.shape
        h = heads
        # Guard: hd_total must be divisible by heads
        # SD3.5 cross-attention passes q with hd_total == head_dim (not heads*head_dim)
        if hd_total % h != 0:
            raise ValueError(
                f"[SAD] SA3: hd_total={hd_total} not divisible by heads={h} "
                f"q.shape={list(q.shape)} — cross-attention not supported by SA3"
            )
        d = hd_total // h
        h_k = k.shape[2] // d if len(k.shape) == 3 and k.shape[2] % d == 0 else h
        h_v = v.shape[2] // d if len(v.shape) == 3 and v.shape[2] % d == 0 else h
        q = q.reshape(b, s, h,   d).transpose(1, 2).contiguous()
        k = k.reshape(b, k.shape[1], h_k, d).transpose(1, 2).contiguous()
        v = v.reshape(b, v.shape[1], h_v, d).transpose(1, 2).contiguous()

    # FP4 packs two values per uint8 byte → head_dim must be even
    # Defensive check: if _SA3_VALID_HD ever extends to odd values, fail loudly
    if d % 2 != 0:
        raise ValueError(f"[SAD] SA3 requires even head_dim, got d={d}")

    # sageattn3_blackwell HND mask layout is not yet documented
    # passing None to avoid silent artifacts on inpainting / cross-attn
    out = fn(
        q, k, v,
        is_causal=False,
        attn_mask=None,
        tensor_layout=_HND,
        per_block_mean=use_pbm,
    ).to(in_dtype)

    if skip_output_reshape:
        return out
    return out.transpose(1, 2).contiguous().reshape(b, s, h * d)


# ── Dynamic ───────────────────────────────────────────────────────────────────

def _current_step(transformer_options: dict, _cache: dict = None) -> tuple:
    """
    Extract (step_index, total_steps) from transformer_options.
    Returns (0, 1) if not determinable — treated as boundary → SA2.
    Sigma comparison done on CPU to avoid GPU stalls.
    """
    if not transformer_options:
        return (0, 1)
    sigmas = transformer_options.get("sample_sigmas")
    if sigmas is None or len(sigmas) < 2:
        return (0, 1)
    total  = len(sigmas) - 1
    cur_ts = transformer_options.get("sigmas")
    if cur_ts is None or cur_ts.numel() == 0:
        return (0, total)

    if _cache is not None:
        sid = id(sigmas)
        first_val = float(sigmas[0])
        if (sid not in _cache
                or _cache.get("len") != total
                or _cache.get("first") != first_val):
            _cache.clear()
            _cache[sid]     = sigmas[:total].cpu().float()
            _cache["len"]   = total
            _cache["first"] = first_val
        sigmas_cpu = _cache[sid]
    else:
        sigmas_cpu = sigmas[:total].cpu().float()

    cur_val = cur_ts[0].cpu().float()
    idx     = int((sigmas_cpu - cur_val).abs().argmin().item())
    return (idx, total)


# ── Wrapper builder ────────────────────────────────────────────────────────────

def _build_wrapper(effective: str, sa2_kernel: str, use_pbm: bool):
    """
    Build and return the attention override function for the resolved mode.
    All wrappers share the ComfyUI-required signature.

    Parameters
    ----------
    effective  : _EFF_SDPA | _EFF_SA2 | _EFF_SA3 | _EFF_DYNAMIC
    sa2_kernel : _K_FP16 | _K_FP8 | _K_FP8PP | _K_TRITON | "" | "sdpa"
    use_pbm    : use per_block_mean=True for SA3 (sa3_kernel=per_block_mean)
    """
    seen_shapes = set()

    def _log(q, heads, skip_reshape):
        if not log.isEnabledFor(logging.DEBUG):
            return
        if len(seen_shapes) > _SEEN_SHAPES_CAP:
            if len(seen_shapes) == _SEEN_SHAPES_CAP + 1:
                log.debug(f"[SAD] shape log limit ({_SEEN_SHAPES_CAP}) reached — further shapes suppressed")
                seen_shapes.add("__cap_logged__")
            return
        h = q.shape[1] if skip_reshape else heads
        d = q.shape[3] if skip_reshape else (q.shape[2] // heads if heads else 0)
        s = q.shape[2] if skip_reshape else q.shape[1]
        key = (h, d, s, str(q.dtype))
        if key not in seen_shapes:
            seen_shapes.add(key)
            log.debug(f"[SAD] tensor  heads={h:3d}  hd={d:4d}  seq={s:6d}  dtype={q.dtype}")

    # ── SDPA ──
    if effective == _EFF_SDPA:
        def _w(func, q, k, v, heads, mask=None, attn_precision=None,
               skip_reshape=False, skip_output_reshape=False, **kw):
            _log(q, heads, skip_reshape)
            return _call_sdpa(q, k, v, heads, mask=mask,
                              skip_reshape=skip_reshape,
                              skip_output_reshape=skip_output_reshape, **kw)
        return _w

    # ── SA2 ──
    if effective == _EFF_SA2:
        if sa2_kernel not in _SA2_FN:
            log.error(f"[SAD] unknown SA2 kernel: {sa2_kernel!r} → SDPA fallback")
            def _w(func, q, k, v, heads, mask=None, attn_precision=None,
                   skip_reshape=False, skip_output_reshape=False, **kw):
                _log(q, heads, skip_reshape)
                return _call_sdpa(q, k, v, heads, mask=mask,
                                  skip_reshape=skip_reshape,
                                  skip_output_reshape=skip_output_reshape, **kw)
            return _w

        call_fn   = _SA2_FN[sa2_kernel]
        seen_warn = set()

        def _w(func, q, k, v, heads, mask=None, attn_precision=None,
               skip_reshape=False, skip_output_reshape=False, **kw):
            _log(q, heads, skip_reshape)
            # Silent passthrough for empty tensors (e.g. SD3.5 cross-attn zero context)
            if q.numel() == 0:
                return _call_sdpa(q, k, v, heads, mask=mask,
                                  skip_reshape=skip_reshape,
                                  skip_output_reshape=skip_output_reshape, **kw)
            try:
                return call_fn(q, k, v, heads, mask=mask,
                               skip_reshape=skip_reshape,
                               skip_output_reshape=skip_output_reshape, **kw)
            except Exception as exc:
                hd  = _get_head_dim(q, heads, skip_reshape)
                exc_str = str(exc)
                key = (heads, hd, exc_str)
                if key not in seen_warn:
                    seen_warn.add(key)
                    log.warning(f"[SAD] {sa2_kernel} error heads={heads} hd={hd}: {exc} → SDPA")
                return _call_sdpa(q, k, v, heads, mask=mask,
                                  skip_reshape=skip_reshape,
                                  skip_output_reshape=skip_output_reshape, **kw)
        return _w

    # ── SA3 ──
    if effective == _EFF_SA3:
        seen_warn = set()

        def _w(func, q, k, v, heads, mask=None, attn_precision=None,
               skip_reshape=False, skip_output_reshape=False, **kw):
            _log(q, heads, skip_reshape)
            # Silent passthrough for empty tensors
            if q.numel() == 0:
                return _call_sdpa(q, k, v, heads, mask=mask,
                                  skip_reshape=skip_reshape,
                                  skip_output_reshape=skip_output_reshape, **kw)
            hd = _get_head_dim(q, heads, skip_reshape)
            if hd not in _SA3_VALID_HD:
                key = (heads, hd)
                if key not in seen_warn:
                    seen_warn.add(key)
                    log.warning(f"[SAD] SA3 requires hd in {_SA3_VALID_HD}, got hd={hd} → SDPA")
                return _call_sdpa(q, k, v, heads, mask=mask,
                                  skip_reshape=skip_reshape,
                                  skip_output_reshape=skip_output_reshape, **kw)
            try:
                return _call_sa3(q, k, v, heads,
                                 skip_reshape=skip_reshape,
                                 skip_output_reshape=skip_output_reshape,
                                 use_pbm=use_pbm, **kw)
            except Exception as exc:
                exc_str = str(exc)
                key = ("sa3_error", hd, exc_str)
                if key not in seen_warn:
                    seen_warn.add(key)
                    log.warning(f"[SAD] SA3 error hd={hd}: {exc} → SDPA")
                return _call_sdpa(q, k, v, heads, mask=mask,
                                  skip_reshape=skip_reshape,
                                  skip_output_reshape=skip_output_reshape, **kw)
        return _w

    # ── Dynamic ──
    if effective == _EFF_DYNAMIC:
        # sa2_kernel may be "sdpa" for SDPA-SA3-SDPA mode
        if sa2_kernel == "sdpa":
            sa2_fn  = _call_sdpa
            sa2_lbl = "sdpa"
        elif sa2_kernel not in _SA2_FN:
            log.error(f"[SAD] DYNAMIC: unknown SA2 kernel {sa2_kernel!r} → SDPA fallback")
            def _w(func, q, k, v, heads, mask=None, attn_precision=None,
                   skip_reshape=False, skip_output_reshape=False, **kw):
                _log(q, heads, skip_reshape)
                return _call_sdpa(q, k, v, heads, mask=mask,
                                  skip_reshape=skip_reshape,
                                  skip_output_reshape=skip_output_reshape, **kw)
            return _w
        else:
            sa2_fn  = _SA2_FN[sa2_kernel]
            sa2_lbl = sa2_kernel

        seen_warn    = set()
        _sigma_cache = {}

        def _w(func, q, k, v, heads, mask=None, attn_precision=None,
               skip_reshape=False, skip_output_reshape=False, **kw):
            _log(q, heads, skip_reshape)
            # Silent passthrough for empty tensors
            if q.numel() == 0:
                return _call_sdpa(q, k, v, heads, mask=mask,
                                  skip_reshape=skip_reshape,
                                  skip_output_reshape=skip_output_reshape, **kw)
            hd = _get_head_dim(q, heads, skip_reshape)
            if hd not in _SA3_VALID_HD:
                try:
                    return sa2_fn(q, k, v, heads,
                                  skip_reshape=skip_reshape,
                                  skip_output_reshape=skip_output_reshape, **kw)
                except Exception as exc:
                    exc_str = str(exc)
                    key = ("sa2_fallback", heads, hd, exc_str)
                    if key not in seen_warn:
                        seen_warn.add(key)
                        log.warning(f"[SAD] {sa2_lbl} hd={hd}: {exc} → SDPA")
                    return _call_sdpa(q, k, v, heads, mask=mask,
                                      skip_reshape=skip_reshape,
                                      skip_output_reshape=skip_output_reshape, **kw)

            step, total = _current_step(kw.get("transformer_options"), _sigma_cache)
            on_boundary = step == 0 or step >= total - 1
            label       = sa2_lbl if on_boundary else "sa3"
            try:
                if on_boundary:
                    return sa2_fn(q, k, v, heads, mask=mask,
                                  skip_reshape=skip_reshape,
                                  skip_output_reshape=skip_output_reshape, **kw)
                else:
                    return _call_sa3(q, k, v, heads, mask=mask,
                                     skip_reshape=skip_reshape,
                                     skip_output_reshape=skip_output_reshape,
                                     use_pbm=use_pbm, **kw)
            except Exception as exc:
                exc_str = str(exc)
                key = (label, step, hd, exc_str)
                if key not in seen_warn:
                    seen_warn.add(key)
                    log.warning(f"[SAD] {label} step {step}/{total}: {exc} → SDPA")
                return _call_sdpa(q, k, v, heads, mask=mask,
                                  skip_reshape=skip_reshape,
                                  skip_output_reshape=skip_output_reshape, **kw)
        return _w

    # Defensive fallback
    def _w(func, q, k, v, heads, mask=None, attn_precision=None,
           skip_reshape=False, skip_output_reshape=False, **kw):
        return func(q, k, v, heads, mask=mask, attn_precision=attn_precision,
                    skip_reshape=skip_reshape,
                    skip_output_reshape=skip_output_reshape, **kw)
    return _w


# ── Mode resolver — legacy, not used in 3.0.0 execute() ─────────────────────
# Kept for backwards compatibility and potential external use.
# Note: logic differs from execute() — does not have combine/sdpa_kernel inputs.

def _resolve_mode(mode: str, env: Environment) -> tuple:
    """
    Resolve UI mode to (effective, sa2_kernel, reason).

    Returns
    -------
    effective  : _EFF_SDPA | _EFF_SA2 | _EFF_SA3 | _EFF_DYNAMIC
    sa2_kernel : _K_FP16 | _K_FP8 | _K_TRITON | ""
    reason     : human-readable string for console/status
    """
    if not env.ok or mode == _MODE_SDPA:
        reason = "forced by user" if mode == _MODE_SDPA \
                 else (env.fail_reason or "environment check failed")
        return (_EFF_SDPA, "", reason)

    if mode == _MODE_SA2:
        k = env.best_sa2_kernel()
        if k == _K_NONE:
            return (_EFF_SDPA, "", "SA2 not available → SDPA")
        return (_EFF_SA2, k, f"SA2 ({k})")

    if mode == _MODE_SA3:
        if not env.has_sa3():
            k = env.best_sa2_kernel()
            if k != _K_NONE:
                return (_EFF_SA2, k, f"SA3 not available → SA2 ({k})")
            return (_EFF_SDPA, "", "SA3 and SA2 not available → SDPA")
        k = env.best_sa2_kernel()
        if k == _K_NONE:
            return (_EFF_SA3, "", "SA3 only (SA2 unavailable)")
        return (_EFF_DYNAMIC, k, f"Blackwell — SA3 dynamic + SA2 boundaries ({k})")

    return (_EFF_SDPA, "", "unknown mode → SDPA")


# ── Global patch (ERNIE / ACE-Step) ──────────────────────────────────────────

def _restore_global_patch():
    """Restore original optimized_attention for all patched modules."""
    global _GLOBAL_PATCH_ACTIVE, _ORIGINAL_MODULE_ATTN
    comfy_attn.optimized_attention = _ORIGINAL_OPTIMIZED
    for mod, orig_fn in _ORIGINAL_MODULE_ATTN.items():
        if "optimized_attention" in getattr(mod, "__dict__", {}):
            mod.optimized_attention = orig_fn
    _ORIGINAL_MODULE_ATTN.clear()
    _GLOBAL_PATCH_ACTIVE = False


def _apply_global_patch(wrapper_fn):
    """
    Replace optimized_attention globally.
    Patches ALL modules in sys.modules that hold a local reference to
    optimized_attention — covers Qwen, Ernie, ACE-Step and any future model
    that imports optimized_attention directly at load time.
    """
    global _GLOBAL_PATCH_ACTIVE, _ORIGINAL_MODULE_ATTN

    def patched(q, k, v, heads, mask=None, attn_precision=None,
                skip_reshape=False, skip_output_reshape=False,
                low_precision_attention=True, **kwargs):
        return wrapper_fn(_ORIGINAL_OPTIMIZED, q, k, v, heads,
                          mask=mask, attn_precision=attn_precision,
                          skip_reshape=skip_reshape,
                          skip_output_reshape=skip_output_reshape,
                          low_precision_attention=low_precision_attention,
                          **kwargs)

    comfy_attn.optimized_attention = patched

    def _patch_module(mod):
        if "optimized_attention" in getattr(mod, "__dict__", {}):
            if mod not in _ORIGINAL_MODULE_ATTN:
                _ORIGINAL_MODULE_ATTN[mod] = mod.__dict__["optimized_attention"]
            mod.optimized_attention = patched

    # Patch ALL modules that hold a local reference — not just ernie/ace_step.
    # Qwen and other models import optimized_attention at load time and hold
    # their own reference — comfy_attn patch alone does not reach them.
    patched_count = 0
    try:
        for mod_name, mod in list(sys.modules.items()):
            try:
                # Use __dict__ to avoid triggering __getattr__ in HuggingFace
                # modules that define optimized_attention as a deprecated alias
                if "optimized_attention" in getattr(mod, "__dict__", {}):
                    _patch_module(mod)
                    patched_count += 1
            except Exception:
                pass
        log.debug(f"[SAD] global patch — {patched_count} modules patched")
    except Exception as e:
        log.warning(f"[SAD] global patch sys.modules scan failed: {e}")

    _GLOBAL_PATCH_ACTIVE = True


# ---------------------------------------------------------------------------
# Node class
# ---------------------------------------------------------------------------

class SmartAttentionDispatcher:
    """
    Detects GPU and software environment, then patches the model to use
    the optimal attention kernel automatically.

    Inputs
    ------
    model       : MODEL
    sdpa_kernel : bool  — force SDPA, overrides all SA settings
    sa2_kernel  : str   — SA2 kernel dropdown
    combine     : bool  — combine SA2+SA3 dynamic mode
    sa3_kernel  : str   — SA3 kernel dropdown

    Outputs
    -------
    model : MODEL — patched model with attention override applied

    Notes
    -----
    Global patch (sys.modules scan) is applied automatically — no user toggle needed.
    Tensor shape logging is available via Python logging at DEBUG level.
    """

    CATEGORY    = _CATEGORY
    FUNCTION    = "execute"
    DESCRIPTION = _DESCRIPTION
    MIN_WIDTH   = _MIN_WIDTH

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)

    @classmethod
    def INPUT_TYPES(cls) -> dict:
        return {
            "required": {
                "model": ("MODEL",),
                "sdpa_kernel": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "Force PyTorch SDPA. Overrides all SA2/SA3 settings.",
                    },
                ),
                "sa2_kernel": (
                    _SA2_MODES,
                    {
                        "default": _SA2_DISABLE,
                        "tooltip": (
                            "SA2 kernel: disable | auto (best for GPU) | "
                            "fp16 (Turing/Ampere) | fp8 (Ada+) | "
                            "fp8++ (Ada+, fp32+fp16 accum) | triton (fallback)."
                        ),
                    },
                ),
                "combine": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": (
                            "Combine SA2 and SA3: SA2 on first and last step, "
                            "SA3 on middle steps. Set sa2_kernel=disable for "
                            "SDPA-SA3-SDPA mode."
                        ),
                    },
                ),
                "sa3_kernel": (
                    _SA3_MODES,
                    {
                        "default": _SA3_DISABLE,
                        "tooltip": (
                            "SA3 kernel: disable | standard (per_block_mean=False) | "
                            "per_block_mean (per_block_mean=True)."
                        ),
                    },
                ),
            },
            "hidden": {"unique_id": "UNIQUE_ID"},
        }

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def execute(self, model, sdpa_kernel: bool, sa2_kernel: str,
                combine: bool, sa3_kernel: str, unique_id=None) -> dict:
        """
        Parameters
        ----------
        model       : ComfyUI model patcher
        sdpa_kernel : force SDPA — overrides all SA settings
        sa2_kernel  : SA2 mode dropdown
        combine     : combine SA2+SA3 dynamic mode
        sa3_kernel  : SA3 mode dropdown

        Returns
        -------
        dict with ui and result keys
        """
        # Global patch always applied — covers Qwen, Ernie, ACE-Step and any
        # model that imports optimized_attention locally at load time
        patch_global = True
        # 1. Detect environment (cached per session)
        env = _get_env()

        # 2. Detect model architecture and attention parameters
        params = _detect_model_params(model)
        hd     = params["head_dim"]
        heads  = params["num_heads"]
        arch   = params["arch"]

        # 3. Resolve effective mode by priority rules
        sa2_k        = ""   # default — overridden by each branch below
        real_eff     = _EFF_SDPA
        config_error = ""

        # Priority 1: sdpa_kernel=True → always SDPA
        if sdpa_kernel:
            real_eff = _EFF_SDPA
            sa2_k    = ""

        # Priority 2: combine=True validation
        elif combine:
            if sa3_kernel == _SA3_DISABLE:
                real_eff     = _EFF_SDPA
                sa2_k        = ""
                config_error = "combine requires sa3_kernel to be selected >>> SDPA"
            else:
                if sa2_kernel == _SA2_DISABLE:
                    # SDPA-SA3-SDPA mode
                    sa2_k = "sdpa"
                    if not env.has_sa3():
                        real_eff = _EFF_SDPA
                        sa2_k    = ""
                    else:
                        real_eff = _EFF_DYNAMIC
                else:
                    if sa2_kernel == _SA2_AUTO:
                        sa2_k = env.best_sa2_kernel()
                    else:
                        sa2_k = _SA2_MODE_TO_KERNEL.get(sa2_kernel, _K_NONE)
                    if sa2_k == _K_NONE or not env.has_any_sa2():
                        real_eff     = _EFF_SDPA
                        sa2_k        = ""
                        config_error = "combine requires SA2 (or set sa2_kernel to disable for SDPA-SA3-SDPA) >>> SDPA"
                    elif not env.has_sa3():
                        real_eff = _EFF_SA2
                    else:
                        real_eff = _EFF_DYNAMIC

        # Priority 3: SA3 only (sa3_kernel selected, sa2_kernel disabled, combine=False)
        elif sa3_kernel != _SA3_DISABLE and sa2_kernel == _SA2_DISABLE:
            if not env.has_sa3():
                real_eff = _EFF_SDPA
            else:
                real_eff = _EFF_SA3

        # Priority 4: SA2 only
        elif sa2_kernel != _SA2_DISABLE:
            if sa2_kernel == _SA2_AUTO:
                sa2_k = env.best_sa2_kernel()
            else:
                sa2_k = _SA2_MODE_TO_KERNEL.get(sa2_kernel, _K_NONE)
            if sa2_k == _K_NONE or not env.has_any_sa2():
                real_eff = _EFF_SDPA
                sa2_k    = ""
            else:
                real_eff = _EFF_SA2

        # Priority 5: everything disabled → SDPA
        else:
            real_eff = _EFF_SDPA
            sa2_k    = ""

        # 4. Architecture constraints
        # SA3 is untested on UNet (SDXL) — force fallback to SA2 or SDPA.
        # SA2 on UNet is supported (hd=64, heads=10/20 confirmed for SDXL).
        if arch == "unet":
            if real_eff == _EFF_SA3:
                # SA3 only mode → fallback to SA2 if available, else SDPA
                if env.has_any_sa2():
                    sa2_k    = env.best_sa2_kernel()
                    real_eff = _EFF_SA2
                else:
                    real_eff = _EFF_SDPA
                    sa2_k    = ""
            elif real_eff == _EFF_DYNAMIC:
                # Dynamic (SA2+SA3) → drop SA3, use SA2 only
                real_eff = _EFF_SA2

        # 4b. SA3 + hd not in valid set → fallback
        if real_eff == _EFF_DYNAMIC and hd is not None and hd not in _SA3_VALID_HD:
            if sa2_kernel == _SA2_DISABLE:
                real_eff = _EFF_SDPA
                sa2_k    = ""
            else:
                sa2_k    = env.best_sa2_kernel()
                real_eff = _EFF_SA2

        # 5. patch_global always True — SA3 in dynamic mode still supported
        # since we patch both transformer_options AND sys.modules globally.
        # No SA3 restriction needed here — wrapper handles it correctly.

        # 6. Clone model
        model_clone = model.clone()
        model_clone.model_options.setdefault("transformer_options", {})

        # 7. Build wrapper and inject override
        use_pbm = (sa3_kernel == _SA3_PBM_MODE)
        wrapper = _build_wrapper(real_eff, sa2_k, use_pbm)

        # Always restore previous global patch first
        if _GLOBAL_PATCH_ACTIVE:
            _restore_global_patch()

        if real_eff == _EFF_SDPA:
            # Pure SDPA mode — no patch needed.
            # Global patch already restored above. Only inject transformer_options
            # so ComfyUI override chain is also cleared.
            model_clone.model_options["transformer_options"].pop(
                "optimized_attention_override", None
            )
        else:
            # SA2 / SA3 / Dynamic — apply global patch (covers Qwen, Ernie,
            # ACE-Step) and inject transformer_options override.
            _apply_global_patch(wrapper)
            model_clone.model_options["transformer_options"][
                "optimized_attention_override"
            ] = wrapper

        # 8. Build status and log
        if sdpa_kernel:
            req_mode = "SDPA"
        elif combine:
            req_mode = "SDPA-SA3-SDPA" if sa2_kernel == _SA2_DISABLE else "SA2-SA3-SA2"
        elif sa3_kernel != _SA3_DISABLE and sa2_kernel == _SA2_DISABLE:
            req_mode = "SA3"
        elif sa2_kernel != _SA2_DISABLE:
            req_mode = "SA2"
        else:
            req_mode = "SDPA"

        status = env.format_status(
            req_mode, real_eff, hd, heads, arch,
        )

        if config_error:
            lines = status.split("\n")
            lines[0] = f"Mode: ERROR — {config_error}"
            status = "\n".join(lines)

        console_mode = status.split("\n")[0].replace("Mode: ", "").strip()
        # Kernel detail — console only, not shown in node UI
        k_parts = []
        if sa2_k and sa2_k not in ("", "sdpa", _K_NONE):
            k_parts.append(f"sa2={sa2_k}")
        if real_eff in (_EFF_SA3, _EFF_DYNAMIC) and sa3_kernel != _SA3_DISABLE:
            k_parts.append(f"sa3={sa3_kernel}")
        kernel_detail = f"  [{', '.join(k_parts)}]" if k_parts else ""
        print(f"[SAD v{_VERSION}] {console_mode}{kernel_detail}")

        # Send instant UI update via WebSocket
        if unique_id is not None:
            try:
                from server import PromptServer
                PromptServer.instance.send_sync(
                    "rogala/sad_status", {"node": unique_id, "text": status}
                )
            except Exception:
                pass

        # 9. Return
        return {
            "ui"    : {"text": (status,)},
            "result": (model_clone,),
        }


# ---------------------------------------------------------------------------
# Web extensions
# ---------------------------------------------------------------------------

WEB_DIRECTORY = "./js"

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS: dict = {
    _NODE_NAME: SmartAttentionDispatcher,
}

NODE_DISPLAY_NAME_MAPPINGS: dict = {
    _NODE_NAME: "Smart Attention Dispatcher",
}
