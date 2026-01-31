#!/usr/bin/env python3
import os, time, struct
import numpy as np
import torch
import brotli

# Prefer APEX submission layout: solution.py in cwd
try:
    import solution as solution  # expected in APEX submission layout
except Exception:
    # fallback for local experiments (only if you actually have v2solution.py)
    import v2solution as solution  # type: ignore

SAMPLES_DIR = "samples"
MAGIC12 = b"BF12"
MAGIC14 = b"BF14"
MAGICRAW = b"RAWB"

# Guard used to emulate APEX "zero if below" behavior
REQUIRE = float(os.environ.get("APEX_REQUIRE", "0.99"))

# If set to 1, proxy=min(combined, chunk8192, chunk4096). Default (0) uses only
# chunk-level proxies: proxy=min(chunk8192, chunk4096).
USE_COMBINED_IN_PROXY = os.environ.get("APEX_USE_COMBINED_IN_PROXY", "0") not in (
    "0", "false", "False", "no", "NO", ""
)

# Informational only sanity checks (do NOT affect PASS/FAIL unless you decide to)
CHECK_PREFIX = os.environ.get("APEX_CHECK_PREFIX", "0") not in ("0", "false", "False", "no", "NO", "")
CHECK_LEN = os.environ.get("APEX_CHECK_LEN", "1") not in ("0", "false", "False", "no", "NO", "")

def parse_payload_info(comp_bytes: bytes):
    payload = brotli.decompress(comp_bytes)
    if len(payload) < 4:
        return ("UNK", None, len(payload))
    magic = payload[:4]
    if magic in (MAGIC12, MAGIC14):
        if len(payload) < 10:
            return (magic.decode("ascii"), None, len(payload))
        bs = struct.unpack_from("<H", payload, 8)[0]
        return (magic.decode("ascii"), bs, len(payload))
    if magic == MAGICRAW:
        return ("RAWB", None, len(payload))

    # If payload isn't tagged, but looks like raw BF16 (even length), label it explicitly.
    # This does NOT change any metrics; it's only for readability in the log.
    if (len(payload) % 2) == 0 and len(payload) >= 2:
        return ("RAWBF16", None, len(payload))

    return ("RAW", None, len(payload))

@torch.no_grad()
def cosine_full(a: torch.Tensor, b: torch.Tensor) -> float:
    na = torch.linalg.norm(a)
    nb = torch.linalg.norm(b)
    if na.item() == 0.0 and nb.item() == 0.0:
        return 1.0
    if na.item() == 0.0 or nb.item() == 0.0:
        return 0.0
    return float((torch.dot(a, b) / (na * nb)).item())

@torch.no_grad()
def min_cosine_chunked(a: torch.Tensor, b: torch.Tensor, chunk: int, offsets=(0, 1024, 2048, 4096)) -> float:
    n = min(a.numel(), b.numel())
    if n == 0:
        return 1.0
    worst = 1.0
    for off in offsets:
        if off >= n:
            continue
        aa = a[off:n]
        bb = b[off:n]
        m = min(aa.numel(), bb.numel())
        aa = aa[:m]
        bb = bb[:m]
        for i in range(0, m, chunk):
            c = cosine_full(aa[i:i + chunk], bb[i:i + chunk])
            if c < worst:
                worst = c
                if worst < (REQUIRE - 0.01):
                    return worst
    return worst

@torch.no_grad()
def combined_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    # matches "top score" style: cosine * (min_norm/max_norm)
    na = torch.linalg.norm(a)
    nb = torch.linalg.norm(b)
    if na.item() == 0.0 and nb.item() == 0.0:
        return 1.0
    if na.item() == 0.0 or nb.item() == 0.0:
        return 0.0
    cos = float((torch.dot(a, b) / (na * nb)).item())
    nmin = float(torch.minimum(na, nb).item())
    nmax = float(torch.maximum(na, nb).item())
    ratio = (nmin / nmax) if nmax != 0.0 else 0.0
    return float(cos * ratio)

def mse_maxabs(orig_bytes: bytes, dec_bytes: bytes):
    o = torch.frombuffer(bytearray(orig_bytes), dtype=torch.bfloat16).float()
    d = torch.frombuffer(bytearray(dec_bytes), dtype=torch.bfloat16).float()
    m = min(o.numel(), d.numel())
    o = o[:m]
    d = d[:m]
    diff = o - d
    mse = float((diff * diff).mean().item()) if m else 0.0
    max_abs = float(diff.abs().max().item()) if m else 0.0
    return mse, max_abs

def apexlike_metrics(orig_bytes: bytes, dec_bytes: bytes):
    o = torch.frombuffer(bytearray(orig_bytes), dtype=torch.bfloat16).float()
    d = torch.frombuffer(bytearray(dec_bytes), dtype=torch.bfloat16).float()
    m = min(o.numel(), d.numel())
    o = o[:m]
    d = d[:m]

    cos_bf16 = cosine_full(o, d)
    cos_fp16 = cosine_full(o.half().float(), d)

    c8192 = min_cosine_chunked(o, d, 8192)
    c4096 = min_cosine_chunked(o, d, 4096)

    comb = combined_similarity(o, d)

    proxy = float(min(c8192, c4096))
    if USE_COMBINED_IN_PROXY:
        proxy = float(min(proxy, comb))

    return {
        "proxy": proxy,
        "bf16": cos_bf16,
        "fp16": cos_fp16,
        "combined": comb,
        "chunk8192": c8192,
        "chunk4096": c4096,
    }

def apex_score(sim: float, csize: int, osize: int) -> float:
    if sim < REQUIRE:
        return 0.0
    return 1.0 - (csize / float(osize))

def test_one(path: str):
    raw = open(path, "rb").read()
    t0 = time.time()
    c = solution.compress_data(raw)
    t1 = time.time()
    d = solution.decompress_data(c)
    t2 = time.time()

    # IMPORTANT: enc is only available after parsing payload.
    enc, bs, inner = parse_payload_info(c)

    # optional roundtrip sanity checks (informational only)
    # prefix check only makes sense for RAW-like encoders (lossless)
    if CHECK_PREFIX and enc.startswith("RAW"):
        prefix_ok = (d[:32] == raw[:32])
    else:
        prefix_ok = True  # not applicable for lossy encoders (BF14/BF12)

    len_ok = (len(d) == len(raw)) if CHECK_LEN else True

    met = apexlike_metrics(raw, d)
    sim = met["proxy"]

    csize = len(c)
    osize = len(raw)
    ratio = csize / float(osize)
    eff = 1.0 - ratio
    score = apex_score(sim, csize, osize)
    mse, max_abs = mse_maxabs(raw, d)

    ok = "PASS" if sim >= REQUIRE else "FAIL"
    msg = (
        f"{os.path.basename(path)} enc={enc}({bs}) "
        f"ratio={ratio:.4f} eff={eff:.4f} "
        f"proxySim={sim:.6f} "
        f"bf16={met['bf16']:.6f} fp16={met['fp16']:.6f} "
        f"combined={met['combined']:.6f} "
        f"chunk8192={met['chunk8192']:.6f} chunk4096={met['chunk4096']:.6f} "
        f"score={score:.6f} csize={csize} inner={inner} osize={osize} "
        f"mse={mse:.3e} max_abs={max_abs:.6f}"
    )
    if CHECK_PREFIX:
        msg += f" prefixOK={int(prefix_ok)}"
    if CHECK_LEN:
        msg += f" lenOK={int(len_ok)}"
    msg += f" t(ms)={(t2-t0)*1000:.2f} (c={(t1-t0)*1000:.2f}, d={(t2-t1)*1000:.2f}) {ok}"
    print(msg)
    return ratio, sim, score

def main():
    files = [os.path.join(SAMPLES_DIR, f) for f in sorted(os.listdir(SAMPLES_DIR)) if f.endswith(".bin")]
    if not files:
        raise SystemExit("No .bin samples in ./samples")

    ratios, sims, scores = [], [], []
    for p in files:
        r, s, sc = test_one(p)
        ratios.append(r)
        sims.append(s)
        scores.append(sc)

    print("\nResumo (apex-like v2):")
    print(f"ratio médio: {sum(ratios)/len(ratios):.4f}")
    print(f"proxySim médio: {sum(sims)/len(sims):.6f}  min: {min(sims):.6f}  max: {max(sims):.6f}")
    print(f"score médio: {sum(scores)/len(scores):.6f}  min: {min(scores):.6f}")

    print("\n" + ("✅ PASS proxy (>=%.2f)" % REQUIRE if min(sims) >= REQUIRE else "❌ FAIL proxy (<%.2f)" % REQUIRE))

if __name__ == "__main__":
    main()
