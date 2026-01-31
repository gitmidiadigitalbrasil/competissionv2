import os
import shutil
import argparse
import numpy as np

def f32_to_bf16_bytes(x_f32: np.ndarray) -> bytes:
    """Converte float32 -> bf16 bytes (truncation)."""
    u = x_f32.astype(np.float32).view(np.uint32)
    bf16 = (u >> 16).astype(np.uint16)
    return bf16.tobytes()

def write_bf16(path: str, x_f32: np.ndarray):
    raw = f32_to_bf16_bytes(x_f32)
    with open(path, "wb") as f:
        f.write(raw)
    return len(raw)

def pat_random_normal(n: int, seed: int, scale: float = 1.0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n, dtype=np.float32) * np.float32(scale)
    return x

def pat_random_uniform(n: int, seed: int, lo=-1.0, hi=1.0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(lo, hi, size=n).astype(np.float32)
    return x

def pat_sign_stripes(H: int, W: int, col_period: int = 17, col_scale_step: float = 0.07):
    # adversarial: alterna sinais e escalas por coluna
    x = np.zeros(H * W, dtype=np.float32)
    for r in range(H):
        for c in range(W):
            scale = np.float32(1.0 + (c % col_period) * col_scale_step)
            sign = np.float32(1.0 if ((r + c) & 1) == 0 else -1.0)
            x[r * W + c] = scale * sign
    return x

def pat_row_scale(H: int, W: int, row_period: int = 31, row_scale_step: float = 0.05):
    # row-wise amplitude modulation (pega min-cosine por linha)
    x = np.zeros(H * W, dtype=np.float32)
    for r in range(H):
        s = np.float32(1.0 + (r % row_period) * row_scale_step)
        # mistura sinal dentro da linha
        row = np.linspace(-1.0, 1.0, W, dtype=np.float32)
        if r & 1:
            row = -row
        x[r * W:(r + 1) * W] = row * s
    return x

def pat_col_scale(H: int, W: int, col_period: int = 29, col_scale_step: float = 0.06):
    # col-wise amplitude modulation (pega min-cosine por coluna)
    x = np.zeros(H * W, dtype=np.float32)
    base = np.linspace(-1.0, 1.0, H, dtype=np.float32)
    for c in range(W):
        s = np.float32(1.0 + (c % col_period) * col_scale_step)
        col = base.copy()
        if c & 1:
            col = -col
        x[c::W] = col * s
    return x

def pat_tile_checker(H: int, W: int, tile_h: int = 16, tile_w: int = 16):
    # padrão em tiles (pega métricas por janela/stride)
    x = np.zeros((H, W), dtype=np.float32)
    for r in range(H):
        for c in range(W):
            tr = (r // tile_h)
            tc = (c // tile_w)
            val = 1.0 if ((tr + tc) & 1) == 0 else -1.0
            # dentro do tile: rampa leve
            val *= (1.0 + 0.002 * (r % tile_h) + 0.001 * (c % tile_w))
            x[r, c] = np.float32(val)
    return x.reshape(-1)

def pat_sparse_spikes(n: int, seed: int, density: float = 0.002, spike: float = 50.0):
    rng = np.random.default_rng(seed)
    x = np.zeros(n, dtype=np.float32)
    k = max(1, int(n * density))
    idx = rng.choice(n, size=k, replace=False)
    signs = rng.choice([-1.0, 1.0], size=k).astype(np.float32)
    x[idx] = signs * np.float32(spike)
    return x

def pat_mixed_distribution(n: int, seed: int):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal(n, dtype=np.float32)
    # mistura heavy-tail
    heavy = (rng.standard_normal(n, dtype=np.float32) ** 3) * np.float32(0.3)
    mask = rng.uniform(0, 1, size=n).astype(np.float32) < np.float32(0.08)
    x = np.where(mask, heavy, x).astype(np.float32)
    return x

def gen_hidden409600(samples_dir: str, seed0: int):
    H, W = 512, 400
    n = H * W
    assert n * 2 == 409600

    items = [
        ("sign_stripes", pat_sign_stripes(H, W)),
        ("row_scale", pat_row_scale(H, W)),
        ("col_scale", pat_col_scale(H, W)),
        ("tile_checker", pat_tile_checker(H, W, 16, 16)),
        ("randomN", pat_random_normal(n, seed0, scale=1.0)),
        ("randomU", pat_random_uniform(n, seed0 + 1, -2.0, 2.0)),
        ("sparse_spikes", pat_sparse_spikes(n, seed0 + 2, density=0.002, spike=60.0)),
        ("mixed", pat_mixed_distribution(n, seed0 + 3)),
    ]

    for name, x in items:
        path = os.path.join(samples_dir, f"hidden_409600_512x400_{name}.bin")
        sz = write_bf16(path, x)
        print(f"wrote {path} bytes={sz}")

def gen_small_sanity(samples_dir: str, seed0: int):
    sizes = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    # sizes are BYTES, must be even
    for i, sz in enumerate(sizes):
        n = sz // 2  # bf16 elems
        x = pat_mixed_distribution(n, seed0 + 100 + i)
        path = os.path.join(samples_dir, f"sanity_{sz}.bin")
        out = write_bf16(path, x)
        print(f"wrote {path} bytes={out}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="samples", help="output folder")
    ap.add_argument("--reset", action="store_true", help="delete samples dir first")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    if args.reset and os.path.isdir(args.dir):
        shutil.rmtree(args.dir)

    os.makedirs(args.dir, exist_ok=True)

    gen_small_sanity(args.dir, args.seed)
    gen_hidden409600(args.dir, args.seed)

    print("\nOK. Now run:")
    print("  python local_test_hiddenlike.py")

if __name__ == "__main__":
    main()
