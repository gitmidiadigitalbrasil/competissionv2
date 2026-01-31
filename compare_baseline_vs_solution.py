#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SERVER-EQUIVALENT COMPARATOR (baseline vs solution)

O que ele faz:
- Lê samples .bin (raw BF16 bytes)
- Roda compress/decompress em solution_baseline.py e solution.py
- Calcula métricas "server-like" por slices aleatórios:
    proxy  = cosine
    norm   = min(||a||,||b||)/max(||a||,||b||)
    combined = proxy * norm
- Reporta WORST (size/off) por arquivo e o pior global
- Gate final: solution minStressCombined >= baseline minStressCombined  => deve evitar ZERO

Vars:
- APEX_SAMPLES_DIR (opcional): diretório de samples
- APEX_STRESS_RANDOM_SLICES (default 128): número de slices aleatórios por arquivo
- APEX_STRESS_MIN_SIZE (default 64)
- APEX_STRESS_MAX_SIZE (default 8192)
- APEX_STRESS_SEED (default 12345)
"""

import os
import sys
import glob
import time
import importlib.util
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch


# ------------------------
# load modules
# ------------------------

def _load_module(mod_name: str, path: str):
    spec = importlib.util.spec_from_file_location(mod_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module from {path}")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ------------------------
# samples dir
# ------------------------

def _find_samples_dir() -> str:
    env = os.getenv("APEX_SAMPLES_DIR", "").strip()
    cands = []
    if env:
        cands.append(env)
    cands += ["./samples", os.path.join(os.getcwd(), "samples")]

    for d in cands:
        if d and os.path.isdir(d):
            if glob.glob(os.path.join(d, "*.bin")):
                return d
    for d in cands:
        if d and os.path.isdir(d):
            return d
    return ""


# ------------------------
# math metrics
# ------------------------

def _as_bf16_float(raw: bytes) -> torch.Tensor:
    # usa bytearray para evitar buffer non-writable
    if raw is None:
        raw = b""
    if len(raw) % 2 != 0:
        # não é BF16 alinhado; retorna float vazio para não explodir
        return torch.empty(0, dtype=torch.float32)
    return torch.frombuffer(bytearray(raw), dtype=torch.bfloat16).float()

def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 or b.numel() == 0:
        return 0.0
    m = min(a.numel(), b.numel())
    a = a[:m]; b = b[:m]
    na = torch.linalg.norm(a)
    nb = torch.linalg.norm(b)
    if na == 0 and nb == 0:
        return 1.0
    if na == 0 or nb == 0:
        return 0.0
    return float(torch.dot(a, b) / (na * nb))

def _norm_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 or b.numel() == 0:
        return 0.0
    m = min(a.numel(), b.numel())
    a = a[:m]; b = b[:m]
    na = torch.linalg.norm(a)
    nb = torch.linalg.norm(b)
    if na == 0 and nb == 0:
        return 1.0
    if na == 0 or nb == 0:
        return 0.0
    return float(torch.min(na, nb) / torch.max(na, nb))

def _combined(a: torch.Tensor, b: torch.Tensor) -> Tuple[float, float, float]:
    p = _cos(a, b)
    n = _norm_sim(a, b)
    return (p * n), p, n


# ------------------------
# eval one module on one sample
# ------------------------

@dataclass
class Worst:
    combined: float
    proxy: float
    norm: float
    size: int
    off: int

@dataclass
class RunRes:
    t_comp_ms: float
    t_decomp_ms: float
    worst_stress: Worst
    hard_worst: Optional[Worst]  # opcional (pode ficar None)

def _stress_scan(a: torch.Tensor, b: torch.Tensor, nslices: int, min_size: int, max_size: int, seed: int) -> Worst:
    m = min(a.numel(), b.numel())
    if m <= 0:
        return Worst(combined=0.0, proxy=0.0, norm=0.0, size=0, off=0)

    rng = np.random.default_rng(seed)
    worstC = 1.0
    worstP = 1.0
    worstN = 1.0
    worstSz = min_size
    worstOff = 0

    max_size_eff = max(min(max_size, m), min_size)

    for _ in range(nslices):
        sz = int(rng.integers(min_size, max_size_eff + 1))
        if sz > m:
            sz = m
        off = int(rng.integers(0, max(1, m - sz + 1)))
        aa = a[off:off+sz]
        bb = b[off:off+sz]
        c, p, n = _combined(aa, bb)
        if c < worstC:
            worstC, worstP, worstN = c, p, n
            worstSz, worstOff = sz, off

    return Worst(combined=worstC, proxy=worstP, norm=worstN, size=worstSz, off=worstOff)

def _hard_scan_fixed(a: torch.Tensor, b: torch.Tensor, chunks, n_offsets: int, seed: int) -> Worst:
    # HARD: min combined em tamanhos fixos e offsets aleatórios + offset 0
    m = min(a.numel(), b.numel())
    if m <= 0:
        return Worst(combined=0.0, proxy=0.0, norm=0.0, size=0, off=0)

    rng = np.random.default_rng(seed)
    offsets = set([0])
    for _ in range(max(0, n_offsets)):
        offsets.add(int(rng.integers(0, m)))

    worstC = 1.0
    worstP = 1.0
    worstN = 1.0
    worstSz = chunks[0] if chunks else 0
    worstOff = 0

    for sz in chunks:
        if sz <= 0:
            continue
        for off in offsets:
            if off >= m:
                off = off % m
            end = min(off + sz, m)
            aa = a[off:end]
            bb = b[off:end]
            c, p, n = _combined(aa, bb)
            if c < worstC:
                worstC, worstP, worstN = c, p, n
                worstSz, worstOff = sz, off

    return Worst(combined=worstC, proxy=worstP, norm=worstN, size=worstSz, off=worstOff)

def _run_module(mod, raw: bytes, nslices: int, min_size: int, max_size: int, seed: int,
                hard_chunks, hard_offsets: int) -> RunRes:
    # compress/decompress
    t0 = time.perf_counter()
    comp = mod.compress_data(raw)
    t1 = time.perf_counter()
    dec = mod.decompress_data(comp)
    t2 = time.perf_counter()

    a = _as_bf16_float(raw)
    b = _as_bf16_float(dec)

    worst_stress = _stress_scan(a, b, nslices=nslices, min_size=min_size, max_size=max_size, seed=seed)
    hard_worst = _hard_scan_fixed(a, b, chunks=hard_chunks, n_offsets=hard_offsets, seed=seed) if hard_chunks else None

    return RunRes(
        t_comp_ms=(t1 - t0) * 1000.0,
        t_decomp_ms=(t2 - t1) * 1000.0,
        worst_stress=worst_stress,
        hard_worst=hard_worst,
    )


# ------------------------
# main
# ------------------------

def main():
    nslices = int(os.getenv("APEX_STRESS_RANDOM_SLICES", "128") or "128")
    min_size = int(os.getenv("APEX_STRESS_MIN_SIZE", "64") or "64")
    max_size = int(os.getenv("APEX_STRESS_MAX_SIZE", "8192") or "8192")
    seed = int(os.getenv("APEX_STRESS_SEED", "12345") or "12345")

    # Reaproveita tuas knobs “hard” existentes, mas agora em combined/proxy/norm
    hard_chunks_env = os.getenv("APEX_HARD_CHUNKS", "").strip()
    if hard_chunks_env:
        hard_chunks = [int(x) for x in hard_chunks_env.split(",") if x.strip().isdigit()]
    else:
        hard_chunks = [1024, 2048, 4096, 8192]
    hard_offsets = int(os.getenv("APEX_HARD_RANDOM_OFFSETS", "256") or "256")

    samples_dir = _find_samples_dir()
    if not samples_dir:
        print("ERRO: não achei samples dir. Use APEX_SAMPLES_DIR=/caminho/para/samples")
        return 2

    bins = sorted(glob.glob(os.path.join(samples_dir, "*.bin")))
    if not bins:
        print(f"ERRO: nenhum .bin em {samples_dir}")
        return 2

    base_path = os.path.join(os.getcwd(), "solution_baseline.py")
    sol_path = os.path.join(os.getcwd(), "solution.py")
    if not os.path.isfile(base_path):
        print(f"ERRO: não achei {base_path}")
        return 2
    if not os.path.isfile(sol_path):
        print(f"ERRO: não achei {sol_path}")
        return 2

    baseline = _load_module("solution_baseline", base_path)
    solution = _load_module("solution_current", sol_path)
    for nm, mod in [("baseline", baseline), ("solution", solution)]:
        if not hasattr(mod, "compress_data") or not hasattr(mod, "decompress_data"):
            print(f"ERRO: {nm} não tem compress_data/decompress_data")
            return 2

    print("=== SERVER-EQUIVALENT COMPARATOR ===")
    print(f"samples_dir={samples_dir}  files={len(bins)}")
    print(f"stress_slices={nslices}  size=[{min_size},{max_size}] seed={seed}")
    print(f"hard_chunks={hard_chunks} hard_random_offsets={hard_offsets}")
    print("")

    base_min_stress = 1.0
    sol_min_stress = 1.0
    base_worst_tag = ""
    sol_worst_tag = ""

    for fp in bins:
        name = os.path.basename(fp)
        raw = open(fp, "rb").read()

        print(f"== {name} ==")
        try:
            rb = _run_module(baseline, raw, nslices, min_size, max_size, seed, hard_chunks, hard_offsets)
            rs = _run_module(solution,  raw, nslices, min_size, max_size, seed, hard_chunks, hard_offsets)
        except Exception as e:
            print(f"ERRO ao avaliar {name}: {e}")
            return 2

        # prints compatíveis com tua leitura (hardC/stressC + proxy)
        hb = rb.hard_worst
        hs = rs.hard_worst
        if hb and hs:
            print(f"BASE hardC={hb.combined:.6f} hardP={hb.proxy:.6f}  stressC={rb.worst_stress.combined:.6f} stressP={rb.worst_stress.proxy:.6f}")
            print(f"SOL  hardC={hs.combined:.6f} hardP={hs.proxy:.6f}  stressC={rs.worst_stress.combined:.6f} stressP={rs.worst_stress.proxy:.6f}")
        else:
            print(f"BASE stressC={rb.worst_stress.combined:.6f} stressP={rb.worst_stress.proxy:.6f}")
            print(f"SOL  stressC={rs.worst_stress.combined:.6f} stressP={rs.worst_stress.proxy:.6f}")

        # WORST info (mostra proxy e norm — raiz do problema)
        w = rs.worst_stress
        print(f"WORST: STRESS size={w.size} off={w.off} combined={w.combined:.6f} proxy={w.proxy:.6f} norm={w.norm:.6f}")
        print("")

        if rb.worst_stress.combined < base_min_stress:
            base_min_stress = rb.worst_stress.combined
            bw = rb.worst_stress
            base_worst_tag = f"{name} | STRESS size={bw.size} off={bw.off} combined={bw.combined:.6f} proxy={bw.proxy:.6f} norm={bw.norm:.6f}"

        if rs.worst_stress.combined < sol_min_stress:
            sol_min_stress = rs.worst_stress.combined
            sw = rs.worst_stress
            sol_worst_tag = f"{name} | STRESS size={sw.size} off={sw.off} combined={sw.combined:.6f} proxy={sw.proxy:.6f} norm={sw.norm:.6f}"

    print("===== FINAL GATE =====")
    print(f"baseline min stressCombined = {base_min_stress:.6f}  (worst: {base_worst_tag})")
    print(f"solution  min stressCombined = {sol_min_stress:.6f}  (worst: {sol_worst_tag})")

    if sol_min_stress + 1e-12 >= base_min_stress:
        print("✅ PASSA SERVER GATE — NÃO DEVE TOMAR ZERO")
        return 0
    else:
        print("❌ FALHA SERVER GATE — ALTO RISCO DE ZERO")
        return 1


if __name__ == "__main__":
    sys.exit(main())
