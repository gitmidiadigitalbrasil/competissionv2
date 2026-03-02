#!/usr/bin/env python3
"""
Script de avaliação oficial para o problema de compressão (rodada 21).
Calcula:
- Similaridade: cos_similarity * norm_similarity (conforme runner oficial)
- task_score = max(0, 1 - compressed_size / original_size) se similaridade >= 0.99, senão 0
- Final score = mediana dos task_scores
"""

import os
import sys
import time
import argparse
import numpy as np
import torch

# Importa as funções de compressão/descompressão da solução
from solution import compress_data, decompress_data

# =============================================================================
# Métrica oficial de similaridade (exatamente como no runner)
# =============================================================================
def official_similarity(orig_bytes: bytes, rec_bytes: bytes, eps: float = 1e-12) -> float:
    """
    Similaridade estável (cos * norm_sim), mas com redução em float64 + eps + clamp
    para reduzir falso-negativo perto do limiar 0.99.
    """
    a = torch.frombuffer(bytearray(orig_bytes), dtype=torch.bfloat16).to(torch.float32)
    b = torch.frombuffer(bytearray(rec_bytes), dtype=torch.bfloat16).to(torch.float32)

    a64 = a.to(torch.float64)
    b64 = b.to(torch.float64)

    na = torch.linalg.norm(a64)
    nb = torch.linalg.norm(b64)

    if na <= eps:
        return 1.0 if nb <= eps else 0.0
    if nb <= eps:
        return 0.0

    cos = torch.dot(a64, b64) / (na * nb)

    diff = torch.linalg.norm(a64 - b64)
    norm_sim = 1.0 - (diff / na)

    norm_sim = torch.clamp(norm_sim, 0.0, 1.0)
    sim = cos * norm_sim
    sim = torch.clamp(sim, 0.0, 1.0)

    return float(sim.item())
# =============================================================================
# Processamento de um arquivo
# =============================================================================
def process_file(filepath: str, sim_threshold: float = 0.99):
    """
    Retorna (task_score, compression_ratio, similarity, time_taken)
    """
    with open(filepath, "rb") as f:
        original = f.read()

    if len(original) % 2 != 0:
        print(f"⚠️  Arquivo {filepath} tem tamanho ímpar – ignorando.", file=sys.stderr)
        return None, None, None, None

    t0 = time.time()
    compressed = compress_data(original)
    decompressed = decompress_data(compressed)
    dt = time.time() - t0

    if len(decompressed) != len(original):
        # Erro de tamanho: similaridade zero
        sim = 0.0
    else:
        sim = official_similarity(original, decompressed)

    ratio = len(compressed) / max(1, len(original))
    task_score = max(0.0, 1.0 - ratio) if sim >= sim_threshold else 0.0

    return task_score, ratio, sim, dt

# =============================================================================
# Principal
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Calcula o score oficial (mediana dos task_scores) para a rodada 21")
    parser.add_argument("--samples_dir", type=str, default="round21",
                        help="Diretório com os arquivos .bin (padrão: round21)")
    parser.add_argument("--limit", type=int, default=0,
                        help="Número máximo de arquivos a processar (0 = todos)")
    parser.add_argument("--sim_threshold", type=float, default=0.99,
                        help="Limiar de similaridade (padrão: 0.99)")
    parser.add_argument("--quiet", action="store_true",
                        help="Não imprime progresso detalhado")
    args = parser.parse_args()

    # Lista arquivos
    if not os.path.isdir(args.samples_dir):
        print(f"Erro: diretório '{args.samples_dir}' não encontrado.", file=sys.stderr)
        sys.exit(1)

    files = sorted([
        f for f in os.listdir(args.samples_dir)
        if os.path.isfile(os.path.join(args.samples_dir, f)) and f.endswith('.bin')
    ])
    if args.limit and args.limit < len(files):
        files = files[:args.limit]

    if not files:
        print("Nenhum arquivo .bin encontrado.")
        return

    print(f"Processando {len(files)} arquivos de {args.samples_dir}...\n")

    task_scores = []
    ratios = []
    sims = []
    times = []
    fails = []  # arquivos com similaridade abaixo do limiar

    total_start = time.time()

    for i, fn in enumerate(files, 1):
        fullpath = os.path.join(args.samples_dir, fn)
        score, ratio, sim, dt = process_file(fullpath, args.sim_threshold)

        if score is None:  # arquivo inválido
            continue

        task_scores.append(score)
        ratios.append(ratio)
        sims.append(sim)
        times.append(dt)

        if sim < args.sim_threshold:
            fails.append((fn, sim))

        if not args.quiet:
            status = "✓" if sim >= args.sim_threshold else "✗"
            print(f"[{i:4d}/{len(files)}] {fn[:40]:40} score={score:.6f}  ratio={ratio:.4f}  sim={sim:.6f}  tempo={dt:.3f}s {status}")

        if i % 50 == 0 and not args.quiet:
            median_sofar = float(np.median(task_scores)) if task_scores else 0
            print(f"   → mediana parcial: {median_sofar:.6f}")

    total_time = time.time() - total_start

    # Estatísticas finais
    print("\n" + "="*60)
    print("RESULTADO FINAL")
    print("="*60)

    if task_scores:
        task_scores_arr = np.array(task_scores)
        final_score = float(np.median(task_scores_arr))
        print(f"Arquivos válidos: {len(task_scores)}")
        print(f"FINAL SCORE (mediana dos task_scores): {final_score:.8f}")
        print(f"Média dos task_scores: {np.mean(task_scores_arr):.8f}")
        print(f"Desvio padrão: {np.std(task_scores_arr):.6f}")
        print(f"Mínimo: {np.min(task_scores_arr):.6f}")
        print(f"Máximo: {np.max(task_scores_arr):.6f}")
        print(f"Percentis: 10%={np.percentile(task_scores_arr, 10):.6f}  90%={np.percentile(task_scores_arr, 90):.6f}")

        if sims:
            print("\n--- Estatísticas de similaridade ---")
            sims_arr = np.array(sims)
            print(f"Similaridade: mediana={np.median(sims_arr):.6f}  média={np.mean(sims_arr):.6f}")
            print(f"  min={np.min(sims_arr):.6f}  max={np.max(sims_arr):.6f}")
            abaixo = np.sum(sims_arr < args.sim_threshold)
            print(f"  abaixo de {args.sim_threshold}: {abaixo} ({abaixo/len(sims_arr)*100:.1f}%)")
            if fails:
                print("\n  Piores casos (similaridade mais baixa):")
                fails_sorted = sorted(fails, key=lambda x: x[1])[:10]
                for fn, s in fails_sorted:
                    print(f"    {fn}: {s:.6f}")

        if ratios:
            print("\n--- Estatísticas de compression ratio ---")
            ratios_arr = np.array(ratios)
            print(f"Ratio: mediana={np.median(ratios_arr):.6f}  média={np.mean(ratios_arr):.6f}")
            print(f"  min={np.min(ratios_arr):.6f}  max={np.max(ratios_arr):.6f}")

        if times:
            print("\n--- Tempo de processamento ---")
            times_arr = np.array(times)
            print(f"Tempo por arquivo: mediana={np.median(times_arr):.3f}s  média={np.mean(times_arr):.3f}s")
            print(f"  min={np.min(times_arr):.3f}s  max={np.max(times_arr):.3f}s")
            print(f"Tempo total: {total_time:.2f}s (~{total_time/len(times):.3f}s/arquivo)")
    else:
        print("Nenhum task_score válido foi gerado.")

if __name__ == "__main__":
    main()