import torch
import numpy as np
from solution import compress_data, decompress_data

WINDOW = 256          # Apex usa janelas locais (128 / 256)
THRESHOLD = 0.99      # Mesmo threshold do erro

def sliding_cosine(a: torch.Tensor, b: torch.Tensor, w: int):
    mins = 1.0
    for i in range(0, len(a) - w + 1, w):
        x = a[i:i+w]
        y = b[i:i+w]
        na = torch.linalg.norm(x)
        nb = torch.linalg.norm(y)
        if na == 0 or nb == 0:
            c = 1.0 if na == nb else 0.0
        else:
            c = torch.dot(x, y) / (na * nb)
        mins = min(mins, float(c))
    return mins


def run_test(raw: bytes):
    comp = compress_data(raw)
    dec = decompress_data(comp)

    a = torch.frombuffer(raw, dtype=torch.bfloat16).float()
    b = torch.frombuffer(dec, dtype=torch.bfloat16).float()

    min_slide = sliding_cosine(a, b, WINDOW)
    print(f"slide_min={min_slide:.6f}")

    if min_slide < THRESHOLD:
        print("❌ REPRODUZIU ERRO DA TASK 317")
    else:
        print("✅ PASSARIA NA TASK 317")


if __name__ == "__main__":
    # usa exatamente um padrão instável (igual mixed.bin)
    rng = torch.randn(409600, dtype=torch.float32) * torch.linspace(0.1, 10, 409600)
    raw = rng.to(torch.bfloat16).view(torch.int16).numpy().tobytes()
    run_test(raw)
