# pre_submit_check.py
import importlib
import sys

print("🔍 Pre-submit: import solution")

try:
    sol = importlib.import_module("solution")
except Exception as e:
    print("❌ ERRO: não foi possível importar `solution`")
    print(e)
    sys.exit(1)

required = ["compress_data", "decompress_data"]

missing = [fn for fn in required if not hasattr(sol, fn)]
if missing:
    print("❌ ERRO: funções obrigatórias ausentes:", missing)
    sys.exit(1)

print("✅ OK: solution.py importado e funções encontradas")
