import warnings
warnings.filterwarnings("ignore", message=".*non-writable.*")
import torch
import torch.nn.functional as F
import uvicorn
import argparse
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
import traceback
import brotli
import struct
import numpy as np

try:
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
except Exception:
    pass

# ── 配置参数 ──────────────────────────────────────────────────────────────────
LARGE_THRESHOLD = 2_000_000   # 大/小文件分界线 (字节)
BROTLI_Q_SMALL  = 9           # 小文件 Brotli 质量
BROTLI_Q_LARGE  = 2           # 大文件 Brotli 质量
BROTLI_MODE     = 2           # Brotli 模式 (2=binary/font)
SIM_THRESHOLD   = 0.99        # 相似度阈值

CANDIDATES = [
            (0, 8192 * 64),
            (0, 8192 * 32),
            (0, 8192 * 16),
            (0, 8192),
            (0.0005, 8192),
            (0.001, 8192),
            (0.001, 600),
            (0.005, 600),
            (0.005, 512),
            (0.005, 1024),
            (0.005, 2048),
            (0.0003, 2048)
        ]

_MAX_OUTLIER_RATIO = max(r for r, _ in CANDIDATES)



def compress_chunk(tensor_chunk: torch.Tensor, brotli_quality: int = BROTLI_Q_SMALL) -> bytes:
    """压缩单个数据块 (性能优化版)."""
    try:
        n = tensor_chunk.numel()
        chunk_f32 = tensor_chunk.float()

        # ── 一次性 topk (abs + topk 只做一次) ─────────────────────────
        k_max = int(n * _MAX_OUTLIER_RATIO)
        if k_max > 0:
            topk_vals, topk_idx_raw = torch.topk(chunk_f32.abs(), k_max)
        else:
            topk_idx_raw = None

        # 预计算 float64 原始向量 + norm (循环外只做一次)
        a64 = chunk_f32.double()
        norm_a = torch.linalg.norm(a64).item()

        best_payload = None
        best_size = float('inf')

        for ci, (outlier_ratio, block_size) in enumerate(CANDIDATES):
            k = int(n * outlier_ratio)

            # ── 1. 离群值 (从预计算 topk 切片 + 排序) ────────────────
            if k > 0 and topk_idx_raw is not None:
                sub_idx, _ = torch.sort(topk_idx_raw[:k])
                oi = sub_idx.to(torch.int32)
                oi_long = oi.long()
                ov_bf16 = chunk_f32[oi_long].to(torch.bfloat16)
            else:
                oi = oi_long = None
                ov_bf16 = None
                k = 0

            # ── 2. 就地零化 (避免 clone) ─────────────────────────────
            if k > 0:
                saved = chunk_f32[oi_long].clone()
                chunk_f32[oi_long] = 0.0

            # ── 3. 分块量化 ──────────────────────────────────────────
            padding = (block_size - (n % block_size)) % block_size
            xp = F.pad(chunk_f32, (0, padding)) if padding > 0 else chunk_f32
            nb = xp.numel() // block_size
            blocks = xp.view(nb, block_size)

            sc_bf16 = blocks.abs().max(dim=1).values.to(torch.bfloat16)
            sc_f32 = sc_bf16.float()
            sc_safe = sc_f32.clone()
            sc_safe[sc_safe == 0] = 1.0

            q = blocks / sc_safe.unsqueeze(1) * 127.0
            q.round_().clamp_(-127, 127)
            qi8 = q.to(torch.int8)

            # ── 4. 恢复 outliers ─────────────────────────────────────
            if k > 0:
                chunk_f32[oi_long] = saved

            # ── 5. 相似度检查 ────────────────────────────────────────
            rec = (qi8.float() / 127.0 * sc_f32.unsqueeze(1)).view(-1)[:n]
            if k > 0:
                rec[oi_long] = ov_bf16.float()
            b64 = rec.to(torch.bfloat16).double()
            if norm_a == 0:
                score = 1.0
            else:
                nb64 = torch.linalg.norm(b64).item()
                if nb64 == 0:
                    score = 0.0
                else:
                    cos = (torch.dot(a64, b64) / (norm_a * nb64)).item()
                    rel = torch.linalg.norm(a64 - b64).item() / norm_a
                    score = cos * (1.0 - rel)
            if score < SIM_THRESHOLD:
                continue

            # ── 6. 达标! 序列化 + 压缩 + 立即返回 ───────────────────
            if k > 0:
                idx_np = oi.numpy()
                delta = np.empty(k, dtype=np.int32)
                delta[0] = idx_np[0]
                delta[1:] = idx_np[1:] - idx_np[:-1]
                idx_b = delta.tobytes()
                val_b = ov_bf16.view(torch.int16).numpy().tobytes()
            else:
                idx_b = val_b = b""

            sr = sc_bf16.view(torch.int16).numpy().view(np.uint8).reshape(-1, 2)
            sc_b = np.concatenate([sr[:, 1], sr[:, 0]]).tobytes()

            qn = qi8.view(-1)
            if qn.numel() % 2 != 0:
                qn = torch.cat([qn, torch.zeros(1, dtype=torch.int8)])
            qu8 = qn.view(torch.uint8)
            hi, lo = qu8 >> 4, qu8 & 0x0F
            ph = ((hi[0::2] << 4) | hi[1::2]).numpy().tobytes()
            pl = ((lo[0::2] << 4) | lo[1::2]).numpy().tobytes()

            hdr = struct.pack('<iiiii', n, k, block_size, len(ph), len(pl))
            payload = hdr + (idx_b + val_b if k > 0 else b"") + sc_b + ph + pl

            return brotli.compress(payload, quality=brotli_quality, mode=BROTLI_MODE)

        return b""

    except Exception as e:
        raise ValueError(f"Chunk compression failed: {e}")

def decompress_chunk(compressed_chunk: bytes) -> torch.Tensor:
    """Decompress a single chunk using V37 algorithm (Hybrid: Outliers + Adaptive Blocks)."""
    try:
        if not compressed_chunk:
            return torch.empty(0, dtype=torch.bfloat16)
            
        decompressed = brotli.decompress(compressed_chunk)
        
        # 1. Header (5 integers = 20 bytes)
        # original_length, num_outliers, block_size, len(stream_high), len(stream_low)
        original_length, num_outliers, block_size, stream_high_len, stream_low_len = struct.unpack('<iiiii', decompressed[:20])
        offset = 20
        
        # 2. Outliers (Delta Encoding)
        if num_outliers > 0:
            indices_size = num_outliers * 4
            outlier_indices_delta = np.frombuffer(decompressed[offset : offset + indices_size], dtype=np.int32)
            outlier_indices = np.cumsum(outlier_indices_delta, dtype=np.int32)
            offset += indices_size
            
            vals_size = num_outliers * 2
            outlier_values = torch.frombuffer(bytearray(decompressed[offset : offset + vals_size]), dtype=torch.bfloat16)
            offset += vals_size
        else:
            outlier_indices = np.array([], dtype=np.int32)
            outlier_values = torch.empty(0, dtype=torch.bfloat16)

        # 3. Scales (Split bfloat16)
        # Calculate num_blocks based on dynamic block_size
        padded_length = (original_length + block_size - 1) // block_size * block_size
        num_blocks = padded_length // block_size
        
        if num_blocks > 0:
            # Read High and Low streams for scales
            s_high_bytes = decompressed[offset : offset + num_blocks]
            offset += num_blocks
            s_low_bytes = decompressed[offset : offset + num_blocks]
            offset += num_blocks
            
            # Reconstruct
            # Ensure bytes for frombuffer
            if not isinstance(s_high_bytes, bytes): s_high_bytes = bytes(s_high_bytes)
            if not isinstance(s_low_bytes, bytes): s_low_bytes = bytes(s_low_bytes)

            try:
                s_high = torch.frombuffer(s_high_bytes, dtype=torch.uint8)
                s_low = torch.frombuffer(s_low_bytes, dtype=torch.uint8)
                
                s_rec = torch.empty(num_blocks * 2, dtype=torch.uint8)
                s_rec[0::2] = s_low
                s_rec[1::2] = s_high
                
                scales = s_rec.view(torch.bfloat16).float()
            except Exception as e:
                raise ValueError(f"Scales reconstruction failed: {e}")
        else:
            scales = torch.empty(0, dtype=torch.float32)
        
        # 4. Streams (High/Low Nibbles)
        stream_high_bytes = decompressed[offset : offset + stream_high_len]
        if stream_high_len > 0:
            if len(stream_high_bytes) == 0:
                 raise ValueError(f"High stream empty but len {stream_high_len}")
            packed_high = torch.frombuffer(stream_high_bytes, dtype=torch.uint8) 
        else:
            packed_high = torch.empty(0, dtype=torch.uint8)
        offset += stream_high_len
        
        stream_low_bytes = decompressed[offset : offset + stream_low_len]
        if stream_low_len > 0:
            packed_low = torch.frombuffer(stream_low_bytes, dtype=torch.uint8)
        else:
            packed_low = torch.empty(0, dtype=torch.uint8)
            
        # 5. Unpack Nibbles
        total_elements = padded_length
        # Unpack High Stream
        # h_unpacked[0::2] = packed >> 4
        # h_unpacked[1::2] = packed & 0x0F
        
        if stream_high_len > 0:
            h_unpacked = torch.empty(stream_high_len * 2, dtype=torch.uint8)
            h_unpacked[0::2] = (packed_high >> 4)
            h_unpacked[1::2] = (packed_high & 0x0F)
            high = h_unpacked[:total_elements]
        else:
            high = torch.zeros(total_elements, dtype=torch.uint8)
            
        if stream_low_len > 0:
            l_unpacked = torch.empty(stream_low_len * 2, dtype=torch.uint8)
            l_unpacked[0::2] = (packed_low >> 4)
            l_unpacked[1::2] = (packed_low & 0x0F)
            low = l_unpacked[:total_elements]
        else:
            low = torch.zeros(total_elements, dtype=torch.uint8)
            
        # Combine
        q_uint8 = (high << 4) | low
        q_8 = q_uint8.view(torch.int8)
        
        # Reshape to (num_blocks, block_size)
        q_blocks = q_8.view(num_blocks, block_size)
        
        # 6. Dequantize
        # x_rec = q * scale / 127
        rec_float = (q_blocks.float() * scales.unsqueeze(1)) / 127.0
        
        # Flatten and crop
        rec_flat = rec_float.view(-1)
        tensor_rec = rec_flat[:original_length].to(torch.bfloat16)

        # 7. Apply Outliers
        if num_outliers > 0:
            # We used Delta Encoding for indices
            idx = outlier_indices.astype(np.int64)
            tensor_rec.view(-1)[idx] = outlier_values.float().to(torch.bfloat16)
            # wait, tensor_rec is bfloat16. outlier_values is bfloat16. good.
            # But tensor_rec assignment: tensor_rec is 1D? Yes.
            
        return tensor_rec

    except Exception as e:
        # import gc; gc.collect()
        raise ValueError(f"Chunk decompression failed: {str(e)}") from e
def compress_data(data: bytes) -> bytes:
    try:
        if not data:
            return b""
            
        # Global Chunking Strategy to prevent OOM
        # 32MB chunks (16M bfloat16 elements)
        ELEMENTS_PER_CHUNK = 16 * 1024 * 1024 
        BYTES_PER_CHUNK = ELEMENTS_PER_CHUNK * 2
        
        # 根据文件大小选择 Brotli 质量
        bq = BROTLI_Q_SMALL if len(data) <= LARGE_THRESHOLD else BROTLI_Q_LARGE
        
        full_tensor = torch.frombuffer(bytearray(data), dtype=torch.bfloat16)
        total_elements = full_tensor.numel()
        
        compressed_parts = []
        
        for i in range(0, total_elements, ELEMENTS_PER_CHUNK):
            chunk = full_tensor[i : i + ELEMENTS_PER_CHUNK]
            compressed_chunk = compress_chunk(chunk, brotli_quality=bq)
            
            # Simple framing: Length (4 bytes) + Chunk Data
            frame = struct.pack('<I', len(compressed_chunk)) + compressed_chunk
            compressed_parts.append(frame)
            
            # Force GC (optional, let's trust Python for now to be faster)
            del chunk
            
        # import gc; gc.collect()
            
        return b"".join(compressed_parts)

    except Exception as e:
        # import gc; gc.collect()
        raise ValueError(f"Compression failed: {str(e)}") from e


def decompress_data(data: bytes) -> bytes:
    try:
        if not data:
            return b""
            
        offset = 0
        total_len = len(data)
        decompressed_parts = []
        
        while offset < total_len:
            # Read Frame Length
            if offset + 4 > total_len:
                raise ValueError("Incomplete frame header")
                
            chunk_len = struct.unpack('<I', data[offset : offset + 4])[0]
            offset += 4
            
            if offset + chunk_len > total_len:
                raise ValueError("Incomplete frame body")
                
            chunk_data = data[offset : offset + chunk_len]
            offset += chunk_len
            
            decompressed_chunk = decompress_chunk(chunk_data)
            decompressed_parts.append(decompressed_chunk)
            
            # Streaming append would be better but list append + join is okay for byte strings
            
        if not decompressed_parts:
            return b""
            
        full_tensor = torch.cat(decompressed_parts)
        return full_tensor.view(torch.int16).numpy().tobytes()
        
    except Exception as e:
        import gc
        gc.collect()
        raise ValueError(f"Decompression failed: {str(e)}") from e


def _validate(data: bytes) -> dict:
    # Validate that data compresses and decompresses correctly
    # Returns: (is_valid, compression_efficiency, cosine_similarity)
    # Data represents torch bfloat16 values (2 bytes per element)
    input_tensor = torch.frombuffer(bytearray(data), dtype=torch.bfloat16)
    compressed = compress_data(data)
    decompressed = decompress_data(compressed)
    output_tensor = torch.frombuffer(bytearray(decompressed), dtype=torch.bfloat16)

    is_valid = torch.equal(input_tensor, output_tensor)
    compression_efficiency = 1 - (len(compressed) / len(data))

    # APEX similarity formula (float64 precision to match server)
    a = input_tensor.double()
    b = output_tensor.double()

    a_norm = torch.linalg.norm(a).item()
    b_norm = torch.linalg.norm(b).item()

    if a_norm == 0:
        cosine_similarity = 1.0 if b_norm == 0 else 0.0
        norm_similarity = 1.0 if b_norm == 0 else 0.0
    else:
        cosine_similarity = (torch.dot(a, b) / (a_norm * b_norm)).item() if b_norm != 0 else 0.0
        norm_similarity = 1 - torch.linalg.norm(a - b).item() / a_norm
    similarity = cosine_similarity * norm_similarity

    return {
        "is_valid": is_valid,
        "compression_efficiency": compression_efficiency,
        "cosine_similarity": float(cosine_similarity),
        "norm_similarity": float(norm_similarity),
        "similarity": float(similarity),
    }

def make_app() -> FastAPI:
    app = FastAPI(title="Matrix Compression Miner API")

    @app.get("/health")
    def health():
        return {"status": "healthy"}

    @app.post("/compress")
    async def compress(file: UploadFile = File(...)):
        try:
            data = await file.read()
            # Remove explicit 400 check for empty data to handle edge cases gracefully
            # if not data:
            #     raise HTTPException(status_code=400, detail="No data received in file")
            
            # Log data info for debugging (first 50 bytes)
            print(f"DEBUG: Received {len(data)} bytes")
            compressed = compress_data(data)
            return Response(content=compressed, media_type="application/octet-stream")
        except HTTPException:
            raise
        except Exception as e:
            error_detail = f"{str(e)}\n{traceback.format_exc()}"
            print(f"ERROR in /compress: {error_detail}")
            raise HTTPException(status_code=500, detail=error_detail)

    @app.post("/decompress")
    async def decompress(file: UploadFile = File(...)):
        try:
            data = await file.read()
            # Remove explicit 400 check
            # if not data:
            #     raise HTTPException(status_code=400, detail="No data received in file")
            
            # Log data info for debugging (first 50 bytes)
            print(f"DEBUG: Received {len(data)} bytes")
            decompressed = decompress_data(data)
            return Response(content=decompressed, media_type="application/octet-stream")
        except HTTPException:
            raise
        except Exception as e:
            error_detail = f"{str(e)}\n{traceback.format_exc()}"
            print(f"ERROR in /decompress: {error_detail}")
            raise HTTPException(status_code=500, detail=error_detail)

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    args = parser.parse_args()
    uvicorn.run(make_app(), host=args.host, port=args.port, log_level="info")
