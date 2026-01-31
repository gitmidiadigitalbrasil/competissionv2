import torch
import uvicorn
import argparse
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import Response
import traceback
import brotli
import struct
import numpy as np


# =========================
# BFP CONFIG (tune these)
# =========================
BFP_BLOCK_SIZE_DEFAULT = 209  # Default aggressive, fallback to 160 if needed
BROTLI_QUALITY = 10           # optimal: 11 causes timeout

MAGIC = b"BF14"               # BF14 format (bf16 scales, nibble-split int8)


def bfp_encode_bf16_bytes(data: bytes, block_size: int) -> bytes:
    """
    Encode bf16 bytes -> BFP payload bytes with byte-plane separation for scales
    and nibble separation for quantized payload.

    Payload format (all little-endian):
      MAGIC (4) = b"BF14"
      n_elems (u32)
      block_size (u16)
      scales (num_blocks * bf16, byte-plane separated)
      quantized (n_elems * int8, nibble-separated)
    """
    if len(data) % 2 != 0:
        raise ValueError("Input data length is not a multiple of 2 (bf16 bytes expected).")

    n_elems = len(data) // 2
    if n_elems == 0:
        raise ValueError("Empty bf16 matrix.")

    if block_size <= 0:
        raise ValueError("block_size must be positive.")

    # Read as bf16 then cast to float32 for quantization
    x_bf16 = torch.frombuffer(bytearray(data), dtype=torch.bfloat16)
    x = x_bf16.float()

    # Pad to multiple of block_size
    num_blocks = (n_elems + block_size - 1) // block_size
    padded_len = num_blocks * block_size
    if padded_len > n_elems:
        x = torch.nn.functional.pad(x, (0, padded_len - n_elems), value=0.0)

    # Reshape into blocks
    x_blocks = x.view(num_blocks, block_size)

    # Compute per-block scales (max absolute value)
    scales = x_blocks.abs().max(dim=1).values  # shape: (num_blocks,)

    # Use bfloat16 for scales
    scales_bf16 = scales.to(torch.bfloat16)
    
    # CRITICAL: Use the stored scale values for quantization
    scales_safe = scales_bf16.float().clone()
    scales_safe[scales_safe == 0] = 1.0

    # Quantize to [-127, 127] range (int8)
    Q = 127.0
    x_normalized = x_blocks / scales_safe.unsqueeze(1)  # normalize to [-1, 1]
    x_quantized = torch.round(x_normalized * Q).clamp(-127, 127).to(torch.int8)

    # Flatten and trim to original length
    x_quantized_flat = x_quantized.view(-1)[:n_elems]

    # Build header
    header = bytearray()
    header += MAGIC
    header += struct.pack("<I", n_elems)
    header += struct.pack("<H", block_size)

    # SCALES: Byte-plane separation
    scales_view = scales_bf16.view(torch.int16).numpy()
    scales_bytes = scales_view.tobytes()
    scales_arr = np.frombuffer(scales_bytes, dtype=np.uint8).reshape(-1, 2)
    scales_reordered = np.concatenate([scales_arr[:, i] for i in range(2)])

    # PAYLOAD: Nibble Splitting
    # Split int8 into high nibble stream and low nibble stream.
    # This groups the 'sign/magnitude' (high nibbles) together, which compresses very well.
    q_np = x_quantized_flat.numpy().view(np.uint8)
    
    # Handle odd length by padding temporarily
    n_q = q_np.size
    pad_byte = False
    if n_q % 2 != 0:
        q_np = np.append(q_np, 0)
        pad_byte = True

    # Reshape to pairs for packing
    pairs = q_np.reshape(-1, 2)
    
    # Extract nibbles (vectorized)
    # pair[0] -> h0, l0 | pair[1] -> h1, l1
    # stream_high = (h0 << 4) | h1
    # stream_low  = (l0 << 4) | l1
    
    h0 = pairs[:, 0] >> 4
    h1 = pairs[:, 1] >> 4
    high_packed = (h0 << 4) | h1
    
    l0 = pairs[:, 0] & 0x0F
    l1 = pairs[:, 1] & 0x0F
    low_packed = (l0 << 4) | l1
    
    payload = bytes(header) + scales_reordered.tobytes() + high_packed.tobytes() + low_packed.tobytes()
    return payload


def bfp_decode_to_bf16_bytes(payload: bytes) -> bytes:
    """
    Decode BFP payload bytes -> bf16 bytes.
    Vectorized for speed.
    """
    if len(payload) >= 4:
        magic = payload[:4]
        if magic == MAGIC:
            return _decode_bfp14(payload)
        if magic == b"BF12":
            return _decode_bfp12(payload)
        if magic == b"BF11":
            return _decode_bfp11(payload)
        if magic == b"BF10":
            return _decode_bfp10(payload)
        if magic == b"BFP9":
            return _decode_bfp9(payload)
        if magic == b"BFP6":
            return _decode_bfp6(payload)
        if magic == b"BFP2":
            return _decode_bfp2(payload)
        if magic == b"BFP1":
            return _decode_bfp1(payload)

    # If not our format, assume raw bf16 bytes
    if len(payload) % 2 == 0:
        return payload
    raise ValueError("Unknown payload format and not valid bf16 bytes.")


def _decode_bfp14(payload: bytes) -> bytes:
    """Decode BF14 format (bf16 byte-split scales + nibble-split int8)."""
    min_header = 4 + 4 + 2
    if len(payload) < min_header:
        raise ValueError("Payload too small to be valid BF14.")

    offset = 4
    n_elems = struct.unpack_from("<I", payload, offset)[0]
    offset += 4
    block_size = struct.unpack_from("<H", payload, offset)[0]
    offset += 2

    if n_elems == 0:
        return b""

    num_blocks = (n_elems + block_size - 1) // block_size

    # 1. Read scales (bf16, byte-plane reordered)
    scales_bytes_len = num_blocks * 2
    scales_end = offset + scales_bytes_len
    if scales_end > len(payload):
        raise ValueError("Payload ended early while reading scales.")
    
    scales_reordered = np.frombuffer(payload[offset:scales_end], dtype=np.uint8)
    scales_arr = np.zeros((num_blocks, 2), dtype=np.uint8)
    for i in range(2):
        scales_arr[:, i] = scales_reordered[i*num_blocks:(i+1)*num_blocks]
    
    scales_int16 = np.frombuffer(scales_arr.tobytes(), dtype=np.int16)
    scales_bf16 = torch.from_numpy(scales_int16.copy()).view(torch.bfloat16)
    scales = scales_bf16.float()
    offset = scales_end

    # 2. Read Quantized Data (Nibble Split)
    # We must calculate the size of the packed streams.
    # n_elems might be odd.
    n_packed = (n_elems + 1) // 2
    
    high_end = offset + n_packed
    low_end = high_end + n_packed
    if low_end > len(payload):
         raise ValueError("Payload ended early while reading quantized data.")

    high_packed = np.frombuffer(payload[offset:high_end], dtype=np.uint8)
    low_packed = np.frombuffer(payload[high_end:low_end], dtype=np.uint8)
    
    # Vectorized Unpack
    # high_packed[i] => h0, h1
    h0 = high_packed >> 4
    h1 = high_packed & 0x0F
    
    l0 = low_packed >> 4
    l1 = low_packed & 0x0F
    
    # Reconstruct bytes
    # byte0 = (h0 << 4) | l0
    # byte1 = (h1 << 4) | l1
    
    q0 = (h0 << 4) | l0
    q1 = (h1 << 4) | l1
    
    # Interleave
    # We create an array of size 2 * n_packed
    q_rec = np.empty(n_packed * 2, dtype=np.uint8)
    q_rec[0::2] = q0
    q_rec[1::2] = q1
    
    # Trim padding if necessary
    q_rec = q_rec[:n_elems]
    
    quantized = torch.from_numpy(q_rec.view(np.int8).copy()).float()

    # Pad quantized to multiple of block_size for reshaping
    padded_len = num_blocks * block_size
    if padded_len > n_elems:
        quantized = torch.nn.functional.pad(quantized, (0, padded_len - n_elems), value=0.0)

    # Reshape and dequantize
    Q = 127.0
    quantized_blocks = quantized.view(num_blocks, block_size)
    x_rec_blocks = (quantized_blocks / Q) * scales.unsqueeze(1)

    # Flatten and trim
    x_rec = x_rec_blocks.view(-1)[:n_elems]

    # Convert to bf16
    x_bf16 = x_rec.to(torch.bfloat16)
    return x_bf16.view(torch.int16).numpy().tobytes()


def _decode_bfp12(payload: bytes) -> bytes:
    """Decode BF12 format (transposed int8 + delta-bf16 scales)."""
    min_header = 4 + 4 + 2
    if len(payload) < min_header:
        raise ValueError("Payload too small to be valid BF12.")

    offset = 4
    n_elems = struct.unpack_from("<I", payload, offset)[0]
    offset += 4
    block_size = struct.unpack_from("<H", payload, offset)[0]
    offset += 2

    if n_elems == 0:
        return b""

    num_blocks = (n_elems + block_size - 1) // block_size

    # Read scales (bf16 delta, byte-plane reordered)
    scales_bytes_len = num_blocks * 2
    scales_end = offset + scales_bytes_len
    if scales_end > len(payload):
        raise ValueError("Payload ended early while reading scales.")
    
    # Reverse byte-plane
    scales_reordered = np.frombuffer(payload[offset:scales_end], dtype=np.uint8)
    scales_arr = np.zeros((num_blocks, 2), dtype=np.uint8)
    for i in range(2):
        scales_arr[:, i] = scales_reordered[i*num_blocks:(i+1)*num_blocks]
    
    # Reconstruct from Deltas
    scales_delta = np.frombuffer(scales_arr.tobytes(), dtype=np.int16)
    # cumsum to restore values. 
    scales_int16 = np.cumsum(scales_delta, dtype=np.int16)
    
    scales_bf16 = torch.from_numpy(scales_int16.copy()).view(torch.bfloat16)
    scales = scales_bf16.float()
    offset = scales_end

    # Read transposed quantized data (PADDED)
    padded_len = num_blocks * block_size
    quantized_end = offset + padded_len
    
    if quantized_end > len(payload):
         # Try reading non-padded length if padded read fails (legacy support?)
         # But BF12 was just introduced and failed, so it doesn't really matter.
         # Stick to strict reading.
         raise ValueError("Payload ended early while reading quantized data.")

    quantized_transposed = np.frombuffer(payload[offset:quantized_end], dtype=np.int8)
    quantized_transposed_t = torch.from_numpy(quantized_transposed.copy()).float()
    
    # Restore shape: (block_size, num_blocks) -> (num_blocks, block_size)
    quantized = quantized_transposed_t.view(block_size, num_blocks).t().contiguous()
    
    # Reshape and dequantize
    Q = 127.0
    x_rec_blocks = (quantized / Q) * scales.unsqueeze(1)

    # Flatten and trim
    x_rec = x_rec_blocks.view(-1)[:n_elems]

    # Convert to bf16
    x_bf16 = x_rec.to(torch.bfloat16)
    return x_bf16.view(torch.int16).numpy().tobytes()


def _decode_bfp11(payload: bytes) -> bytes:
    """Decode BF11 format (byte-split bf16 scales + int8 quantized)."""
    min_header = 4 + 4 + 2  # MAGIC + n_elems + block_size
    if len(payload) < min_header:
        raise ValueError("Payload too small to be valid BF11.")

    offset = 4  # skip MAGIC
    n_elems = struct.unpack_from("<I", payload, offset)[0]
    offset += 4
    block_size = struct.unpack_from("<H", payload, offset)[0]
    offset += 2

    if n_elems == 0:
        return b""

    num_blocks = (n_elems + block_size - 1) // block_size

    # Read scales (bf16, byte-plane reordered)
    scales_bytes_len = num_blocks * 2
    scales_end = offset + scales_bytes_len
    if scales_end > len(payload):
        raise ValueError("Payload ended early while reading scales.")
    
    # Reverse byte-plane reordering
    scales_reordered = np.frombuffer(payload[offset:scales_end], dtype=np.uint8)
    scales_arr = np.zeros((num_blocks, 2), dtype=np.uint8)
    for i in range(2):
        scales_arr[:, i] = scales_reordered[i*num_blocks:(i+1)*num_blocks]
    
    # Bytes -> int16 view -> bfloat16 -> float32
    scales_int16 = np.frombuffer(scales_arr.tobytes(), dtype=np.int16)
    # Important: Create a copy to ensure memory alignment for torch
    scales_bf16 = torch.from_numpy(scales_int16.copy()).view(torch.bfloat16)
    scales = scales_bf16.float()
    offset = scales_end

    # Read quantized values (int8, not transposed)
    quantized_end = offset + n_elems
    if quantized_end > len(payload):
        raise ValueError("Payload ended early while reading quantized data.")
    
    quantized = np.frombuffer(payload[offset:quantized_end], dtype=np.int8)
    quantized = torch.from_numpy(quantized.copy()).float()

    # Pad quantized to multiple of block_size for reshaping
    padded_len = num_blocks * block_size
    if padded_len > n_elems:
        quantized = torch.nn.functional.pad(quantized, (0, padded_len - n_elems), value=0.0)

    # Reshape and dequantize
    Q = 127.0
    quantized_blocks = quantized.view(num_blocks, block_size)
    x_rec_blocks = (quantized_blocks / Q) * scales.unsqueeze(1)

    # Flatten and trim
    x_rec = x_rec_blocks.view(-1)[:n_elems]

    # Convert to bf16
    x_bf16 = x_rec.to(torch.bfloat16)
    return x_bf16.view(torch.int16).numpy().tobytes()


def _decode_bfp10(payload: bytes) -> bytes:
    """Decode BF10 format (byte-split fp16 scales + int8 quantized)."""
    min_header = 4 + 4 + 2  # MAGIC + n_elems + block_size
    if len(payload) < min_header:
        raise ValueError("Payload too small to be valid BF10.")

    offset = 4  # skip MAGIC
    n_elems = struct.unpack_from("<I", payload, offset)[0]
    offset += 4
    block_size = struct.unpack_from("<H", payload, offset)[0]
    offset += 2

    if n_elems == 0:
        return b""

    num_blocks = (n_elems + block_size - 1) // block_size

    # Read scales (fp16, byte-plane reordered)
    scales_bytes_len = num_blocks * 2
    scales_end = offset + scales_bytes_len
    if scales_end > len(payload):
        raise ValueError("Payload ended early while reading scales.")
    
    # Reverse byte-plane reordering
    scales_reordered = np.frombuffer(payload[offset:scales_end], dtype=np.uint8)
    scales_arr = np.zeros((num_blocks, 2), dtype=np.uint8)
    for i in range(2):
        scales_arr[:, i] = scales_reordered[i*num_blocks:(i+1)*num_blocks]
    scales = np.frombuffer(scales_arr.tobytes(), dtype=np.float16)
    scales = torch.from_numpy(scales.copy().astype(np.float32)).float()
    offset = scales_end

    # Read quantized values (int8, not transposed)
    quantized_end = offset + n_elems
    if quantized_end > len(payload):
        raise ValueError("Payload ended early while reading quantized data.")
    
    quantized = np.frombuffer(payload[offset:quantized_end], dtype=np.int8)
    quantized = torch.from_numpy(quantized.copy()).float()

    # Pad quantized to multiple of block_size for reshaping
    padded_len = num_blocks * block_size
    if padded_len > n_elems:
        quantized = torch.nn.functional.pad(quantized, (0, padded_len - n_elems), value=0.0)

    # Reshape and dequantize
    Q = 127.0
    quantized_blocks = quantized.view(num_blocks, block_size)
    x_rec_blocks = (quantized_blocks / Q) * scales.unsqueeze(1)

    # Flatten and trim
    x_rec = x_rec_blocks.view(-1)[:n_elems]

    # Convert to bf16
    x_bf16 = x_rec.to(torch.bfloat16)
    return x_bf16.view(torch.int16).numpy().tobytes()


def _decode_bfp9(payload: bytes) -> bytes:
    """Decode BFP9 format (byte-split scales + int8 quantized)."""
    min_header = 4 + 4 + 2 + 4  # MAGIC + n_elems + block_size + num_blocks
    if len(payload) < min_header:
        raise ValueError("Payload too small to be valid BFP9.")

    offset = 4  # skip MAGIC
    n_elems = struct.unpack_from("<I", payload, offset)[0]
    offset += 4
    block_size = struct.unpack_from("<H", payload, offset)[0]
    offset += 2
    num_blocks = struct.unpack_from("<I", payload, offset)[0]
    offset += 4

    if n_elems == 0:
        return b""

    # Read scales (fp32, byte-plane reordered)
    scales_bytes_len = num_blocks * 4
    scales_end = offset + scales_bytes_len
    if scales_end > len(payload):
        raise ValueError("Payload ended early while reading scales.")
    
    # Reverse byte-plane reordering
    scales_reordered = np.frombuffer(payload[offset:scales_end], dtype=np.uint8)
    scales_arr = np.zeros((num_blocks, 4), dtype=np.uint8)
    for i in range(4):
        scales_arr[:, i] = scales_reordered[i*num_blocks:(i+1)*num_blocks]
    scales = np.frombuffer(scales_arr.tobytes(), dtype=np.float32)
    scales = torch.from_numpy(scales.copy()).float()
    offset = scales_end

    # Read quantized values (int8, not transposed)
    quantized_end = offset + n_elems
    if quantized_end > len(payload):
        raise ValueError("Payload ended early while reading quantized data.")
    
    quantized = np.frombuffer(payload[offset:quantized_end], dtype=np.int8)
    quantized = torch.from_numpy(quantized.copy()).float()

    # Pad quantized to multiple of block_size for reshaping
    padded_len = num_blocks * block_size
    if padded_len > n_elems:
        quantized = torch.nn.functional.pad(quantized, (0, padded_len - n_elems), value=0.0)

    # Reshape and dequantize
    Q = 127.0
    quantized_blocks = quantized.view(num_blocks, block_size)
    x_rec_blocks = (quantized_blocks / Q) * scales.unsqueeze(1)

    # Flatten and trim
    x_rec = x_rec_blocks.view(-1)[:n_elems]

    # Convert to bf16
    x_bf16 = x_rec.to(torch.bfloat16)
    return x_bf16.view(torch.int16).numpy().tobytes()


def _decode_bfp6(payload: bytes) -> bytes:
    """Decode BFP6 format (int8 quantized + fp16 scales, vectorized)."""
    min_header = 4 + 4 + 2 + 4  # MAGIC + n_elems + block_size + num_blocks
    if len(payload) < min_header:
        raise ValueError("Payload too small to be valid BFP6.")

    offset = 4  # skip MAGIC
    n_elems = struct.unpack_from("<I", payload, offset)[0]
    offset += 4
    block_size = struct.unpack_from("<H", payload, offset)[0]
    offset += 2
    num_blocks = struct.unpack_from("<I", payload, offset)[0]
    offset += 4

    if n_elems == 0:
        return b""

    # Read scales (fp16 - 2 bytes each)
    scales_bytes_len = num_blocks * 2
    scales_end = offset + scales_bytes_len
    if scales_end > len(payload):
        raise ValueError("Payload ended early while reading scales.")
    
    scales = np.frombuffer(payload[offset:scales_end], dtype=np.float16)
    scales = torch.from_numpy(scales.copy().astype(np.float32)).float()
    offset = scales_end

    # Read quantized values (int8)
    quantized_end = offset + n_elems
    if quantized_end > len(payload):
        raise ValueError("Payload ended early while reading quantized data.")
    
    quantized = np.frombuffer(payload[offset:quantized_end], dtype=np.int8)
    quantized = torch.from_numpy(quantized.copy()).float()

    # Pad quantized to multiple of block_size for reshaping
    padded_len = num_blocks * block_size
    if padded_len > n_elems:
        quantized = torch.nn.functional.pad(quantized, (0, padded_len - n_elems), value=0.0)

    # Reshape and dequantize
    Q = 127.0
    quantized_blocks = quantized.view(num_blocks, block_size)
    x_rec_blocks = (quantized_blocks / Q) * scales.unsqueeze(1)

    # Flatten and trim
    x_rec = x_rec_blocks.view(-1)[:n_elems]

    # Convert to bf16
    x_bf16 = x_rec.to(torch.bfloat16)
    return x_bf16.view(torch.int16).numpy().tobytes()


def _decode_bfp2(payload: bytes) -> bytes:
    """Decode BFP2 format (int8 quantized, vectorized)."""
    min_header = 4 + 4 + 2 + 4  # MAGIC + n_elems + block_size + num_blocks
    if len(payload) < min_header:
        raise ValueError("Payload too small to be valid BFP2.")

    offset = 4  # skip MAGIC
    n_elems = struct.unpack_from("<I", payload, offset)[0]
    offset += 4
    block_size = struct.unpack_from("<H", payload, offset)[0]
    offset += 2
    num_blocks = struct.unpack_from("<I", payload, offset)[0]
    offset += 4

    if n_elems == 0:
        return b""

    # Read scales (fp32)
    scales_bytes_len = num_blocks * 4
    scales_end = offset + scales_bytes_len
    if scales_end > len(payload):
        raise ValueError("Payload ended early while reading scales.")
    
    scales = np.frombuffer(payload[offset:scales_end], dtype=np.float32)
    scales = torch.from_numpy(scales.copy()).float()
    offset = scales_end

    # Read quantized values (int8)
    quantized_end = offset + n_elems
    if quantized_end > len(payload):
        raise ValueError("Payload ended early while reading quantized data.")
    
    quantized = np.frombuffer(payload[offset:quantized_end], dtype=np.int8)
    quantized = torch.from_numpy(quantized.copy()).float()

    # Pad quantized to multiple of block_size for reshaping
    padded_len = num_blocks * block_size
    if padded_len > n_elems:
        quantized = torch.nn.functional.pad(quantized, (0, padded_len - n_elems), value=0.0)

    # Reshape and dequantize
    Q = 127.0
    quantized_blocks = quantized.view(num_blocks, block_size)
    x_rec_blocks = (quantized_blocks / Q) * scales.unsqueeze(1)

    # Flatten and trim
    x_rec = x_rec_blocks.view(-1)[:n_elems]

    # Convert to bf16
    x_bf16 = x_rec.to(torch.bfloat16)
    return x_bf16.view(torch.int16).numpy().tobytes()


def _decode_bfp1(payload: bytes) -> bytes:
    """Decode legacy BFP1 format for backward compatibility."""
    min_header = 4 + 4 + 2 + 1 + 1 + 4
    if len(payload) < min_header:
        raise ValueError("Payload too small to be valid BFP1.")

    offset = 4  # skip MAGIC
    n_elems = struct.unpack_from("<I", payload, offset)[0]
    offset += 4
    block_size = struct.unpack_from("<H", payload, offset)[0]
    offset += 2
    bits = struct.unpack_from("<B", payload, offset)[0]
    offset += 1
    scale_dtype = struct.unpack_from("<B", payload, offset)[0]
    offset += 1
    packed_q_len = struct.unpack_from("<I", payload, offset)[0]
    offset += 4

    if n_elems == 0:
        return b""

    num_blocks = (n_elems + block_size - 1) // block_size

    # Read scales
    scale_bytes_per_block = 4 if scale_dtype == 1 else 2
    scale_np_dtype = np.float32 if scale_dtype == 1 else np.float16
    
    scales_bytes_len = num_blocks * scale_bytes_per_block
    scales_end = offset + scales_bytes_len
    scales = np.frombuffer(payload[offset:scales_end], dtype=scale_np_dtype).astype(np.float32)
    scales = torch.from_numpy(scales.copy())
    offset = scales_end

    # Read and unpack quantized data
    packed_end = offset + packed_q_len
    packed_q = payload[offset:packed_end]
    
    # Fast unpack for 8-bit case
    if bits == 8:
        quantized = np.frombuffer(packed_q, dtype=np.int8)
        quantized = torch.from_numpy(quantized.copy()).float()
    else:
        # Slow path for other bit widths
        quantized = torch.tensor(_unpack_signed_ints_fast(packed_q, n_elems, bits), dtype=torch.float32)

    Q = (1 << (bits - 1)) - 1

    # Dequantize block by block
    x_rec = torch.empty((n_elems,), dtype=torch.float32)
    idx = 0
    for bi in range(num_blocks):
        start = bi * block_size
        end = min(start + block_size, n_elems)
        s = float(scales[bi].item())
        block_len = end - start

        if s == 0.0:
            x_rec[start:end] = 0.0
        else:
            x_rec[start:end] = s * (quantized[idx:idx + block_len] / Q)

        idx += block_len

    x_bf16 = x_rec.to(torch.bfloat16)
    return x_bf16.view(torch.int16).numpy().tobytes()


def _unpack_signed_ints_fast(packed: bytes, count: int, bits: int):
    """Unpack signed integers - vectorized where possible."""
    if bits == 8:
        return list(np.frombuffer(packed, dtype=np.int8)[:count])
    
    # For non-8-bit, use numpy-based approach
    mask = (1 << bits) - 1
    sign_bit = 1 << (bits - 1)
    
    packed_arr = np.frombuffer(packed, dtype=np.uint8)
    values = []
    acc = 0
    acc_bits = 0
    idx = 0

    for _ in range(count):
        while acc_bits < bits:
            if idx >= len(packed_arr):
                break
            acc |= int(packed_arr[idx]) << acc_bits
            acc_bits += 8
            idx += 1

        raw = acc & mask
        acc >>= bits
        acc_bits -= bits

        if raw & sign_bit:
            raw -= (1 << bits)
        values.append(raw)

    return values


# =========================
# API compression methods
# =========================
def _calculate_combined_score(original_bytes: bytes, payload: bytes) -> float:
    """
    Calculate the combined similarity score:
    Score = (Percent Similarity in Norm) * (Cosine Similarity)
    """
    try:
        decoded_bytes = bfp_decode_to_bf16_bytes(payload)
        
        # Load tensors
        original = torch.frombuffer(bytearray(original_bytes), dtype=torch.bfloat16).float()
        decoded = torch.frombuffer(bytearray(decoded_bytes), dtype=torch.bfloat16).float()
        
        # Handle length mismatch (truncation/padding)
        if len(decoded) != len(original):
            min_len = min(len(original), len(decoded))
            original = original[:min_len]
            decoded = decoded[:min_len]
            
        norm_a = torch.linalg.norm(original)
        norm_b = torch.linalg.norm(decoded)
        
        # 1. Percent Similarity in Norm
        # Avoid division by zero
        if norm_a == 0 and norm_b == 0:
            norm_sim = 1.0
        elif norm_a == 0 or norm_b == 0:
            norm_sim = 0.0
        else:
            # min / max approach
            norm_sim = (torch.min(norm_a, norm_b) / torch.max(norm_a, norm_b)).item()
            
        # 2. Cosine Similarity
        if norm_a == 0 or norm_b == 0:
             # handled above effectively, but for completeness
             if norm_a == norm_b: cosine_sim = 1.0
             else: cosine_sim = 0.0
        else:
            cosine_sim = (torch.dot(original, decoded) / (norm_a * norm_b)).item()
            
        return norm_sim * cosine_sim

    except Exception:
        # Fail safe
        return 0.0

def compress_data(data: bytes) -> bytes:
    """
    Lossy compression with aggressive adaptive block size.
    Maximizes block size while maintaining Combined Score >= 0.99.
    """
    try:
        if not data:
            raise ValueError("Empty data received")

        # Candidate 44: AGGRESSIVE Single [216]
        # Only use 216 block size - the proven winner starting point
        # Ultra-minimum overhead, maximum compression
        candidates = [239, 237, 227, 218, 188]
        thresholds = [0.9999, 0.9999, 0.9999, 0.9999, 0.9999]
        
        selected_payload = None
        
        for bs in candidates:
            # 1. Encode
            payload = bfp_encode_bf16_bytes(data, block_size=bs)
            # If we reached the last candidate and it still failed...
            if bs == candidates[-1]:
                # We have to return something.
                selected_payload = payload
                break
            
            # 2. Validate
            score = _calculate_combined_score(data, payload)
            
            # Use a slightly stricter threshold locally to account for floating point jitter
            # and ensure we pass the external 0.99 check comfortably.
            if score >= thresholds[candidates.index(bs)]:
                selected_payload = payload
                break
            
            # If we reached the last candidate and it still failed...
            if bs == candidates[-1]:
                # We have to return something.
                selected_payload = payload

        compressed = brotli.compress(selected_payload, quality=BROTLI_QUALITY)
        return compressed

    except Exception as e:
        raise ValueError(f"Compression failed: {str(e)}") from e


def decompress_data(data: bytes) -> bytes:
    """
    Decompression:
      brotli -> BFP payload -> bf16 bytes
    """
    try:
        if not data:
            raise ValueError("Empty data received")

        payload = brotli.decompress(data)
        out = bfp_decode_to_bf16_bytes(payload)

        # Must be bf16-aligned
        if len(out) % 2 != 0:
            raise ValueError("Decoded output is not bf16-aligned (length not multiple of 2).")

        return out

    except Exception as e:
        raise ValueError(f"Decompression failed: {str(e)}") from e


# =========================
# FastAPI app
# =========================
def make_app() -> FastAPI:
    app = FastAPI(title="Matrix Compression Miner API (BFP + zstd, vectorized)")

    @app.get("/health")
    def health():
        return {"status": "healthy"}

    @app.post("/compress")
    async def compress(file: UploadFile = File(...)):
        try:
            data = await file.read()
            if not data:
                raise HTTPException(status_code=400, detail="No data received in file")

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
            if not data:
                raise HTTPException(status_code=400, detail="No data received in file")

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