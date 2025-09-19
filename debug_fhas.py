import time
import torch
from torch.nn.attention import sdp_kernel, SDPBackend

def time_one_step(model, batch, device, backend: SDPBackend, desc=""):
    # Force a specific backend to run this step (Flash / MemEff / Math)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with sdp_kernel(backend):
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
            out = model(**batch)
            loss = out.loss
        loss.backward()
    torch.cuda.synchronize()
    dur = time.perf_counter() - t0
    return dur

def sanity_sdpa_backends(model, batch, device):
    b = {k: v.to(device) for k, v in batch.items()}
    # Warm-up to allocate kernels
    for _ in range(3):
        with torch.amp.autocast(device_type='cuda', dtype=torch.float16, enabled=True):
            model(**b).loss.backward()
    torch.cuda.synchronize()

    for name, bk in [("FLASH", SDPBackend.FLASH_ATTENTION),
                     ("MEM_EFF", SDPBackend.EFFICIENT_ATTENTION),
                     ("MATH", SDPBackend.MATH)]:
        try:
            dur = time_one_step(model, b, device, bk, desc=name)
            print(f"[SDPA={name}] step time: {dur*1000:.2f} ms")
        except Exception as e:
            print(f"[SDPA={name}] not available: {repr(e)}")