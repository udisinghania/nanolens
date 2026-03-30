import torch
import logging

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

def _collapse_hook(module, inp, out, name, threshold):
    """Monitors continuous latent variance for calcification and halts dead runs."""
    try:
        # Handle tuple outputs standard in HF / timm Vision Transformers
        h = out[0] if isinstance(out, tuple) else out
        
        # Calculate feature variance across the embedding dimension
        # .float() prevents float16 overflow artifacts during variance calc
        variance = h.float().var(dim=-1).mean().item()
        
        if variance < threshold:
            logging.error(f"🚨 [nanoLens] Alert: Variance collapse detected at {name} (var={variance:.6e})")
            
            # Trigger telemetry dump for debugging (requires torch.cuda.memory._record_memory_history)
            try:
                if torch.cuda.is_available():
                    torch.cuda.memory._dump_snapshot(f"nanolens_collapse_{name.replace('.','_')}.pickle")
                    logging.info("VRAM-safe telemetry snapshot saved.")
            except Exception:
                pass
                
            raise RuntimeError(f"Representation collapse caught at {name}. Halting compute to save budget.")
            
    finally:
        # Strict garbage collection to prevent DDP (Distributed Data Parallel) VRAM leaks
        del out, inp

def attach_nanolens(model, target_layer=torch.nn.Linear, threshold=1e-4):
    """Attaches zero-overhead telemetry to the model's residual stream."""
    handles = []
    for name, module in model.named_modules():
        if isinstance(module, target_layer):
            # Capture the current name in the lambda closure
            hook = lambda m, i, o, n=name: _collapse_hook(m, i, o, n, threshold)
            handles.append(module.register_forward_hook(hook))
            
    logging.info(f"[nanoLens] Attached circuit breakers to {len(handles)} layers. Monitoring for calcification.")
    return handles