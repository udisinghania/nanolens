import torch
import logging

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

def _collapse_hook(module, inp, out, name, threshold, collapsed_layers):
    """Monitors continuous latent variance for calcification and halts dead runs."""
    if name in collapsed_layers:
        return

    # Handle tuple outputs standard in HF / timm Vision Transformers
    h = out[0] if isinstance(out, tuple) else out
    
    if not isinstance(h, torch.Tensor):
        return

    # var(dim=0) captures whether different inputs in the batch map to the same representation
    # correction=0 forces population variance to prevent silent NaN failures on batch_size=1
    variance = h.float().var(dim=0, correction=0).mean().item()
    
    if variance < threshold:
        collapsed_layers.add(name)
        logging.error(f"🚨 [nanoLens] Variance collapse at '{name}' (var={variance:.6e} < threshold={threshold:.6e})")
        
        # Trigger telemetry dump for debugging
        if torch.cuda.is_available():
            try:
                # Must initialize memory history before dumping snapshot
                torch.cuda.memory._record_memory_history(max_entries=100_000)
                torch.cuda.memory._dump_snapshot(f"nanolens_collapse_{name.replace('.','_')}.pickle")
                logging.info("[nanoLens] VRAM-safe telemetry snapshot saved.")
            except Exception as e:
                logging.warning(f"[nanoLens] Snapshot failed: {e}")
                
        raise RuntimeError(f"[nanoLens] Representation collapse caught at '{name}'. Halting compute.")

def attach_nanolens(model, target_layer=torch.nn.Linear, threshold=1e-4):
    """Attaches zero-overhead telemetry to the model's residual stream."""
    handles = []
    collapsed_layers = set()  # Scoped per attach call, prevents global state pollution
    
    for name, module in model.named_modules():
        if isinstance(module, target_layer):
            # Capture the current name and the scoped set in the lambda closure
            hook = lambda m, i, o, n=name: _collapse_hook(m, i, o, n, threshold, collapsed_layers)
            handles.append(module.register_forward_hook(hook))
            
    logging.info(f"[nanoLens] Attached circuit breakers to {len(handles)} layers (threshold={threshold:.6e}).")
    return handles