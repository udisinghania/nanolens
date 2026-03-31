import torch
import torch.distributed as dist
import logging

logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)

def _collapse_hook(module, inp, out, name, threshold, collapsed_layers, variance_dim, should_stop):
    """Internal hook to monitor variance and flag the DDP-safe shutdown tensor."""
    if name in collapsed_layers:
        return
        
    h = out[0] if isinstance(out, tuple) else out
    if not isinstance(h, torch.Tensor):
        return
        
    variance = h.float().var(dim=variance_dim, correction=0).mean().item()
    
    if variance < threshold:
        collapsed_layers.add(name)
        logging.error(f" 🚨 [nanoLens] Variance collapse at '{name}' (var={variance:.2e}, dim={variance_dim})")
        
        # Telemetry dump
        if torch.cuda.is_available():
            try:
                torch.cuda.memory._record_memory_history(max_entries=100_000)
                torch.cuda.memory._dump_snapshot(f"nanolens_collapse_{name.replace('.','_')}.pickle")
            except Exception:
                pass
                
        # DDP-Safe Shutdown: Flip the flag instead of raising an exception mid-forward
        should_stop.fill_(1)

def attach_nanolens(model, target_layer=torch.nn.Linear, threshold=1e-4, variance_dim=0):
    """
    variance_dim=0  : inter-sample spread (detects mode collapse, higher CPU memory cost).
    variance_dim=-1 : intra-token spread (weaker signal, ~6x lower overhead).
    """
    # Initialize the shutdown tensor on the same device as the model
    try:
        device = next(model.parameters()).device
    except StopIteration:
        device = torch.device("cpu")
        
    should_stop = torch.zeros(1, dtype=torch.bool, device=device)
    handles = []
    collapsed_layers = set()
    
    for name, module in model.named_modules():
        if isinstance(module, target_layer):
            hook = lambda m, i, o, n=name: _collapse_hook(
                m, i, o, n, threshold, collapsed_layers, variance_dim, should_stop
            )
            handles.append(module.register_forward_hook(hook))
            
    logging.info(f"[nanoLens] Attached to {len(handles)} layers (threshold={threshold:.2e}, variance_dim={variance_dim}).")
    return handles, should_stop

def check_nanolens(should_stop, process_group=None):
    """
    Call at the top of every training step. Coordinates shutdown across all ranks via all_reduce(MAX).
    """
    if not dist.is_initialized():
        # Single-GPU fallback
        if should_stop.item():
            raise RuntimeError("[nanoLens] Representation collapse detected. Halting compute.")
        return
        
    # All ranks participate in the collective reduction
    dist.all_reduce(should_stop, op=dist.ReduceOp.MAX, group=process_group)
    
    if should_stop.item():
        rank = dist.get_rank()
        raise RuntimeError(f"[nanoLens] Rank {rank}: Collapse detected across cluster. Halting compute.")

@torch.no_grad()
def calibrate(model, sample_input, target_layer=torch.nn.Linear, percentile=10, variance_dim=0):
    """Runs a dummy forward pass to profile layer variances and suggests a safe threshold."""
    variances = {}
    handles = []
    
    def probe(name):
        def _hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            if isinstance(h, torch.Tensor):
                variances[name] = h.float().var(dim=variance_dim, correction=0).mean().item()
        return _hook

    for name, module in model.named_modules():
        if isinstance(module, target_layer):
            handles.append(module.register_forward_hook(probe(name)))
            
    model(sample_input)
    
    for h in handles:
        h.remove()
        
    if not variances:
        raise RuntimeError("[nanoLens] No target layers found during calibration.")
        
    sorted_variances = sorted(variances.values())
    idx = max(0, int(len(sorted_variances) * percentile / 100) - 1)
    threshold = sorted_variances[idx] * 0.1  # 10th percentile divided by 10
    
    logging.info(f"[nanoLens] Calibration complete. Variance range: [{min(sorted_variances):.2e}, {max(sorted_variances):.2e}]. Suggested threshold: {threshold:.2e}")
    return threshold, variances