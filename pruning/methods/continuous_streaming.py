from __future__ import annotations

from pathlib import Path
import math
import shutil
import tempfile
import time
import torch
from pruning.paths import SCRATCH
from pruning.methods.continuous import Continuous
from pruning.models.opt import decoder_layers, linear_layers, layer_prefix, capture_first_inputs, augmented_linear, copy_augmented
from pruning.curvature import estimate_b_covariance, collect_a
from pruning.nnprune.factorized_fisher import complement_update
from pruning.nnprune.streaming import disk_masks


class StreamingContinuous(Continuous):
    def __init__(self, problem, options):
        if not options.get("affine") or options.get("deletion_selection") != "global_obs":
            raise ValueError("Streaming continuous requires global affine OBS")
        super().__init__(problem, options)

    @torch.no_grad()
    def propose(self, problem, target, index, progress):
        started = time.perf_counter()
        settings, options, model = problem.settings, self.options, problem.model
        required = max(0, math.ceil(target*problem.total) - (problem.total-problem.remaining()))
        if not required:
            return {"requested_new_deletions": 0, "selected_new_deletions": 0}
        root = Path(problem.cfg.runtime.scratch or SCRATCH)
        root.mkdir(parents=True, exist_ok=True)
        modules = problem.affine_modules()
        entries = sum(module.weight.numel()+sum(p.numel() for p in [module.bias] if p is not None) for module in modules.values())
        activation_groups = {}
        for name,module in modules.items():
            prefix,_,projection = name.rpartition(".")
            key = prefix if projection in {"q_proj","k_proj","v_proj","gate_proj","up_proj"} else name
            activation_groups[key,module.bias is not None] = (module.in_features+int(module.bias is not None))**2
        factor_entries = sum(activation_groups.values()) + sum(module.out_features**2 for module in modules.values())
        needed = int(1.1*(8*factor_entries+9*entries))
        if shutil.disk_usage(root).free < needed:
            raise RuntimeError(f"Streaming OBS needs about {needed/2**30:.1f} GiB of free scratch space")
        with tempfile.TemporaryDirectory(prefix="obs-", dir=root) as directory:
            cache = Path(directory)
            factor_paths, score_paths, layout = [], [], []
            hidden, kwargs = capture_first_inputs(model, problem.calibration)
            for layer_index, layer in enumerate(decoder_layers(model)[:problem.layers_count]):
                covariance, counts, _ = estimate_b_covariance(model, problem.b_sequences, problem.names,
                    settings.b_tokens, settings.seed, problem.layers_count, layer_index,
                    batch_size=settings.b_batch_size)
                activations, hidden = collect_a(layer, hidden, kwargs, problem.names, settings.activation_batch_size,
                    return_hidden=True, include_bias=True, dtype=torch.float64)
                factors, shared_a = {}, {}
                for name in problem.names:
                    module = linear_layers(layer)[name]
                    weight = augmented_linear(module).double()
                    raw = activations.pop(name)
                    key = id(raw)
                    a = shared_a[key].to(problem.device) if key in shared_a else self.damped_a(raw,weight)
                    b = covariance.pop((layer_index, name)) / counts[layer_index, name]
                    b.diagonal().add_(options.b_damp_ratio*b.diag().mean().clamp_min(torch.finfo(b.dtype).eps))
                    ai, bi = (torch.cholesky_inverse(torch.linalg.cholesky(factor)) for factor in (a,b))
                    scores = .5*weight.square()/bi.diag()[:,None]/ai.diag()[None,:]
                    scores.masked_fill_(~(self.affine_eligible(problem, layer_index, name, module) & weight.ne(0)), torch.inf)
                    path = cache / f"scores-{len(score_paths)}.pt"
                    torch.save(scores.cpu(), path)
                    score_paths.append(path)
                    layout.append((layer_index, name))
                    if key not in shared_a:
                        shared_a[key] = a.cpu()
                    factors[name] = shared_a[key], b.cpu()
                    del a,b,ai,bi,scores,weight
                path = cache / f"factors-{layer_index}.pt"
                torch.save(factors,path)
                factor_paths.append(path)
                del factors,activations,covariance,shared_a,raw
                print(f"OBS factor snapshot: {layer_index+1}/{problem.layers_count} blocks",flush=True)
                progress(layer_index+1,[{"phase":"factor_snapshot","layer":layer_index}],0,time.perf_counter()-started)
            del hidden
            masks, score_sum = disk_masks(score_paths,required,cache,device=problem.cfg.runtime.selection_device)
            for path in score_paths:
                path.unlink()
            history, offset = [], 0
            for layer_index, layer in enumerate(decoder_layers(model)[:problem.layers_count]):
                factors = torch.load(factor_paths[layer_index],map_location=problem.device,weights_only=True)
                for name in problem.names:
                    if layout[offset] != (layer_index,name):
                        raise RuntimeError("Streaming OBS factor/mask layout changed")
                    module = linear_layers(layer)[name]
                    prefix = f"{layer_prefix(model,layer_index)}.{name}"
                    live = problem.extra["obs_live"][prefix]
                    chosen = torch.load(masks[offset],map_location=problem.device,weights_only=True)
                    deleted = chosen | ~live.to(problem.device)
                    weight = augmented_linear(module).double()
                    a,b = factors.pop(name)
                    ai,bi = (torch.cholesky_inverse(torch.linalg.cholesky(factor)) for factor in (a,b))
                    delta,record = complement_update(weight,a,b,deleted,rtol=options.cg_rtol,
                        max_iterations=options.cg_max_iterations,a_inverse=ai,b_inverse=bi)
                    copy_augmented(module,(weight+delta).masked_fill(deleted,0.))
                    live &= ~chosen.to(live.device)
                    history.append({"layer":layer_index,"module":name,**record})
                    offset += 1
                    del a,b,ai,bi,delta,weight,chosen,deleted
                factor_paths[layer_index].unlink()
                progress(layer_index+1,history,0,time.perf_counter()-started)
            cleanup = self.affine_cleanup(problem)
        return {"layers":history,"requested_new_deletions":required,"selected_new_deletions":required,
                "singleton_score_sum":score_sum,"canonical_savings":cleanup,"deletion_selection":"global_obs",
                "factor_backend":"one unchanged-model block at a time; exact disk global selection"}
