from __future__ import annotations

from pathlib import Path
import math
import shutil
import tempfile
import time
import torch
from pruning.paths import SCRATCH
from pruning.methods.discrete_simple import DiscreteSimple
from pruning.models.opt import augmented_linear,copy_augmented,layer_prefix
from pruning.nnprune.kronecker import KroneckerFisher
from pruning.nnprune.l0 import initial_curvature,minimize_l0
from pruning.nnprune.l0_fast import lipschitz_estimate,minimize_l0 as minimize_fast
from pruning.nnprune.streaming import disk_kth
from pruning.nnprune.network import parameter_name
from pruning.nnprune.canonicalize.mlp import canonicalize_mlp_tensors
from pruning.nnprune.canonicalize.gated import GatedGroup,canonicalize_gated


class StreamingDiscreteSimple(DiscreteSimple):
    def __init__(self,problem,options):
        if options.b_refresh != "step":
            raise ValueError("Streaming L0 currently refreshes factors at each proposal")
        super().__init__(problem,options)

    @torch.no_grad()
    def propose(self,problem,target,index,progress,*,capture=None):
        if capture is not None:
            raise ValueError("Full-memory diagnostic capture is unavailable for streamed L0")
        options,started = self.options,time.perf_counter()
        before = problem.remaining()
        fixed_penalty = options.get("penalty_mode","retune") == "fixed"
        stored = problem.extra.get("l0_penalty") if fixed_penalty else None
        calibrate = options.penalty is None and stored is None
        if calibrate and fixed_penalty:
            target = options.get("initial_sparsity",.05)
        root = Path(problem.cfg.runtime.scratch or SCRATCH)
        root.mkdir(parents=True,exist_ok=True)
        width = torch.finfo(self.dtype).bits//8
        parameters = sum(module.weight.numel()+int(module.bias.numel() if module.bias is not None else 0) for module in self.modules.values())
        factor_entries = sum((module.in_features+int(module.bias is not None))**2+module.out_features**2 for module in self.modules.values())
        needed = int(1.1*width*(factor_entries+3*parameters))
        if shutil.disk_usage(root).free < needed:
            raise RuntimeError(f"Streamed L0 needs about {needed/2**30:.1f} GiB of free scratch space")
        with tempfile.TemporaryDirectory(prefix="l0-",dir=root) as directory:
            cache = Path(directory)
            factor_paths,score_paths,curvatures = [None]*len(self.units),[],{}
            x,_ = problem.fisher_data(options.fisher_n,options.fisher_seed)
            b_inputs = torch.cat(problem.b_sequences).to(problem.device)
            for block_index in range(problem.layers_count):
                prefix = layer_prefix(problem.model,block_index)+"."
                indices = [index for index,(names,_) in enumerate(self.units) if all(name.startswith(prefix) for name in names)]
                block_names = [name for unit_index in indices for name in self.units[unit_index][0]]
                metric = KroneckerFisher(problem.model,x,options,modules={name:self.modules[name] for name in block_names},
                    b_inputs=b_inputs,max_b_tokens=problem.cfg.sampling.b_tokens or None,
                    a_batch_size=problem.cfg.sampling.activation_batch_size,b_batch_size=problem.cfg.sampling.b_batch_size)
                for unit_index in indices:
                    names,_ = self.units[unit_index]
                    factors,scales = {},{}
                    for name in names:
                        a,b = metric.factors.pop(name)
                        a,b = a.to(dtype=self.dtype),b.to(dtype=self.dtype)
                        sa,sb = [torch.where(factor.diag()>0,factor.diag().clamp_min(0).sqrt(),torch.ones_like(factor.diag()))
                                 if options.standardize else torch.ones_like(factor.diag()) for factor in (a,b)]
                        a,b = a/sa[:,None]/sa[None,:],b/sb[:,None]/sb[None,:]
                        factors[name],scales[name] = (a.cpu(),b.cpu()),(sa.cpu(),sb.cpu())
                        curvature = lipschitz_estimate(a,b) if self.solver == "fast" else initial_curvature(a,b)
                        curvatures[name] = curvature
                        if calibrate:
                            reference = augmented_linear(self.modules[name]).to(dtype=self.dtype)*sb[:,None]*sa[None,:]
                            eligible = self.penalized[name].to(problem.device) & reference.ne(0)
                            score = .5*curvature*reference[eligible].square()
                            path = cache/f"scores-{len(score_paths)}.pt"
                            torch.save(score.cpu(),path)
                            score_paths.append(path)
                            del reference,eligible,score
                        del a,b,sa,sb
                    path = cache/f"factors-{unit_index}.pt"
                    torch.save({"factors":factors,"scales":scales},path)
                    factor_paths[unit_index] = path
                    del factors,scales
                del metric
                print(f"L0 factor snapshot: {block_index+1}/{problem.layers_count} blocks",flush=True)
                progress(block_index+1,[{"phase":"factor_snapshot","layer":block_index}],0,time.perf_counter()-started)
            if any(path is None for path in factor_paths):
                raise ValueError("Streaming L0 requires each canonical group to stay inside one decoder block")
            if calibrate:
                if target is None:
                    raise ValueError("Specify an L0 sparsity hint or penalty")
                available = sum(torch.load(path,map_location="cpu",weights_only=True,mmap=True).numel() for path in score_paths)
                rank = max(1,min(available,math.ceil(target*problem.total)-(problem.total-before)))
                penalty = max(disk_kth(score_paths,rank,device=problem.cfg.runtime.selection_device)[0],torch.finfo(self.dtype).tiny) if available else 0.
                for path in score_paths:
                    path.unlink()
            else:
                penalty = options.penalty if options.penalty is not None else stored
            factor_seconds = time.perf_counter()-started
            lower,upper,best,trials = 0.,None,None,[]
            best_directory = cache/"best"
            for trial in range(options.penalty_trials if calibrate else 1):
                trial_directory = cache/"trial"
                trial_directory.mkdir()
                history,direct,removed,completed = [],0,0,0
                for unit_index,(names,pair) in enumerate(self.units):
                    payload = torch.load(factor_paths[unit_index],map_location=problem.device,weights_only=True)
                    factors,scales = payload["factors"],payload["scales"]
                    reference = {name:augmented_linear(self.modules[name]).to(dtype=self.dtype)*scales[name][1][:,None]*scales[name][0][None,:] for name in names}
                    penalized = {name:self.penalized[name].to(problem.device) for name in names}
                    live = ({name:problem.extra["l0_live"][name].to(problem.device) for name in names}
                            if options.get("permanent_deletions",False) else None)
                    settings = ({"lipschitz":{name:curvatures[name] for name in names}} if self.solver == "fast" else {})
                    solver = minimize_fast if self.solver == "fast" else minimize_l0
                    values,record = solver(reference,factors,penalty,penalized,options,pair=pair,
                        fixed={name:~mask for name,mask in live.items()} if live is not None else None,**settings)
                    if self.solver == "fast":
                        for name in names:
                            curvatures[name] = record["lipschitz"]
                    updated = {name:live[name]&~(values[name].eq(0)&penalized[name]) for name in names} if live is not None else None
                    actual = {name:(values[name]/scales[name][1][:,None]/scales[name][0][None,:]).to(self.modules[name].weight.dtype) for name in names}
                    direct += self.count_removed(actual,None)
                    self.cleanup(names,pair,actual,updated)
                    removed += self.count_removed(actual,updated)
                    torch.save({"values":{name:value.cpu() for name,value in actual.items()},
                                "live":{name:mask.cpu() for name,mask in updated.items()} if updated is not None else None},
                               trial_directory/f"{unit_index}.pt")
                    history.append({"modules":list(names),**record})
                    completed += len(names)
                    progress(completed,history,removed-direct,time.perf_counter()-started)
                    del payload,factors,scales,reference,penalized,live,values,actual,updated
                sparsity = removed/problem.total
                damage = sum(row["predicted_fisher_damage"] for row in history)
                summary = {"penalty":penalty,"removed_fraction":sparsity,"predicted_fisher_damage":damage,
                    "penalized_objective":sum(row["penalized_objective"] for row in history),"direct_removed":direct,"canonical_savings":removed-direct}
                trials.append(summary)
                distance = abs(sparsity-target) if target is not None else 0.
                if best is None or (distance,damage) < best[0]:
                    if best_directory.exists():
                        shutil.rmtree(best_directory)
                    trial_directory.rename(best_directory)
                    best = ((distance,damage),history,summary)
                else:
                    shutil.rmtree(trial_directory)
                print(f"L0 penalty trial {trial+1}: lambda={penalty:.6g}, removal={100*sparsity:.3f}%",flush=True)
                if not calibrate or distance <= options.sparsity_tolerance:
                    break
                if sparsity < target:
                    lower = penalty
                    penalty = math.sqrt(lower*upper) if upper is not None else penalty*options.penalty_growth
                else:
                    upper = penalty
                    penalty = math.sqrt(lower*upper) if lower else penalty/options.penalty_growth
            _,history,summary = best
            for unit_index,(names,_) in enumerate(self.units):
                payload = torch.load(best_directory/f"{unit_index}.pt",map_location="cpu",weights_only=True)
                for name in names:
                    value = payload["values"][name]
                    copy_augmented(self.modules[name],value)
                    live = payload["live"][name] if payload["live"] is not None else value.ne(0)
                    if payload["live"] is not None:
                        problem.extra["l0_live"][name].copy_(live)
                    module = self.modules[name]
                    for suffix,parameter in module.named_parameters(recurse=False):
                        key = parameter_name(name,suffix)
                        local = live[:,:module.in_features] if suffix == "weight" else live[:,-1]
                        problem.extra["pruned_coordinates"][key].copy_(local | ~self.initial_nonzero[key].cpu())
            if fixed_penalty:
                problem.extra["l0_penalty"] = summary["penalty"]
                if problem.remaining() >= before:
                    return None
        return {"layers":history,**summary,"penalty_trials":trials,
            "target_within_tolerance":abs(summary["removed_fraction"]-target)<=options.sparsity_tolerance if target is not None else None,
            "new_removed_percent":100*(before-problem.remaining())/problem.total,"penalty_fixed":fixed_penalty,
            "factor_seconds":factor_seconds,"optimized_modules":len(self.modules),"canonical_chains":len(self.chains),
            "canonical_gated_groups":len(self.gated_groups),"metric":"axb","factor_backend":"streaming"}

    def count_removed(self,values,live):
        count = 0
        for name,value in values.items():
            module = self.modules[name]
            for suffix,_ in module.named_parameters(recurse=False):
                key = parameter_name(name,suffix)
                mask = live[name] if live is not None else value.ne(0)
                mask = mask[:,:module.in_features] if suffix == "weight" else mask[:,-1]
                count += int((self.initial_nonzero[key].cpu() & ~mask.cpu()).sum())
        return count

    def cleanup(self,names,pair,values,live):
        if isinstance(pair,GatedGroup):
            dead = canonicalize_gated(*(values[name] for name in pair.names))
            if live is not None:
                live[pair.gate][dead] = False
                live[pair.up][dead] = False
                live[pair.down][:,dead] = False
        elif pair is not None:
            chain = next(chain for chain in self.chains if tuple(name.rpartition('.')[0] for name in chain.weights) == names)
            weights = [values[name][:,:self.modules[name].in_features] for name in names]
            biases = [values[name][:,-1] if self.modules[name].bias is not None else None for name in names]
            canonicalize_mlp_tensors(weights,biases,chain.activations)
            if live is not None:
                for left,right in zip(names,names[1:]):
                    dead = values[left].eq(0).all(1) & values[right][:,:len(values[left])].eq(0).all(0)
                    live[left][dead] = False
                    live[right][:,:len(dead)][:,dead] = False
