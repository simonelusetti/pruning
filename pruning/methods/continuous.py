from __future__ import annotations
import math
import time
import torch
from pruning.models.opt import (linear_layers, capture_first_inputs, forward_layer,
    augmented_linear, copy_augmented, neuron_capacity, neuron_size, neuron_removed_count, parameter_count,
    decoder_layers, layer_prefix)
from pruning.curvature import collect_a, estimate_b_covariance, inverse_factors, neuron_factor_snapshot
from pruning.nnprune.factorized_fisher import complement_update
from pruning.nnprune.neuron_fisher import neuron_obs_scores, neuron_complement, select_neuron_cuts
from pruning.nnprune.canonicalize.mlp import canonicalize_linear_chain
from pruning.nnprune.canonicalize.gated import canonicalize_gated


@torch.inference_mode()
def blockwise_obs_mask(module, a, b_inverse_diagonal, sparsity, percdamp, blocksize):
    weight = module.weight.detach().float().clone()
    a = a.to(weight)
    a = ((a + a.T) / 2).clone()
    b_inverse_diagonal = b_inverse_diagonal.to(device=weight.device, dtype=torch.float64)
    dead = a.diag() == 0
    a[dead, dead] = 1
    weight[:, dead] = 0
    diagonal = torch.arange(len(a), device=weight.device)
    a[diagonal, diagonal] += percdamp * a.diag().mean()
    factor, info = torch.linalg.cholesky_ex(a)
    if int(info):
        raise RuntimeError("damped activation factor is not positive definite")
    inverse_factor = torch.linalg.cholesky(
        torch.cholesky_inverse(factor), upper=True)
    for start in range(0, weight.shape[1], blocksize):
        end = min(start + blocksize, weight.shape[1])
        current = weight[:, start:end].clone()
        errors = torch.zeros_like(current)
        local_inverse = inverse_factor[start:end, start:end]
        input_inverse = local_inverse.diag().double().square()
        denominator = b_inverse_diagonal[:, None] * input_inverse[None, :]
        denominator = denominator.clamp_min(1e-30)
        score = current.double().square() / denominator
        count = min(int(score.numel() * sparsity), score.numel() - 1)
        threshold = torch.kthvalue(score.flatten(), count + 1).values
        mask = score <= threshold
        # With a fixed block mask, the column recurrence is a triangular system per row.
        counts = mask.sum(1)
        width = int(counts.max())
        # Dense row systems cost more than the recurrence at OPT dimensions.
        if width > (end - start) // 4:
            result = torch.empty_like(current)
            for column in range(end - start):
                values = current[:, column]
                quantized = values.masked_fill(mask[:, column], 0.)
                result[:, column] = quantized
                error = (values - quantized) / local_inverse[column, column]
                current[:, column:] -= error[:, None] @ local_inverse[column, column:][None]
                errors[:, column] = error
            weight[:, start:end] = result
            weight[:, end:] -= errors @ inverse_factor[start:end, end:]
            continue
        indices = mask.argsort(dim=1, descending=True, stable=True)[:, :width]
        active = torch.arange(width, device=weight.device)[None] < counts[:, None]
        for first in range(0, len(current), 256):
            last = min(first + 256, len(current))
            selected = indices[first:last]
            live = active[first:last]
            triangular = local_inverse[selected[:, :, None], selected[:, None, :]]
            triangular.masked_fill_(~(live[:, :, None] & live[:, None, :]), 0.)
            triangular.diagonal(dim1=-2, dim2=-1).copy_(
                torch.where(live, local_inverse.diag()[selected], 1.))
            solved = torch.linalg.solve_triangular(
                triangular.transpose(-1, -2),
                current[first:last].gather(1, selected).masked_fill(~live, 0.).unsqueeze(-1),
                upper=False).squeeze(-1)
            errors[first:last].scatter_(1, selected, solved)
        weight[:, start:end] = (current - errors @ local_inverse).masked_fill_(mask, 0.)
        weight[:, end:] -= errors @ inverse_factor[start:end, end:]
    return weight == 0


class Continuous:
    uses_selection = False

    def __init__(self, problem, options):
        self.options = options
        self.frozen = None
        self.affine = options.get("affine", False)
        self.cleanup = options.get("canonicalization", False)
        self.global_ranking = options.get("deletion_selection", "blockwise_obs") == "global_obs"
        if problem.model.config.model_type == "llama" and not (self.global_ranking and self.affine):
            raise ValueError("LLaMA continuous uses the global affine OBS preset")
        if options.get("deletion_selection", "blockwise_obs") not in {"blockwise_obs", "global_obs"}:
            raise ValueError("unknown continuous deletion selection")
        if options.deletion_unit not in {"weight", "hidden_neuron"}:
            raise ValueError("continuous deletion_unit must be weight or hidden_neuron")
        if (min(options.mu, options.percdamp, options.b_damp_ratio) < 0 or
                options.blocksize < 1 or not 0 < options.cg_rtol < 1 or options.cg_max_iterations < 1):
            raise ValueError("invalid continuous solve settings")
        if options.deletion_unit == "hidden_neuron" and (
                options.mu != 0 or options.b_refresh != "step" or options.get("modules", "all") != "all"):
            raise ValueError("hidden-neuron complements require mu=0 and one factor snapshot per proposal")
        if self.global_ranking and (options.deletion_unit != "weight" or options.mu != 0 or options.b_refresh != "step"):
            raise ValueError("global OBS requires weight deletions, mu=0, and one factor snapshot per proposal")
        if (self.affine or self.cleanup) and not self.global_ranking:
            raise ValueError("Affine compensation and optional cleanup require global OBS")
        if self.cleanup and not self.affine:
            raise ValueError("Global OBS cleanup requires affine compensation for constant folding")
        if options.deletion_unit == "weight" and not self.global_ranking and options.b_refresh not in {"layer", "frozen"}:
            raise ValueError("weight complements support layer or frozen B")
        if options.b_refresh == "frozen" and problem.sampling.refresh != "fixed":
            raise ValueError("frozen B requires sampling.refresh=fixed")
        if self.affine:
            self.modules = problem.affine_modules()
            state_device = problem.cfg.runtime.get("state_device", problem.device)
            self.initial_nonzero = {f"{name}.{suffix}": value.detach().ne(0).to(state_device)
                                    for name, module in self.modules.items()
                                    for suffix, value in module.named_parameters(recurse=False)}
            problem.extra.setdefault("obs_live", {name: torch.ones((module.out_features, module.in_features+int(module.bias is not None)), device=state_device, dtype=torch.bool)
                                                   for name, module in self.modules.items()})
            problem.extra["pruned_coordinate_convention"] = "live"
            problem.extra.setdefault("pruned_coordinates", {
                name: torch.ones_like(mask) for name, mask in self.initial_nonzero.items()})

    def affine_masks(self, problem):
        masks = problem.extra["pruned_coordinates"]
        for name, module in self.modules.items():
            live = problem.extra["obs_live"][name]
            masks[f"{name}.weight"].copy_(live[:, :module.in_features] | ~self.initial_nonzero[f"{name}.weight"])
            if module.bias is not None:
                masks[f"{name}.bias"].copy_(live[:, -1] | ~self.initial_nonzero[f"{name}.bias"])

    def affine_eligible(self, problem, index, name, module):
        prefix = f"{layer_prefix(problem.model, index)}.{name}"
        original = self.initial_nonzero[f"{prefix}.weight"]
        if module.bias is not None:
            original = torch.cat((original, self.initial_nonzero[f"{prefix}.bias"][:, None]), 1)
        eligible = problem.extra["obs_live"][prefix] & original
        if self.cleanup and name == "fc2" and {"fc1", "fc2"} <= set(problem.names) and module.bias is not None:
            eligible[:, -1] = False
        return eligible.to(module.weight.device)

    def affine_cleanup(self, problem):
        self.affine_masks(problem)
        before = problem.remaining()
        if self.cleanup and problem.model.config.model_type == "llama":
            for group in problem.gated_groups():
                if not set(group.names) <= self.modules.keys():
                    raise ValueError("Gated cleanup requires all three MLP projections")
                gate, up, down = (self.modules[name] for name in group.names)
                if gate.bias is not None or up.bias is not None:
                    raise ValueError("Gated cleanup currently requires bias-free gate and up projections")
                dead = canonicalize_gated(gate.weight, up.weight, down.weight)
                live = problem.extra["obs_live"]
                dead = dead.to(live[group.gate].device)
                live[group.gate][dead] = False
                live[group.up][dead] = False
                live[group.down][:, :down.in_features][:, dead] = False
        if self.cleanup and {"fc1", "fc2"} <= set(problem.names):
            for index, layer in enumerate(problem.model.model.decoder.layers[:problem.layers_count]):
                canonicalize_linear_chain((layer.fc1, layer.fc2), (layer.activation_fn,))
                dead = layer.fc1.weight.eq(0).all(1) & layer.fc2.weight.eq(0).all(0)
                if layer.fc1.bias is not None:
                    dead &= layer.fc1.bias.eq(0)
                live = problem.extra["obs_live"]
                dead = dead.to(live[f"model.decoder.layers.{index}.fc1"].device)
                live[f"model.decoder.layers.{index}.fc1"][dead] = False
                live[f"model.decoder.layers.{index}.fc2"][:, :layer.fc2.in_features][:, dead] = False
            self.affine_masks(problem)
        return before - problem.remaining()

    def propose(self, problem, target, index, progress):
        if problem.neurons:
            return self.neurons(problem, target, progress)
        return (self.global_weights if self.global_ranking else self.weights)(problem, target, progress)

    @torch.no_grad()
    def global_weights(self, problem, target, progress):
        options, model, settings = self.options, problem.model, problem.settings
        started = time.perf_counter()
        required = max(0, math.ceil(target * problem.total) - (problem.total - problem.remaining()))
        modules = [(index, name, linear_layers(layer)[name])
                   for index, layer in enumerate(decoder_layers(model)[:problem.layers_count])
                   for name in problem.names]
        available = sum(int((augmented_linear(module).ne(0) & self.affine_eligible(problem, index, name, module)).sum())
                        if self.affine else int(module.weight.count_nonzero()) for index, name, module in modules)
        if required > available:
            raise ValueError("target exceeds the remaining prunable weights")
        print(f"Global OBS: snapshot for {required:,} new deletions", flush=True)
        covariance, counts, b_seconds = estimate_b_covariance(
            model, problem.b_sequences, problem.names, settings.b_tokens,
            settings.seed, problem.layers_count, batch_size=settings.b_batch_size,
            on_sequence=lambda completed, *_: print(
                f"Global B snapshot: {completed}/{len(problem.b_sequences)} sequences", flush=True))
        factors = inverse_factors(covariance, counts, options.b_damp_ratio)
        hidden, kwargs = capture_first_inputs(model, problem.calibration)
        activations = {}
        for index, layer in enumerate(decoder_layers(model)[:problem.layers_count]):
            local, hidden = collect_a(layer, hidden, kwargs, problem.names,
                                      settings.activation_batch_size, return_hidden=True,
                                      include_bias=self.affine, dtype=torch.float64 if self.affine else None)
            activations.update({(index, name): value for name, value in local.items()})
            print(f"Global A snapshot: block {index + 1}/{problem.layers_count}", flush=True)
        del hidden
        scores = torch.empty(sum(module.weight.numel() + (module.bias.numel() if self.affine and module.bias is not None else 0)
                                 for _, _, module in modules),
                             dtype=torch.float64, device=problem.device)
        offset, activation_inverses = 0, {}
        for index, name, module in modules:
            weight = (augmented_linear(module) if self.affine else module.weight).double()
            a = self.damped_a(activations[index, name], weight)
            a_inverse = torch.cholesky_inverse(torch.linalg.cholesky(a))
            b_inverse = factors[index, name][1]
            local = .5 * weight.square() / b_inverse.diag()[:, None] / a_inverse.diag()[None, :]
            local.masked_fill_(weight == 0, float("inf"))
            if self.affine:
                local.masked_fill_(~self.affine_eligible(problem, index, name, module), float("inf"))
                activation_inverses[index, name] = a_inverse
            scores[offset:offset + weight.numel()].copy_(local.flatten())
            offset += weight.numel()
        selected = torch.topk(scores, required, largest=False, sorted=False).indices
        score_sum = float(scores[selected].sum())
        mask = torch.zeros_like(scores, dtype=torch.bool)
        mask[selected] = True
        del scores, selected, local, weight, a, a_inverse, b_inverse
        print(f"Global OBS: selected {required:,} weights; starting local compensation", flush=True)
        history, offset = [], 0
        for index, layer in enumerate(decoder_layers(model)[:problem.layers_count]):
            layer_started, records = time.perf_counter(), {}
            for name in problem.names:
                module = linear_layers(layer)[name]
                shape = augmented_linear(module) if self.affine else module.weight
                deleted = mask[offset:offset + shape.numel()].view_as(shape)
                b, inverse = factors.pop((index, name))
                if self.affine:
                    live = problem.extra["obs_live"][f"{layer_prefix(model, index)}.{name}"]
                    records[name] = self.compensate(module, activations.pop((index, name)), b, inverse,
                                                    deleted, f"{index}:{name}", live=live.to(module.weight.device),
                                                    a_inverse=activation_inverses.pop((index, name)))
                    live &= ~deleted.to(live.device)
                else:
                    records[name] = self.compensate(module, activations.pop((index, name)), b, inverse,
                                                    deleted, f"{index}:{name}")
                records[name]["deletion_selection"] = "global singleton OBS; one unchanged-model factor snapshot"
                offset += shape.numel()
            history.append({"layer": index, "modules": records,
                            "b_estimation_seconds": b_seconds if index == 0 else 0.,
                            "seconds": time.perf_counter() - layer_started})
            progress(index + 1, history, 0, time.perf_counter() - started)
        cleanup = self.affine_cleanup(problem) if self.affine else 0
        return {"layers": history, "b_estimation_seconds": b_seconds,
                "requested_new_deletions": required, "selected_new_deletions": int(mask.sum()),
                "singleton_score_sum": score_sum, "deletion_selection": "global_obs", "canonical_savings": cleanup}

    def damped_a(self, a, weight):
        a = a.to(weight)
        a = ((a + a.T) / 2).clone()
        a.diagonal().add_(self.options.percdamp * a.diag().mean())
        return a

    def compensate(self, module, a, b, inverse, mask, label, *, live=None, a_inverse=None):
        weight = (augmented_linear(module) if live is not None else module.weight.detach()).double().clone()
        a = self.damped_a(a, weight)
        deleted = mask.to(weight.device) | (~live if live is not None else weight == 0)
        delta, record = complement_update(
            weight, a, b, deleted, mu=self.options.mu, rtol=self.options.cg_rtol,
            max_iterations=self.options.cg_max_iterations, b_inverse=inverse, a_inverse=a_inverse,
            progress=lambda iteration, residual: print(
                f"{label} CG {iteration}, residual {residual:.3g}", flush=True))
        revised = (weight + delta).masked_fill(deleted, 0.)
        if live is not None:
            copy_augmented(module, revised)
        else:
            module.weight.copy_(revised.to(module.weight))
        return record | {"actual_zeros": int(deleted.sum()),
                         "deleted": int((deleted & (weight != 0)).sum()),
                         "updates": int(delta.masked_fill(deleted, 0).count_nonzero())}

    @torch.no_grad()
    def weights(self, problem, target, progress):
        options, model = self.options, problem.model
        settings = problem.settings
        started = time.perf_counter()
        module_target = target * problem.total / parameter_count(model, problem.layers_count, problem.names)
        hidden, kwargs = capture_first_inputs(model, problem.calibration)
        history = []
        frozen_seconds = 0.
        if options.b_refresh == "frozen" and self.frozen is None:
            covariance, counts, frozen_seconds = estimate_b_covariance(
                model, problem.b_sequences, problem.names, settings.b_tokens,
                settings.seed, problem.layers_count, batch_size=settings.b_batch_size)
            self.frozen = inverse_factors(covariance, counts, options.b_damp_ratio)
        for layer_index, layer in enumerate(model.model.decoder.layers[:problem.layers_count]):
            layer_started = time.perf_counter()
            b_seconds = 0.
            if self.frozen is None:
                covariance, counts, b_seconds = estimate_b_covariance(
                    model, problem.b_sequences, problem.names, settings.b_tokens,
                    settings.seed, problem.layers_count, layer_index, batch_size=settings.b_batch_size)
                factors = inverse_factors(covariance, counts, options.b_damp_ratio)
            else:
                factors = self.frozen
            a_factors = collect_a(layer, hidden, kwargs, problem.names, settings.activation_batch_size)
            records = {}
            for name in problem.names:
                module = linear_layers(layer)[name]
                b, inverse = factors[layer_index, name]
                mask = blockwise_obs_mask(module, a_factors[name], inverse.diag(),
                                          module_target, options.percdamp, options.blocksize)
                records[name] = self.compensate(module, a_factors[name], b, inverse, mask,
                                                f"{layer_index}:{name}")
                records[name].update(module_target_sparsity=module_target,
                                     deletion_selection="blockwise generalized OBS; temporary updates discarded")
            hidden = forward_layer(layer, hidden, kwargs, settings.activation_batch_size)
            history.append({"layer": layer_index, "modules": records,
                            "b_estimation_seconds": b_seconds + (frozen_seconds if layer_index == 0 else 0.),
                            "seconds": time.perf_counter() - layer_started})
            progress(layer_index + 1, history, 0, time.perf_counter() - started)
        return {"layers": history, "b_estimation_seconds": sum(x["b_estimation_seconds"] for x in history)}

    @torch.no_grad()
    def neurons(self, problem, target, progress):
        model, count, settings = problem.model, problem.layers_count, problem.settings
        if target * problem.total > neuron_capacity(model, count):
            raise ValueError("target exceeds the removable hidden-neuron parameters")
        started = time.perf_counter()
        trace = getattr(problem, "trace", None)
        snapshots, counts, b_seconds = neuron_factor_snapshot(
            model, problem.calibration, problem.b_sequences, count, settings,
            on_sequence=trace.fisher_progress if trace is not None else None)
        layers = model.model.decoder.layers[:count]
        scores, savings = [], []
        for index, (layer, snapshot) in enumerate(zip(layers, snapshots)):
            a, b = snapshot["a"], snapshot["b"]
            scores.append(neuron_obs_scores(augmented_linear(layer.fc1), augmented_linear(layer.fc2),
                                            a["fc1"], b["fc1"], a["fc2"], b["fc2"], problem.deleted_neurons[index]))
            savings.append(torch.full((layer.fc1.out_features,), neuron_size(layer),
                                      device=layer.fc1.weight.device, dtype=torch.long))
        required = max(0, math.ceil(target * problem.total) - (problem.total - problem.remaining()))
        cuts = select_neuron_cuts(scores, savings, required)
        history, canonical_savings = [], 0
        for index, (layer, snapshot, (incoming_cut, outgoing_cut)) in enumerate(zip(layers, snapshots, cuts)):
            a, b = snapshot["a"], snapshot["b"]
            incoming, outgoing = augmented_linear(layer.fc1), augmented_linear(layer.fc2)
            surviving = ~(problem.deleted_neurons[index] | incoming_cut | outgoing_cut)
            changed = bool((incoming_cut | outgoing_cut).any())
            if trace is not None and changed:
                cut_in, cut_out = incoming.clone(), outgoing.clone()
                cut_in[incoming_cut] = 0
                cut_out[:, :len(incoming)][:, outgoing_cut] = 0
                copy_augmented(layer.fc1, cut_in)
                copy_augmented(layer.fc2, cut_out)
                trace.measure("cut_only", index, surviving,
                              incoming_cut_count=int(incoming_cut.sum()),
                              outgoing_cut_count=int(outgoing_cut.sum()))
                copy_augmented(layer.fc1, incoming)
                copy_augmented(layer.fc2, outgoing)
            revised_in, revised_out, record = neuron_complement(
                incoming, outgoing,
                a["fc1"], b["fc1"], a["fc2"], b["fc2"],
                problem.deleted_neurons[index], incoming_cut, outgoing_cut)
            copy_augmented(layer.fc1, revised_in)
            copy_augmented(layer.fc2, revised_out)
            if trace is not None and changed:
                update_in = (revised_in - incoming)[surviving]
                free = torch.cat((surviving, surviving.new_tensor([True])))
                update_out = (revised_out - outgoing)[:, free]
                selected = scores[index][incoming_cut, 0].sum() + scores[index][outgoing_cut, 1].sum()
                record.update(singleton_score_sum=float(selected),
                              incoming_update_norm=float(update_in.norm()),
                              outgoing_update_norm=float(update_out.norm()),
                              incoming_relative_update=float(update_in.norm() / incoming[surviving].norm().clamp_min(1e-30)),
                              outgoing_relative_update=float(update_out.norm() / outgoing[:, free].norm().clamp_min(1e-30)))
                trace.measure("compensated", index, surviving, **record)
            probe = snapshot["probe"]
            before = layer.fc2(layer.activation_fn(layer.fc1(probe)))
            canonicalize_linear_chain((layer.fc1, layer.fc2), (layer.activation_fn,))
            after = layer.fc2(layer.activation_fn(layer.fc1(probe)))
            if not torch.allclose(after, before, rtol=1e-5, atol=1e-6):
                raise RuntimeError("canonicalization changed the proposed MLP function")
            actual = ((layer.fc1.weight == 0).all(1) & (layer.fc1.bias == 0) & (layer.fc2.weight == 0).all(0))
            requested = problem.deleted_neurons[index] | incoming_cut | outgoing_cut
            if bool((requested & ~actual).any()):
                raise RuntimeError("canonicalization did not remove the selected neurons")
            saved = int((actual & ~problem.deleted_neurons[index]).sum()) * neuron_size(layer)
            direct = int(incoming_cut.sum()) * (layer.fc1.in_features + 1) + int(outgoing_cut.sum()) * layer.fc2.out_features
            problem.deleted_neurons[index] = actual
            canonical_savings += saved - direct
            record.update(direct_deletions=direct, removed_after_canonicalization=saved,
                          canonical_savings=saved-direct,
                          canonicalization_max_error=float((after-before).abs().max()))
            if trace is not None and changed:
                row = trace.measure("canonicalized", index, surviving, **record)
                trace.archive_layer(index, snapshot, incoming, outgoing, revised_in, revised_out,
                                    (incoming_cut, outgoing_cut), record, row)
            history.append({"layer": index, "neuron_move": record,
                            "b_estimation_seconds": b_seconds if index == 0 else 0.,
                            "b_observations": min(counts[index, name] for name in ("fc1", "fc2")),
                            "a_estimation_seconds": snapshot["a_estimation_seconds"]})
            progress(index + 1, history, canonical_savings, time.perf_counter() - started)
        return {"layers": history, "canonical_savings": canonical_savings,
                "b_estimation_seconds": b_seconds}
