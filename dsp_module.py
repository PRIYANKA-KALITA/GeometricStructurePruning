import copy
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class Hook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input_size = 1
        self.flops = 1
        for s in module.weight.size():
            self.flops *= s
        self.flops *= output.size(2) * output.size(3)
        for i in input[0].size():
            self.input_size *= i
        module.flops = self.flops
        module.input_size = self.input_size / (16 * 32 * 32)

    def close(self):
        self.hook.remove()


def _prunable_convs(model: nn.Module) -> List[nn.Conv2d]:
    layers = []
    l = -1
    exclude = ["downsample"]
    for name, layer in model.named_modules():
        if isinstance(layer, nn.Conv2d) and all(e not in name for e in exclude):
            if l == -1:
                l += 1
                continue
            layers.append(layer)
            l += 1
    return layers


class GroupWrapper(nn.Module):
    """
    Geometric DSP grouping:
    - Spherical filter representation
    - Learnable spherical centroids
    - Direction-aware Gumbel-Softmax assignments
    - Orthogonality regularization across group centroids
    """

    def __init__(
        self,
        model,
        n_groups=4,
        tau_geo=0.5,
        group_optimizer_lr=1e-3,
        projection_dim=0,
        kmeans_iters=20,
        kmeans_restarts=4,
        rank=0,
    ):
        super().__init__()
        self.rank = rank
        self.model = model
        self.n_groups = n_groups
        self.tau_geo = tau_geo
        self.projection_dim = projection_dim
        self.kmeans_iters = kmeans_iters
        self.kmeans_restarts = kmeans_restarts
        self.layers = _prunable_convs(model)
        self.centroid_params = []
        self.projectors = nn.ModuleList()

        self.print("Initializing G-DSP GroupWrapper...")
        self.print("Running pre-training K-Means centroid initialization pipeline...")
        self.print("Finding layers to be grouped")
        self.print("=" * 80)
        for idx, layer in enumerate(self.layers):
            cout, cin, k1, k2 = layer.weight.shape
            d = cin * k1 * k2
            proj_d = projection_dim if (projection_dim > 0 and projection_dim < d) else d

            if not hasattr(layer, "group"):
                layer.register_buffer("group", torch.zeros(n_groups, cout, device=layer.weight.device))
            if not hasattr(layer, "prob"):
                layer.register_buffer("prob", torch.zeros(n_groups, cout, device=layer.weight.device))

            projector = nn.Linear(d, proj_d, bias=False).to(layer.weight.device)
            with torch.no_grad():
                projector.weight.zero_()
                projector.weight[:, :proj_d] = torch.eye(proj_d, device=projector.weight.device)
            self.projectors.append(projector)

            centroids = nn.Parameter(torch.randn(n_groups, proj_d, device=layer.weight.device))
            self._run_pretrained_kmeans_pipeline(centroids, layer.weight.data, projector)
            layer.register_parameter("centroids", centroids)
            self.centroid_params.append(layer.centroids)
            self.centroid_params.append(projector.weight)

            self.print(f"[{idx}] {cout} filters, {cin} channels, {k1}x{k2} kernels, proj_dim={proj_d}")

        self.print("=" * 80)
        self.print(f"Number of groups: {self.n_groups}")
        self.print(f"Geometric temperature (tau_geo): {self.tau_geo}")
        self.print(f"Projection dim: {self.projection_dim if self.projection_dim > 0 else 'disabled'}")
        self.print("=" * 80)

        self.group_optimizer = torch.optim.Adam(self.centroid_params, lr=group_optimizer_lr, eps=1e-12)

        hook = []
        for layer in self.layers:
            hook.append(Hook(layer))
        self.model.eval()
        with torch.no_grad():
            self.model(torch.randn(1, 3, 32, 32, device=self.layers[-1].weight.device))
        self.model.train()
        for h in hook:
            h.close()

    @torch.no_grad()
    def _run_pretrained_kmeans_pipeline(self, centroids: torch.Tensor, weight: torch.Tensor, projector: nn.Linear):
        # Strict spherical K-Means on pre-trained filters before differentiable training.
        w = weight.view(weight.size(0), -1)
        z = projector(w)
        z = torch.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
        z = F.normalize(z, dim=1, eps=1e-8)
        n_groups = centroids.size(0)

        best_obj = None
        best_centroids = None
        for _ in range(self.kmeans_restarts):
            c = self._kmeans_pp_init(z, n_groups)
            for _ in range(self.kmeans_iters):
                sim = torch.matmul(z, F.normalize(c, dim=1, eps=1e-8).t())
                assign = sim.argmax(dim=1)
                for g in range(n_groups):
                    idx = (assign == g).nonzero(as_tuple=False).view(-1)
                    if idx.numel() > 0:
                        c[g].copy_(F.normalize(z[idx].mean(dim=0, keepdim=True), dim=1, eps=1e-8).squeeze(0))
            sim = torch.matmul(z, F.normalize(c, dim=1, eps=1e-8).t())
            obj = sim.max(dim=1)[0].mean()
            if (best_obj is None) or (obj > best_obj):
                best_obj = obj
                best_centroids = c.clone()

        if best_centroids is None:
            # Fallback if all restarts become numerically degenerate.
            best_centroids = torch.randn_like(centroids)
        centroids.copy_(F.normalize(torch.nan_to_num(best_centroids, nan=0.0, posinf=0.0, neginf=0.0), dim=1, eps=1e-8))

    @torch.no_grad()
    def _kmeans_pp_init(self, z: torch.Tensor, n_groups: int) -> torch.Tensor:
        # K-Means++ style seeding in cosine space.
        n = z.size(0)
        first = torch.randint(0, n, (1,), device=z.device).item()
        centers = [z[first]]
        for _ in range(1, n_groups):
            c = F.normalize(torch.stack(centers, dim=0), dim=1, eps=1e-8)
            sim = torch.matmul(z, c.t())
            min_dist = 1.0 - sim.max(dim=1)[0]
            min_dist = torch.nan_to_num(min_dist, nan=0.0, posinf=0.0, neginf=0.0)
            min_dist = torch.clamp(min_dist, min=0.0)
            denom = min_dist.sum()
            if (not torch.isfinite(denom)) or (denom <= 1e-12):
                probs = torch.full_like(min_dist, 1.0 / max(1, min_dist.numel()))
            else:
                probs = min_dist / denom
            nxt = torch.multinomial(probs, 1).item()
            centers.append(z[nxt])
        out = torch.stack(centers, dim=0)
        out = torch.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return F.normalize(out, dim=1, eps=1e-8)

    def print(self, *args):
        if self.rank == 0:
            print(*args)

    def _filter_centroid_cosine(self, idx: int, layer: nn.Conv2d) -> torch.Tensor:
        # Returns cosine matrix with shape [Cout, N_groups].
        w = layer.weight.view(layer.weight.size(0), -1)
        z = self.projectors[idx](w)
        w_hat = F.normalize(z, dim=1)
        c_hat = F.normalize(layer.centroids, dim=1)
        return torch.matmul(w_hat, c_hat.t())

    def initialize(self):
        # Keep a detached cache for pruning/statistics only.
        with torch.no_grad():
            for idx, layer in enumerate(self.layers):
                cosine = self._filter_centroid_cosine(idx, layer)
                logits = cosine / self.tau_geo
                prob = F.gumbel_softmax(logits, dim=1)
                layer.prob.copy_(prob.t())

    @torch.no_grad()
    def harden_assignments(self):
        for layer in self.layers:
            index = layer.prob.max(dim=0, keepdim=True)[1]
            layer.group.copy_(torch.zeros_like(layer.prob).scatter_(0, index, 1.0))
            layer.prob.copy_(layer.group)

    def geometric_alignment_loss(self):
        loss = 0.0
        n = 0
        for idx, layer in enumerate(self.layers):
            cosine = self._filter_centroid_cosine(idx, layer)
            # Build a fresh differentiable assignment each step to avoid
            # reusing stale autograd graphs from cached tensors.
            logits = cosine / self.tau_geo
            prob_t = F.gumbel_softmax(logits, dim=1)
            with torch.no_grad():
                layer.prob.copy_(prob_t.t())
            loss = loss - (prob_t * cosine).sum(dim=1).mean()
            n += 1
        return loss / max(1, n)

    def orthogonality_loss(self):
        loss = 0.0
        n = 0
        for layer in self.layers:
            c_hat = F.normalize(layer.centroids, dim=1)
            gram = torch.matmul(c_hat, c_hat.t())
            off_diag = gram - torch.eye(gram.size(0), device=gram.device)
            loss = loss + (off_diag ** 2).sum() / 2.0
            n += 1
        return loss / max(1, n)

    def zero_grad(self):
        self.group_optimizer.zero_grad(True)

    def step(self):
        self.group_optimizer.step()

    @torch.no_grad()
    def stats(self):
        conf = [layer.prob.max(dim=0)[0].mean().item() for layer in self.layers]
        return sum(conf) / len(conf)

    def forward(self, x):
        return self.model(x)


class PruneWrapper(nn.Module):
    def __init__(self, model, n_groups=None, fp_every_nth_conv=None, fp_layer_indices=None, rank=0):
        super(PruneWrapper, self).__init__()
        self.rank = rank
        self.print("Initializing...")
        self.model = model
        self.layers = []
        self.fp_layers = []
        if fp_layer_indices is not None:
            fp_every_nth_conv = 2 ** 32
        else:
            fp_layer_indices = []
            if fp_every_nth_conv is None:
                self.print("Please provide one of fp_every_nth_conv and fp_layer_indices.")
                self.print("If you don't want filter pruning, please set fp_every_nth_conv=-1 or fp_layer_indices=[]")
                raise ValueError
            elif fp_every_nth_conv == -1:
                fp_every_nth_conv = 2 ** 32
        exclude = ["downsample"]
        self.beta = 0

        l = -1
        self.print("Finding layers to be pruned")
        self.print("=" * 80)
        for name, layer in model.named_modules():
            if isinstance(layer, nn.Conv2d) and all(e not in name for e in exclude):
                if l == -1:
                    l += 1
                    continue
                layer.register_buffer("group", torch.zeros(n_groups, layer.weight.size(0), device=layer.weight.device))
                layer.register_buffer("mask", torch.ones(layer.weight.size(0), layer.weight.size(1), 1, 1, device=layer.weight.device))
                self.layers.append(layer)
                w_dim = layer.weight.size()
                self.print(f"[{l}] {name}: {w_dim[0]} filters, {w_dim[1]} channels, {w_dim[2]}x{w_dim[3]} kernels")
                if ((l + 1) % fp_every_nth_conv == 0) or (l in fp_layer_indices):
                    self.fp_layers.append(layer)
                l += 1

        self.n_groups = n_groups

        hook = []
        for layer in self.layers:
            hook.append(Hook(layer))
        self.model.eval()
        with torch.no_grad():
            self.model(torch.randn(1, 3, 32, 32, device=layer.weight.device))
        self.model.train()
        for h in hook:
            h.close()

    @torch.no_grad()
    def set_arch_hard(self, layer):
        index = layer.group.max(dim=0, keepdim=True)[1]
        layer.prob = torch.zeros_like(layer.group).scatter_(0, index, 1.0)

    @torch.no_grad()
    def find_mask(self, layer):
        layer.mask.fill_(1)
        importance = layer.weight.data ** 2

        imp = torch.stack([((p.view(-1, 1, 1, 1) ** 2) * importance).sum(dim=(3, 2, 0)) for p in layer.prob], dim=0)
        imp = imp / (imp.sum(dim=1, keepdim=True) + 1e-12)
        rank = imp.sort(dim=1)[0]
        csoi = rank.cumsum(dim=1)
        count = (csoi < self.beta).long().sum(dim=1)
        th = rank[torch.arange(rank.size(0)), count - 1].unsqueeze(1)
        mask = (layer.prob.unsqueeze(2) * (imp > th).float().unsqueeze(1)).sum(0)

        layer.mask.copy_(mask.view(mask.size(0), mask.size(1), 1, 1))

    @torch.no_grad()
    def find_mask_fp(self, layer):
        importance = layer.weight.data ** 2

        imp = importance.sum(dim=(3, 2, 1))
        imp = imp / (imp.sum() + 1e-12)
        rank = imp.sort(dim=0)[0]
        csoi = rank.cumsum(dim=0)
        count = (csoi < self.beta).long().sum(dim=0)
        th = rank[count - 1]
        mask = (imp > th).float().unsqueeze(1)

        layer.mask.mul_(mask.view(mask.size(0), mask.size(1), 1, 1))

    @torch.no_grad()
    def apply_mask(self, layer):
        layer.weight.mul_(layer.mask)

    @torch.no_grad()
    def residual_bn_proc(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.bias.mul_((m.weight.abs() > 0).float())

    def apply(self, func, inputs):
        return list(map(func, inputs))

    def print(self, *args):
        if self.rank == 0:
            print(*args)

    def initialize(self, rate, n_iter=10):
        self.print("=" * 80)
        self.print("Finding pruning settings to achieve the target pruning rate")
        self.print("=" * 80)
        self.apply(self.set_arch_hard, self.layers)
        checkpoints = copy.deepcopy(self.model.state_dict())
        self.beta = 0.15
        lower, upper = 0, 1.0
        for _ in range(n_iter):
            pflops, pparams = self.prune()
            if pflops > rate * 100:
                temp = self.beta
                self.beta = (self.beta + lower) / 2
                upper = temp
            else:
                temp = self.beta
                self.beta = (self.beta + upper) / 2
                lower = temp
            self.model.load_state_dict(checkpoints)
        pflops, pparams = self.prune(True)
        return pflops, pparams

    def prune(self, verbose=False):
        self.apply(self.find_mask, self.layers)
        self.apply(self.find_mask_fp, self.fp_layers)
        self.apply(self.apply_mask, self.layers)
        for _ in range(125):
            out = self.model(torch.randn(80, 3, 32, 32).cuda())
            F.cross_entropy(out, torch.randint(0, out.size(1), (80,)).cuda()).backward()

        with torch.no_grad():
            for m in self.model.modules():
                if isinstance(m, nn.Conv2d):
                    m.weight.mul_((m.weight.grad.abs().sum(dim=(3, 2, 1), keepdim=True) > 0).float())
                    m.weight.mul_((m.weight.grad.abs().sum(dim=(3, 2, 0), keepdim=True) > 0).float())
                    if hasattr(m, "mask"):
                        m.mask.mul_((m.weight.grad.abs().sum(dim=(3, 2, 1), keepdim=True) > 0).float())
                        m.mask.mul_((m.weight.grad.abs().sum(dim=(3, 2, 0), keepdim=True) > 0).float())
                elif isinstance(m, nn.BatchNorm2d):
                    m.weight.mul_((m.weight.grad.abs() > 0).float())
                    m.bias.mul_((m.weight.grad.abs() > 0).float())

            pflops, pparams = self.summary(verbose)
        self.model.zero_grad(True)
        return pflops, pparams

    def summary(self, verbose=False, init=False):
        if init:
            self.apply(self.set_arch_hard, self.layers)
        remaining_flops = 0
        remaining_params = 0
        total_flops = 0
        total_params = 0
        for n, layer in enumerate(self.layers):
            kernels = (layer.weight.abs().sum(dim=(3, 2)) > 0).float()
            remaining = torch.mm(layer.prob, kernels)
            r_ch = (remaining > 0).float().sum(dim=1)
            r_f = (remaining.sum(1) / (r_ch + 1e-8)).round()
            remaining_flops += layer.flops * kernels.sum().item() / kernels.numel()
            remaining_params += layer.weight.numel() * kernels.sum().item() / kernels.numel()
            total_flops += layer.flops
            total_params += layer.weight.numel()
            if verbose:
                self.print("[%d] FLOPS: %2.2f%%" % (n, 100 * kernels.sum().item() / kernels.numel()), "Structure:", *list(zip(r_f.long().tolist(), r_ch.long().tolist())))
        pflops = 100 * (1 - remaining_flops / total_flops)
        pparams = 100 * (1 - remaining_params / total_params)
        if verbose:
            self.print("=" * 80)
            self.print("Summary")
            self.print(f"Beta: {self.beta}")
            self.print(f"FLOPS: {int(remaining_flops)} ({pflops}% pruned)")
            self.print(f"PARAMS: {int(remaining_params)} ({pparams}% pruned)")
            self.print("=" * 80)
        return pflops, pparams

    def after_step(self):
        self.apply(self.apply_mask, self.layers)
        self.residual_bn_proc()

    def forward(self, x):
        return self.model(x)


@torch.no_grad()
def _ensure_conv_masks(model: nn.Module):
    for conv in _prunable_convs(model):
        if not hasattr(conv, "mask"):
            conv.register_buffer(
                "mask",
                torch.ones(conv.weight.size(0), conv.weight.size(1), 1, 1, device=conv.weight.device),
            )


def _set_module_by_name(root: nn.Module, name: str, module: nn.Module):
    parts = name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    setattr(parent, parts[-1], module)


@torch.no_grad()
def build_compact_model_from_masks(model: nn.Module) -> nn.Module:
    # Structurally rebuild ResNetBasicblock modules into Compactblock modules.
    from cifar_model import Compactblock, ResNetBasicblock

    for name, m in list(model.named_modules()):
        if isinstance(m, ResNetBasicblock) and hasattr(m.conv_a, "mask") and hasattr(m.conv_b, "mask"):
            compact = Compactblock().to(next(m.parameters()).device)
            compact.compact(m)
            _set_module_by_name(model, name, compact)
    return model


@torch.no_grad()
def adaptive_geometric_fusion(model: nn.Module, n_groups: int, gamma_merge: float = 0.95) -> Tuple[nn.Module, List[Tuple[int, int, int]]]:
    """
    Post-training adaptive intra-group filter fusion:
    - Find highly similar filters within each learned group.
    - Merge filter j into i with compensation on next layer channels.
    - Zero pruned filter/channel (structural compaction can be run later).
    Returns list of (layer_idx, keep_i, prune_j) merges.
    """
    merges = []
    _ensure_conv_masks(model)
    convs = _prunable_convs(model)
    if len(convs) < 2:
        return model, merges

    for l in range(len(convs) - 1):
        curr = convs[l]
        nxt = convs[l + 1]
        if not hasattr(curr, "group"):
            continue
        assign = curr.group.argmax(dim=0)
        alive = torch.ones(curr.weight.size(0), dtype=torch.bool, device=curr.weight.device)
        w_flat = curr.weight.view(curr.weight.size(0), -1)

        for p in range(n_groups):
            idx = (assign == p).nonzero(as_tuple=False).view(-1).tolist()
            changed = True
            while changed:
                changed = False
                best_pair = None
                best_sim = -1.0
                for a in range(len(idx)):
                    i = idx[a]
                    if not alive[i]:
                        continue
                    wi = w_flat[i]
                    for b in range(a + 1, len(idx)):
                        j = idx[b]
                        if not alive[j]:
                            continue
                        wj = w_flat[j]
                        sim = F.cosine_similarity(wi.unsqueeze(0), wj.unsqueeze(0)).item()
                        if sim > gamma_merge and sim > best_sim:
                            best_sim = sim
                            best_pair = (i, j)
                if best_pair is None:
                    continue

                i, j = best_pair
                wi_norm = torch.norm(w_flat[i], p=2)
                wj_norm = torch.norm(w_flat[j], p=2)
                beta = (wj_norm / (wi_norm + 1e-12)).item()

                nxt.weight[:, i, :, :].add_(beta * nxt.weight[:, j, :, :])
                curr.weight[j].zero_()  # temporary; compact pass will remove physically
                nxt.weight[:, j].zero_()  # temporary; compact pass will remove physically
                curr.mask[j].zero_()
                nxt.mask[:, j].zero_()
                curr.group[:, j].zero_()
                alive[j] = False
                merges.append((l, i, j))
                changed = True

    compact_model = build_compact_model_from_masks(model)
    return compact_model, merges
