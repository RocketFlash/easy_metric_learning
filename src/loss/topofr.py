import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_distance_matrix(x, p=2):
    x_flat = x.reshape(x.size(0), -1)
    return torch.cdist(x_flat, x_flat, p=p)


class UnionFind:
    def __init__(self, n_vertices):
        self.parent = np.arange(n_vertices, dtype=np.int64)

    def find(self, vertex):
        parent = self.parent[vertex]
        if parent != vertex:
            self.parent[vertex] = self.find(parent)
        return self.parent[vertex]

    def merge(self, source, target):
        source_parent = self.find(source)
        target_parent = self.find(target)
        if source_parent != target_parent:
            self.parent[source_parent] = target_parent


class PersistentHomologyEdges:
    def __call__(self, distance_matrix):
        matrix = distance_matrix.detach().cpu().numpy()
        n_vertices = matrix.shape[0]
        if n_vertices <= 1:
            return np.empty((0, 2), dtype=np.int64)

        union_find = UnionFind(n_vertices)
        triu_indices = np.triu_indices(n_vertices, k=1)
        edge_weights = matrix[triu_indices]
        edge_indices = np.argsort(edge_weights, kind="stable")

        pairs = []
        for edge_index in edge_indices:
            u = triu_indices[0][edge_index]
            v = triu_indices[1][edge_index]
            younger_component = union_find.find(u)
            older_component = union_find.find(v)
            if younger_component == older_component:
                continue
            if younger_component > older_component:
                union_find.merge(v, u)
            else:
                union_find.merge(u, v)
            pairs.append((min(u, v), max(u, v)))
            if len(pairs) == n_vertices - 1:
                break

        return np.asarray(pairs, dtype=np.int64)


class TopologicalSignatureDistance(nn.Module):
    def __init__(self, match_edges=None):
        super(TopologicalSignatureDistance, self).__init__()
        if match_edges not in {None, "symmetric"}:
            raise ValueError("match_edges must be None or 'symmetric'")
        self.match_edges = match_edges
        self.signature_calculator = PersistentHomologyEdges()

    @staticmethod
    def _select_distances(distance_matrix, pairs):
        if pairs.size == 0:
            return distance_matrix.new_empty(0)
        rows = torch.as_tensor(
            pairs[:, 0], dtype=torch.long, device=distance_matrix.device
        )
        cols = torch.as_tensor(
            pairs[:, 1], dtype=torch.long, device=distance_matrix.device
        )
        return distance_matrix[rows, cols]

    @staticmethod
    def _signature_error(signature1, signature2):
        return ((signature1 - signature2) ** 2).sum()

    def forward(self, distances1, distances2):
        pairs1 = self.signature_calculator(distances1)
        pairs2 = self.signature_calculator(distances2)

        signature1 = self._select_distances(distances1, pairs1)
        signature2 = self._select_distances(distances2, pairs2)
        if self.match_edges is None:
            return self._signature_error(signature1, signature2)

        signature1_on_2 = self._select_distances(distances2, pairs1)
        signature2_on_1 = self._select_distances(distances1, pairs2)
        return self._signature_error(
            signature1, signature1_on_2
        ) + self._signature_error(
            signature2_on_1,
            signature2,
        )


class TopoFRLoss(nn.Module):
    """Persistent topology alignment loss over image and embedding spaces."""

    def __init__(
        self,
        p=2,
        match_edges="symmetric",
        max_samples=64,
        normalize_input=True,
        normalize_features=True,
        eps=1e-12,
    ):
        super(TopoFRLoss, self).__init__()
        self.p = p
        self.max_samples = max_samples
        self.normalize_input = normalize_input
        self.normalize_features = normalize_features
        self.eps = eps
        self.topological_distance = TopologicalSignatureDistance(
            match_edges=match_edges
        )

    def _select_samples(self, input_space, feature_space):
        if self.max_samples is None or input_space.size(0) <= self.max_samples:
            return input_space, feature_space

        indices = torch.linspace(
            0,
            input_space.size(0) - 1,
            steps=self.max_samples,
            device=input_space.device,
        ).long()
        return input_space.index_select(0, indices), feature_space.index_select(
            0, indices
        )

    def _normalize_distances(self, distances):
        max_distance = distances.detach().max().clamp_min(self.eps)
        return distances / max_distance

    def forward(self, input_space, feature_space, labels=None):
        del labels
        if input_space.size(0) != feature_space.size(0):
            raise ValueError(
                "input_space and feature_space must have the same batch size"
            )
        if input_space.size(0) <= 1:
            return feature_space.sum() * 0.0

        input_space, feature_space = self._select_samples(input_space, feature_space)
        input_distances = compute_distance_matrix(input_space, p=self.p)
        feature_distances = compute_distance_matrix(feature_space, p=self.p)

        if self.normalize_input:
            input_distances = self._normalize_distances(input_distances)
        if self.normalize_features:
            feature_distances = self._normalize_distances(feature_distances)

        topo_error = self.topological_distance(input_distances, feature_distances)
        return topo_error / float(input_space.size(0))


def gauss_uniform_mixture_1d(values, n_iter=1000, eps=1e-10):
    x = values.reshape(-1, 1).astype(float)
    if x.shape[0] == 0:
        return np.empty(0, dtype=np.float32), 0.0, 0.0

    x_std = x.std(axis=0, keepdims=True)
    x = (x - x.mean(axis=0, keepdims=True)) / np.maximum(x_std, eps)

    n_samples, n_dims = x.shape
    pi = 0.1
    mu = np.zeros((1, n_dims))
    epsilon = x - mu
    sigma = (epsilon.reshape(-1, n_dims, 1) * epsilon.reshape(-1, 1, n_dims)).mean(
        axis=0
    )
    c = 0.2
    previous_log_like = 0.0

    for _ in range(n_iter):
        variance = np.maximum(np.diag(sigma), 1e-3).reshape(1, -1)
        norm = np.sqrt(np.prod(2 * np.pi * variance))
        phi = np.exp(-0.5 * ((x - mu) ** 2 / variance).sum(axis=1)) / norm
        gamma = phi * pi / (phi * pi + (1 - pi) * c + eps)

        n_gaussian = max(gamma.sum(), eps)
        epsilon = x - mu
        sigma = (
            gamma.reshape(-1, 1, 1)
            * epsilon.reshape(-1, n_dims, 1)
            * epsilon.reshape(-1, 1, n_dims)
        ).sum(axis=0) / n_gaussian
        pi = min(n_gaussian / n_samples, 0.5)

        c1 = (((1 - gamma.reshape(-1, 1)) / (1 - pi + eps)) * epsilon).sum(
            axis=0
        ) / n_gaussian
        c2 = (((1 - gamma.reshape(-1, 1)) / (1 - pi + eps)) * (epsilon**2)).sum(
            axis=0
        ) / n_gaussian
        support = 3 * c2 - c1**2
        c = (
            0.001
            if np.any(np.abs(support) < 0.001)
            else 1.0 / (np.prod(2 * np.sqrt(np.maximum(support, eps))) + eps)
        )

        log_like = np.sum(pi * phi + (1 - pi) * c)
        if abs(log_like - previous_log_like) < eps:
            break
        previous_log_like = log_like

    return gamma.astype(np.float32), float(pi), float(np.diag(sigma)[0])


class TopoFRSDECrossEntropyLoss(nn.Module):
    """TopoFR SDE/GUM sample-weighted classification loss."""

    def __init__(
        self,
        temperature=1.0,
        probability_power=1.0,
        alternate_entropy_sign=True,
        ignore_index=-100,
        gum_iters=1000,
        eps=1e-12,
    ):
        super(TopoFRSDECrossEntropyLoss, self).__init__()
        self.temperature = temperature
        self.probability_power = probability_power
        self.alternate_entropy_sign = alternate_entropy_sign
        self.ignore_index = ignore_index
        self.gum_iters = gum_iters
        self.eps = eps

    def forward(self, logits, labels):
        labels = labels.reshape(-1).long()
        valid_mask = (labels != self.ignore_index) & (labels != -1)
        if valid_mask.sum() == 0:
            return logits.sum() * 0.0

        logits = logits[valid_mask]
        labels = labels[valid_mask]
        per_sample_loss = F.cross_entropy(logits, labels, reduction="none")
        probabilities = F.softmax(logits.detach(), dim=1)
        entropy = -(probabilities * torch.log(probabilities.clamp_min(self.eps))).sum(
            dim=1
        )

        entropy_np = entropy.cpu().numpy()
        if self.alternate_entropy_sign and entropy_np.shape[0] > 1:
            entropy_np[2 * np.arange(entropy_np.shape[0] // 2)] *= -1

        sample_weight_np, _, _ = gauss_uniform_mixture_1d(
            entropy_np,
            n_iter=self.gum_iters,
        )
        sample_weight = torch.as_tensor(
            sample_weight_np,
            device=logits.device,
            dtype=logits.dtype,
        )
        probability_gt = probabilities.gather(1, labels.view(-1, 1)).flatten()

        w1 = torch.pow(2.0 - sample_weight, self.temperature)
        w2 = torch.pow(1.0 - probability_gt, self.probability_power)
        return (w1 * w2 * per_sample_loss).mean()
