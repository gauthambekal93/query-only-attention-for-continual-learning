# -*- coding: utf-8 -*-
"""
RanDumb replacement for neural_networks.py in the user's Permuted-MNIST code.

Core interface:
    net.transform(x)
    net.track_estimates(x, y)
    net.inference(x)

No learned neural network, optimizer, loss, backward pass, or FIFO support set.
"""

import math
import torch
import torch.nn as nn


class RanDumb(nn.Module):
    """
    RanDumb-style online classifier.

    Random representation:
        z(x) = sqrt(2 / D) * cos(x @ W + b)

        W_ij ~ N(0, 2 * gamma)
        b_j  ~ Uniform(0, 2*pi)

    W and b are sampled once and remain fixed.

    Online state:
        - running class means in random-feature space
        - shared cumulative within-class scatter matrix

    Inference:
        shared-covariance SLDA discriminant scores.

    NOTE
    ----
    The paper reports 25K Random Fourier Features, gamma=1.0, and
    lambda=1e-6 for MNIST. This implementation retains the full shared
    within-class covariance and applies OAS shrinkage before SLDA scoring.
    """

    def __init__(
        self,
        input_size,
        num_outputs =10,
        num_features=25000,
        gamma=1.0,
        ridge=1e-6,
        seed=0,
        device=None,
    ):
        super().__init__()

        self.input_size = input_size
        self.num_outputs = num_outputs
        self.num_features = num_features
        self.gamma = gamma
        self.ridge = ridge
        self.seed = seed

        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(device)

        self.device_ = device

        # ------------------------------------------------------------
        # Fixed Random Fourier transform.
        #
        # This is the usual sklearn.RBFSampler-equivalent form:
        #
        #   W ~ N(0, 2*gamma)
        #   b ~ Uniform(0, 2*pi)
        #   phi(x) = sqrt(2/D) cos(xW + b)
        #
        # W and b are buffers, not parameters, so they never train.
        # ------------------------------------------------------------
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)

        random_weights = torch.randn(
            input_size,
            num_features,
            generator=gen,
            dtype=torch.float32,
            device=device,
        ) * math.sqrt(2.0 * gamma)

        random_offset = torch.rand(
            num_features,
            generator=gen,
            dtype=torch.float32,
            device=device,
        ) * (2.0 * math.pi)

        self.register_buffer("random_weights", random_weights)
        self.register_buffer("random_offset", random_offset)

        # ------------------------------------------------------------
        # Running class statistics.
        # ------------------------------------------------------------
        self.register_buffer(
            "class_counts",
            torch.zeros(num_outputs, dtype=torch.long, device=device),
        )

        self.register_buffer(
            "class_means",
            torch.zeros(
                num_outputs,
                num_features,
                dtype=torch.float32,
                device=device,
            ),
        )

        # ------------------------------------------------------------
        # Cumulative shared within-class scatter. For D=25,000 this buffer
        # alone occupies about 2.33 GiB in float32, as required by the
        # paper's full-covariance SLDA classifier.
        # ------------------------------------------------------------
        self.register_buffer(
            "within_class_scatter",
            torch.zeros(
                num_features,
                num_features,
                dtype=torch.float32,
                device=device,
            ),
        )

        self.register_buffer(
            "num_seen",
            torch.zeros((), dtype=torch.long, device=device),
        )

        # Cache the D x C discriminant weights rather than a second D x D
        # precision matrix. Any statistics update invalidates this cache.
        self.register_buffer(
            "discriminant_weights",
            torch.zeros(num_features, num_outputs, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "discriminant_bias",
            torch.zeros(num_outputs, dtype=torch.float32, device=device),
        )
        self.register_buffer(
            "cache_valid",
            torch.tensor(False, dtype=torch.bool, device=device),
        )

    @property
    def device(self):
        return self.random_weights.device

    # ================================================================
    # 1. FIXED TRANSFORMATION
    # ================================================================
    @torch.no_grad()
    def transform(self, x):
        """
        Apply the fixed Random Fourier Feature transformation.

        x:
            [batch, input_size]

        returns:
            [batch, num_features]
        """
        x = x.to(self.device, dtype=torch.float32)

        if x.ndim > 2:
            x = x.reshape(x.shape[0], -1)

        if x.shape[1] != self.input_size:
            raise ValueError(
                f"Expected input dimension {self.input_size}, "
                f"but received {x.shape[1]}."
            )

        projection = x @ self.random_weights
        projection = projection + self.random_offset

        return math.sqrt(2.0 / self.num_features) * torch.cos(projection)

    # ================================================================
    # 2. ONLINE ESTIMATE UPDATE
    # ================================================================
    @torch.no_grad()
    def track_estimates(self, x, y):
        """
        Replace optimizer/backprop training.

        For every arriving batch:
            1. transform x with the fixed RFF map
            2. update cumulative class means and within-class scatter

        Nothing is learned with gradient descent.
        """
        y = y.to(self.device, dtype=torch.long)
        z = self.transform(x)

        self._update_statistics(z, y)
        self.cache_valid.fill_(False)

    @torch.no_grad()
    def _update_statistics(self, z, y):
        """
        Exact batch-parallel Welford update of class means and the pooled
        within-class scatter matrix.

        For every class c, the accumulated scatter is

            sum_{i:y_i=c} (z_i - mu_c)(z_i - mu_c)^T.

        It therefore excludes between-class separation from the covariance.
        """
        labels = torch.unique(y)

        for label_tensor in labels:
            label = int(label_tensor.item())

            mask = y == label
            z_c = z[mask]

            batch_count = z_c.shape[0]
            if batch_count == 0:
                continue

            old_count = int(self.class_counts[label].item())
            new_count = old_count + batch_count

            batch_mean = z_c.mean(dim=0)

            centered = z_c - batch_mean
            batch_scatter = centered.T @ centered

            old_mean = self.class_means[label].clone()
            delta = batch_mean - old_mean

            if old_count > 0:
                correction = torch.outer(delta, delta) * (
                    old_count * batch_count / float(new_count)
                )
                batch_scatter.add_(correction)

            self.within_class_scatter.add_(batch_scatter)
            self.class_means[label].copy_(
                old_mean + delta * (batch_count / float(new_count))
            )

            self.class_counts[label] = new_count
            self.num_seen.add_(batch_count)

    # ================================================================
    # 3. COVARIANCE / PRECISION ESTIMATE
    # ================================================================
    @torch.no_grad()
    def get_covariance(self):
        """
        Return the full OAS-shrunk shared within-class covariance.

        This is the same OAS formula used for an assume-centered residual
        matrix, computed from streaming sufficient statistics.
        """
        n = int(self.num_seen.item())

        if n <= 1:
            return torch.eye(
                self.num_features, dtype=torch.float32, device=self.device
            )

        empirical = self.within_class_scatter / float(n)
        empirical = 0.5 * (empirical + empirical.T)

        p = self.num_features
        mu = torch.trace(empirical) / float(p)
        alpha = torch.mean(empirical.square())
        denominator = (n + 1.0) * (alpha - mu.square() / float(p))

        if denominator.item() <= 0.0:
            shrinkage = empirical.new_tensor(1.0)
        else:
            shrinkage = torch.clamp(
                (alpha + mu.square()) / denominator,
                min=0.0,
                max=1.0,
            )

        covariance = (1.0 - shrinkage) * empirical
        covariance.diagonal().add_(shrinkage * mu + self.ridge)
        return covariance

    @torch.no_grad()
    def _refresh_discriminant(self):
        """Solve Sigma W = M and cache the SLDA weights and intercepts."""
        covariance = self.get_covariance()
        means_t = self.class_means.T

        # lstsq is the scoring form used in the authors' released SLDA code
        # and is safer than explicitly forming an inverse.
        weights = torch.linalg.lstsq(covariance, means_t).solution
        bias = -0.5 * torch.sum(means_t * weights, dim=0)

        self.discriminant_weights.copy_(weights)
        self.discriminant_bias.copy_(bias)
        self.cache_valid.fill_(True)

    # ================================================================
    # 4. INFERENCE
    # ================================================================
    @torch.no_grad()
    def inference(self, query_x):
        """
        Return class scores with shape:

            [batch_size, num_outputs]

        Higher score = more likely class.

        This shape intentionally matches the old ERNetwork output so
        existing code can still use:

            predictions.argmax(dim=1)
        """
        z = self.transform(query_x)

        batch_size = z.shape[0]

        # Start every class at -inf so unseen classes can never win.
        scores = torch.full(
            (batch_size, self.num_outputs),
            float("-inf"),
            dtype=z.dtype,
            device=self.device,
        )

        valid_classes = torch.where(self.class_counts > 0)[0]

        if valid_classes.numel() == 0:
            # No estimate exists yet. Returning equal scores gives a
            # deterministic argmax of class 0, but more importantly it
            # lets prequential evaluation run before the first update.
            return torch.zeros(
                (batch_size, self.num_outputs),
                dtype=z.dtype,
                device=self.device,
            )

        if not bool(self.cache_valid.item()):
            self._refresh_discriminant()

        scores[:, valid_classes] = (
            z @ self.discriminant_weights[:, valid_classes]
            + self.discriminant_bias[valid_classes]
        )

        return scores

    # Optional compatibility alias.
    @torch.no_grad()
    def prediction(self, data_manager_obj, query_x):
        """
        Compatibility with the old call signature:

            net.prediction(data_manager_obj, query_x)

        data_manager_obj is deliberately unused; RanDumb does not need
        FIFO support examples.
        """
        return self.inference(query_x)

    # ================================================================
    # OPTIONAL RESET
    # ================================================================
    @torch.no_grad()
    def reset_estimates(self):
        """
        Reset online statistics but KEEP the fixed random transform.
        """
        self.class_counts.zero_()
        self.class_means.zero_()
        self.within_class_scatter.zero_()
        self.num_seen.zero_()
        self.discriminant_weights.zero_()
        self.discriminant_bias.zero_()
        self.cache_valid.fill_(False)
