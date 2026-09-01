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
        - global running per-feature variance for Mahalanobis-style scaling

    Inference:
        nearest class mean using negative diagonal Mahalanobis distance.

    NOTE
    ----
    The paper reports 25K Random Fourier Features, gamma=1.0, and
    lambda=1e-6 for MNIST. A full 25K x 25K covariance is impractical,
    so this implementation keeps the per-feature variance (diagonal
    covariance). The rest of the continual interface is faithful:
    fixed random features + online class statistics + covariance
    statistics + NCM inference.
    """

    def __init__(
        self,
        input_size,
        num_outputs =10,
        num_features=25000,
        gamma=1.0,
        ridge=1e-6,
        device=None,
    ):
        super().__init__()

        self.input_size = input_size
        self.num_outputs = num_outputs
        self.num_features = num_features
        self.gamma = gamma
        self.ridge = ridge

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
        # Global running variance statistics (Welford).
        # ------------------------------------------------------------
        self.register_buffer(
            "global_mean",
            torch.zeros(num_features, dtype=torch.float32, device=device),
        )

        self.register_buffer(
            "global_M2",
            torch.zeros(num_features, dtype=torch.float32, device=device),
        )

        self.register_buffer(
            "num_seen",
            torch.zeros((), dtype=torch.long, device=device),
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
            2. update running class means
            3. update global covariance/variance statistics

        Nothing is learned with gradient descent.
        """
        y = y.to(self.device, dtype=torch.long)
        z = self.transform(x)

        self._update_class_means(z, y)
        self._update_variance(z)

    @torch.no_grad()
    def _update_class_means(self, z, y):
        """
        Batch-equivalent online class-mean update.
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

            if old_count == 0:
                self.class_means[label].copy_(batch_mean)
            else:
                self.class_means[label].add_(
                    (batch_mean - self.class_means[label])
                    * (batch_count / float(new_count))
                )

            self.class_counts[label] = new_count

    @torch.no_grad()
    def _update_variance(self, z):
        """
        Parallel/batch Welford update for the global feature-wise variance.
        """
        batch_n = z.shape[0]
        if batch_n == 0:
            return

        batch_mean = z.mean(dim=0)

        if batch_n > 1:
            centered = z - batch_mean
            batch_M2 = (centered * centered).sum(dim=0)
        else:
            batch_M2 = torch.zeros_like(batch_mean)

        old_n = int(self.num_seen.item())

        if old_n == 0:
            self.global_mean.copy_(batch_mean)
            self.global_M2.copy_(batch_M2)
            self.num_seen.fill_(batch_n)
            return

        total_n = old_n + batch_n
        delta = batch_mean - self.global_mean

        self.global_mean.add_(delta * (batch_n / float(total_n)))

        self.global_M2.add_(
            batch_M2
            + delta.pow(2) * (old_n * batch_n / float(total_n))
        )

        self.num_seen.fill_(total_n)

    # ================================================================
    # 3. COVARIANCE / PRECISION ESTIMATE
    # ================================================================
    @torch.no_grad()
    def get_variance(self):
        """
        Return diagonal covariance estimate plus ridge regularization.
        """
        n = int(self.num_seen.item())

        if n <= 1:
            return torch.ones(
                self.num_features,
                dtype=torch.float32,
                device=self.device,
            )

        variance = self.global_M2 / float(n - 1)

        # Avoid a nearly-zero dimension dominating Mahalanobis distance.
        variance = variance + self.ridge

        return variance

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
        variance = self.get_variance()

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

        means = self.class_means[valid_classes]

        # [B, C_seen, D]
        diff = z.unsqueeze(1) - means.unsqueeze(0)

        # Diagonal Mahalanobis distance:
        #
        #   sum_j (z_j - mu_cj)^2 / variance_j
        #
        distances = (
            diff.pow(2) / variance.view(1, 1, -1)
        ).sum(dim=-1)

        scores[:, valid_classes] = -distances

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
        self.global_mean.zero_()
        self.global_M2.zero_()
        self.num_seen.zero_()
