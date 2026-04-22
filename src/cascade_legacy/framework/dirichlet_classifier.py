"""
Dirichlet Evidential Classifier for uncertainty-aware cascade routing.

Predicts Dirichlet distribution parameters over class probabilities,
enabling decomposition of uncertainty into aleatoric (data noise)
vs epistemic (lack of evidence) components.

Components:
    _DirichletMLP   - MLP mapping features to concentration parameters
    EvidentialLoss   - Type II ML + KL regularization loss
    DirichletClassifier - Public numpy-in/numpy-out API
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import QuantileTransformer


class _DirichletMLP(nn.Module):
    """MLP that outputs positive Dirichlet concentration parameters.

    Architecture: Linear -> ReLU -> Linear -> ReLU -> Linear -> Softplus + 1
    The +1 ensures all alpha_k > 1 so the Dirichlet mode exists.
    """

    def __init__(self, n_features, n_classes, hidden_dims=(128, 64), dropout=0.1,
                 spectral_norm=True, max_logit=20.0):
        super().__init__()
        self.max_logit = max_logit
        layers = []
        prev_dim = n_features
        for h in hidden_dims:
            linear = nn.Linear(prev_dim, h)
            if spectral_norm:
                linear = nn.utils.spectral_norm(linear)
            layers.append(linear)
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = h
        final = nn.Linear(prev_dim, n_classes)
        if spectral_norm:
            final = nn.utils.spectral_norm(final)
        layers.append(final)
        self.net = nn.Sequential(*layers)
        self.softplus = nn.Softplus()

    def forward(self, x):
        """Returns concentration parameters alpha, all > 1."""
        logits = self.net(x)
        logits = logits.clamp(-self.max_logit, self.max_logit)
        return self.softplus(logits) + 1.0


class EvidentialLoss(nn.Module):
    """Evidential deep learning loss combining fit and regularization terms.

    Term 1 (Type II ML): Expected cross-entropy under the Dirichlet.
        L_fit = sum_k y_k * (digamma(S) - digamma(alpha_k))

    Term 2 (KL regularization): Penalizes evidence on wrong classes.
        L_kl = KL(Dir(alpha_tilde) || Dir(1))
        where alpha_tilde removes evidence for the correct class.

    KL weight is annealed linearly from 0 to kl_weight over annealing_steps.
    """

    def __init__(self, kl_weight=0.1, annealing_steps=50, class_weights=None):
        super().__init__()
        self.kl_weight = kl_weight
        self.annealing_steps = annealing_steps
        self.class_weights = class_weights  # (K,) tensor or None

    def forward(self, alpha, y_onehot, epoch):
        """Compute combined loss.

        Args:
            alpha: (batch, K) concentration parameters, all > 1
            y_onehot: (batch, K) one-hot encoded labels
            epoch: current training epoch (for KL annealing)

        Returns:
            Scalar loss tensor.
        """
        S = alpha.sum(dim=1, keepdim=True)  # (batch, 1)

        # Term 1: Type II Maximum Likelihood (Dirichlet-Multinomial)
        l_fit = (y_onehot * (torch.digamma(S) - torch.digamma(alpha))).sum(dim=1)

        # Term 2: KL regularization on wrong-class evidence
        # Remove evidence for correct class before penalizing
        alpha_tilde = y_onehot + (1.0 - y_onehot) * alpha
        l_kl = self._kl_dirichlet_uniform(alpha_tilde)

        # Anneal KL weight
        lam = self.kl_weight * min(1.0, epoch / max(self.annealing_steps, 1))

        per_sample = l_fit + lam * l_kl

        # Apply per-sample class weights (inverse frequency)
        if self.class_weights is not None:
            # y_onehot @ class_weights -> per-sample weight based on true class
            sample_w = (y_onehot * self.class_weights.unsqueeze(0)).sum(dim=1)
            per_sample = per_sample * sample_w

        return per_sample.mean()

    @staticmethod
    def _kl_dirichlet_uniform(alpha):
        """KL(Dir(alpha) || Dir(1,...,1)).

        Args:
            alpha: (batch, K) concentration parameters

        Returns:
            (batch,) KL divergence values.
        """
        K = alpha.shape[1]
        ones = torch.ones_like(alpha)
        S_alpha = alpha.sum(dim=1, keepdim=True)
        S_ones = float(K)

        kl = (
            torch.lgamma(S_alpha.squeeze(1)) - torch.lgamma(torch.tensor(S_ones, device=alpha.device))
            - torch.lgamma(alpha).sum(dim=1) + torch.lgamma(ones).sum(dim=1)
            + ((alpha - ones) * (torch.digamma(alpha) - torch.digamma(S_alpha))).sum(dim=1)
        )
        return kl


class DirichletClassifier:
    """Evidential classifier using Dirichlet distribution over class probabilities.

    Trains an MLP to predict Dirichlet concentration parameters, enabling
    decomposition of prediction uncertainty into aleatoric and epistemic
    components for cascade routing decisions.

    Args:
        n_classes: Number of output classes.
        hidden_dims: Tuple of hidden layer sizes for the MLP.
        lr: Learning rate for Adam optimizer.
        epochs: Number of training epochs.
        batch_size: Mini-batch size.
        kl_weight: Maximum weight for KL regularization term.
        annealing_steps: Epochs over which to linearly anneal KL weight from 0.
        device: Torch device ('cuda', 'cpu', or None for auto-detect).
    """

    def __init__(self, n_classes, hidden_dims=(128, 64), lr=1e-3,
                 epochs=200, batch_size=256, kl_weight=0.1,
                 annealing_steps=50, class_weight='balanced', device=None):
        self.n_classes = n_classes
        self.hidden_dims = hidden_dims
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.kl_weight = kl_weight
        self.annealing_steps = annealing_steps
        self.class_weight = class_weight  # 'balanced', None, or array of K weights

        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self._model = None
        self._n_features = None
        self._scaler = None

    def fit(self, X, y):
        """Train the evidential classifier.

        Args:
            X: np.ndarray of shape (n_samples, n_features).
            y: np.ndarray of shape (n_samples,) with integer labels 0..K-1.

        Returns:
            self
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        self._n_features = X.shape[1]

        # Quantile-transform: maps each feature to a standard normal via ranks.
        # Handles extreme skew (e.g. t_value spans 14 orders of magnitude)
        # that z-score normalization cannot tame.
        self._scaler = QuantileTransformer(
            output_distribution='normal', random_state=42,
            n_quantiles=min(len(X), 1000),
        )
        X = self._scaler.fit_transform(X)

        # One-hot encode labels
        y_onehot = np.zeros((len(y), self.n_classes), dtype=np.float32)
        y_onehot[np.arange(len(y)), y] = 1.0

        X_t = torch.from_numpy(X).to(self.device)
        y_t = torch.from_numpy(y_onehot).to(self.device)

        dataset = TensorDataset(X_t, y_t)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self._model = _DirichletMLP(
            self._n_features, self.n_classes, self.hidden_dims
        ).to(self.device)

        # Compute class weights
        cw_tensor = None
        if self.class_weight is not None:
            if isinstance(self.class_weight, str) and self.class_weight == 'balanced':
                counts = np.bincount(y, minlength=self.n_classes).astype(np.float32)
                counts = np.maximum(counts, 1.0)
                cw = len(y) / (self.n_classes * counts)
                cw_tensor = torch.from_numpy(cw).to(self.device)
            else:
                cw_tensor = torch.as_tensor(self.class_weight, dtype=torch.float32,
                                            device=self.device)

        criterion = EvidentialLoss(
            kl_weight=self.kl_weight, annealing_steps=self.annealing_steps,
            class_weights=cw_tensor,
        )
        optimizer = torch.optim.Adam(self._model.parameters(), lr=self.lr,
                                     weight_decay=1e-4)

        self._model.train()
        for epoch in range(self.epochs):
            for X_batch, y_batch in loader:
                alpha = self._model(X_batch)
                loss = criterion(alpha, y_batch, epoch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        return self

    def predict(self, X):
        """Predict with full uncertainty decomposition.

        Args:
            X: np.ndarray of shape (n_samples, n_features).

        Returns:
            dict with keys:
                'class': (n_samples,) predicted class labels
                'proba': (n_samples, K) expected class probabilities
                'alpha': (n_samples, K) Dirichlet concentration parameters
                'strength': (n_samples,) total evidence S = sum(alpha)
                'uncertainty': dict with:
                    'total': (n_samples,) entropy of expected probabilities
                    'aleatoric': (n_samples,) expected entropy under Dirichlet
                    'epistemic': (n_samples,) total - aleatoric (mutual info)
        """
        alpha = self._predict_alpha(X)
        S = alpha.sum(axis=1)
        proba = alpha / S[:, None]

        # Total uncertainty: H[E[p]] = -sum p_k log p_k
        total = -np.sum(proba * np.log(proba + 1e-10), axis=1)

        # Aleatoric: E[H[p]] = -sum (alpha_k/S)(digamma(alpha_k+1) - digamma(S+1))
        from scipy.special import digamma
        aleatoric = -np.sum(
            (alpha / S[:, None]) * (digamma(alpha + 1) - digamma(S[:, None] + 1)),
            axis=1
        )

        epistemic = total - aleatoric

        return {
            'class': np.argmax(proba, axis=1),
            'proba': proba,
            'alpha': alpha,
            'strength': S,
            'uncertainty': {
                'total': total,
                'aleatoric': aleatoric,
                'epistemic': epistemic,
            }
        }

    def predict_proba(self, X):
        """Return expected class probabilities (sklearn-compatible).

        Args:
            X: np.ndarray of shape (n_samples, n_features).

        Returns:
            np.ndarray of shape (n_samples, K) with rows summing to 1.
        """
        alpha = self._predict_alpha(X)
        S = alpha.sum(axis=1, keepdims=True)
        return alpha / S

    def _predict_alpha(self, X):
        """Run forward pass and return alpha as numpy array."""
        if self._model is None:
            raise RuntimeError("Call fit() before predict()")
        X = np.asarray(X, dtype=np.float32)
        X = self._scaler.transform(X).astype(np.float32)
        X_t = torch.from_numpy(X).to(self.device)
        self._model.eval()
        with torch.no_grad():
            alpha = self._model(X_t).cpu().numpy()
        return alpha
