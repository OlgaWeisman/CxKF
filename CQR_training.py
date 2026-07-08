import copy
import sys
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


class all_q_model(nn.Module):
    """Conditional quantile estimator."""

    def __init__(self, quantiles, in_shape=1, hidden_size=64, dropout=0.5):
        super().__init__()

        self.quantiles = quantiles
        self.num_quantiles = len(quantiles)
        self.hidden_size = hidden_size
        self.in_shape = in_shape
        self.out_shape = len(quantiles)
        self.dropout = dropout

        self.build_model()
        self.init_weights()

    def build_model(self):
        self.base_model = nn.Sequential(
            nn.Linear(self.in_shape, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),

            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(self.dropout),

            nn.Linear(self.hidden_size, self.num_quantiles),
        )

    def init_weights(self):
        for m in self.base_model:
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        return self.base_model(x)


class AllQuantileLoss(nn.Module):
    """Pinball loss for several quantiles."""

    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = torch.tensor(quantiles).float()

    def forward(self, preds, target):
        """
        preds:  [batch, num_quantiles]
        target: [batch]
        """

        if target.ndim > 1:
            target = target.squeeze()

        quantiles = self.quantiles.to(preds.device)

        losses = []
        for i, q in enumerate(quantiles):
            errors = target - preds[:, i]
            loss_q = torch.max((q - 1) * errors, q * errors)
            losses.append(loss_q.unsqueeze(1))

        losses = torch.cat(losses, dim=1)
        return torch.mean(torch.sum(losses, dim=1))


def epoch_internal_train(model, loss_func, x_train, y_train, batch_size, optimizer):
    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    total_loss = 0.0
    T = y_train.shape[-1]
    loss_per_seq = torch.zeros(T, dtype=torch.float)
    for xb, yb in loader:
        optimizer.zero_grad()
        for t in range(T):

            preds = model(xb[:,:,t])
            loss_per_seq[t] = loss_func(preds, yb[:,t])

        loss = loss_per_seq.mean()

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / xb.shape[0]


def compute_coverage_len(y, y_lower, y_upper):
    y = np.squeeze(y)
    T = y.shape[-1]
    y_lower_rep = np.tile(y_lower[:, None], (1, T))
    y_upper_rep = np.tile(y_upper[:, None], (1, T))
    covered = (y >= y_lower_rep) & (y <= y_upper_rep)
    coverage = np.mean(covered)
    avg_length = np.mean(y_upper - y_lower)

    return coverage, avg_length


class AllQuantileRegressor:
    def __init__(
        self,
        quantiles,
        in_shape=1,
        hidden_size=64,
        dropout=0.5,
        lr=1e-3,
        test_ratio=0.2,
        random_state=0,
        target_coverage=0.9,
        device=None,
    ):
        self.quantiles = np.array(quantiles)
        self.quantile_low = np.min(self.quantiles)
        self.quantile_high = np.max(self.quantiles)

        self.in_shape = in_shape
        self.hidden_size = hidden_size
        self.dropout = dropout

        self.test_ratio = test_ratio
        self.random_state = random_state
        self.target_coverage = target_coverage

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.model = all_q_model(
            quantiles=self.quantiles,
            in_shape=self.in_shape,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
        ).to(self.device)

        self.loss_func = AllQuantileLoss(self.quantiles)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.loss_history = []
        self.test_loss_history = []
        self.full_loss_history = []

    def fit(self, x, y, epochs=1000, batch_size=64, verbose=False):
        sys.stdout.flush()

        x = np.asarray(x).astype(np.float32)
        y = np.asarray(y).astype(np.float32).squeeze()

        x_train, x_val, y_train, y_val = train_test_split(
            x,
            y,
            test_size=self.test_ratio,
            random_state=self.random_state,
        )

        x_train = torch.from_numpy(x_train).float().to(self.device)
        y_train = torch.from_numpy(y_train).float().to(self.device)

        x_val_t = torch.from_numpy(x_val).float().to(self.device)
        y_val_t = torch.from_numpy(y_val).float().to(self.device)

        best_model_state = None
        best_avg_length = np.inf
        best_coverage = 0
        best_epoch = 0

        for e in range(epochs):
            self.model.train()

            epoch_loss = epoch_internal_train(
                self.model,
                self.loss_func,
                x_train,
                y_train,
                batch_size,
                self.optimizer,
            )

            self.loss_history.append(epoch_loss)
            val_loss_t = torch.zeros(x_val_t.shape[-1], dtype=torch.float)
            self.model.eval()
            with torch.no_grad():
                for t in range(x_val_t.shape[-1]):
                    preds = self.model(x_val_t[:,:,t])
                    val_loss_t[t] = self.loss_func(preds, y_val_t[:,t]).item()
            val_loss = val_loss_t.mean()
            self.test_loss_history.append(val_loss)

            test_preds = preds.cpu().numpy()

            y_lower = np.min(test_preds, axis=1)
            y_upper = np.max(test_preds, axis=1)

            coverage, avg_length = compute_coverage_len(y_val, y_lower, y_upper)

            if coverage >= self.target_coverage and avg_length < best_avg_length:
                best_avg_length = avg_length
                best_coverage = coverage
                best_epoch = e
                best_model_state = copy.deepcopy(self.model.state_dict())

            if verbose and (e + 1) % 100 == 0:
                print(
                    f"Epoch {e+1}: "
                    f"Train loss={epoch_loss:.4f}, "
                    f"Val loss={val_loss:.4f}, "
                    f"Coverage={coverage:.4f}, "
                    f"Best coverage={best_coverage:.4f}, "
                    f"Best length={best_avg_length:.4f}, "
                    f"Best epoch={best_epoch}"
                )
                sys.stdout.flush()

        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

    def predict(self, x):
        x = np.asarray(x).astype(np.float32)

        self.model.eval()

        with torch.no_grad():
            preds = np.zeros((x.shape[0], 2, x.shape[-1]))
            for t in range(x.shape[-1]):
                x_t = torch.from_numpy(x[:,:,t]).float().to(self.device)
                preds[:, :, t] = self.model(x_t)
        y_lower = np.min(preds, axis=1)
        y_upper = np.max(preds, axis=1)

        interval_preds = np.stack([y_lower, y_upper], axis=1)

        return interval_preds, preds