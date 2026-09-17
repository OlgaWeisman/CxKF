import sys
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np

from tqdm import tqdm
from types import SimpleNamespace
import os
#
# import lightning.pytorch as pl
# from lightning.pytorch import Trainer
# from lightning.pytorch.callbacks import EarlyStopping


# ============================================================
# Add the GitHub repository to Python's path
# ============================================================
y_grid_size_per_y_dim = {
    2: 1e5,
    3: 1e5,
    4: 2e5
}
z_grid_size_per_z_dim = {
    1: 1e4,
    2: 1e4,
    3: 4e4,
    4: 1e5
}
if os.name == "nt":  # Windows
    DR_CP_REPOSITORY = Path(
        r"C:\Users\owner\OneDrive - post.bgu.ac.il\מסמכים\PythonCode"
        r"\multi-output-conformal-regression-master"
        r"\multi-output-conformal-regression-master"
    )
else:  # Linux (BGU server)
    DR_CP_REPOSITORY = Path(
        "/home/weismano/projects/multi-output-conformal-regression-master/multi-output-conformal-regression-master"
    )

if not DR_CP_REPOSITORY.exists():
    raise FileNotFoundError(
        f"Repository was not found:\n{DR_CP_REPOSITORY}"
    )

repository_path = str(DR_CP_REPOSITORY)

if repository_path not in sys.path:
    sys.path.insert(0, repository_path)


import importlib.util
from pathlib import Path

mixture_file = (
    DR_CP_REPOSITORY
    / "moc"
    / "models"
    / "mixture"
    / "mixture_model.py"
)

spec = importlib.util.spec_from_file_location(
    "mixture_model_direct",
    mixture_file,
)

mixture_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mixture_module)

MixtureLightningModule = mixture_module.MixtureLightningModule

from moc.conformal.conformalizers import DR_CP


class DRCPMDN:
    def __init__(
            self,
            input_dim,
            output_dim,
            mixture_size=5,
            hidden_size=100,
            num_layers=3,
            learning_rate=1e-4,
            batch_size=128,
            num_ep=100,
            device=None,
    ):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.batch_size = batch_size
        self.num_ep = num_ep
        self.all_time = []

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        self.model = MixtureLightningModule(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            loss="nll",
            mixture_size=mixture_size,
            lr=learning_rate,
        ).to(self.device)

        # self.model.device = self.device

        # The repository's forward() reads:
        #
        # self.trainer.datamodule.output_dim
        #
        # In this manual training loop there is no Lightning Trainer,
        # so we provide only the attribute required by forward().
        self.model._trainer = SimpleNamespace(
            datamodule=SimpleNamespace(
                output_dim=output_dim
            )
        )

        # Use the repository's optimizer definition
        self.optimizer = self.model.configure_optimizers()

        self.train_losses = []
        self.val_losses = []

    def fit(
            self,
            x_train,
            y_train,
            x_val,
            y_val,
            load_path=None,
    ):
        """
        Train the repository MixtureLightningModule over trajectories.

        Expected shapes
        ----------------
        x_train: [N_train, d_x, T]
        y_train: [N_train, d_y, T]

        x_val:   [N_val, d_x, T]
        y_val:   [N_val, d_y, T]

        For each batch:

            loss = (1 / T) * sum_t loss_t

        where loss_t is computed by the repository MDN step() on
        (x_batch[:, :, t], y_batch[:, :, t]).
        """

        self._validate_trajectory_data(
            x_train,
            y_train,
            name="training",
        )

        self._validate_trajectory_data(
            x_val,
            y_val,
            name="validation",
        )

        if x_train.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, "
                f"but received {x_train.shape[1]}."
            )

        if y_train.shape[1] != self.output_dim:
            raise ValueError(
                f"Expected output_dim={self.output_dim}, "
                f"but received {y_train.shape[1]}."
            )

        if x_train.shape[-1] != y_train.shape[-1]:
            raise ValueError(
                "x_train and y_train must have the same T."
            )

        if x_val.shape[-1] != y_val.shape[-1]:
            raise ValueError(
                "x_val and y_val must have the same T."
            )

        if x_train.shape[-1] != x_val.shape[-1]:
            raise ValueError(
                "Training and validation trajectories must have the same T."
            )

        if load_path is not None:
            checkpoint = torch.load(load_path)

            self.model.load_state_dict(
                checkpoint["model_state_dict"]
            )

            self.model.eval()

            print(f"Loaded model from {load_path}")
            return
        T = x_train.shape[-1]

        train_dataset = TensorDataset(
            x_train.detach().float(),
            y_train.detach().float(),
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        x_val = x_val.detach().float().to(self.device)
        y_val = y_val.detach().float().to(self.device)

        self.train_losses = []
        self.val_losses = []


        for ep in tqdm(
                range(self.num_ep),
                desc="Training repository MDN",
        ):

            self.model.train()

            epoch_train_losses = []
            epoch_train_losses_per_t = []

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                self.optimizer.zero_grad()

                total_loss = torch.zeros(
                    (),
                    device=self.device,
                )

                per_t_train_losses = []

                for t in range(T):
                    x_t = x_batch[:, :, t]  # [B, d_x]
                    y_t = y_batch[:, :, t]  # [B, d_y]

                    # Use the repository's original step():
                    #
                    # def step(self, batch):
                    #     x, y = batch
                    #     dist = self(x)
                    #     loss = self.compute_loss(dist, y)
                    #     return loss
                    # print(f"t = {t}, epoch = {ep}")
                    loss_t = self.model.step(
                        (x_t, y_t)
                    )
                    # print(f"loss = {loss_t}")
                    if not torch.isfinite(loss_t):
                        raise RuntimeError(
                            f"Non-finite loss at epoch {ep + 1}, "
                            f"time {t}: {loss_t.item()}"
                        )

                    total_loss = total_loss + loss_t
                    # print(type(loss_t))
                    per_t_train_losses.append(
                        loss_t.detach().item()
                    )

                # One scalar loss over the whole trajectory.
                loss = total_loss / T

                loss.backward()
                self.optimizer.step()

                epoch_train_losses.append(
                    loss.detach().item()
                )

                epoch_train_losses_per_t.append(
                    per_t_train_losses
                )

            mean_train_loss = float(
                np.mean(epoch_train_losses)
            )

            mean_train_loss_per_t = np.mean(
                np.asarray(epoch_train_losses_per_t),
                axis=0,
            )

            self.train_losses.append(
                mean_train_loss
            )

            # ----------------------------------------------------
            # Validation
            # ----------------------------------------------------

            self.model.eval()

            with torch.no_grad():
                total_val_loss = 0.0
                per_t_val_losses = []

                for t in range(T):
                    x_val_t = x_val[:, :, t]  # [N_val, d_x]
                    y_val_t = y_val[:, :, t]  # [N_val, d_y]

                    val_loss_t = self.model.step(
                        (x_val_t, y_val_t)
                    )

                    if not torch.isfinite(val_loss_t):
                        raise RuntimeError(
                            f"Non-finite validation loss at epoch {ep + 1}, "
                            f"time {t}: {val_loss_t.item()}"
                        )

                    total_val_loss += val_loss_t.item()

                    per_t_val_losses.append(
                        val_loss_t.item()
                    )

                mean_val_loss = total_val_loss / T

            self.val_losses.append(
                mean_val_loss
            )

            tqdm.write(
                f"Epoch {ep + 1:4d}/{self.num_ep} | "
                f"train={mean_train_loss:.6f} | "
                f"val={mean_val_loss:.6f}"
            )

            # Optional: inspect time-wise losses.
            # print("Train losses per t:", mean_train_loss_per_t)
            # print("Val losses per t:", per_t_val_losses)

        self.model.eval()

        return self.model


    @staticmethod
    def _validate_trajectory_data(
            x,
            y,
            name,
    ):
        if not isinstance(x, torch.Tensor):
            raise TypeError(
                f"x_{name} must be a torch.Tensor."
            )

        if not isinstance(y, torch.Tensor):
            raise TypeError(
                f"y_{name} must be a torch.Tensor."
            )

        if x.ndim != 3:
            raise ValueError(
                f"x_{name} must have shape [N, d_x, T], "
                f"received {tuple(x.shape)}."
            )

        if y.ndim != 3:
            raise ValueError(
                f"y_{name} must have shape [N, d_y, T], "
                f"received {tuple(y.shape)}."
            )

        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"x_{name} and y_{name} must contain the "
                "same number of trajectories."
            )

    def calibrate(
        self,
        x_cal: Tensor,
        y_cal: Tensor,
        alpha: Optional[float] = None,
    ) -> Tensor:
        """
        Calibrate using the repository's original DR_CP class.

        Constructing DR_CP computes:

            calib_scores_i
                = DR_CP.get_score(x_i, y_i)
                = nll(model, x_i, y_i)
                = -log f_hat(y_i | x_i)

        Then get_q(alpha) calculates the conformal threshold.
        """

        self._check_model_trained()

        self._validate_xy(
            x_cal,
            y_cal,
            name="calibration",
        )

        self._validate_dimensions(
            x_cal,
            y_cal,
        )

        if alpha is not None:
            if not 0.0 < alpha < 1.0:
                raise ValueError(
                    "alpha must satisfy 0 < alpha < 1."
                )

            self.alpha = alpha

        calibration_dataset = TensorDataset(
            x_cal.detach().float().cpu(),
            y_cal.detach().float().cpu(),
        )

        calibration_loader = DataLoader(
            calibration_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )

        self.model.eval()
        self.model.to(self.device)
        # self.model.device = self.device

        with torch.no_grad():
            self.dr_cp = DR_CP(
                calibration_loader,
                self.model,
            )

            self.q_hat = self.dr_cp.get_q(
                self.alpha
            )

        print(
            f"MDN DR-CP calibrated: "
            f"alpha={self.alpha}, "
            f"q_hat={self.q_hat.item():.6f}"
        )

        return self.q_hat

    def get_calib_scores_trajectory(self, alpha):
        """
        Compute one conformal score per calibration trajectory.

        Per-time score:
            R_{i,t} = get_score(x_{i,t}, y_{i,t})

        Trajectory score:
            R_i = max_t R_{i,t}
        """

        key = ("trajectory", alpha)

        if key not in self.scores_dict:

            trajectory_scores = []

            for (x, y), cache_cal in zip(
                    self.dl_calib,
                    self.cache_calib
            ):
                # Expected shapes:
                # x: [B, d_x, T]
                # y: [B, d_y, T]

                x = x.to(self.model.device)
                y = y.to(self.model.device)

                T = x.shape[-1]

                scores_per_t = []

                for t in range(T):
                    x_t = x[:, :, t]
                    y_t = y[:, :, t]

                    score_t = self.dr_cp.get_score(
                        x_t,
                        y_t,
                        alpha,
                        {}
                    )

                    # score_t: [B]
                    scores_per_t.append(score_t)

                # [B, T]
                scores_per_t = torch.stack(
                    scores_per_t,
                    dim=1
                )

                # One score per trajectory: [B]
                scores_traj = scores_per_t.max(
                    dim=1
                ).values

                trajectory_scores.append(scores_traj)

            # [N_cal]
            trajectory_scores = torch.cat(
                trajectory_scores,
                dim=0
            )

            self.scores_dict[key] = trajectory_scores

        return self.scores_dict[key]

    def get_q_trajectory(self, alpha):
        """
        Compute trajectory-wise conformal quantile.
        """

        return self.dr_cp.conformal_quantile(
            self.get_calib_scores_trajectory(alpha),
            alpha
        )

    def conformal_quantile(self, scores, alpha):
        n = scores.shape[0]
        scores = torch.cat([scores, torch.full((1,) + scores.shape[1:], torch.inf, device=scores.device)], dim=0)
        level = torch.tensor((1 - alpha) * (n + 1)) / (n + 1)
        level = level.type(scores.dtype).to(scores.device)
        return torch.quantile(scores, level, interpolation='higher', dim=0)
    def calibrate_trajectory(
            self,
            x_cal: Tensor,
            y_cal: Tensor,
            alpha: Optional[float] = None,
    ) -> Tensor:
        """
        Trajectory-wise DR-CP calibration.

        Expected shapes:
            x_cal: [N_cal, d_x, T]
            y_cal: [N_cal, d_y, T]

        The trajectory score is:

            R_i = max_t R_{i,t}

        where R_{i,t} is the original DR-CP score.
        """

        self._check_model_trained()

        if alpha is not None:
            if not 0.0 < alpha < 1.0:
                raise ValueError(
                    "alpha must satisfy 0 < alpha < 1."
                )

            self.alpha = alpha

        N, d_x, T = x_cal.shape
        d_y = y_cal.shape[1]

        x_flat = (
            x_cal.permute(0, 2, 1)
            .reshape(N * T, d_x)
        )

        y_flat = (
            y_cal.permute(0, 2, 1)
            .reshape(N * T, d_y)
        )

        calibration_dataset = TensorDataset(
            x_flat.detach().float().cpu(),
            y_flat.detach().float().cpu(),
        )

        calibration_loader = DataLoader(
            calibration_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
        )

        self.model.eval()
        self.model.to(self.device)

        with torch.no_grad():

            self.dr_cp = DR_CP(
                calibration_loader,
                self.model,
            )

            scores = self.dr_cp.calib_scores

            scores = scores.reshape(N, T)

            trajectory_scores = scores.max(dim=1).values
            # Create the dictionary if DR_CP did not create it
            if not hasattr(self.dr_cp, "scores_dict"):
                self.dr_cp.scores_dict = {}

            # Make the original get_calib_scores(alpha)
            # return the trajectory-wise scores
            self.dr_cp.scores_dict[self.alpha] = trajectory_scores
            self.dr_cp.calib_scores = trajectory_scores
            # Now original get_q(alpha) becomes trajectory-wise
            self.q_hat = self.dr_cp.get_q(
                self.alpha
            )

            # # Now this is trajectory-wise q_hat
            # self.q_hat = self.dr_cp.get_q(
            #     self.alpha
            # )

            return self.q_hat
    # def get_score(
    #     self,
    #     x: Tensor,
    #     y: Tensor,
    # ) -> Tensor:
    #     """
    #     Call the repository's original DR_CP.get_score:
    #
    #         score = -log f_hat(y | x).
    #     """
    #
    #     self._check_calibrated()
    #
    #     self._validate_xy(
    #         x,
    #         y,
    #         name="inference",
    #     )
    #
    #     self._validate_dimensions(
    #         x,
    #         y,
    #     )
    #
    #     x = x.detach().float().to(self.device)
    #     y = y.detach().float().to(self.device)
    #
    #     self.model.eval()
    #
    #     with torch.no_grad():
    #         scores = self.dr_cp.get_score(
    #             x,
    #             y,
    #         )
    #
    #     return scores

    def is_in_region(
        self,
        x: Tensor,
        y: Tensor,
    ) -> Tensor:
        """
        Check whether each (x, y) pair belongs to the calibrated DR-CP region.

        Inputs
        ------
        x : [N, d_x]
        y : [N, d_y]

        Returns
        -------
        inside : [N] (bool)
        """

        self._check_calibrated()

        self._validate_xy(
            x,
            y,
            name="inference",
        )

        self._validate_dimensions(
            x,
            y,
        )

        x = x.detach().float().to(self.device)
        y = y.detach().float().to(self.device)

        self.model.eval()

        with torch.no_grad():
            inside = self.dr_cp.is_in_region(
                x=x,
                y=y,
                alpha=self.alpha,
            )
            # inside =self.dr_cp.get_score(x=x,y=y) <= self.q_hat

        return inside.reshape(-1)

    def coverage(
        self,
        x_test: Tensor,
        y_test: Tensor,
    ) -> float:
        """
        Compute empirical coverage:

            mean(1{y_i in C_alpha(x_i)}).
        """

        inside = self.is_in_region(
            x=x_test,
            y=y_test,
        )

        return inside.float().mean().item()

    def save(
        self,
        filename: str,
    ):
        """
        Save the trained MDN parameters and DR-CP settings.
        """

        self._check_model_trained()

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "alpha": self.alpha,
            "mixture_size": self.mixture_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "learning_rate": self.learning_rate,
            "loss": self.loss,
            "q_hat": (
                None
                if self.q_hat is None
                else self.q_hat.detach().cpu()
            ),
        }

        torch.save(
            checkpoint,
            filename,
        )

    def _check_model_trained(self):
        if self.model is None:
            raise RuntimeError(
                "The MDN model has not been trained. "
                "Call fit(x_train, y_train) first."
            )

    def _check_calibrated(self):
        self._check_model_trained()

        if self.dr_cp is None or self.q_hat is None:
            raise RuntimeError(
                "DR-CP has not been calibrated. "
                "Call calibrate(x_cal, y_cal) first."
            )

    def _validate_dimensions(
        self,
        x: Tensor,
        y: Tensor,
    ):
        if x.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected x dimension {self.input_dim}, "
                f"received {x.shape[1]}."
            )

        if y.shape[1] != self.output_dim:
            raise ValueError(
                f"Expected y dimension {self.output_dim}, "
                f"received {y.shape[1]}."
            )

    @staticmethod
    def _validate_xy(
        x: Tensor,
        y: Tensor,
        name: str,
    ):
        if not isinstance(x, Tensor):
            raise TypeError(
                f"x_{name} must be a torch.Tensor."
            )

        if not isinstance(y, Tensor):
            raise TypeError(
                f"y_{name} must be a torch.Tensor."
            )

        if x.ndim != 2:
            raise ValueError(
                f"x_{name} must have shape [N, input_dim], "
                f"received {tuple(x.shape)}."
            )

        if y.ndim != 2:
            raise ValueError(
                f"y_{name} must have shape [N, output_dim], "
                f"received {tuple(y.shape)}."
            )

        if x.shape[0] != y.shape[0]:
            raise ValueError(
                f"x_{name} and y_{name} must have the same "
                "number of samples."
            )

        if x.shape[0] == 0:
            raise ValueError(
                f"The {name} dataset cannot be empty."
            )
    def get_distance_from_quantile_region_new(self,
                                          x,
                                          y,
                                          z_grid,
                                          stride):

        device = self.device

        x = x.to(device).float()
        y = y.to(device).float()
        z_grid = z_grid.to(device).float()

        z_grid_width = (z_grid[:, 0].max() - z_grid[:, 0].min())#/stride[0]
        z_grid_height = (z_grid[:, 1].max() - z_grid[:, 1].min())#/stride[1]
        if z_grid.shape[1] == 3:
            z_grid_depth = z_grid[:, 2].max() - z_grid[:, 2].min()
            grid_volume = z_grid_width * z_grid_height * z_grid_depth
        else:
            grid_volume = z_grid_width * z_grid_height

        z_in_region_mask = self.is_in_region(
            x.repeat(len(z_grid), 1),
            z_grid
        )
        y_in_region_mask = self.is_in_region(
            x,
            y
        )

        # z_in_region_mask = z_in_region_mask.squeeze(1).bool()
        z_in_region = z_grid[z_in_region_mask]

        covered_area = torch.ceil(grid_volume*(len(z_in_region)/len(z_grid)))
        return (
            covered_area,
            y_in_region_mask,
            z_in_region_mask
        )

    def inference_new(self, x_test, y_test, y_train, before_cal_flag=False):
        # if before_cal_flag:
        #     self.lambda_hat_used = torch.tensor(0.0)
        # else:
        #     self.lambda_hat_used = self.lambda_hat
        device = self.device
        x = x_test.to(device).float()
        y_train = y_train.to(device).float()
        grid_step = y_grid_size_per_y_dim[y_train.shape[1]] ** (1 / y_train.shape[1])

        q = 0.01
        border_min = y_train.quantile(q, dim=0)
        border_max = y_train.quantile(1 - q, dim=0)

        border_min -= 2
        border_max += 2

        stride = (border_max - border_min) / grid_step
        shifts = [torch.arange(
            border_min[i], border_max[i], step=stride[i], dtype=torch.float32, device=device
        ) for i in range(border_max.shape[0])]

        z_grid = torch.cartesian_prod(*shifts)

        n = len(x)

        covered_area = torch.zeros(n, device=device)
        in_region = torch.zeros(n, device=device)
        s = torch.zeros(n, device=device)

        for i in tqdm(range(n)):
            (covered_area[i],
             in_region[i], z_in_mask) = self.get_distance_from_quantile_region_new(
                x[i].unsqueeze(0),
                y_test[i].unsqueeze(0),
                z_grid,
                stride
            )
        #     if i == 0:
        #         z_in_region = z_grid[z_in_mask]
        # if not before_cal_flag:
        #     save_dir = "inference_containers"
        #     os.makedirs(save_dir, exist_ok=True)
        #
        #     model_name = f"DR_CP"
        #
        #
        #     save_path = os.path.join(save_dir, f"{model_name}_inference_container.pt")
        #
        #     torch.save(
        #         {
        #             "model_name": model_name,
        #             "z_in_region": z_in_region.detach().cpu(),
        #             "z_grid": z_grid.detach().cpu(),
        #             "x0": x[0].detach().cpu(),
        #             "y0": y_test[0].detach().cpu(),
        #         },
        #         save_path
        #     )
        #
        #     print(f"Saved {save_path}")

        error = (1 - in_region).mean()
        s = 1 - in_region
        mean_total_covered_area = covered_area.mean()
        return error, mean_total_covered_area, s