import torch
import torch.nn as nn
from scipy.stats import chi2

class MahalanobisEllipticalRegionNoLearning(nn.Module):

    def __init__(self, dim, alpha=0.1, eps=1e-6):
        super().__init__()

        self.dim = dim
        self.alpha = alpha
        self.eps = eps

        r0_value = chi2.ppf(1.0 - alpha, df=dim)

        r0 = torch.tensor(
            r0_value ** 0.5,
            dtype=torch.float32
        )

        self.register_buffer("r0", r0)
        self.register_buffer("lambda_hat", torch.tensor(0.0))

    def mahalanobis_distance(self, y, mu, Sigma):

        diff = y - mu  # [batch, dim]

        eye = torch.eye(
            self.dim,
            device=y.device,
            dtype=y.dtype
        ).unsqueeze(0)

        Sigma = Sigma + self.eps * eye

        sol = torch.linalg.solve(
            Sigma,
            diff.unsqueeze(-1)
        ).squeeze(-1)

        dist_sq = torch.sum(diff * sol, dim=1)

        return torch.sqrt(
            torch.clamp(dist_sq, min=0.0)
        )

    def forward(self, mu, Sigma):

        radius = self.r0 + self.lambda_hat

        return {
            "center": mu,
            "covariance": Sigma,
            "radius": radius
        }

    def is_in_region(self, y, mu, Sigma):

        dist = self.mahalanobis_distance(
            y,
            mu,
            Sigma
        )

        radius = self.r0 + self.lambda_hat

        mask = (dist <= radius).unsqueeze(1)

        return mask

    def calibrate_crc(self, y_cal, mu_cal, Sigma_cal):

        n = y_cal.shape[0]

        B = 1.0
        threshold = self.alpha - B / n

        with torch.no_grad():

            dist = self.mahalanobis_distance(
                y_cal,
                mu_cal,
                Sigma_cal
            )

            residuals = dist - self.r0

            candidate_lambdas, _ = torch.sort(
                residuals
            )

            for lam in candidate_lambdas:

                radius = self.r0 + lam

                risk = (
                    dist > radius
                ).float().mean()

                if risk <= threshold:

                    self.lambda_hat.copy_(lam)

                    return lam

            self.lambda_hat.copy_(
                candidate_lambdas[-1]
            )

            return self.lambda_hat