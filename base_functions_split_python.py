import numpy as np
from typing import Dict, List, Union, Any, Optional, Callable
from dataclasses import dataclass
from sklearn.cluster import KMeans


@dataclass
class PredictionBandsFit:
    """Class to represent fitted prediction bands model"""
    density_fit: Any
    cum_dist_evaluated_train: np.ndarray
    conformity_score_train: np.ndarray
    conformity_score_train_hpd: np.ndarray
    t_grid: np.ndarray
    band: float
    g_train: np.ndarray
    centers_kmeans: np.ndarray


@dataclass
class PredictionBands:
    """Class to represent prediction bands results"""
    y_grid: np.ndarray
    densities: np.ndarray
    ths: Union[np.ndarray, float]
    prediction_bands_which_belong: List[np.ndarray]
    intervals: List[Union[List[tuple], str]]
    type: str
    alpha: float
    th_hpd: Optional[np.ndarray] = None


def kmeanspp(data: np.ndarray, k: int, **kwargs) -> Any:
    """
    K-means++ initialization (fallback to regular k-means if not available)
    """
    try:
        # Try k-means++ initialization
        return KMeans(n_clusters=k, init='k-means++', **kwargs).fit(data)
    except Exception:
        # Fallback to regular k-means
        return KMeans(n_clusters=k, **kwargs).fit(data)


def cum_dist(z: np.ndarray, cde: np.ndarray, y_observed: np.ndarray) -> np.ndarray:
    """
    Compute cumulative distribution distances
    """
    cum_dists = []
    dz = np.diff(z)[0] if len(z) > 1 else 1.0

    for i, y_obs in enumerate(y_observed):
        # Find the cumulative distribution up to y_obs
        cdf = np.cumsum(cde[i, :]) * dz
        # Find the index closest to y_obs
        idx = np.argmin(np.abs(z - y_obs))
        cum_dists.append(cdf[idx])

    return np.array(cum_dists)


def flexcode_fit(x_train: np.ndarray, z_train: np.ndarray,
                 x_validation: np.ndarray, z_validation: np.ndarray,
                 regression_function: Optional[Callable] = None, **kwargs) -> Any:
    """
    Placeholder for FlexCoDE fitting function
    You'll need to implement this or use actual FlexCoDE library
    """

    # This is a placeholder - replace with actual FlexCoDE implementation
    class MockFlexCodeFit:
        def __init__(self):
            self.x_train = x_train
            self.z_train = z_train

    return MockFlexCodeFit()


def flexcode_predict(fit: Any, x_new: np.ndarray) -> Any:
    """
    Placeholder for FlexCoDE prediction function
    """

    # This is a placeholder - replace with actual FlexCoDE implementation
    class MockPrediction:
        def __init__(self, n_samples: int):
            self.z = np.linspace(-3, 3, 100)
            self.CDE = np.random.rand(n_samples, len(self.z))
            # Normalize to make it look like proper densities
            self.CDE = self.CDE / np.sum(self.CDE, axis=1, keepdims=True)

    return MockPrediction(len(x_new))
    """
    Helper function to compute profile density (placeholder implementation)
    This would need to be implemented based on the specific requirements
    """
    # This is a placeholder - you'll need to implement based on your specific needs
    return cde


def which_neighbors(centers: np.ndarray, data: np.ndarray, k: int) -> np.ndarray:
    """
    Find nearest neighbors/partitions (placeholder implementation)
    """
    # Simple nearest neighbor assignment based on Euclidean distance
    distances = np.linalg.norm(data[:, np.newaxis] - centers[np.newaxis, :], axis=2)
    return np.argmin(distances, axis=1)


def compute_intervals(which_belong: np.ndarray, z: np.ndarray) -> List[tuple]:
    """
    Compute intervals from boolean array indicating which points belong
    """
    if not np.any(which_belong):
        return []

    # Find contiguous regions
    diff = np.diff(np.concatenate(([False], which_belong, [False])).astype(int))
    starts = np.where(diff == 1)[0]
    ends = np.where(diff == -1)[0]

    intervals = []
    for start, end in zip(starts, ends):
        intervals.append((z[start], z[end - 1]))

    return intervals


def find_threshold_hpd(band: float, density: np.ndarray, coverage: float) -> float:
    """
    Find threshold for Highest Posterior Density (HPD) intervals
    Placeholder implementation
    """
    # Sort density values in descending order
    sorted_density = np.sort(density)[::-1]
    cumsum_density = np.cumsum(sorted_density)
    # Normalize to get cumulative probability
    cumsum_prob = cumsum_density / cumsum_density[-1]
    # Find threshold where cumulative probability exceeds coverage
    idx = np.where(cumsum_prob >= coverage)[0]
    if len(idx) > 0:
        return sorted_density[idx[0]]
    else:
        return np.min(density)


def predict_prediction_bands(cd_split_fit: PredictionBandsFit,
                             xnew: np.ndarray,
                             type: str = "dist",
                             alpha: float = 0.1) -> PredictionBands:
    """
    Compute conformal prediction bands based on density estimation on new samples

    Parameters:
    -----------
    cd_split_fit : PredictionBandsFit
        Object fitted using fit_prediction_bands function

    xnew : np.ndarray
        New covariates (one per row) where prediction bands are to be computed

    type : str, default="dist"
        Type of prediction bands: "dist" for dist-split, "cd" for cd-split, "hpd" for HPD

    alpha : float, default=0.1
        Miscoverage level (10% by default)

    Returns:
    --------
    PredictionBands
        Object containing:
        - y_grid: Grid of values for y
        - densities: Matrix with estimated densities
        - ths: Thresholds for conformal scores
        - prediction_bands_which_belong: Boolean arrays indicating band membership
        - intervals: Prediction intervals for each input
        - type: Type of prediction band
        - alpha: Miscoverage level
    """

    # Predict using FlexCoDE
    pred_test = flexcode_predict(cd_split_fit.density_fit, xnew)

    if type == "cd":
        prediction_bands_which_belong = []
        intervals = []

        ths = np.full(len(cd_split_fit.conformity_score_train), np.nan)
        g_test = np.full((len(xnew), len(cd_split_fit.t_grid)), np.nan)

        for ii in range(len(xnew)):
            g_test[ii, :] = profile_density(cd_split_fit.t_grid,
                                            pred_test.z,
                                            pred_test.CDE[ii, :])

        which_partition_test = which_neighbors(cd_split_fit.centers_kmeans,
                                               g_test, 1)
        which_partition_train = which_neighbors(cd_split_fit.centers_kmeans,
                                                cd_split_fit.g_train, 1)

        ths_partition = np.full(len(cd_split_fit.centers_kmeans), np.nan)
        for ii in range(len(cd_split_fit.centers_kmeans)):
            partition_scores = cd_split_fit.conformity_score_train[which_partition_train == ii]
            if len(partition_scores) > 0:
                ths_partition[ii] = np.quantile(partition_scores, alpha)

        # Fill NaN values with overall quantile
        nan_mask = np.isnan(ths_partition)
        if np.any(nan_mask):
            overall_quantile = np.quantile(cd_split_fit.conformity_score_train, alpha)
            ths_partition[nan_mask] = overall_quantile

        ths = ths_partition[which_partition_test]

        for ii in range(len(xnew)):
            which_belong = pred_test.CDE[ii, :] >= ths[ii]
            prediction_bands_which_belong.append(which_belong)
            intervals.append(compute_intervals(which_belong, pred_test.z))

        return PredictionBands(
            y_grid=pred_test.z,
            densities=pred_test.CDE,
            ths=ths,
            prediction_bands_which_belong=prediction_bands_which_belong,
            intervals=intervals,
            type="cd",
            alpha=alpha
        )

    elif type == "dist":
        ths = np.quantile(cd_split_fit.cum_dist_evaluated_train,
                          [alpha / 2, 1 - alpha / 2])
        prediction_bands_which_belong = []
        intervals = []
        f_test = np.full((len(xnew), len(pred_test.z)), np.nan)

        dz = np.diff(pred_test.z)[0] if len(pred_test.z) > 1 else 1.0

        for ii in range(len(xnew)):
            f_test[ii, :] = np.cumsum(pred_test.CDE[ii, :]) * dz
            which_belong = (f_test[ii, :] >= ths[0]) & (f_test[ii, :] <= ths[1])
            prediction_bands_which_belong.append(which_belong)

            if np.any(which_belong):
                min_val = np.min(pred_test.z[which_belong])
                max_val = np.max(pred_test.z[which_belong])
                intervals.append(f"({min_val},{max_val})")
            else:
                intervals.append("()")

        return PredictionBands(
            y_grid=pred_test.z,
            densities=pred_test.CDE,
            ths=ths,
            prediction_bands_which_belong=prediction_bands_which_belong,
            intervals=intervals,
            type="dist",
            alpha=alpha
        )

    elif type == "hpd":
        th = np.quantile(cd_split_fit.conformity_score_train_hpd, alpha)
        prediction_bands_which_belong = []
        intervals = []
        th_hpd = np.full(len(xnew), np.nan)

        for ii in range(len(xnew)):
            th_hpd[ii] = find_threshold_hpd(cd_split_fit.band,
                                            pred_test.CDE[ii, :],
                                            1 - th)
            which_belong = pred_test.CDE[ii, :] >= th_hpd[ii]
            prediction_bands_which_belong.append(which_belong)
            intervals.append(compute_intervals(which_belong, pred_test.z))

        return PredictionBands(
            y_grid=pred_test.z,
            densities=pred_test.CDE,
            ths=th,
            prediction_bands_which_belong=prediction_bands_which_belong,
            intervals=intervals,
            type="hpd",
            alpha=alpha,
            th_hpd=th_hpd
        )

    else:
        raise ValueError(f"Type of distribution not implemented ({type})")


# Example usage:
if __name__ == "__main__":
    # Example of how to use the function
    # You would need to prepare cd_split_fit dictionary with appropriate data

    # Mock data for demonstration
    cd_split_fit = {
        'conformity_score_train': np.random.rand(100),
        't_grid': np.linspace(-2, 2, 50),
        'centers_kmeans': np.random.rand(5, 50),
        'g_train': np.random.rand(100, 50),
        'cum_dist_evaluated_train': np.random.rand(100),
        'conformity_score_train_hpd': np.random.rand(100),
        'band': 0.9
    }

    xnew = np.random.rand(10, 5)  # 10 new samples with 5 features each

    # Compute prediction bands
    result = predict_prediction_bands(fitted_model, xnew, type="dist", alpha=0.1)

    print(f"Model fitted with {len(x)} training samples")
    print(f"Prediction bands computed for {len(xnew)} new samples")
    print(f"Type: {result.type}, Alpha: {result.alpha}")
    print(f"Y grid shape: {result.y_grid.shape}")
    print(f"Densities shape: {result.densities.shape}")