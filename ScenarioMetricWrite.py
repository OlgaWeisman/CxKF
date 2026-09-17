import csv
import os
import re


class ScenarioMetricsWriter:
    """
    Per-scenario CSV with multiple algorithms.
    Columns:
    [scenario, algo, snr_db, C_mean, C_svd, S, WI_mean, WI_svd]
    """

    def __init__(self, scenario: str,
                 coverage_mode: str,
                 outdir: str = "outputs",
                 prefix: str = "metrics"):

        self.scenario = scenario
        self.coverage_mode = coverage_mode
        self.outdir = outdir
        self.prefix = prefix

        self.fieldnames = [
            "scenario",
            "coverage_mode",
            "algo",
            "snr_db",
            "C_mean",
            "C_svd",
            "S",
            "WI_mean",
            "WI_svd",
        ]

        os.makedirs(self.outdir, exist_ok=True)

        safe_scenario = re.sub(r"[^A-Za-z0-9._-]+", "_", scenario)
        self.path = os.path.join(
            self.outdir,
            f"{self.prefix}_{safe_scenario}.csv"
        )

        if (not os.path.exists(self.path)) or (os.path.getsize(self.path) == 0):
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                csv.DictWriter(f, fieldnames=self.fieldnames).writeheader()

    def append(self,
               *,
               algo: str,
               snr_db: float,
               C_mean: float,
               C_svd: float,
               S: float,
               WI_mean: float,
               WI_svd: float):

        row = {
            "scenario": self.scenario,
            "coverage_mode": self.coverage_mode,
            "algo": algo,
            "snr_db": snr_db,
            "C_mean": C_mean,
            "C_svd": C_svd,
            "S": S,
            "WI_mean": WI_mean,
            "WI_svd": WI_svd,
        }

        with open(self.path, "a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=self.fieldnames).writerow(row)

    def append_metrics(self,
                       *,
                       algo: str,
                       snr_db: float,
                       C,
                       S,
                       WI):
        """
        C  : tensor/ndarray of coverage values
        S  : scalar
        WI : tensor/ndarray of interval widths
        """

        self.append(
            algo=algo,
            snr_db=snr_db,
            C_mean=float(C.mean()),
            C_svd=float(C.std()),
            S=float(S),
            WI_mean=float(WI.mean()),
            WI_svd=float(WI.std()),
        )

    def filepath(self):
        return self.path