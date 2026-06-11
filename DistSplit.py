import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import numpy2ri, default_converter
from rpy2.robjects.conversion import localconverter
import warnings

# Suppress R warnings for cleaner output
warnings.filterwarnings('ignore')


def install_predictionBands_packages():
    """
    Install the required R packages: FlexCoDE and predictionBands
    """
    print("Installing R packages...")

    # Install devtools if not already installed
    robjects.r('''
        if (!require("devtools", quietly = TRUE)) {
            install.packages("devtools", repos="https://cran.r-project.org")
        }
    ''')

    # Install FlexCoDE and predictionBands from GitHub
    robjects.r('''
        library(devtools)

        # Install FlexCoDE (required dependency)
        if (!require("FlexCoDE", quietly = TRUE)) {
            install_github("rizbicki/FlexCoDE")
        }

        # Install predictionBands
        if (!require("predictionBands", quietly = TRUE)) {
            install_github("rizbicki/predictionBands")
        }

        # Load the libraries
        library(FlexCoDE)
        library(predictionBands)
    ''')

    print("✓ R packages installed successfully!")


# def setup_prediction_bands_example(x, y, xnew, ynew):
#     """
#     Complete working example with predictionBands
#     """
#
#     # First, install the packages (run this once)
#     try:
#         install_predictionBands_packages()
#     except Exception as e:
#         print(f"Package installation failed: {e}")
#         print("Please install manually in R console:")
#         print('devtools::install_github("rizbicki/FlexCoDE")')
#         print('devtools::install_github("rizbicki/predictionBands")')
#         return
#
#     print(f"Training data: {x.shape}, {y.shape}")
#     print(f"Test data: {xnew.shape}, {ynew.shape}")
#
#     # Convert numpy arrays to R and load libraries
#     with localconverter(default_converter + numpy2ri.converter):
#         robjects.globalenv["x"] = x
#         robjects.globalenv["y"] = y
#         robjects.globalenv["xnew"] = xnew
#         robjects.globalenv["ynew"] = ynew
#
#     print("\n" + "=" * 50)
#     print("RUNNING PREDICTION BANDS")
#     print("=" * 50)
#     robjects.r('''
#         library(FlexCoDE)
#         library(predictionBands)
#
#
#         # Fit prediction bands model
#         cat("Fitting predictionBands model...\n")
#         fit <- fit_predictionBands(x, y, 0.5, 0.4, 0.1)
#         cat("Model fitted successfully!\n\n")
#
#         # Predict with dist-split method
#         cat("=== DIST-SPLIT PREDICTION BANDS ===\n")
#         bands_dist <- predict(fit, xnew, type="dist")
#
#         # Extract intervals - they're stored as strings in bands_dist$intervals
#         intervals_char <- bands_dist$intervals
#
#         # Convert string intervals to numeric bounds
#         extract_bounds <- function(interval_str) {
#         # Remove parentheses and split by comma
#         clean_str <- gsub("[()]", "", interval_str)
#         bounds <- as.numeric(strsplit(clean_str, ",")[[1]])
#         return(bounds)
#         }
#
#         # Extract lower and upper bounds
#         lower_bounds <- numeric(length(intervals_char))
#         upper_bounds <- numeric(length(intervals_char))
#
#         for(i in 1:length(intervals_char)) {
#         bounds <- extract_bounds(intervals_char[i])
#         lower_bounds[i] <- bounds[1]
#         upper_bounds[i] <- bounds[2]
#         }
#
#         cat("Lower bounds (first 10 values):\n")
#         print(head(lower_bounds, 10))
#         cat("Upper bounds (first 10 values):\n")
#         print(head(upper_bounds, 10))
#
#         # Calculate coverage
#         coverage_dist <- mean((ynew >= lower_bounds) & (ynew <= upper_bounds))
#         cat("Dist-split coverage:", coverage_dist, "\n")
#
#         # You can still use the plot methods
#         plot(bands_dist)
#         plot(bands_dist, ynew)
#             ''')
#     with localconverter(default_converter + numpy2ri.converter):
#         lower_dist = robjects.globalenv["lower_bounds"]
#         upper_dist = robjects.globalenv["upper_bounds"]
#     # Convert to numpy arrays
#     lower_dist_np = np.array(lower_dist)
#     upper_dist_np = np.array(upper_dist)
#
#     print(f"Dist-split bands shape: {lower_dist_np.shape}, {upper_dist_np.shape}")
#
#     # Calculate coverage in Python
#     coverage_dist_py = np.mean((ynew >= lower_dist_np) & (ynew <= upper_dist_np))
#
#     print(f"\nPython verification:")
#     print(f"Dist-split coverage: {coverage_dist_py:.3f}")
#
#     return {
#         'lower_dist': lower_dist_np,
#         'upper_dist': upper_dist_np,
#
#         'y_true': ynew,
#         'coverage_dist': coverage_dist_py,
#     }
class DistSplit:

    # Alternative function if you already have the packages installed
    def run_prediction_bands_only(x, y):
        """
        Run prediction bands assuming packages are already installed
        """
        print("Running predictionBands (assuming packages are installed)...")

        # Convert to R
        with localconverter(default_converter + numpy2ri.converter):
            robjects.globalenv["x"] = x
            robjects.globalenv["y"] = y

        # Run R code
        robjects.r('''
                        library(FlexCoDE)
                        library(predictionBands)

                        # Fit prediction bands model
                        cat("Fitting predictionBands model...\n")
                        fit <- fit_predictionBands(x,y,0.5,0.4,0.1)
                        cat("Model fitted successfully!\n\n")
                    ''')
        return robjects.globalenv["fit"]


    def predict_dist_split_r(fit, xnew, ynew,alpha):
        """
        Runs dist-split predictionBands using an already fitted model.
        Returns lower/upper bounds and coverage.
        """
        with localconverter(default_converter + numpy2ri.converter):
            robjects.globalenv["xnew"] = xnew
            robjects.globalenv["ynew"] = ynew
            robjects.globalenv["fit"] = fit
            robjects.globalenv["alpha"] = alpha

        robjects.r('''
                        library(FlexCoDE)
                        library(predictionBands) 

                        cat("Predicting with dist-split...\n")
                        bands_dist <- predict(fit, xnew, type="dist", alpha)

                        intervals_char <- bands_dist$intervals

                        extract_bounds <- function(interval_str) {
                            clean_str <- gsub("[()]", "", interval_str)
                            bounds <- as.numeric(strsplit(clean_str, ",")[[1]])
                            return(bounds)
                        }

                        lower_bounds <- numeric(length(intervals_char))
                        upper_bounds <- numeric(length(intervals_char))

                        for(i in 1:length(intervals_char)) {
                            bounds <- extract_bounds(intervals_char[i])
                            lower_bounds[i] <- bounds[1]
                            upper_bounds[i] <- bounds[2]
                        }

                        coverage_dist <- mean((ynew >= lower_bounds) & (ynew <= upper_bounds))
                    ''')

        with localconverter(default_converter + numpy2ri.converter):
            lower_bounds = np.array(robjects.globalenv["lower_bounds"])
            upper_bounds = np.array(robjects.globalenv["upper_bounds"])
            coverage = float(robjects.globalenv["coverage_dist"])

        return {
            'lower_dist': lower_bounds,
            'upper_dist': upper_bounds,
            'coverage_dist': coverage
        }
    def predict_dist_split_py(fit, xnew, ynew, ths):
        with localconverter(default_converter + numpy2ri.converter):
            robjects.globalenv["xnew"] = xnew
            robjects.globalenv["ynew"] = ynew
            robjects.globalenv["fit"] = fit
        robjects.r('''
                        library(FlexCoDE)
                        library(predictionBands) 
                        pred_test <- FlexCoDE::predict.FlexCoDE(fit$density_fit, xnew)
                        ''')
        with localconverter(default_converter + numpy2ri.converter):
            pred_test = np.array(robjects.globalenv["pred_test"])
        prediction_bands_which_belong = list()
        # intervals = list()
        # Convert R matrix to NumPy array
        CDE = np.array(pred_test[0])  # Shape: (n_Z, n_samples) usually
        # Convert R vector to NumPy array
        z = np.array(pred_test[1])  # Shape: (n_Z,)
        dz = np.diff(z)[0]  # assume uniform grid

        FTest = np.cumsum(CDE, axis=1) * dz  # shape: (n_samples, len(z))
        interval = np.zeros([xnew.shape[0], 2])
        for ii in range(xnew.shape[0]):
            lower = ths[0][ii]
            upper = ths[1][ii]
            belongs = (FTest[ii] >= lower) & (FTest[ii] <= upper)
            prediction_bands_which_belong.append(belongs)

            if np.any(belongs):
                interval[ii, 0] = z[belongs].min()
                interval[ii, 1] = z[belongs].max()
            else:
                interval[ii, :] = [np.nan, np.nan]  # fallback for empty band
            # intervals.append(interval)
        coverage_dist =np.mean((ynew >= interval[:, 0]) & (ynew <= interval[:, 1]))
        return {
            'y_grid': z,
            'ths': ths,
            'densities': CDE,
            'prediction_bands_which_belong': prediction_bands_which_belong,
            'intervals': interval,
            'type': "dist",
        }

        ## Example
        # if __name__ == "__main__":
        #     # Sample data generation
        #     print("\nGenerating sample data...")
        #     n, d, n_new = 1000, 10, 50
        #     x = np.random.randn(n, d)
        #     y = x[:, 0] + np.random.normal(0, 0.1, n)
        #     xnew = np.random.randn(n_new, d)
        #     ynew = xnew[:, 0] + np.random.normal(0, 0.1, n_new)
        #     # Run the complete setup and example
        #     try:
        #         fit = run_prediction_bands_only(x, y)
        #         #results = predict_dist_split_r(fit, xnew, ynew)
        #         predict_dist_split_py(fit, xnew, ynew)
        #         # results = setup_prediction_bands_example(x,y,xnew,ynew)
        #         print("\n✓ Complete example finished successfully!")
        #     except Exception as e:
        #         print(f"Error occurred: {e}")
