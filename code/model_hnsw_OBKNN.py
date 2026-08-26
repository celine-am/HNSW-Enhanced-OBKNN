import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.utils.validation import check_is_fitted
from sklearn.exceptions import NotFittedError
from capymoa.base import AnomalyDetector
from capymoa.instance import Instance
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
from scipy.stats import norm
import sys, os
from pathlib import Path
from scipy.linalg import inv
from scipy.spatial import distance
from scipy.stats import zscore
from capymoa.instance import Instance
import logging
import hnswlib

JITTER = 1e-6


def featurewise_distance(vec1, vec2, metric, p=2):

    vec1 = np.squeeze(vec1)
    vec2 = np.squeeze(vec2)

    if metric == "cityblock":
        return np.abs(vec1 - vec2)

    elif metric == "euclidean":
        return (vec1 - vec2) ** 2

    elif metric == "minkowski":
        return np.abs(vec1 - vec2) ** p

    elif metric == "chebyshev":
        diffs = np.abs(vec1 - vec2)
        vec = np.zeros_like(diffs, dtype=float)
        max_index = np.argmax(diffs)
        vec[max_index] = diffs[max_index]
        return vec

    elif metric == "canberra":
        denom = np.abs(vec1) + np.abs(vec2)
        with np.errstate(divide='ignore', invalid='ignore'):
            vec = np.abs(vec1 - vec2) / denom
            vec[np.isnan(vec)] = 0
        return vec

    elif metric == "cosine":
        denom = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        if denom == 0:
            return np.zeros_like(vec1)
        cosine_similarity = (vec1 * vec2) / denom
        return 1 - cosine_similarity

    elif metric == "braycurtis":
        num = np.abs(vec1 - vec2)
        denom = np.abs(vec1 + vec2)
        with np.errstate(divide='ignore', invalid='ignore'):
            vec = num / denom
            vec[np.isnan(vec)] = 0
        return vec
    else:
        raise ValueError(f"Featurewise distance for '{metric}' is not supported.")


def transform_instance(instance: Instance, transf, prev_instance: Instance = None):

    if transf == "MA":
        t_instance = pd.Series(instance.x).rolling(window=5, min_periods=None).mean().to_numpy()
        t_instance = Instance.from_array(instance.schema, t_instance.reshape(-1))
        return t_instance
    elif transf == "LOG":
        t_instance = np.log(instance.x)
        t_instance = Instance.from_array(instance.schema, t_instance)
        return t_instance
    elif transf == "POW":
        t_instance = np.power(instance.x, 2)
        t_instance = Instance.from_array(instance.schema, t_instance)
        return t_instance
    elif transf == "SQRT":
        t_instance = np.power(instance.x, 0.5)
        t_instance = Instance.from_array(instance.schema, t_instance)
        return t_instance
    elif transf == "DIL":
        t_instance = instance.x[::5]
        t_instance = Instance.from_array(instance.schema, t_instance.reshape(-1))
        return t_instance
    elif transf == "FOD":
        diff_data = np.diff(instance.x)
        t_instance_data = np.insert(diff_data, 0, 0)
        t_instance = Instance.from_array(instance.schema, t_instance_data)
        return t_instance
    elif transf == "SOD":
        t_instance = np.diff(np.diff(instance.x))
        t_instance = Instance.from_array(instance.schema, t_instance)
        return t_instance
    elif transf == "FT":
        t_instance = np.fft.fft(instance.x)
        t_instance = Instance.from_array(instance.schema, t_instance.reshape(-1))
        return t_instance
    elif transf == "iFT":
        t_instance = np.fft.ifft(instance.x).real
        t_instance = Instance.from_array(instance.schema, t_instance.reshape(-1))
        return t_instance
    elif transf == "SQRT&ZNORM":
        t_instance = np.power(instance.x, 0.5)
        t_instance = zscore(t_instance, nan_policy='propagate', axis=0)
        t_instance = Instance.from_array(instance.schema, t_instance.reshape(-1))
        return t_instance
    elif transf == "ZNORM":
        t_instance = zscore(instance.x, nan_policy='propagate', axis=0)
        t_instance = Instance.from_array(instance.schema, t_instance.reshape(-1))
        return t_instance
    elif transf == "DEN&ZNORM":
        minimum_threshold = 100
        instance.x[instance.x < minimum_threshold] = 0
        t_instance = zscore(instance.x, nan_policy='propagate', axis=0)
        t_instance = Instance.from_array(instance.schema, t_instance.reshape(-1))
        return t_instance
    elif transf == "DEN":
        minimum_threshold = 100
        instance.x[instance.x < minimum_threshold] = 0
        t_instance = instance.x
        t_instance = Instance.from_array(instance.schema, t_instance.reshape(-1))
        return t_instance
    else:
        return instance


def median_of_means(data):
    k = int(np.sqrt(len(data)))
    arr = np.array(data)
    np.random.shuffle(arr)
    try:
        blocks = np.array_split(arr, k)
    except ValueError as e:
        logging.debug(f"Error: k ({k}) is likely larger than the number of data points.")
        logging.debug("Setting k=1 (standard mean).")
        blocks = [arr]
    block_means = [np.mean(block) for block in blocks]
    return np.median(block_means)


def build_hnsw(data_chunk, space="l2", M=16, ef_construction=200, ef_search=50):
    """Build an HNSW index from a data chunk."""
    dim = data_chunk.shape[1]
    index = hnswlib.Index(space=space, dim=dim)
    index.init_index(max_elements=len(data_chunk)+1000, ef_construction=ef_construction, M=M)
    index.add_items(data_chunk, np.arange(len(data_chunk)))
    index.set_ef(ef_search)
    return index


class OnlineBootKNN(AnomalyDetector):
    """
    Anomaly detection using an online ensemble of k-nearest neighbors (KNN) or HNSW.

    Parameters:
    - schema: Optional, schema for data processing (if needed).
    - random_seed: Optional, seed for random number generation.
    - chunk_size: Size of each data chunk to process.
    - ensemble_size: Number of KNN/HNSW models in the ensemble.
    - algorithm: KNN algorithm to use ('brute', 'kd_tree', etc.).
    - n_jobs: Number of CPU cores to use for parallel processing.
    - window_size: Size of the sliding window for statistics.
    - dmetric: Distance metric for KNN (e.g., 'cityblock', 'euclidean').
    - transf: Optional, transformation to apply to data.
    - alpha_z_test: Significance level for z-score test (default 0.05).
    - type_dist: Distance strategy — 'largest' (KNN), 'mean' (centroid), or 'hnsw'.
    - hnsw_space: HNSW space — 'l2', 'ip', or 'cosine' (used when type_dist='hnsw').
    - hnsw_M: HNSW M parameter (graph connectivity).
    - hnsw_ef_construction: HNSW ef_construction parameter.
    - hnsw_ef_search: HNSW ef search parameter.
    - rebuild_on_drift: Trigger full HNSW rebuild when |z| > drift_threshold.
    - drift_threshold: |z| threshold that flags a drift event.
    """

    def __init__(
        self,
        schema=None,
        random_seed=None,
        chunk_size=240,
        ensemble_size=240,
        algorithm='brute',
        n_jobs=-1,
        window_size=120,
        dmetric="cityblock",
        transf="ZNORM",
        type_dist="largest",
        alpha_z_test=0.05,
        alpha_ema=0.01,
        no_bootstrapp=False,
        no_z_score=False,
        update_distance_with_abnormal=True,
        update_mode_stats="ema",
        k=1,
        # HNSW-specific parameters (to check)
        hnsw_space="l2",
        hnsw_M=16,
        hnsw_ef_construction=200,
        hnsw_ef_search=50
    ):
        super().__init__(schema=schema, random_seed=random_seed)

        self.random_generator = np.random.RandomState(random_seed)

        self.data_window = np.array([])
        self.chunks = []

        
        self.chunk_size = chunk_size
        self.ensemble_size = ensemble_size
        self.n_jobs = n_jobs
        self.window_size = window_size
        self.dmetric = dmetric
        self.transf = transf
        self.alpha_z_test = alpha_z_test
        self.update_mode_stats = update_mode_stats
        self.alpha_ema = alpha_ema
        self.algorithm = algorithm
        self.k = k
        self.ensemble = []
        self.inv_cov = None
        self.type_dist = type_dist

        # HNSW params
        self.hnsw_space = hnsw_space
        self.hnsw_M = hnsw_M
        self.hnsw_ef_construction = hnsw_ef_construction
        self.hnsw_ef_search = hnsw_ef_search

        self.init = True
        self.last_value_is_anomaly = False
        self.reset_threshold = 4200
        self.count_reset = None
        self.normal_reference_ch = None
        self.abnormal_reference_ch = None
        self.c_normal_reference_w = None
        self.p_normal_reference_w = None
        self.abnormal_reference_w = None
        self.z_critical_one_tail = norm.ppf(1 - self.alpha_z_test)

        
        self.z = 0.0
        self.mean = np.nan
        self.mean_of_anomalies = np.nan
        self.std_dev = np.nan
        self.std_dev_anomalies = np.nan
        self.min_dist = np.nan
        self.n = 0
        self.n_anomalies = 0
        self.accum_error = np.nan
        self.accum_error_anomalies = np.nan
        self.max_p_random_number = np.nan

     
        self.z_scores_to_monitor = []
        self.means_to_monitor = []
        self.std_devs_to_monitor = []
        self.min_dists_to_monitor = []
        self.z_thresholds_to_monitor = []
        self.p_random_number_to_monitor = []
        self.ground_truth_to_monitor = []

        self.no_bootstrapp = no_bootstrapp
        self.no_z_score = no_z_score
        self.update_distance_with_abnormal = update_distance_with_abnormal


        self.detected_anomalies = 0

    def __str__(self):
        return "OnlineBootKNN"

   
    def train(self, instance: Instance):
        instance = transform_instance(instance, self.transf)
        self._learn_batch(instance.x)

    def _learn_batch(self, data):

        if self.init:
           
            if self.data_window.size == 0:
                self.data_window = data.reshape(1, -1)
            else:
                self.data_window = np.vstack([self.data_window, data])

            if len(self.data_window) >= self.window_size:

                for i in range(self.ensemble_size):

                    if self.no_bootstrapp:
                        data_chunk = self.data_window
                    else:
                        indices = self.random_generator.choice(
                            len(self.data_window), size=self.chunk_size, replace=True
                        )
                        data_chunk = self.data_window[indices]

                    self.chunks.append(data_chunk)

                    if self.type_dist in ("largest", "mean"):
                        metric_params = {}
                        nn_model = NearestNeighbors(
                            n_neighbors=self.k,
                            algorithm=self.algorithm,
                            n_jobs=self.n_jobs,
                            metric=self.dmetric,
                            metric_params=metric_params,
                        )
                        nn_model.fit(data_chunk)
                        self.ensemble.append(nn_model)

                    elif self.type_dist == "hnsw":
                        hnsw_index = build_hnsw(
                            data_chunk,
                            space=self.hnsw_space,
                            M=self.hnsw_M,
                            ef_construction=self.hnsw_ef_construction,
                            ef_search=self.hnsw_ef_search,
                        )
                        self.ensemble.append(hnsw_index)

                self.init = False

        else:
            #Slide the window 
            if self.type_dist == "hnsw":
                # roll instead of vstack 
                self.data_window = np.roll(self.data_window, -1, axis=0)
                self.data_window[-1] = data
            else:
                self.data_window = self.data_window[1:]
                self.data_window = np.vstack([self.data_window, data])

            # Update ensemble
            for i in range(self.ensemble_size):

                self.max_p_random_number = 0

                if self.no_bootstrapp:
                    p_random_number = 1
                elif self.last_value_is_anomaly:
                    p_random_number = 0
                else:
                    p_random_number = self.random_generator.poisson(1)

                if self.max_p_random_number < p_random_number:
                    self.max_p_random_number = p_random_number

                # KNN
                if self.type_dist in ("largest", "mean"):

                    if p_random_number > 0:

                        for _ in range(p_random_number):
                            self.chunks[i] = self.chunks[i][1:]
                            self.chunks[i] = np.vstack([self.chunks[i], data])

                        if self.type_dist == "largest":
                            self.ensemble[i].fit(self.chunks[i])

                # HNSW
                elif self.type_dist == "hnsw":

                    # if the chunck is not updated by the bootstrapping
                    # we don't change the chunck neither its hnsw index 

                    #if we should update we enter this condition 
                    if p_random_number > 0:

                        # updating the chunck (as usual)
                        for _ in range(p_random_number):
                            self.chunks[i] = self.chunks[i][1:]
                            self.chunks[i] = np.vstack([self.chunks[i], data])

                        # as the chunck changed, we rebuild only the hnsw index of the updated chunck 
                        self.ensemble[i] = build_hnsw(
                            self.chunks[i],
                            space=self.hnsw_space,
                            M=self.hnsw_M,
                            ef_construction=self.hnsw_ef_construction,
                            ef_search=self.hnsw_ef_search,
                        )

   
    def score_instance(self, instance: Instance):
        instance = transform_instance(instance, self.transf)
        data = instance.x.reshape(1, -1)

        distances = []
        references = []

        for i in range(self.ensemble_size):
            try:
                if self.type_dist == "largest":
                    check_is_fitted(self.ensemble[i])
                    dist, idx = self.ensemble[i].kneighbors(data)
                    d = dist[0][-1]
                    index_f = idx[0][-1]
                    vf_ch = self.chunks[i][index_f]

                elif self.type_dist == "mean":
                    vf_ch = np.mean(self.chunks[i], axis=0)
                    d = featurewise_distance(vf_ch, data, metric=self.dmetric).sum()

                elif self.type_dist == "hnsw":
                    #knn query using hnsw index 
                    labels, dist = self.ensemble[i].knn_query(data, k=1)
                    d = dist[0][0]
                    vf_ch = self.chunks[i][labels[0][0]]

                distances.append(d)
                references.append(vf_ch)

            except NotFittedError:
                logging.debug(f"Model {i} is not fitted.")
                distances.append(0)
            except IndexError:
                logging.debug(f"Vector {i} has not completed the minimum window.")
                distances.append(0)
            except Exception as exc:
                logging.error(f"An error occurred while scoring instance: {exc}")
                distances.append(None)

        self.min_dist = np.min(distances)

        if self.no_z_score:
            return self.min_dist

        if not self.init:
            if (self.n == 0) | (self.n > self.reset_threshold):
                self.count_reset = 0 if self.count_reset is None else self.count_reset + 1
                self.start_statistics(self.min_dist)
            elif self.n == 1:
                self.update_statistics_normal(self.min_dist)
            else:
                self.update_z_score(self.min_dist)

                if not self.last_value_is_anomaly:
                    self.update_statistics_normal(self.min_dist)
                else:
                    self.update_statistics_abnormal(self.min_dist)
                    min_pos_dist = np.argmin(distances)
                    self.normal_reference_ch = references[min_pos_dist]
                    self.abnormal_reference_ch = data.reshape(-1)

        return self.z

   
    def start_statistics(self, new_dist):
        self.z = 0.0
        self.n = 1
        self.mean = new_dist
        self.accum_error = JITTER
        self.last_value_is_anomaly = False

    def update_statistics_normal(self, new_dist):
        self.n += 1
        if self.update_mode_stats == 'welford':
            delta = new_dist - self.mean
            self.mean += delta / self.n
            delta2 = new_dist - self.mean
            self.accum_error += delta * delta2
            self.std_dev = math.sqrt(self.accum_error / (self.n - 1))
        else:
            delta = new_dist - self.mean
            self.mean += self.alpha_ema * delta
            self.accum_error = (1 - self.alpha_ema) * (self.accum_error + self.alpha_ema * delta ** 2)
            self.std_dev = math.sqrt(self.accum_error)

    def update_statistics_abnormal(self, new_dist):
        self.n += 1
        self.n_anomalies += 1

        if self.update_distance_with_abnormal:
            if self.update_mode_stats == 'welford':
                delta = new_dist - self.mean
                self.mean += delta / self.n
                delta2 = new_dist - self.mean
                self.accum_error += delta * delta2
                self.std_dev = math.sqrt(self.accum_error / (self.n - 1))
            else:
                delta = new_dist - self.mean
                self.mean += self.alpha_ema * delta
                self.accum_error = (1 - self.alpha_ema) * (self.accum_error + self.alpha_ema * delta ** 2)
                self.std_dev = math.sqrt(self.accum_error)
        else:
            if self.n_anomalies == 1:
                self.mean_of_anomalies = new_dist
                self.accum_error_anomalies = JITTER
            else:
                if self.update_mode_stats == 'welford':
                    delta = new_dist - self.mean_of_anomalies
                    self.mean_of_anomalies += delta / self.n_anomalies
                    delta2 = new_dist - self.mean_of_anomalies
                    self.accum_error_anomalies += delta * delta2
                    if self.n_anomalies > 1:
                        self.std_dev_anomalies = math.sqrt(
                            self.accum_error_anomalies / (self.n_anomalies - 1)
                        )
                else:
                    delta = new_dist - self.mean_of_anomalies
                    self.mean_of_anomalies += self.alpha_ema * delta
                    self.accum_error_anomalies = (1 - self.alpha_ema) * (
                        self.accum_error_anomalies + self.alpha_ema * delta ** 2
                    )
                    self.std_dev_anomalies = math.sqrt(self.accum_error_anomalies)

    def update_z_score(self, new_dist):
        if self.std_dev != 0 and not pd.isna(self.std_dev):
            self.z = (new_dist - self.mean) / self.std_dev
        else:
            self.z = np.nan

        if self.z > self.z_critical_one_tail and not pd.isna(self.z):
            self.last_value_is_anomaly = True
            self.detected_anomalies += 1
        else:
            self.last_value_is_anomaly = False

    def predict(self, data: np.ndarray):
        raise NotImplementedError("The 'predict' method must be implemented by subclasses.")

 
    def explain(self, headers, region_study_list, path: str, file_name: str):
        headers = headers.astype(float)
        region_study_list = np.array(region_study_list).astype(str)

        if not self.last_value_is_anomaly:
            return

        if self.normal_reference_ch is None or self.abnormal_reference_ch is None:
            raise ValueError("Error: Normal reference or input data is None.")

        if len(self.normal_reference_ch) != len(self.abnormal_reference_ch):
            raise ValueError("Error: Normal reference and input data must have the same length.")

        differences = featurewise_distance(
            self.abnormal_reference_ch, self.normal_reference_ch, metric=self.dmetric
        )

        fig, ax1 = plt.subplots(figsize=(14, 7))
        ax1.bar(
            headers, differences,
            label=f"Feature Differences (Z: {round(self.z, 2)})",
            color='orange', alpha=1.0,
        )
        ax1.set_ylabel('Feature Differences', fontsize=14)
        ax1.set_xlabel('Wavelengths (nm)', fontsize=14)

        min_diff = differences.min()
        max_diff = differences.max()
        lower_limit = min_diff * 1.05
        upper_limit = max_diff * 1.05
        ax1.set_ylim(lower_limit, upper_limit)
        ax1.grid(True, which='both', axis='y', linestyle='--', alpha=0.5)

        for i, rs in enumerate(region_study_list):
            rs_s = float(rs.split(":")[0])
            rs_f = float(rs.split(":")[1])
            comp = str(rs.split(":")[2])
            ax1.axvline(x=rs_s, color='grey', linestyle='dotted', linewidth=1.5, alpha=0.8)
            ax1.axvline(x=rs_f, color='grey', linestyle='dotted', linewidth=1.5, alpha=0.8)
            ax1.text(
                rs_f,
                upper_limit * 0.85 * (len(region_study_list) - i) / len(region_study_list),
                f' RS{i+1} ({comp}) at: {rs_s}', fontsize=10,
            )
            ax1.legend(loc="upper left", fontsize=10)
            ax1.legend(loc="upper right", fontsize=10)

        plt.title(
            f"Feature Differences (Abnormal vs. Normal Instances). \n"
            f"# Total Anomalies: {self.n_anomalies} - "
            f"# Total Instances: {self.count_reset * self.reset_threshold + self.n + self.window_size}",
            fontsize=14,
        )

        plot_path = os.path.join(path, f"{file_name}_anomaly_explanation.pdf")
        plt.savefig(plot_path, format="pdf", bbox_inches='tight')
        logging.debug(f"Plot saved at {plot_path}")
        plt.close()

    def monitor_core_statistics_training(self):
        logging.debug(
            f"\n{'='*30}\n"
            f"Training Stats.\n"
            f"Status: [Initial Phase: {self.init}]\n"
            f"shape - data_window: {np.shape(self.data_window)}\n"
            f"shape - chunks: {np.shape(self.chunks)}\n"
            f"{'='*30}"
        )

    def monitor_core_statistics_scoring(self):
        logging.debug(
            f"\n{'='*30}\n"
            f"Scoring Stats.\n"
            f"Status: [Anomaly: {self.last_value_is_anomaly}]\n"
            f"Stats:  Acum Mean: {self.mean:.2f} | Std: {self.std_dev:.2f} | "
            f"Min Dist: {self.min_dist:.2f} | N: {self.n}\n"
            f"Anoms:  Count: {self.n_anomalies} | P-Rand: {self.max_p_random_number:.2f}\n"
            f"Z-Test: Score: {self.z:.2f} | Crit: {self.z_critical_one_tail:.4f}\n"
            f"{'='*30}"
        )

    def plot_core_statistics(self, path: str, file_name: str, label: int = 0):
        self.p_random_number_to_monitor.append(self.max_p_random_number)
        self.means_to_monitor.append(self.mean)
        self.std_devs_to_monitor.append(self.std_dev)
        self.min_dists_to_monitor.append(self.min_dist)
        self.ground_truth_to_monitor.append(label)
        self.z_scores_to_monitor.append(self.z)
        self.z_thresholds_to_monitor.append(self.z_critical_one_tail)

        gt_indices = [i for i, x in enumerate(self.ground_truth_to_monitor) if int(x) == 1]
        x_axis = np.arange(len(self.means_to_monitor))
        gt_style = {'color': 'darkorange', 'edgecolors': 'darkorange', 'marker': 'X', 's': 50, 'zorder': 5}

        # Plot 1: Mean / Std / Min Dist
        plt.clf()
        fig, ax1 = plt.subplots(figsize=(10, 6))
        means = np.array(self.means_to_monitor, dtype=float)
        stds = np.array(self.std_devs_to_monitor, dtype=float)
        min_dists = np.array(self.min_dists_to_monitor, dtype=float)

        ax1.plot(x_axis, min_dists, label='Min Dist', color='green', marker='s', markersize=2, alpha=0.7)
        if gt_indices:
            ax1.scatter(gt_indices, min_dists[gt_indices], label='Anomaly (GT)', **gt_style)
        ax1.set_ylabel('Min Distance', color='green')
        ax1.grid(True, linestyle='--', alpha=0.6)

        ax2 = ax1.twinx()
        ax2.plot(x_axis, means, label='Mean', color='blue', linestyle='--', marker='o', markersize=2)
        ax2.fill_between(x_axis, means - stds, means + stds, color='blue', alpha=0.2,
                         label='Confidence Interval (±1 Std)')
        ax2.set_ylabel('Mean Statistics', color='blue')

        plt.title('Min Dist vs Mean & Std Dev')
        fig.legend(loc='upper left')
        plt.savefig(os.path.join(path, f"{file_name}_mean_min.pdf"), format="pdf")
        plt.close()

        # Plot 2: Z-Score + Threshold
        plt.clf()
        fig, ax3 = plt.subplots(figsize=(10, 6))
        z_scores = np.array(self.z_scores_to_monitor)

        ax3.plot(z_scores, label='Z Score', color='red', marker='o', markersize=4)
        ax3.plot(self.z_thresholds_to_monitor, label='Threshold', color='purple',
                 linestyle=':', marker='s', markersize=2)
        if gt_indices:
            ax3.scatter(gt_indices, z_scores[gt_indices], label='Anomaly (GT)', **gt_style)
        ax3.set_ylabel('Z Score & Threshold')
        ax3.grid(True, linestyle='--', alpha=0.6)

        ax4 = ax3.twinx()

        plt.title('Z-Score, Threshold')
        fig.legend(loc='upper left')
        plt.savefig(os.path.join(path, f"{file_name}_zscore_extra.pdf"), format="pdf")
        plt.close()

        # Plot 3: Z-Score + P-Random Number
        plt.clf()
        fig, ax5 = plt.subplots(figsize=(10, 6))

        ax5.plot(z_scores, label='Z Score', color='red', marker='o', markersize=4)
        if gt_indices:
            ax5.scatter(gt_indices, z_scores[gt_indices], label='Anomaly (GT)', **gt_style)
        ax5.set_ylabel('Z Score', color='red')
        ax5.grid(True, linestyle='--', alpha=0.6)

        ax6 = ax5.twinx()
        ax6.plot(self.p_random_number_to_monitor, label='P Random Num', color='black',
                 linestyle='--', marker='^', markersize=2)
        ax6.set_ylabel('Model & P-Number')

        plt.title('Z-Score vs Model & P-Random Number')
        fig.legend(loc='upper left')
        plt.savefig(os.path.join(path, f"{file_name}_zscore_model_pnum.pdf"), format="pdf")
        plt.close()

        logging.debug(f"All reorganized plots saved to {path}")



if __name__ == "__main__":

    current_dir = Path(__file__).resolve().parent.parent.parent

    sys.path.append(str(current_dir))

    current_dir = current_dir.parent

    from capymoa.stream import NumpyStream
    from data_utils import calculate_performance_metrics
    from model_utils import clean_score
    import time
    from model.mDragstream.mdragstream import mDragStream

    from pysad.models import ExactStorm, IForestASD, KitNet, LODA, RobustRandomCutForest, RSHash, xStream
    from capymoa.anomaly import OnlineIsolationForest, HalfSpaceTrees as HStreeCapy
    from dSalmon.outlier import SWKNN, SWLOF

    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


    DATA_PATH = current_dir / 'datasets' / 'raw' / 'ScenariosV3'

    PATH_PLOT_FILE_NAME_INTERPRETATION = current_dir / 'notebooks' / 'img_anomalies'

    PATH_PLOT_FILE_NAME_SCORE = current_dir / 'improvement_assessment' / 'imgs' /'obknn_hnsw'/'score'
    PATH_PLOT_FILE_NAME_TRAIN = current_dir / 'improvement_assessment' / 'imgs' /'obknn_hnsw'/'train'

    PATH_SUMMARY_RESULTS_IMPROVEMENT = current_dir / 'improvement_assessment' / 'csv'
    


    files = [f for f in DATA_PATH.iterdir() if f.suffix == '.csv']

    summary_data = []
    TRANF = "ZNORM"
    P_WINDOW_SIZE = None

    WINDOW_SIZE = 120
    CHUNK_SIZE = 240
    ENSEMBLE_SIZE = 240

    N_JOBS = -1
    NO_BOOTSTRAPP = False
    NO_ZSCORE = False
    UPDATE_WITH_ABNORMAL = False
    DMETRIC = "cityblock"
    ALGO = "brute"
    ALPHA_Z_TEST = 0.95
    ALPHA_EMA = 0.01
    ALPHA_EMA = 0.01
    UPDATE_MODE_STATS = "ema"
    SLEEP_TIME = 0
    MIN_Z_SCORE = 4
    REGION_STUDY = ["386.45:393.38:N2", "773.38:780.40:O2", "652.47:659.53:H",
                    "304.46:311.54:OH", "748.38:752.19:Ar"]
    COLS_POS_FMIN = 1
    COLS_POS_FMAX = 2049
    COLS_POS_LABEL = -1

    NUMBER_RUNS = 10
    SCORE_DIR = "direct"

    #Choose : "largest", "mean", "hnsw" 
    TYPE_DIST = "hnsw"   # using HNSW 

    # HNSW-specific settings (only used when TYPE_DIST == "hnsw")
    HNSW_SPACE = "l2"           # 'l2', 'ip', or 'cosine'
    HNSW_M = 16
    HNSW_EF_CONSTRUCTION = 200
    HNSW_EF_SEARCH = 50

    #DATASETS_LIST = ["HNH_", "LNH_", "TNH_","RNH_", "RAR_", "RMX_","HAR_", "LAR_", "TAR_","SAR1_", "SAR2_", "SAR3_"]

    #DATASETS_LIST = ["HNH_", "LNH_"]

    #DATASETS_LIST = ["RMX_"]
    #DATASETS_LIST = ["RAR_"]
    #DATASETS_LIST = ["RNH_"]

    DATASETS_LIST = ["A1_","A2_","A3_","A4_","A5_","A6_","A7_","A8_","A9_"]


    f_break = False

    for file_name in files:

        if any(substring in file_name.name for substring in DATASETS_LIST):
            logging.debug("File to Use: %s", file_name)
        else:
            logging.debug("File not to Use %s", file_name)
            continue

        file_path = os.path.join(DATA_PATH, file_name)
        try:
            df = pd.read_csv(file_path, sep=',', low_memory=False, dtype={'CURRENTTIMESTAMP': str})
        except Exception:
            df = pd.read_csv(file_path, sep=',', low_memory=False)

        cols = df.iloc[:, COLS_POS_FMIN:COLS_POS_FMAX].columns
        score_column = 'Score'
        error_column = 'Error'
        label_column = df.columns[COLS_POS_LABEL]
        cols = df.columns[slice(COLS_POS_FMIN, COLS_POS_FMAX)]
        col_target = df.columns[-1]

        logging.debug("Name DS: %s", file_name)
        logging.debug("# Total Points: %d", len(df[cols].values))
        logging.debug("# Anomalies: %d", len(df[col_target].values))
        logging.debug("# of Columns: %d", len(df.columns))
        logging.debug("# of Columns to use: %d", len(cols))
        logging.debug("Columns: %s", cols)

        stream = NumpyStream(df[cols].values, df[col_target].astype(int).values,
                             dataset_name='PV', feature_names=cols)
        schema = stream.get_schema()

        if f_break:
            break

        for i in range(NUMBER_RUNS):

            stream.restart()
            scores = []
            raw_scores = []
            errors = []
            row = 0

            if P_WINDOW_SIZE is not None:
                WINDOW_SIZE = max(1, int(len(df) * P_WINDOW_SIZE))

            learner = OnlineBootKNN(
                schema=schema,
                random_seed=i,
                window_size=WINDOW_SIZE,
                type_dist=TYPE_DIST,
                chunk_size=CHUNK_SIZE,
                ensemble_size=ENSEMBLE_SIZE,
                dmetric=DMETRIC,
                transf=TRANF,
                alpha_z_test=ALPHA_Z_TEST,
                algorithm=ALGO,
                no_bootstrapp=NO_BOOTSTRAPP,
                no_z_score=NO_ZSCORE,
                update_mode_stats=UPDATE_MODE_STATS,
                update_distance_with_abnormal=UPDATE_WITH_ABNORMAL,
                alpha_ema=ALPHA_EMA,
                n_jobs=N_JOBS,
                # HNSW params (ignored when type_dist != 'hnsw')
                hnsw_space=HNSW_SPACE,
                hnsw_M=HNSW_M,
                hnsw_ef_construction=HNSW_EF_CONSTRUCTION,
                hnsw_ef_search=HNSW_EF_SEARCH,
            )

            np.random.seed(i)
            if f_break:
                break

            total_train_time = 0.0
            total_score_time = 0.0
            min_train_time = float("inf")
            max_train_time = 0.0
            min_score_time = float("inf")
            max_score_time = 0.0

            train_times = []
            score_times = []

            for row, instance in enumerate(stream):

                time.sleep(SLEEP_TIME)

                logging.debug("########################################################")
                logging.debug(f'The new instance: {instance.x}, index: {instance.y_index}')
                logging.debug("########################################################")

                if hasattr(learner, "fit_partial"):  # Pysad models
                    start_train = time.perf_counter()
                    learner.fit_partial(instance.x)
                    training_time = time.perf_counter() - start_train
                    start_score = time.perf_counter()
                    score = learner.score_partial(instance.x)
                    cleaned_score, error_score = clean_score(score)
                    scoring_time = time.perf_counter() - start_score

                elif hasattr(learner, "train"):  # Capymoa models
                    start_score = time.perf_counter()
                    score = learner.score_instance(instance)
                    cleaned_score, error_score = clean_score(score)
                    scoring_time = time.perf_counter() - start_score
                    min_score_time = min(min_score_time, scoring_time)
                    max_score_time = max(max_score_time, scoring_time)

                    start_train = time.perf_counter()
                    learner.train(instance)
                    training_time = time.perf_counter() - start_train
                    min_train_time = min(min_train_time, training_time)
                    max_train_time = max(max_train_time, training_time)

                    train_times.append(training_time)
                    score_times.append(scoring_time)

                    total_train_time += training_time
                    total_score_time += scoring_time

                    n_instances = row + 1

                elif hasattr(learner, "fit_predict"):  # dSalmon models
                    training_time = 0
                    start_score = time.perf_counter()
                    score = learner.fit_predict(instance.x)
                    cleaned_score, error_score = clean_score(score)
                    scoring_time = time.perf_counter() - start_score

                else:
                    raise AttributeError(
                        f"Model {learner.__class__.__name__} has no recognized training/scoring method."
                    )

                scores.append(cleaned_score)
                raw_scores.append(score)
                errors.append(error_score)

                if np.isnan(cleaned_score):
                    f_break = True
                    break
            """pd.DataFrame({
            "instance": np.arange(len(train_times)),
            "train_time": train_times
            }).to_csv(
             os.path.join("train_times_run{i}.csv"),
            index=False
            )"""
            
          """  # time plots
            # scoring time per instance viz
            x = np.arange(len(score_times))

            plt.figure(figsize=(12, 3))

            plt.vlines(
                x,
                0,
                score_times,
                color="green",
                linewidth=1
            )

            plt.xlabel("Instances / Time steps")
            plt.ylabel("Time (sec)")
            plt.title(f"Scoring Time - OBKNN+HNSW(RNH-R)- Run {i}")

            plt.tight_layout()

            plt.savefig(
                os.path.join(
                    PATH_PLOT_FILE_NAME_SCORE,
                    f"OBKNN+HNSW (RNH-R)_run{i}_score_time.png"
                ),
                dpi=300
            )

            plt.close()"""

           #training time per instance viz 
            """x = np.arange(len(train_times))

            plt.figure(figsize=(12, 3))

            plt.vlines(
                x,
                1e-6,
                train_times,
                color="royalblue",
                linewidth=1,
                alpha=0.6
            )

            plt.scatter(x, train_times, color="black", s=8, alpha=0.6)

            plt.yscale("log")  # 

            plt.xlabel("Instances / Time steps")
            plt.ylabel("Log(Time sec)")
            plt.title(f"Training Time (log scale) - OBKNN+HNSW(RMX) - Run {i}")

            plt.tight_layout()

            plt.savefig(
                os.path.join(
                    PATH_PLOT_FILE_NAME_TRAIN,
                    f"OBKNN+HNSW (RMX)_run{i}_train_time_log.png"
                ),
                dpi=300
            )

            plt.close()"""
            
            """#training time per instance viz
            x = np.arange(len(train_times))
            plt.figure(figsize=(12, 3))
            plt.vlines( x, 0, train_times, color="royalblue", linewidth=1 )
            plt.xlabel("Instances / Time steps") 
            plt.ylabel("Time (sec)")
            plt.title(f"Training Time - OBKNN+HNSW(RNH-R) - Run {i}")
            plt.tight_layout() 
            plt.savefig( os.path.join( PATH_PLOT_FILE_NAME_TRAIN, f"OBKNN+HNSW (RNH-R)_run{i}_train_time.png" ), dpi=300 )
            plt.close()"""
            
            
            n_total = len(df)
            n_real = (df[col_target].astype(int) == 1).sum()

            pct_real = 100 * n_real / n_total
            pct_detected = 100 * learner.detected_anomalies / n_total

    
            df[score_column + str(i)] = list(scores)
            df[score_column + "_raw" + str(i)] = list(raw_scores)
            df[error_column + str(i)] = list(errors)

            try:
                results = calculate_performance_metrics(
                    df, label_column, score_column + str(i),
                    t_window_size=WINDOW_SIZE, score_direction=SCORE_DIR,
                )
                (roc_auc, pr_auc, max_f1, metrics, roc_auc_wtd, pr_auc_wtd,
                 max_f1_wtd, pct_detection, pct_false_positives,
                 tn, fp, fn, tp, best_threshold) = results
                auc_roc = metrics.get('AUC_ROC', None)
                auc_pr = metrics.get('AUC_PR', None)
                precision = metrics.get('Precision', None)
                f = metrics.get('F', None)
                precision_at_k = metrics.get('Precision_at_k', None)
                rprecision = metrics.get('Rprecision', None)
                rrecall = metrics.get('Rrecall', None)
                rf = metrics.get('RF', None)
                r_auc_roc = metrics.get('R_AUC_ROC', None)
                r_auc_pr = metrics.get('R_AUC_PR', None)
                vus_roc = metrics.get('VUS_ROC', None)
                vus_pr = metrics.get('VUS_PR', None)
                affiliation_precision = metrics.get('Affiliation_Precision', None)
                affiliation_recall = metrics.get('Affiliation_Recall', None)

            except Exception as e:
                print(f"Error calculating metrics for index {i}: {e}")
                roc_auc = pr_auc = max_f1 = metrics = roc_auc_wtd = pr_auc_wtd = None
                max_f1_wtd = pct_detection = pct_false_positives = None
                tn = fp = fn = tp = best_threshold = None
                auc_roc = auc_pr = precision = f = precision_at_k = None
                rprecision = rrecall = rf = r_auc_roc = r_auc_pr = None
                vus_roc = vus_pr = affiliation_precision = affiliation_recall = None

            summary_data.append({
                "iteration": i,
                "scenario": file_name.name.split("_")[0],
                "method": str(learner.__class__.__name__),
                "raw_roc_auc": roc_auc,
                "raw_pr_auc": pr_auc,
                "raw_max_f1": max_f1,
                "raw_roc_auc_wtd": roc_auc_wtd,
                "raw_pr_auc_wtd": pr_auc_wtd,
                "raw_max_f1_wtd": max_f1_wtd,
                "raw_pct_detection": pct_detection,
                "raw_pct_false_positives": pct_false_positives,
                "tn": tn, "fp": fp, "fn": fn, "tp": tp,
                "best_threshold": best_threshold,
                "auc_roc": auc_roc,
                "auc_pr": auc_pr,
                "precision": precision,
                "f_metric": f,
                "precision_at_k": precision_at_k,
                "rprecision": rprecision,
                "rrecall": rrecall,
                "rf": rf,
                "r_auc_roc": r_auc_roc,
                "r_auc_pr": r_auc_pr,
                "vus_roc": vus_roc,
                "vus_pr": vus_pr,
                "affiliation_precision": affiliation_precision,
                "affiliation_recall": affiliation_recall,
                "total_train_time": total_train_time,
                "total_score_time": total_score_time,
                "n_instances": n_instances,
                "avg_train_time": total_train_time / n_instances,
                "avg_score_time": total_score_time / n_instances,
                "min_train_time": min_train_time,
                "max_train_time": max_train_time,
                "min_score_time": min_score_time,
                "max_score_time": max_score_time,
                "real_pct": pct_real,
                #"anomaly_pct": pct_detected,
            })

    selected_cols = [
        "iteration", "scenario", "raw_pr_auc",
        "real_pct", "raw_pct_detection",
        "tn","fp","fn","tp",
        "precision",
        "avg_train_time", "avg_score_time",
        "min_score_time", "max_score_time",
        "min_train_time", "max_train_time",
       
    ]

  
   
    logging.debug("########################")
    logging.debug("Complete Results:")
    result = pd.DataFrame(df)
    logging.debug(result[[score_column + str(i), error_column + str(i)]])

    logging.debug("########################")
    logging.debug("Summary (mean AUC-PR - without training scores):")

    summary_data = pd.DataFrame(summary_data)
    
    #summary results csv
    summary_data[selected_cols].to_csv(os.path.join(PATH_SUMMARY_RESULTS_IMPROVEMENT, "summary_results_BV3_hnsw_OBKNN.csv"),index=False )




    #Summary as in the original source code
    pivot = summary_data.pivot_table(
        values=["raw_pr_auc"], columns=['scenario'], index=['method'], aggfunc='mean'
    )
    pivot['Avg'] = pivot.mean(axis=1)
    pivot = pivot.round(3).sort_values(by='Avg', ascending=False)
    logging.debug(pivot)

    logging.debug("########################")
    logging.debug("Summary (mean AUC-PR - with training scores):")

    pivot2 = summary_data.pivot_table(
        values=["raw_pr_auc_wtd"], columns=['scenario'], index=['method'], aggfunc='mean'
    )
    pivot2['Avg'] = pivot2.mean(axis=1)
    pivot2 = pivot2.round(3).sort_values(by='Avg', ascending=False)
    logging.debug(pivot2)

    logging.debug("########################")
    logging.debug("Summary (std AUC-PR):")

    pivot3 = summary_data.pivot_table(
        values=["raw_pr_auc"], columns=['scenario'], index=['method'], aggfunc='std'
    )
    pivot3['Avg'] = pivot3.mean(axis=1)
    pivot3 = pivot3.round(5).sort_values(by='Avg', ascending=False)
    logging.debug(pivot3)
