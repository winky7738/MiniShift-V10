"""
PatchCore logic based on https://github.com/rvorias/ind_knn_ad
"""

from sklearn import random_projection
import json
from utils.utils import KNNGaussianBlur
from utils.utils import set_seeds
import numpy as np
from sklearn.metrics import roc_auc_score
import timm
import torch
from tqdm import tqdm
from utils.au_pro_util import calculate_au_pro
from feature_extractors.pointnet2_utils import *
from pointnet2_ops import pointnet2_utils
import cv2
import os
from utils.mvtec3d_util import *
import time
import open3d as o3d
# from feature_extractors.models import *
from torch.utils.data import DataLoader
from knn_cuda import KNN

def fps(data, number):
    '''
        data B N 3
        number int
    '''
    fps_idx = pointnet2_utils.furthest_point_sample(data, number)
    fps_data = pointnet2_utils.gather_operation(data.transpose(1, 2).contiguous(), fps_idx).transpose(1, 2).contiguous()
    return fps_data, fps_idx


def organized_pc_to_unorganized_pc(organized_pc):
    return organized_pc.reshape(organized_pc.shape[0] * organized_pc.shape[1], organized_pc.shape[2])

def normalize(pred, max_value=None, min_value=None):
    if max_value is None or min_value is None:
        return (pred - pred.min()) / (pred.max() - pred.min())
    else:
        return (pred - min_value) / (max_value - min_value)


def apply_ad_scoremap(image, scoremap, alpha=0.5):
    np_image = np.asarray(image, dtype=float)
    scoremap = (scoremap * 255).astype(np.uint8)
    scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
    scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
    return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)

class Features(torch.nn.Module):


    def unorganized_data_to_organized(self,unorganized_pc, none_zero_data_list):
        '''

        Args:
            unorganized_pc:
            none_zero_data_list:

        Returns:

        '''
        # print(none_zero_data_list[0].shape)
        if not isinstance(none_zero_data_list, list):
            none_zero_data_list = [none_zero_data_list]

        for idx in range(len(none_zero_data_list)):
            none_zero_data_list[idx] = none_zero_data_list[idx].squeeze().detach().cpu().numpy()

        # print("unorganized_pc",unorganized_pc.shape)


        unorganized_pc = unorganized_pc.numpy()
        if self.args.dataset == 'mvtec' or self.args.dataset == 'eyecandies':
            nonzero_indices = np.nonzero(np.all(unorganized_pc != 0, axis=1))[0]
            

        full_data_list = []

        for none_zero_data in none_zero_data_list:
            if none_zero_data.ndim == 1:
                none_zero_data = np.expand_dims(none_zero_data,1)
            full_data = np.zeros((unorganized_pc.shape[0], none_zero_data.shape[1]), dtype=none_zero_data.dtype)
            
            if self.args.dataset == 'mvtec' or self.args.dataset == 'eyecandies':
                full_data[nonzero_indices, :] = none_zero_data
            else:
                full_data = none_zero_data

            full_data_reshaped = full_data.reshape((1, unorganized_pc.shape[0], none_zero_data.shape[1]))
            full_data_tensor = torch.tensor(full_data_reshaped).permute(2, 0, 1).unsqueeze(dim=0)
            full_data_list.append(full_data_tensor)

        return full_data_list

    def normalize(self,pred, max_value=None, min_value=None):
        if max_value is None or min_value is None:
            return (pred - pred.min()) / (pred.max() - pred.min())
        else:
            return (pred - min_value) / (max_value - min_value)


    def apply_ad_scoremap(self,image, scoremap, alpha=0.5):
        np_image = np.asarray(image, dtype=float)
        scoremap = (scoremap * 255).astype(np.uint8)
        scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
        scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
        return (alpha * np_image + (1 - alpha) * scoremap).astype(np.uint8)




    def __init__(self, image_size=224, f_coreset=0.1, coreset_eps=0.9,args = None):
        super().__init__()
        requested_device = getattr(args, "device", "cuda:0")
        self.device = requested_device if torch.cuda.is_available() else "cpu"
        # self.deep_feature_extractor = Model(device=self.device)
        # self.deep_feature_extractor.to(self.device)
        # self.deep_feature_extractor.freeze_parameters(layers=[], freeze_bn=True)
        self.args = args
        self.image_size = image_size
        self.f_coreset = f_coreset
        self.coreset_eps = getattr(args, "coreset_eps", coreset_eps)
        self.average = torch.nn.AvgPool2d(3, stride=1)
        self.blur = KNNGaussianBlur(4)
        self.n_reweight = 3
        set_seeds(0)
        self.patch_lib = []
        self.anomaly_patch_lib = []
        self.pre_patch_lib = []
        self.tmp_patch_lib = []
        self.name_list = []
        self.test_patch_lib = []


        self.image_preds = list()
        self.image_labels = list()
        self.pixel_preds = list()
        self.pixel_labels = list()
        self.gts = []
        self.predictions = []
        self.image_rocauc = 0
        self.pixel_rocauc = 0
        self.au_pro = 0
        self.prototype_density = None
        self.geom_median = None
        self.geom_mad = None
        self.current_class_name = None
        self.train_samples = []
        self.normal_score_stats = {}
        self.sample_diagnostics = []

    def __call__(self, x):
        # Extract the desired feature maps using the backbone model.
        with torch.no_grad():
            feature_maps = self.deep_feature_extractor(x)

        feature_maps = [fmap.to("cpu") for fmap in feature_maps]
        return feature_maps

    def add_sample_to_mem_bank(self, sample):
        raise NotImplementedError

    def predict(self, sample, mask, label):
        raise NotImplementedError

    def robust_object_score(self, point_scores, top_ratio=0.001, min_topk=80, max_topk=2048):
        flat_scores = point_scores.flatten()
        topk = int(round(flat_scores.numel() * top_ratio))
        topk = max(min_topk, min(topk, max_topk, flat_scores.numel()))
        top_values = torch.topk(flat_scores, topk, largest=True, sorted=False).values
        return torch.mean(top_values)

    def _needs_normal_calibration(self):
        return (
            getattr(self.args, "p_map_mode", "legacy") == "dual_center"
            or getattr(self.args, "object_score_mode", "legacy") == "normal_tail_coherence"
        )

    def _safe_quantile(self, values, q, default=1.0):
        if values is None or values.numel() == 0:
            return torch.tensor(default, dtype=torch.float32)
        return torch.quantile(values.float(), q)

    def _safe_topk_mean(self, values, ratio=0.01, min_topk=1):
        flat = values.flatten()
        if flat.numel() == 0:
            return torch.tensor(0.0, device=values.device if torch.is_tensor(values) else "cpu")
        topk = max(min_topk, int(round(flat.numel() * ratio)))
        topk = min(topk, flat.numel())
        return torch.topk(flat, topk, largest=True, sorted=False).values.mean()

    def _normalize_by_stat(self, values, stat_value):
        if stat_value is None:
            return values
        if not torch.is_tensor(stat_value):
            stat_value = torch.tensor(float(stat_value), device=values.device, dtype=values.dtype)
        stat_value = stat_value.to(values.device, dtype=values.dtype)
        return values / torch.clamp(stat_value, min=1e-6)

    def _center_knn_mean(self, center_scores, centers, smooth_k=None, weighted=True):
        if center_scores.ndim == 1:
            center_scores = center_scores.view(1, 1, -1)
        if centers.ndim == 2:
            centers = centers.unsqueeze(0)

        num_centers = centers.shape[1]
        if num_centers <= 1:
            return center_scores.view(-1)

        smooth_k = smooth_k or getattr(self.args, "smooth_k", 8)
        neighbor_k = min(num_centers, smooth_k + 1)
        knn = KNN(k=neighbor_k, transpose_mode=True)
        dists, idx = knn(centers, centers)

        if neighbor_k > 1:
            idx = idx[:, :, 1:]
            dists = dists[:, :, 1:]

        neighbor_scores = index_points(center_scores.permute(0, 2, 1), idx).squeeze(-1)
        if weighted:
            weights = 1.0 / (dists + 1e-8)
            weights = weights / torch.clamp(weights.sum(dim=-1, keepdim=True), min=1e-8)
            smoothed = (neighbor_scores * weights).sum(dim=-1)
        else:
            smoothed = neighbor_scores.mean(dim=-1)
        return smoothed.view(-1)

    def _legacy_object_score(self, score_map):
        if self.args.dataset == 'real':
            return torch.mean(score_map)
        if self.args.dataset == 'shapenet':
            tmp_s, _ = torch.topk(score_map, min(80, score_map.numel()))
            return torch.mean(tmp_s)
        if self.args.dataset in {'mulsen', 'minishift', 'quan'}:
            if getattr(self.args, "use_robust_object_score", False):
                return self.robust_object_score(
                    score_map,
                    top_ratio=getattr(self.args, "object_top_ratio", 0.001),
                    min_topk=getattr(self.args, "object_score_min_topk", 80),
                    max_topk=getattr(self.args, "object_score_max_topk", 2048),
                )
            tmp_s, _ = torch.topk(score_map, min(80, score_map.numel()))
            return torch.mean(tmp_s)
        return torch.max(score_map)

    def _get_diagnostic_path(self):
        diagnostic_dir = getattr(self.args, "diagnostic_dir", "./logs")
        if not (os.path.isdir(diagnostic_dir) or os.path.islink(diagnostic_dir)):
            os.makedirs(diagnostic_dir, exist_ok=True)
        class_name = self.current_class_name or "unknown"
        return os.path.join(diagnostic_dir, f"{self.args.expname}_{class_name}_diagnostics.jsonl")

    def _write_sample_diagnostic(self, record):
        record = {key: (value.item() if torch.is_tensor(value) else value) for key, value in record.items()}
        with open(self._get_diagnostic_path(), "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _min_cdist_chunked(self, query_features, bank_features):
        query_chunk = getattr(self.args, "query_chunk", 1024)
        bank_chunk = getattr(self.args, "bank_chunk", 8192)
        min_vals = []
        min_indices = []

        for query_start in range(0, query_features.shape[0], query_chunk):
            query_end = min(query_start + query_chunk, query_features.shape[0])
            query_chunk_features = query_features[query_start:query_end]
            query_min_vals = torch.full(
                (query_chunk_features.shape[0],),
                float("inf"),
                device=query_features.device,
                dtype=query_features.dtype,
            )
            query_min_indices = torch.zeros(
                query_chunk_features.shape[0],
                device=query_features.device,
                dtype=torch.long,
            )

            for bank_start in range(0, bank_features.shape[0], bank_chunk):
                bank_end = min(bank_start + bank_chunk, bank_features.shape[0])
                bank_chunk_features = bank_features[bank_start:bank_end]
                distances = torch.cdist(query_chunk_features, bank_chunk_features)
                chunk_min_vals, chunk_min_indices = torch.min(distances, dim=1)
                better_mask = chunk_min_vals < query_min_vals
                query_min_vals[better_mask] = chunk_min_vals[better_mask]
                query_min_indices[better_mask] = chunk_min_indices[better_mask] + bank_start

            min_vals.append(query_min_vals)
            min_indices.append(query_min_indices)

        return torch.cat(min_vals, dim=0), torch.cat(min_indices, dim=0)

    def _reduce_bank_for_calibration(self, bank):
        if bank.shape[0] <= 1:
            return bank.to(self.device)

        target_n = int(self.f_coreset * bank.shape[0]) if self.f_coreset < 1 else min(int(self.f_coreset), bank.shape[0])
        target_n = max(1, min(target_n, bank.shape[0]))
        if target_n >= bank.shape[0]:
            return bank.to(self.device)

        idx = self.get_coreset_idx_randomp(bank, n=target_n, eps=self.coreset_eps, force_cpu=not torch.cuda.is_available())
        return bank[idx].to(self.device)

    def _compute_normal_score_stats(self):
        self.normal_score_stats = {
            "raw_q99": torch.tensor(1.0, dtype=torch.float32),
            "smooth_q99": torch.tensor(1.0, dtype=torch.float32),
            "normal_q95": torch.tensor(0.0, dtype=torch.float32),
            "normal_q99": torch.tensor(0.0, dtype=torch.float32),
            "normal_q995": torch.tensor(0.0, dtype=torch.float32),
            "normal_q999": torch.tensor(0.0, dtype=torch.float32),
            "normal_tail_median": torch.tensor(0.0, dtype=torch.float32),
            "normal_tail_MAD": torch.tensor(1.0, dtype=torch.float32),
        }

        if not self._needs_normal_calibration() or len(self.train_samples) == 0:
            return

        folds = max(1, min(getattr(self.args, "normal_calibration_folds", 0), len(self.train_samples)))
        if folds <= 1:
            folds = 1

        raw_scores_all = []
        smooth_scores_all = []

        for fold_idx in range(folds):
            holdout = [sample for sample_idx, sample in enumerate(self.train_samples) if sample_idx % folds == fold_idx]
            bank_parts = [sample["features"] for sample_idx, sample in enumerate(self.train_samples) if sample_idx % folds != fold_idx]

            if not bank_parts:
                bank = self.patch_lib
            else:
                bank = torch.cat(bank_parts, dim=0)
                bank = self._reduce_bank_for_calibration(bank)

            for sample in holdout:
                sample_features = sample["features"].to(self.device)
                sample_center = sample["center"].to(self.device)
                raw_center_scores, _ = self._min_cdist_chunked(sample_features, bank)
                smooth_center_scores = self._center_knn_mean(
                    raw_center_scores,
                    sample_center,
                    smooth_k=getattr(self.args, "smooth_k", 8),
                    weighted=True,
                )
                raw_scores_all.append(raw_center_scores.detach().cpu())
                smooth_scores_all.append(smooth_center_scores.detach().cpu())

        if not raw_scores_all:
            return

        raw_values = torch.cat(raw_scores_all).float()
        smooth_values = torch.cat(smooth_scores_all).float() if smooth_scores_all else raw_values
        threshold = self._safe_quantile(raw_values, 0.995, default=0.0)
        tail_values = raw_values[raw_values >= threshold]
        if tail_values.numel() == 0:
            tail_values = raw_values
        tail_median = torch.median(tail_values)
        tail_mad = torch.median(torch.abs(tail_values - tail_median))
        tail_mad = torch.clamp(tail_mad, min=1e-6)

        self.normal_score_stats = {
            "raw_q99": torch.clamp(self._safe_quantile(raw_values, 0.99, default=1.0), min=1e-6),
            "smooth_q99": torch.clamp(self._safe_quantile(smooth_values, 0.99, default=1.0), min=1e-6),
            "normal_q95": self._safe_quantile(raw_values, 0.95, default=0.0),
            "normal_q99": self._safe_quantile(raw_values, 0.99, default=0.0),
            "normal_q995": threshold,
            "normal_q999": self._safe_quantile(raw_values, 0.999, default=0.0),
            "normal_tail_median": tail_median,
            "normal_tail_MAD": tail_mad,
        }

    def _apply_post_smoothing(self, score_map, points, centers, center_scores):
        mode = getattr(self.args, "post_smooth_mode", "legacy")
        if mode == "none":
            return score_map

        if mode == "knn_mean":
            smooth_k = getattr(self.args, "post_smooth_k", 12)
            knn = KNN(k=min(smooth_k, centers.shape[1]), transpose_mode=True)
            _, center_neighbor_idx = knn(centers, centers)
            center_neighbor_scores = index_points(center_scores.permute(0, 2, 1), center_neighbor_idx)
            smoothed_center_scores = center_neighbor_scores.mean(dim=2).permute(0, 2, 1)
            return interpolating_points_chunked(
                points.permute(0, 2, 1).to(self.device),
                centers.permute(0, 2, 1).to(self.device),
                smoothed_center_scores.to(self.device),
                chunk_size=getattr(self.args, "interp_chunk_size", 10000),
            ).permute(0, 2, 1)

        num_group = getattr(self.args, "post_smooth_centers", 1024)
        group_size = getattr(self.args, "post_smooth_k", 12)

        batch_size, num_points, _ = points.contiguous().shape
        smooth_centers, _ = fps(points.contiguous(), num_group)  # B G 3

        knn = KNN(k=group_size, transpose_mode=True)
        _, idx = knn(points, smooth_centers)  # B G M

        idx_base = torch.arange(0, batch_size, device=points.device).view(-1, 1, 1) * num_points
        idx = (idx + idx_base).view(-1)
        neighborhood = score_map.reshape(batch_size * num_points, -1)[idx, :]
        neighborhood = neighborhood.reshape(batch_size, num_group, group_size, -1).contiguous()
        agg_s_map = torch.mean(neighborhood, -2).view(1, 1, -1)

        return interpolating_points_chunked(
            points.permute(0, 2, 1).to(self.device),
            smooth_centers.permute(0, 2, 1).to(self.device),
            agg_s_map.to(self.device),
            chunk_size=getattr(self.args, "interp_chunk_size", 10000),
        ).permute(0, 2, 1)

    def _compute_prototype_density(self):
        if self.patch_lib is None or self.patch_lib.shape[0] == 0:
            self.prototype_density = None
            return

        density_k = getattr(self.args, "prototype_density_k", 5)
        bank_chunk = getattr(self.args, "bank_chunk", 8192)
        prototype_density = torch.empty(self.patch_lib.shape[0], device=self.patch_lib.device, dtype=self.patch_lib.dtype)

        for bank_start in range(0, self.patch_lib.shape[0], bank_chunk):
            bank_end = min(bank_start + bank_chunk, self.patch_lib.shape[0])
            bank_slice = self.patch_lib[bank_start:bank_end]
            all_dist_chunks = []

            for ref_start in range(0, self.patch_lib.shape[0], bank_chunk):
                ref_end = min(ref_start + bank_chunk, self.patch_lib.shape[0])
                ref_slice = self.patch_lib[ref_start:ref_end]
                dist_chunk = torch.cdist(bank_slice, ref_slice)
                if bank_start == ref_start:
                    diag_len = min(bank_slice.shape[0], ref_slice.shape[0])
                    diag_idx = torch.arange(diag_len, device=dist_chunk.device)
                    dist_chunk[diag_idx, diag_idx] = float("inf")
                all_dist_chunks.append(dist_chunk)

            all_distances = torch.cat(all_dist_chunks, dim=1)
            nearest_distances = torch.topk(
                all_distances,
                k=min(density_k, all_distances.shape[1]),
                dim=1,
                largest=False,
                sorted=False,
            ).values
            prototype_density[bank_start:bank_end] = nearest_distances.median(dim=1).values

        clip_low = getattr(self.args, "prototype_density_clip_low", 0.01)
        clip_high = getattr(self.args, "prototype_density_clip_high", 0.99)
        low_q = torch.quantile(prototype_density, clip_low)
        high_q = torch.quantile(prototype_density, clip_high)
        self.prototype_density = prototype_density.clamp(low_q, high_q)

    def _finalize_training_features(self):
        if not (getattr(self.args, "use_geom4d", False) or getattr(self.args, "use_geom4d_for_p", False)) or self.patch_lib.shape[1] < 4:
            return

        geom = self.patch_lib[:, -4:]
        geom_median = torch.median(geom, dim=0).values
        geom_mad = torch.median(torch.abs(geom - geom_median), dim=0).values
        geom_mad = torch.clamp(geom_mad, min=1e-6)
        self.geom_median = geom_median
        self.geom_mad = geom_mad
        self.patch_lib[:, -4:] = ((geom - geom_median) / geom_mad) * self.args.geom_weight
        for sample in self.train_samples:
            sample_geom = sample["features"][:, -4:]
            sample["features"][:, -4:] = ((sample_geom - geom_median.cpu()) / geom_mad.cpu()) * self.args.geom_weight

    def init_para(self):
        self.image_preds = list()
        self.image_labels = list()
        self.pixel_preds = list()
        self.pixel_labels = list()
        self.gts = []
        self.predictions = []
        self.image_rocauc = 0
        self.pixel_rocauc = 0
        self.au_pro = 0
        self.sample_diagnostics = []
        diagnostic_path = self._get_diagnostic_path()
        if os.path.exists(diagnostic_path):
            os.remove(diagnostic_path)

    def _compute_point_score_map(self, center_raw_scores, unorganized_pc, unorganized_pc_no_zeros, center):
        feature_map_dims = center_raw_scores.shape[0]
        center_score_map = center_raw_scores.view(1, 1, feature_map_dims)

        if not self.args.use_LFSA:
            return center_score_map.squeeze(0), center_raw_scores

        if getattr(self.args, "p_map_mode", "legacy") == "dual_center":
            smooth_center = self._center_knn_mean(
                center_raw_scores,
                center,
                smooth_k=getattr(self.args, "smooth_k", 8),
                weighted=True,
            )
            raw_stat = self.normal_score_stats.get("raw_q99", torch.tensor(1.0))
            smooth_stat = self.normal_score_stats.get("smooth_q99", torch.tensor(1.0))
            raw_norm = self._normalize_by_stat(center_raw_scores, raw_stat)
            smooth_norm = self._normalize_by_stat(smooth_center, smooth_stat)
            fused_center = (
                getattr(self.args, "p_raw_weight", 0.65) * raw_norm
                + getattr(self.args, "p_smooth_weight", 0.35) * smooth_norm
            )
            center_score_map = fused_center.view(1, 1, feature_map_dims)
        else:
            fused_center = center_raw_scores

        point_score_map = interpolating_points_chunked(
            unorganized_pc_no_zeros.permute(0, 2, 1).to(self.device),
            center.permute(0, 2, 1).to(self.device),
            center_score_map.to(self.device),
            chunk_size=getattr(self.args, "interp_chunk_size", 10000),
        ).permute(0, 2, 1)

        if getattr(self.args, "p_map_mode", "legacy") != "dual_center":
            point_score_map = self._apply_post_smoothing(
                point_score_map,
                unorganized_pc_no_zeros,
                center,
                center_raw_scores.view(1, 1, feature_map_dims),
            )

        point_score_map = torch.Tensor(self.unorganized_data_to_organized(unorganized_pc, [point_score_map])[0])
        if self.args.dataset in {'mvtec', 'eyecandies'}:
            point_score_map = point_score_map.squeeze().reshape(1, 224, 224)
            point_score_map = self.blur(point_score_map)

        return point_score_map.squeeze(0), fused_center

    def _compute_object_score(self, point_score_map, center_raw_scores, center_o_scores, center):
        if self.args.dataset in {'mulsen', 'minishift', 'quan'} and getattr(self.args, "object_score_mode", "legacy") == "normal_tail_coherence":
            threshold = self.normal_score_stats.get("normal_q995", torch.tensor(0.0)).to(center_o_scores.device, dtype=center_o_scores.dtype)
            tail_mad = self.normal_score_stats.get("normal_tail_MAD", torch.tensor(1.0)).to(center_o_scores.device, dtype=center_o_scores.dtype)
            excess = torch.relu(center_o_scores - threshold)
            excess = excess / torch.clamp(tail_mad, min=1e-6)
            tail_mass = excess.mean()
            tail_top = self._safe_topk_mean(excess, ratio=0.01, min_topk=1)
            local_excess = self._center_knn_mean(
                excess,
                center,
                smooth_k=getattr(self.args, "o_coherence_k", 12),
                weighted=False,
            )
            coherence = self._safe_topk_mean(local_excess, ratio=0.01, min_topk=1)
            object_score = 0.35 * tail_mass + 0.35 * tail_top + 0.30 * coherence
            diagnostic = {
                "raw_top80": self._safe_topk_mean(center_raw_scores, ratio=80.0 / max(1, center_raw_scores.numel()), min_topk=1),
                "raw_top500": self._safe_topk_mean(center_raw_scores, ratio=500.0 / max(1, center_raw_scores.numel()), min_topk=1),
                "normal_q995_excess_mass": tail_mass,
                "normal_q995_excess_top": tail_top,
                "coherence_score": coherence,
                "final_object_score": object_score,
                "raw_score_q99": self._safe_quantile(center_raw_scores.detach().cpu(), 0.99, default=0.0),
                "raw_score_q999": self._safe_quantile(center_raw_scores.detach().cpu(), 0.999, default=0.0),
            }
            return object_score, diagnostic

        object_score = self._legacy_object_score(point_score_map)
        diagnostic = {
            "raw_top80": self._safe_topk_mean(point_score_map, ratio=80.0 / max(1, point_score_map.numel()), min_topk=1),
            "raw_top500": self._safe_topk_mean(point_score_map, ratio=500.0 / max(1, point_score_map.numel()), min_topk=1),
            "normal_q995_excess_mass": 0.0,
            "normal_q995_excess_top": 0.0,
            "coherence_score": 0.0,
            "final_object_score": object_score,
            "raw_score_q99": self._safe_quantile(point_score_map.detach().cpu(), 0.99, default=0.0),
            "raw_score_q999": self._safe_quantile(point_score_map.detach().cpu(), 0.999, default=0.0),
        }
        return object_score, diagnostic




    def compute_anomay_scores(self, patch, mask, label, path, unorganized_pc, unorganized_pc_no_zeros, center):

        patch = patch.to(self.device)
        min_val, min_idx = self._min_cdist_chunked(patch, self.patch_lib)
        center_p_scores = min_val
        if getattr(self.args, "prototype_density_norm_for_p", False) and self.prototype_density is not None:
            center_p_scores = center_p_scores / (self.prototype_density[min_idx] + 1e-6)

        center_o_scores = min_val
        if getattr(self.args, "prototype_density_norm_for_o", False) and self.prototype_density is not None:
            center_o_scores = center_o_scores / (self.prototype_density[min_idx] + 1e-6)

        s_map, fused_center_scores = self._compute_point_score_map(
            center_p_scores,
            unorganized_pc,
            unorganized_pc_no_zeros,
            center,
        )
        s, diagnostic = self._compute_object_score(s_map, min_val, center_o_scores, center)
        

        if self.args.vis_save:
            while isinstance(path,list):
                path = path[0]
            from pathlib import Path
            path_obj = Path(path)
            post_data_path = None

            if "data" in path:
                parts = path.split("data", 1)
                if len(parts) > 1:
                    post_data_path = parts[1].lstrip(os.sep)

            if post_data_path is None and self.args.dataset == "minishift":
                from data.MiniShiftAD import DATASETS_PATH as dataset_root
                try:
                    post_data_path = str(path_obj.resolve().relative_to(Path(dataset_root).resolve()))
                except ValueError:
                    pass

            if post_data_path is None:
                post_data_path = os.path.join(self.args.dataset, path_obj.name)

            save_path = "./vis-results/"+post_data_path

            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            scoremap = normalize(s_map.squeeze())

            scoremap = (scoremap.cpu().numpy() * 255).astype(np.uint8)
            scoremap = cv2.applyColorMap(scoremap, cv2.COLORMAP_JET)
            scoremap = cv2.cvtColor(scoremap, cv2.COLOR_BGR2RGB)
            unorganized_pc = unorganized_pc.squeeze().cpu()
            scoremap = torch.Tensor(scoremap).squeeze()
            outpoints = torch.cat([unorganized_pc,scoremap],1)

            save_path = str(Path(save_path).with_suffix(".txt"))
            np.savetxt(save_path, outpoints.numpy())
            save_path = "./vis-results-GT/"+post_data_path

            save_dir = os.path.dirname(save_path)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            scoremap = scoremap.cpu().numpy().astype(np.uint8)
            scoremap[mask.flatten().numpy()==1]=np.array([255,0,0])
            scoremap[mask.flatten().numpy()==0]=np.array([0,0,255])
            scoremap = torch.Tensor(scoremap).squeeze()
            outpoints = torch.cat([unorganized_pc,scoremap],1)
            save_path = str(Path(save_path).with_suffix(".txt"))
            np.savetxt(save_path, outpoints.numpy())


        self.image_preds.append(s.cpu().numpy())
        self.image_labels.append(label)
        self.pixel_preds.extend(s_map.cpu().flatten().numpy())
        self.pixel_labels.extend(mask.flatten().numpy())

        self.predictions.append(s_map.squeeze().detach().cpu().squeeze().numpy())
        self.gts.append(mask.squeeze().detach().cpu().squeeze().numpy())
        diagnostic_record = {
            "category": self.current_class_name or "unknown",
            "sample_id": str(path[0] if isinstance(path, list) else path),
            "label": int(label.flatten()[0].item() if torch.is_tensor(label) else label[0] if isinstance(label, (list, tuple, np.ndarray)) else label),
            "raw_top80": diagnostic["raw_top80"],
            "raw_top500": diagnostic["raw_top500"],
            "normal_q995_excess_mass": diagnostic["normal_q995_excess_mass"],
            "normal_q995_excess_top": diagnostic["normal_q995_excess_top"],
            "coherence_score": diagnostic["coherence_score"],
            "final_object_score": diagnostic["final_object_score"],
            "raw_score_q99": diagnostic["raw_score_q99"],
            "raw_score_q999": diagnostic["raw_score_q999"],
        }
        self._write_sample_diagnostic(diagnostic_record)








    def calculate_metrics(self,path=None):
        self.image_preds = np.stack(self.image_preds)
        self.image_labels = np.stack(self.image_labels)
        self.pixel_preds = np.array(self.pixel_preds)

        if not path == None:
            numpy_save = normalize(self.image_preds)
            numpy_save = (numpy_save * 255).astype(np.uint8)
            numpy_save_gt = (self.image_labels*255).astype(np.uint8)[:,0]
            numpy_save = np.append(numpy_save, numpy_save_gt, axis=0)
            np.save(path, numpy_save)


        self.image_rocauc = roc_auc_score(self.image_labels, self.image_preds)
        self.pixel_rocauc = roc_auc_score(self.pixel_labels, self.pixel_preds)
        if self.args.dataset == 'mvtec' or self.args.dataset == 'eyecandies':
            self.au_pro, _ = calculate_au_pro(self.gts, self.predictions)
        else:
            self.au_pro = 0



    def run_coreset(self):
        self.patch_lib = torch.cat(self.patch_lib, 0).cpu()
        self._finalize_training_features()

        self.f_coreset = 0.05
        if self.f_coreset < 1:
            self.coreset_idx = self.get_coreset_idx_randomp(self.patch_lib,
                                                            n=int(self.f_coreset * self.patch_lib.shape[0]),
                                                            eps=self.coreset_eps, )
            
            self.patch_lib = self.patch_lib[self.coreset_idx].to(self.device)
            if getattr(self.args, "prototype_density_norm_for_p", False) or getattr(self.args, "prototype_density_norm_for_o", False):
                self._compute_prototype_density()
        else:
            self.patch_lib = self.patch_lib.to(self.device)

        self._compute_normal_score_stats()

               

    def get_coreset_idx_randomp(self, z_lib, n=1000, eps=0.90, float16=True, force_cpu=False):
        """Returns n coreset idx for given z_lib.
        Performance on AMD3700, 32GB RAM, RTX3080 (10GB):
        CPU: 40-60 it/s, GPU: 500+ it/s (float32), 1500+ it/s (float16)
        Args:
            z_lib:      (n, d) tensor of patches.
            n:          Number of patches to select.
            eps:        Agression of the sparse random projection.
            float16:    Cast all to float16, saves memory and is a bit faster (on GPU).
            force_cpu:  Force cpu, useful in case of GPU OOM.
        Returns:
            coreset indices
        """

        print(f"   Fitting random projections. Start dim = {z_lib.shape}.")
        try:
            transformer = random_projection.SparseRandomProjection(eps=eps)
            z_lib = torch.tensor(transformer.fit_transform(z_lib))
            print(f"   DONE.                 Transformed dim = {z_lib.shape}.")
        except ValueError:
            print("   Error: could not project vectors. Please increase `eps`.")

        select_idx = 0
        last_item = z_lib[select_idx:select_idx + 1]
        coreset_idx = [torch.tensor(select_idx)]
        min_distances = torch.linalg.norm(z_lib - last_item, dim=1, keepdims=True)
        # The line below is not faster than linalg.norm, although i'm keeping it in for
        # future reference.
        # min_distances = torch.sum(torch.pow(z_lib-last_item, 2), dim=1, keepdims=True)

        if float16:
            last_item = last_item.half()
            z_lib = z_lib.half()
            min_distances = min_distances.half()
        if torch.cuda.is_available() and not force_cpu:
            last_item = last_item.to(self.device)
            z_lib = z_lib.to(self.device)
            min_distances = min_distances.to(self.device)

        for _ in tqdm(range(n - 1)):
            distances = torch.linalg.norm(z_lib - last_item, dim=1, keepdims=True)  # broadcasting step
            min_distances = torch.minimum(distances, min_distances)  # iterative step
            select_idx = torch.argmax(min_distances)  # selection step

            # bookkeeping
            last_item = z_lib[select_idx:select_idx + 1]
            min_distances[select_idx] = 0
            coreset_idx.append(select_idx.to("cpu"))
        return torch.stack(coreset_idx)
