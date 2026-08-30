"""Low-cost competition optimizations for the V2 Simple3D runtime.

The default path keeps the V2 FPFH extractor, but compares descriptors inside
rotation-insensitive spatial regions and fuses the three FPFH scales after
normal-score rank calibration.  No learned backbone or registration network is
required, which keeps the method practical on a GTX 1660.
"""

import hashlib
import json
import time
from pathlib import Path

import torch

from feature_extractors.FPFH import FPFHFeatures


class CompetitionFPFHFeatures(FPFHFeatures):
    """V2 FPFH with a configurable coreset and lightweight score adapters."""

    def __init__(self, args=None):
        super().__init__(args=args)
        self.bank_region_ids = None
        self.scale_score_references = []
        self._active_query_regions = None

    def _compute_geom4d(self, points, center, idx):
        """Avoid an expensive eigendecomposition when geom4d is disabled.

        The V2 extractor computes this tensor unconditionally and discards it
        later.  At 8192 groups that dead calculation dominates GTX 1660 time.
        """
        if not (
            getattr(self.args, "use_geom4d", False)
            or getattr(self.args, "use_geom4d_for_p", False)
        ):
            return None
        return super()._compute_geom4d(points, center, idx)

    @staticmethod
    def _path_text(path):
        while isinstance(path, (list, tuple)) and path:
            path = path[0]
        return str(path) if path is not None else "unknown"

    def _training_cache_path(self, path):
        if not getattr(self.args, "cache_features", False):
            return None
        signature = {
            "path": self._path_text(path),
            "num_group": self.args.num_group,
            "group_size": self.args.group_size,
            "max_nn": self.args.max_nn,
            "normal_max_nn": getattr(self.args, "normal_max_nn", 10),
            "num_MSND": self.args.num_MSND,
            "scale_fusion": getattr(self.args, "scale_fusion", "legacy_equivalent"),
            "fpfh_normalization": getattr(self.args, "fpfh_normalization", "none"),
            "position": getattr(self.args, "use_position_features", False),
            "position_weight": getattr(self.args, "position_weight", 0.0),
            "geom4d": getattr(self.args, "use_geom4d", False),
        }
        digest = hashlib.sha1(
            json.dumps(signature, sort_keys=True).encode("utf-8")
        ).hexdigest()
        cache_dir = Path(getattr(self.args, "feature_cache_dir", "./feature-cache"))
        return cache_dir / f"{digest}.pt"

    def _test_cache_path(self, path):
        if not getattr(self.args, "cache_test_features", False):
            return None
        training_style_path = self._training_cache_path(path)
        if training_style_path is None:
            return None
        return training_style_path.with_name(f"test-{training_style_path.name}")

    def _normalize_fpfh(self, feature_maps):
        """Normalize each 33-bin FPFH scale without mixing their statistics.

        FPFH is a non-negative histogram descriptor.  Square-rooting an L1
        normalized histogram converts Euclidean distance into the Hellinger
        distance, which is substantially less sensitive to a few large bins.
        Keeping the scales separate also prevents one neighborhood scale from
        dominating the nearest-neighbour search.
        """
        mode = getattr(self.args, "fpfh_normalization", "none")
        if mode == "none":
            return feature_maps
        if mode != "hellinger_per_scale":
            raise ValueError(f"Unsupported fpfh_normalization: {mode}")
        if feature_maps.shape[-1] % 33 != 0:
            raise ValueError("Per-scale FPFH normalization requires 33-bin scales.")

        scales = feature_maps.reshape(feature_maps.shape[0], -1, 33)
        scales = scales.clamp_min(0)
        scales = scales / scales.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        scales = torch.sqrt(scales)
        scales = torch.nn.functional.normalize(scales, p=2, dim=-1, eps=1e-12)

        # Preserve the old duplicated-middle-scale distance weighting while
        # retaining a compact 99-D representation.
        if scales.shape[1] == 3 and getattr(self.args, "scale_fusion", "") == "legacy_equivalent":
            weights = scales.new_tensor([1.0, 2.0 ** 0.5, 1.0]).view(1, 3, 1)
            scales = scales * weights
        return scales.reshape(feature_maps.shape[0], -1)

    def _augment_position(self, feature_maps, points, centers):
        if not getattr(self.args, "use_position_features", True):
            return feature_maps
        if getattr(self.args, "use_geom4d", False) or getattr(self.args, "use_geom4d_for_p", False):
            raise ValueError("Position features and geom4d must not be enabled together.")

        center_xyz = centers.squeeze(0).to(feature_maps.device, dtype=feature_maps.dtype)
        point_xyz = points.squeeze(0).to(feature_maps.device, dtype=feature_maps.dtype)
        if feature_maps.shape[0] != center_xyz.shape[0]:
            raise ValueError("Position features require use_LFSA=True.")

        bbox_diagonal = torch.linalg.norm(point_xyz.amax(dim=0) - point_xyz.amin(dim=0))
        bbox_diagonal = torch.clamp(bbox_diagonal, min=1e-6)
        position = center_xyz / bbox_diagonal
        position = position * float(getattr(self.args, "position_weight", 25.0))
        return torch.cat([feature_maps, position], dim=-1)

    def _region_ids(self, centers):
        """Assign centers to rotation-insensitive radial/height regions.

        The PCA normal is sign ambiguous, so height uses its absolute value.
        Every sample is normalized independently; this avoids treating global
        translation, scale or rotation as an anomaly.
        """
        if centers.ndim == 3:
            centers = centers.squeeze(0)
        centers = centers.float()
        radial_bins = max(1, int(getattr(self.args, "region_radial_bins", 1)))
        height_bins = max(1, int(getattr(self.args, "region_height_bins", 1)))
        if radial_bins == 1 and height_bins == 1:
            return torch.zeros(centers.shape[0], dtype=torch.long, device=centers.device)

        centered = centers - torch.median(centers, dim=0).values
        covariance = centered.T.matmul(centered) / max(1, centered.shape[0] - 1)
        _, eigenvectors = torch.linalg.eigh(covariance)
        normal = eigenvectors[:, 0]
        signed_height = centered.matmul(normal)
        height = signed_height.abs()
        planar = centered - signed_height.unsqueeze(1) * normal.unsqueeze(0)
        radius = torch.linalg.norm(planar, dim=1)

        radius_scale = torch.quantile(radius, 0.95).clamp_min(1e-6)
        height_scale = torch.quantile(height, 0.95).clamp_min(1e-6)
        radius_unit = (radius / radius_scale).clamp(0, 1 - 1e-7)
        height_unit = (height / height_scale).clamp(0, 1 - 1e-7)
        radial_id = torch.floor(radius_unit * radial_bins).long()
        height_id = torch.floor(height_unit * height_bins).long()
        return radial_id * height_bins + height_id

    def _scale_slices(self, feature_dim):
        if not getattr(self.args, "use_scale_rank_fusion", False):
            return [slice(0, feature_dim)]
        if getattr(self.args, "use_position_features", False):
            raise ValueError("Scale-rank fusion and concatenated position features are incompatible.")
        if feature_dim % 33 != 0:
            raise ValueError("Scale-rank fusion requires concatenated 33-bin FPFH descriptors.")
        return [slice(start, start + 33) for start in range(0, feature_dim, 33)]

    def _raw_scale_scores(self, query_features, bank_features, query_regions=None, bank_regions=None):
        """Return one nearest-neighbour score vector for every FPFH scale."""
        scale_scores = []
        first_indices = None
        min_region_bank = max(1, int(getattr(self.args, "region_min_bank", 64)))
        use_regions = (
            getattr(self.args, "use_regional_memory", False)
            and query_regions is not None
            and bank_regions is not None
            and query_regions.numel() == query_features.shape[0]
            and bank_regions.numel() == bank_features.shape[0]
        )

        for scale_slice in self._scale_slices(query_features.shape[1]):
            query_scale = query_features[:, scale_slice]
            bank_scale = bank_features[:, scale_slice]
            if not use_regions:
                scores, indices = FPFHFeatures._min_cdist_chunked(self, query_scale, bank_scale)
            else:
                scores = torch.empty(
                    query_scale.shape[0], device=query_scale.device, dtype=query_scale.dtype
                )
                indices = torch.empty(query_scale.shape[0], device=query_scale.device, dtype=torch.long)
                for region in torch.unique(query_regions).tolist():
                    query_mask = query_regions == region
                    bank_mask = bank_regions == region
                    bank_ids = torch.nonzero(bank_mask, as_tuple=False).flatten()
                    if bank_ids.numel() < min_region_bank:
                        bank_ids = torch.arange(bank_scale.shape[0], device=bank_scale.device)
                    region_scores, local_indices = FPFHFeatures._min_cdist_chunked(
                        self, query_scale[query_mask], bank_scale[bank_ids]
                    )
                    scores[query_mask] = region_scores
                    indices[query_mask] = bank_ids[local_indices]
            scale_scores.append(scores)
            if first_indices is None:
                first_indices = indices
        return scale_scores, first_indices

    def _rank_normalize_scale(self, scores, scale_index):
        if scale_index >= len(self.scale_score_references):
            return scores
        reference = self.scale_score_references[scale_index]
        if reference is None or reference.numel() < 2:
            return scores
        reference = reference.to(scores.device, dtype=scores.dtype)
        ranks = torch.searchsorted(reference, scores.contiguous(), right=True).to(scores.dtype)
        ranks = ranks / float(reference.numel())
        q99 = torch.quantile(reference, 0.99).clamp_min(1e-6)
        tail = torch.relu(scores / q99 - 1.0)
        return ranks + float(getattr(self.args, "rank_tail_weight", 0.15)) * tail

    def _fused_multiscale_scores(
        self,
        query_features,
        bank_features,
        query_regions=None,
        bank_regions=None,
        apply_rank=True,
    ):
        scale_scores, nearest_indices = self._raw_scale_scores(
            query_features, bank_features, query_regions, bank_regions
        )
        weights = tuple(getattr(self.args, "decision_scale_weights", (0.25, 0.50, 0.25)))
        if len(weights) != len(scale_scores):
            weights = tuple(1.0 for _ in scale_scores)
        weight_sum = max(sum(float(weight) for weight in weights), 1e-12)
        fused = torch.zeros_like(scale_scores[0])
        for scale_index, (weight, scores) in enumerate(zip(weights, scale_scores)):
            if apply_rank:
                scores = self._rank_normalize_scale(scores, scale_index)
            fused = fused + float(weight) * scores
        return fused / weight_sum, nearest_indices, scale_scores

    def _fit_scale_score_references(self, features, regions):
        """Estimate normal score distributions without extracting points again."""
        self.scale_score_references = []
        if not getattr(self.args, "use_scale_rank_fusion", False):
            return
        max_samples = max(128, int(getattr(self.args, "scale_calibration_samples", 4096)))
        if features.shape[0] > max_samples:
            sample_idx = torch.linspace(0, features.shape[0] - 1, max_samples).long()
            features = features[sample_idx]
            regions = regions[sample_idx]
        features = features.to(self.device)
        regions = regions.to(self.device)
        _, _, raw_scale_scores = self._fused_multiscale_scores(
            features,
            self.patch_lib,
            regions,
            self.bank_region_ids,
            apply_rank=False,
        )
        for scores in raw_scale_scores:
            normal_scores = scores.detach().float()
            nonzero = normal_scores[normal_scores > 1e-8]
            if nonzero.numel() >= 64:
                normal_scores = nonzero
            self.scale_score_references.append(torch.sort(normal_scores).values)
        summary = [
            round(float(torch.quantile(reference, 0.99).item()), 6)
            for reference in self.scale_score_references
        ]
        print(f"   Scale normal-score q99: {summary}")

    def _compute_normal_score_stats(self):
        """Calibrate from a deterministic, representative sample subset.

        V8 rescored all 108 normal objects against the memory bank. Quantile
        calibration only needs enough center scores to estimate the tails, so
        evenly spaced objects retain dataset coverage at much lower cost.
        """
        original_samples = self.train_samples
        sample_limit = max(1, int(getattr(self.args, "normal_calibration_samples", 16)))
        if len(original_samples) > sample_limit:
            indices = torch.linspace(0, len(original_samples) - 1, sample_limit).long().tolist()
            self.train_samples = [original_samples[index] for index in indices]
            print(
                f"   Normal calibration subset: {len(original_samples)} -> "
                f"{len(self.train_samples)} objects"
            )
        try:
            return super()._compute_normal_score_stats()
        finally:
            self.train_samples = original_samples

    def _min_cdist_chunked(self, query_features, bank_features):
        """Return the mean distance to the k nearest normal prototypes.

        k=1 is the original PatchCore behaviour.  A small k suppresses an
        accidental single-prototype match and follows PointCore's robust 3-NN
        operating point without constructing another memory bank.
        """
        if getattr(self.args, "use_scale_rank_fusion", False):
            query_regions = None
            bank_regions = None
            if (
                self._active_query_regions is not None
                and query_features.shape[0] == self._active_query_regions.numel()
                and bank_features.shape[0] == self.patch_lib.shape[0]
            ):
                query_regions = self._active_query_regions
                bank_regions = self.bank_region_ids
            fused, indices, _ = self._fused_multiscale_scores(
                query_features,
                bank_features,
                query_regions,
                bank_regions,
                apply_rank=True,
            )
            return fused, indices

        num_nn = max(1, int(getattr(self.args, "anomaly_num_nn", 1)))
        num_nn = min(num_nn, bank_features.shape[0])
        if num_nn == 1:
            return super()._min_cdist_chunked(query_features, bank_features)

        query_chunk = int(getattr(self.args, "query_chunk", 1024))
        bank_chunk = int(getattr(self.args, "bank_chunk", 8192))
        score_parts = []
        nearest_index_parts = []
        for query_start in range(0, query_features.shape[0], query_chunk):
            query = query_features[query_start:query_start + query_chunk]
            best_values = torch.full(
                (query.shape[0], num_nn),
                float("inf"),
                device=query.device,
                dtype=query.dtype,
            )
            best_indices = torch.zeros(
                (query.shape[0], num_nn), device=query.device, dtype=torch.long
            )
            for bank_start in range(0, bank_features.shape[0], bank_chunk):
                bank = bank_features[bank_start:bank_start + bank_chunk]
                distances = torch.cdist(query, bank)
                indices = torch.arange(
                    bank_start,
                    bank_start + bank.shape[0],
                    device=query.device,
                    dtype=torch.long,
                ).expand(query.shape[0], -1)
                candidates = torch.cat([best_values, distances], dim=1)
                candidate_indices = torch.cat([best_indices, indices], dim=1)
                best_values, order = torch.topk(
                    candidates, k=num_nn, dim=1, largest=False, sorted=True
                )
                best_indices = torch.gather(candidate_indices, 1, order)
            score_parts.append(best_values.mean(dim=1))
            nearest_index_parts.append(best_indices[:, 0])
        return torch.cat(score_parts), torch.cat(nearest_index_parts)

    @staticmethod
    def _fixed_top_mean(values, topk):
        flat = values.flatten()
        topk = max(1, min(int(topk), flat.numel()))
        return torch.topk(flat, k=topk, largest=True, sorted=False).values.mean()

    def _compute_object_score(self, point_score_map, center_raw_scores, center_o_scores, center):
        """Fuse anomaly evidence at category-appropriate spatial extents."""
        if getattr(self.args, "object_score_mode", "legacy") != "category_multi_tail":
            return super()._compute_object_score(
                point_score_map, center_raw_scores, center_o_scores, center
            )

        top80 = self._fixed_top_mean(point_score_map, 80)
        top500 = self._fixed_top_mean(point_score_map, 500)
        q99 = torch.quantile(point_score_map.float(), 0.99)
        q999 = torch.quantile(point_score_map.float(), 0.999)
        weights = getattr(self.args, "object_tail_weights", (0.25, 0.50, 0.25))
        if len(weights) != 3:
            raise ValueError("object_tail_weights must contain three values")
        object_score = (
            float(weights[0]) * top80
            + float(weights[1]) * top500
            + float(weights[2]) * q999
        )
        diagnostic = {
            "raw_top80": top80,
            "raw_top500": top500,
            "normal_q995_excess_mass": 0.0,
            "normal_q995_excess_top": 0.0,
            "coherence_score": 0.0,
            "final_object_score": object_score,
            "raw_score_q99": q99,
            "raw_score_q999": q999,
        }
        return object_score, diagnostic

    def collect_features(self, pc, path=None):
        cache_path = self._training_cache_path(path)
        if cache_path is not None and cache_path.is_file():
            cached = torch.load(cache_path, map_location="cpu")
            feature_maps = cached["features"]
            centers = cached["center"]
            self.patch_lib.append(feature_maps)
            self.train_samples.append(
                {"features": feature_maps, "center": centers, "path": self._path_text(path)}
            )
            return

        feature_maps, _, points, centers = self.get_features(pc)
        feature_maps = self._normalize_fpfh(feature_maps)
        feature_maps = self._augment_position(feature_maps, points, centers)
        feature_cpu = feature_maps.detach().cpu()
        center_cpu = centers.detach().cpu()
        self.patch_lib.append(feature_cpu)
        self.train_samples.append(
            {
                "features": feature_cpu,
                "center": center_cpu,
                "path": self._path_text(path),
            }
        )
        if cache_path is not None:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"features": feature_cpu, "center": center_cpu}, cache_path)

    def predict(self, pc, mask, label, path=None):
        cache_path = self._test_cache_path(path)
        can_rebuild_geometry = getattr(self.args, "dataset", "") == "minishift"
        if cache_path is not None and cache_path.is_file() and can_rebuild_geometry:
            cached = torch.load(cache_path, map_location="cpu")
            feature_maps = cached["features"].to(self.device)
            centers = cached["center"].to(self.device)
            unorganized_pc = pc.squeeze(0).detach().cpu()
            points = pc.to(self.device)
        else:
            feature_maps, unorganized_pc, points, centers = self.get_features(pc)
            feature_maps = self._normalize_fpfh(feature_maps)
            feature_maps = self._augment_position(feature_maps, points, centers)
            if cache_path is not None and can_rebuild_geometry:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "features": feature_maps.detach().cpu(),
                        "center": centers.detach().cpu(),
                    },
                    cache_path,
                )
        self._active_query_regions = self._region_ids(centers).to(feature_maps.device)
        try:
            self.compute_anomay_scores(
                feature_maps,
                mask,
                label,
                path,
                unorganized_pc,
                points,
                centers,
            )
        finally:
            self._active_query_regions = None

    def run_coreset(self):
        coreset_started = time.perf_counter()
        self.patch_lib = torch.cat(self.patch_lib, 0).cpu()
        self._finalize_training_features()
        all_region_ids = torch.cat(
            [self._region_ids(sample["center"]).cpu() for sample in self.train_samples], dim=0
        )
        calibration_features = self.patch_lib
        calibration_regions = all_region_ids

        self.f_coreset = float(getattr(self.args, "coreset_ratio", 0.03))
        if not 0 < self.f_coreset <= 1:
            raise ValueError("coreset_ratio must be in (0, 1].")

        if self.f_coreset < 1:
            preselect_factor = int(getattr(self.args, "coreset_preselect_factor", 0))
            use_regional_coreset = (
                getattr(self.args, "use_regional_memory", False)
                and getattr(self.args, "use_regional_coreset", True)
            )
            region_values = torch.unique(all_region_ids).tolist() if use_regional_coreset else [None]
            selected_parts = []
            print(
                f"   Coreset selection mode: "
                f"{'regional' if use_regional_coreset else 'global'} "
                f"({len(region_values)} bank partitions)"
            )
            for region in region_values:
                if region is None:
                    candidate_idx = torch.arange(self.patch_lib.shape[0])
                else:
                    candidate_idx = torch.nonzero(
                        all_region_ids == int(region), as_tuple=False
                    ).flatten()
                candidate_lib = self.patch_lib[candidate_idx]
                target = max(1, int(round(self.f_coreset * candidate_lib.shape[0])))
                if preselect_factor > 1 and candidate_lib.shape[0] > target * preselect_factor:
                    generator = torch.Generator().manual_seed(int(region or 0))
                    candidate_count = target * preselect_factor
                    preselect_idx = torch.randperm(
                        candidate_lib.shape[0], generator=generator
                    )[:candidate_count]
                    candidate_idx = candidate_idx[preselect_idx]
                    candidate_lib = candidate_lib[preselect_idx]
                selected_candidate_idx = self.get_coreset_idx_randomp(
                    candidate_lib,
                    n=target,
                    eps=self.coreset_eps,
                )
                selected_parts.append(candidate_idx[selected_candidate_idx])
                if region is not None:
                    print(
                        f"   Region {int(region)}: {candidate_lib.shape[0]} candidates -> "
                        f"{target} prototypes"
                    )
            self.coreset_idx = torch.cat(selected_parts)
            self.patch_lib = self.patch_lib[self.coreset_idx].to(self.device)
            self.bank_region_ids = all_region_ids[self.coreset_idx].to(self.device)
            if (
                getattr(self.args, "prototype_density_norm_for_p", False)
                or getattr(self.args, "prototype_density_norm_for_o", False)
            ):
                self._compute_prototype_density()
        else:
            self.patch_lib = self.patch_lib.to(self.device)
            self.bank_region_ids = all_region_ids.to(self.device)

        selection_finished = time.perf_counter()
        self._fit_scale_score_references(calibration_features, calibration_regions)
        scale_calibration_finished = time.perf_counter()
        self._compute_normal_score_stats()
        normal_calibration_finished = time.perf_counter()
        self.optimization_timing = {
            "coreset_selection_seconds": round(selection_finished - coreset_started, 3),
            "scale_calibration_seconds": round(
                scale_calibration_finished - selection_finished, 3
            ),
            "normal_calibration_seconds": round(
                normal_calibration_finished - scale_calibration_finished, 3
            ),
        }
        print(f"   Optimization timing: {self.optimization_timing}")
