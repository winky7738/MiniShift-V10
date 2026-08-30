"""Category-routed final score adapters on top of the unchanged V2 backbone.

Feature extraction, 4096-point LFSA, 128-neighbour aggregation and 5% coreset
selection remain the default V2 operating point.  Each category chooses either
the proven global legacy path or a geometry-aware regional/rank path.  The two
hardest geometry families use a slightly larger memory bank while staying close
to the measured two-hour GTX 1660 budget.
"""


# Object weights are ordered as: top-80 mean, top-500 mean, 99.9% quantile.
# The four locally available competition categories receive explicit profiles;
# the remaining official categories use conservative geometry-family defaults.
CATEGORY_PROFILES = {
    # Ring-like rigid part: V8 full-run diagnostics show top80 AUC=0.7883,
    # while adding q999 lowered the final AUC to 0.7858.
    "nut": {
        "category_profile_name": "localized_ring",
        "object_tail_weights": (1.00, 0.00, 0.00),
        "p_raw_weight": 0.70,
        "p_smooth_weight": 0.30,
        "smooth_k": 8,
    },
    # Smooth free-form body: q999 is more stable than a fixed-area average.
    "piggy": {
        "category_profile_name": "smooth_volumetric",
        "object_tail_weights": (0.05, 0.05, 0.90),
        "p_raw_weight": 0.65,
        "p_smooth_weight": 0.35,
        "smooth_k": 8,
    },
    # Planar parts benefit from stronger denoising and broader-area evidence.
    "flat_pad": {
        "category_profile_name": "planar_broad",
        "object_tail_weights": (0.10, 0.65, 0.25),
        "p_raw_weight": 0.55,
        "p_smooth_weight": 0.45,
        "smooth_k": 12,
    },
    "screen": {
        "category_profile_name": "planar_edge_aware",
        "object_tail_weights": (0.15, 0.60, 0.25),
        "p_raw_weight": 0.60,
        "p_smooth_weight": 0.40,
        "smooth_k": 10,
    },
    # Conservative profiles for the other eight official categories.
    "spring_pad": {
        "category_profile_name": "planar_broad",
        "object_tail_weights": (0.10, 0.65, 0.25),
        "p_raw_weight": 0.55,
        "p_smooth_weight": 0.45,
        "smooth_k": 12,
    },
    "light": {
        "category_profile_name": "planar_broad",
        "object_tail_weights": (0.10, 0.65, 0.25),
        "p_raw_weight": 0.60,
        "p_smooth_weight": 0.40,
        "smooth_k": 10,
    },
    "screw": {
        "category_profile_name": "localized_rotational",
        "object_tail_weights": (0.60, 0.10, 0.30),
        "p_raw_weight": 0.70,
        "p_smooth_weight": 0.30,
        "smooth_k": 6,
    },
    "button_cell": {
        "category_profile_name": "localized_rotational",
        "object_tail_weights": (0.55, 0.15, 0.30),
        "p_raw_weight": 0.65,
        "p_smooth_weight": 0.35,
        "smooth_k": 8,
    },
    "capsule": {
        "category_profile_name": "smooth_volumetric",
        "object_tail_weights": (0.15, 0.35, 0.50),
        "p_raw_weight": 0.65,
        "p_smooth_weight": 0.35,
        "smooth_k": 8,
    },
    "plastic_cylinder": {
        "category_profile_name": "smooth_volumetric",
        "object_tail_weights": (0.15, 0.35, 0.50),
        "p_raw_weight": 0.65,
        "p_smooth_weight": 0.35,
        "smooth_k": 8,
    },
    "cube": {
        "category_profile_name": "edged_volumetric",
        "object_tail_weights": (0.20, 0.50, 0.30),
        "p_raw_weight": 0.65,
        "p_smooth_weight": 0.35,
        "smooth_k": 8,
    },
    "toothbrush": {
        "category_profile_name": "thin_elongated",
        "object_tail_weights": (0.45, 0.25, 0.30),
        "p_raw_weight": 0.70,
        "p_smooth_weight": 0.30,
        "smooth_k": 6,
    },
}


# Profiles are deliberately small: they only alter score-time grouping and
# fusion, so changing CATEGORY does not load another model or add an epoch.
SPATIAL_SCORE_PROFILES = {
    "localized_ring": {
        "region_radial_bins": 4,
        "region_height_bins": 1,
        "decision_scale_weights": (0.25, 0.50, 0.25),
    },
    "smooth_volumetric": {
        "region_radial_bins": 3,
        "region_height_bins": 2,
        "decision_scale_weights": (0.20, 0.45, 0.35),
    },
    "planar_broad": {
        "region_radial_bins": 4,
        "region_height_bins": 2,
        "decision_scale_weights": (0.20, 0.55, 0.25),
    },
    "planar_edge_aware": {
        "region_radial_bins": 4,
        "region_height_bins": 2,
        "decision_scale_weights": (0.15, 0.55, 0.30),
    },
    "localized_rotational": {
        "region_radial_bins": 4,
        "region_height_bins": 2,
        "decision_scale_weights": (0.25, 0.50, 0.25),
    },
    "edged_volumetric": {
        "region_radial_bins": 3,
        "region_height_bins": 2,
        "decision_scale_weights": (0.20, 0.50, 0.30),
    },
    "thin_elongated": {
        "region_radial_bins": 5,
        "region_height_bins": 2,
        "decision_scale_weights": (0.30, 0.50, 0.20),
    },
}


# Full-pipeline routes are based on local full-run evidence:
# - global_legacy reproduces the path that reached piggy 0.799/0.690 and avoids
#   disturbing already-strong capsule/cylinder/button-cell object scores;
# - regional_rank keeps the V9 path that raised screen P-ROC 0.578 -> 0.753;
# - regional_rank_extra spends some remaining time on the low-baseline screw,
#   toothbrush and cube categories by retaining 20% more normal prototypes.
PIPELINE_ROUTES = {
    "global_legacy": {
        "strategy_name": "global_legacy_preserve",
        "use_regional_memory": False,
        "use_regional_coreset": False,
        "use_scale_rank_fusion": False,
        "object_score_mode": "legacy",
        "coreset_ratio": 0.05,
        "normal_calibration_samples": 16,
        "scale_calibration_samples": 4096,
    },
    "regional_rank": {
        "strategy_name": "regional_rank_v9",
        "use_regional_memory": True,
        "use_regional_coreset": True,
        "use_scale_rank_fusion": True,
        "object_score_mode": "category_multi_tail",
        "coreset_ratio": 0.05,
        "normal_calibration_samples": 16,
        "scale_calibration_samples": 4096,
    },
    "regional_rank_extra": {
        "strategy_name": "regional_rank_extra_coverage",
        "use_regional_memory": True,
        "use_regional_coreset": True,
        "use_scale_rank_fusion": True,
        "object_score_mode": "category_multi_tail",
        "coreset_ratio": 0.06,
        "normal_calibration_samples": 24,
        "scale_calibration_samples": 8192,
        "region_min_bank": 80,
    },
}


CATEGORY_PIPELINE_ROUTES = {
    "capsule": "global_legacy",
    "cube": "regional_rank_extra",
    "spring_pad": "regional_rank",
    "screw": "regional_rank_extra",
    "screen": "regional_rank",
    "piggy": "global_legacy",
    "nut": "regional_rank",
    "flat_pad": "regional_rank",
    "plastic_cylinder": "global_legacy",
    "button_cell": "global_legacy",
    "toothbrush": "regional_rank_extra",
    "light": "regional_rank",
}


for _category, _profile in CATEGORY_PROFILES.items():
    _spatial = SPATIAL_SCORE_PROFILES[_profile["category_profile_name"]]
    _profile.update(_spatial)
    _route_name = CATEGORY_PIPELINE_ROUTES[_category]
    _profile.update(PIPELINE_ROUTES[_route_name])
    _profile["pipeline_route"] = _route_name


def validate_category_profiles(valid_categories):
    """Fail early if a category is missing or a weight vector is malformed."""
    missing = set(valid_categories) - set(CATEGORY_PROFILES)
    if missing:
        raise ValueError(f"Missing category optimizer profiles: {sorted(missing)}")
    missing_routes = set(valid_categories) - set(CATEGORY_PIPELINE_ROUTES)
    if missing_routes:
        raise ValueError(f"Missing category pipeline routes: {sorted(missing_routes)}")
    for category, profile in CATEGORY_PROFILES.items():
        weights = profile["object_tail_weights"]
        if len(weights) != 3 or any(weight < 0 for weight in weights):
            raise ValueError(f"Invalid object_tail_weights for {category}: {weights}")
        if abs(sum(weights) - 1.0) > 1e-6:
            raise ValueError(f"object_tail_weights must sum to 1 for {category}: {weights}")
        scale_weights = profile["decision_scale_weights"]
        if len(scale_weights) != 3 or any(weight < 0 for weight in scale_weights):
            raise ValueError(f"Invalid decision_scale_weights for {category}: {scale_weights}")
        if abs(sum(scale_weights) - 1.0) > 1e-6:
            raise ValueError(f"decision_scale_weights must sum to 1 for {category}: {scale_weights}")
        ratio = float(profile["coreset_ratio"])
        if not 0 < ratio <= 0.10:
            raise ValueError(f"coreset_ratio is outside the final safety range for {category}: {ratio}")
