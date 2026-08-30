"""Train and evaluate all configured MiniShiftAD categories.

Edit only the CONFIG section below for normal use. The categories in
``CATEGORIES`` are trained sequentially with category-routed final settings.
"""

import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace

import torch


# ============================== CONFIG ==============================
CATEGORIES = (
    "capsule",
    "cube",
    "spring_pad",
    "screw",
    "screen",
    "piggy",
    "nut",
    "flat_pad",
    "plastic_cylinder",
    "button_cell",
    "toothbrush",
    "light",
)
CATEGORY = CATEGORIES[0]          # active category, updated by the batch runner
DATA_ROOT = "/mnt/d/BaiduNetdiskDownload"
V2_CODE_ROOT = "/mnt/d/download/xwechat_files/wxid_lzehrb17j6c221_d011/msg/file/2026-08/MiniShift-Simple3D-main_V2/MiniShift-Simple3D-main"
LEVEL = "ALL"                    # ALL / easy / medium / hard
RUN_TAG = "V10_AI_handoff_12cat" # 12-category handoff configuration
SKIP_COMPLETED_RESULTS = True     # safe resume after interruption or machine restart

# Competition profile: the paper's 4k accuracy/speed operating point.
NUM_GROUP = 4096
GROUP_SIZE = 128
MAX_NN = 40
NORMAL_MAX_NN = 20                 # best measured setting on nut (V2: 0.7555 mean)
USE_LFSA = True
USE_MSND = True
NUM_MSND = 2
FEATURE = "FPFH"

# V2 point-level and object-level scoring options.
SCALE_FUSION = "legacy_equivalent"  # fixes the old duplicated middle scale
P_MAP_MODE = "dual_center"
SMOOTH_K = 8
P_RAW_WEIGHT = 0.65
P_SMOOTH_WEIGHT = 0.35
OBJECT_SCORE_MODE = "category_multi_tail"
NORMAL_CALIBRATION_FOLDS = 0
NORMAL_CALIBRATION_SAMPLES = 16       # 16 x 4096 centers is sufficient for tail quantiles
OBJECT_TOP_RATIO = 0.001
QUERY_CHUNK = 1024
BANK_CHUNK = 8192
INTERP_CHUNK_SIZE = 4096
TARGET_RUNTIME_MINUTES = 120       # reporting target only; training is never force-stopped
CORESET_RATIO = 0.05               # original V2 5% memory bank
CORESET_PRESELECT_FACTOR = 0       # original V2 greedy coreset over all features
USE_REGIONAL_CORESET = True        # faster and preserves coverage in every spatial region
ANOMALY_NUM_NN = 1                 # 3-NN was rejected by fixed-subset A/B
FPFH_NORMALIZATION = "none"       # Hellinger was rejected by fixed-subset A/B
USE_POSITION_FEATURES = False      # raw XYZ is not rotation invariant on MiniShift
POSITION_WEIGHT = 0.0

# Lightweight 3D score adapters. Category-specific bin counts and scale
# weights are supplied by category_optimizer.py after CATEGORY is resolved.
USE_REGIONAL_MEMORY = True
REGION_RADIAL_BINS = 4
REGION_HEIGHT_BINS = 2
REGION_MIN_BANK = 64
USE_SCALE_RANK_FUSION = True
DECISION_SCALE_WEIGHTS = (0.25, 0.50, 0.25)
SCALE_CALIBRATION_SAMPLES = 4096
RANK_TAIL_WEIGHT = 0.15

RUN_TEST_AFTER_TRAIN = True       # False: only build the training memory bank
SAVE_VISUALIZATION = False        # True can create a large amount of output
CACHE_TEST_FEATURES = True        # repeat scoring runs reuse deterministic test FPFH
DEVICE = "cuda:0"
# ====================================================================


os.environ["MINISHIFT_DATASET_PATH"] = DATA_ROOT

configured_v2_root = Path(V2_CODE_ROOT).resolve()
bundled_v2_root = (Path(__file__).resolve().parent.parent / "v2_runtime").resolve()
v2_root = configured_v2_root if configured_v2_root.is_dir() else bundled_v2_root
if not v2_root.is_dir():
    raise FileNotFoundError(
        f"V2 code directory not found at {configured_v2_root} or bundled path {bundled_v2_root}"
    )
sys.path.insert(0, str(v2_root))

from data import MiniShiftAD as minishift_data  # noqa: E402
from data.MiniShiftAD import MiniShiftADTest, MiniShiftADTrain  # noqa: E402
import patchcore_runner as patchcore_module  # noqa: E402
from competition_features import CompetitionFPFHFeatures  # noqa: E402
from category_optimizer import CATEGORY_PROFILES, validate_category_profiles  # noqa: E402

# PatchCore resolves FPFHFeatures from its module globals when each category is
# prepared. Replacing that factory keeps the V2 runner/evaluator unchanged.
patchcore_module.FPFHFeatures = CompetitionFPFHFeatures
PatchCore = patchcore_module.PatchCore

# V2 ships with /mnt/g/MiniShiftAD hard-coded. Override it with this machine's
# real dataset location without changing the V2 source package.
minishift_data.DATASETS_PATH = DATA_ROOT


VALID_CATEGORIES = {
    "capsule", "cube", "spring_pad", "screw", "screen", "piggy", "nut",
    "flat_pad", "plastic_cylinder", "button_cell", "toothbrush", "light",
}
VALID_LEVELS = {"ALL", "easy", "medium", "hard"}
validate_category_profiles(VALID_CATEGORIES)


def validate_config():
    if CATEGORY not in VALID_CATEGORIES:
        choices = ", ".join(sorted(VALID_CATEGORIES))
        raise ValueError(f"CATEGORY={CATEGORY!r} is invalid; choose one of: {choices}")
    if LEVEL not in VALID_LEVELS:
        raise ValueError(f"LEVEL={LEVEL!r} is invalid; choose ALL/easy/medium/hard")
    category_dir = Path(DATA_ROOT) / CATEGORY
    if not category_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {category_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; start this script in the configured WSL environment")
    runtime_file = Path(sys.modules[PatchCore.__module__].__file__).resolve()
    if v2_root not in runtime_file.parents:
        raise RuntimeError(f"V2 PatchCore was not loaded; active module: {runtime_file}")


def make_args():
    return SimpleNamespace(
        expname=f"MiniShift_{RUN_TAG}_{CATEGORY}_{LEVEL}",
        device=DEVICE,
        dataset="minishift",
        max_nn=MAX_NN,
        normal_max_nn=NORMAL_MAX_NN,
        num_group=NUM_GROUP,
        group_size=GROUP_SIZE,
        use_MSND=USE_MSND,
        use_LFSA=USE_LFSA,
        vis_save=SAVE_VISUALIZATION,
        num_MSND=NUM_MSND,
        feature=FEATURE,
        level=LEVEL,
        scale_fusion=SCALE_FUSION,
        legacy_duplicate_scale=False,
        post_smooth_mode="none",
        post_smooth_k=12,
        post_smooth_centers=1024,
        interp_chunk_size=INTERP_CHUNK_SIZE,
        p_map_mode=P_MAP_MODE,
        smooth_k=SMOOTH_K,
        p_raw_weight=P_RAW_WEIGHT,
        p_smooth_weight=P_SMOOTH_WEIGHT,
        object_top_ratio=OBJECT_TOP_RATIO,
        object_score_min_topk=80,
        object_score_max_topk=2048,
        use_robust_object_score=True,
        prototype_density_norm=False,
        prototype_density_norm_for_p=False,
        prototype_density_norm_for_o=False,
        prototype_density_k=5,
        prototype_density_clip_low=0.01,
        prototype_density_clip_high=0.99,
        use_geom4d=False,
        use_geom4d_for_p=False,
        geom_weight=0.05,
        query_chunk=QUERY_CHUNK,
        bank_chunk=BANK_CHUNK,
        use_category_profiles=True,
        category_profiles=CATEGORY_PROFILES,
        geometry_profiles={},
        category_to_geometry={},
        object_score_mode=OBJECT_SCORE_MODE,
        o_coherence_k=12,
        normal_calibration_folds=NORMAL_CALIBRATION_FOLDS,
        normal_calibration_samples=NORMAL_CALIBRATION_SAMPLES,
        cache_features=True,
        cache_test_features=CACHE_TEST_FEATURES,
        feature_cache_dir="./feature-cache",
        diagnostic_dir="./logs",
        coreset_ratio=CORESET_RATIO,
        coreset_preselect_factor=CORESET_PRESELECT_FACTOR,
        use_regional_coreset=USE_REGIONAL_CORESET,
        anomaly_num_nn=ANOMALY_NUM_NN,
        fpfh_normalization=FPFH_NORMALIZATION,
        use_position_features=USE_POSITION_FEATURES,
        position_weight=POSITION_WEIGHT,
        use_regional_memory=USE_REGIONAL_MEMORY,
        region_radial_bins=REGION_RADIAL_BINS,
        region_height_bins=REGION_HEIGHT_BINS,
        region_min_bank=REGION_MIN_BANK,
        use_scale_rank_fusion=USE_SCALE_RANK_FUSION,
        decision_scale_weights=DECISION_SCALE_WEIGHTS,
        scale_calibration_samples=SCALE_CALIBRATION_SAMPLES,
        rank_tail_weight=RANK_TAIL_WEIGHT,
    )


def load_previous_baseline():
    """Load the preserved 4096-center result for an honest before/after report."""
    baseline_path = Path("logs") / f"MiniShift_{CATEGORY}_{LEVEL}_result.json"
    if not baseline_path.is_file():
        return None
    try:
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    metrics = baseline.get("metrics")
    if not isinstance(metrics, dict):
        return None
    return {"path": str(baseline_path), "metrics": metrics}


def main():
    validate_config()
    train_count = len(MiniShiftADTrain(CATEGORY))
    test_count = len(MiniShiftADTest(CATEGORY, level=LEVEL)) if RUN_TEST_AFTER_TRAIN else 0
    args = make_args()
    effective_profile = CATEGORY_PROFILES[CATEGORY]
    baseline = load_previous_baseline()

    print("=" * 72)
    print("MiniShiftAD V2 single-category training")
    print(f"Category       : {CATEGORY}")
    print(f"Dataset        : {DATA_ROOT}")
    print(f"V2 runtime     : {Path(sys.modules[PatchCore.__module__].__file__).resolve()}")
    print(f"GPU            : {torch.cuda.get_device_name(0)}")
    print(f"Train samples  : {train_count}")
    print(f"Test samples   : {test_count}")
    print(f"Test level     : {LEVEL}")
    print(f"Feature config : groups={NUM_GROUP}, group_size={GROUP_SIZE}, max_nn={MAX_NN}")
    print(
        f"Competition opt: coreset={effective_profile['coreset_ratio']:.1%}, "
        f"nn={ANOMALY_NUM_NN}, fpfh_norm={FPFH_NORMALIZATION}, "
        f"position={USE_POSITION_FEATURES}"
    )
    print(
        f"Category route : {effective_profile['strategy_name']} "
        f"({effective_profile['pipeline_route']})"
    )
    print(f"Runtime target : about {TARGET_RUNTIME_MINUTES} minutes (no forced stop)")
    if baseline:
        print(f"Previous result: {baseline['metrics']}")
    print("Progress bars below show processed samples / total samples.")
    print("=" * 72, flush=True)

    started = time.time()
    model = PatchCore(args=args)

    print("\n[1/2] Training: extracting features and building the coreset", flush=True)
    train_started = time.time()
    model.fit(CATEGORY)
    train_elapsed = time.time() - train_started

    metrics = None
    if RUN_TEST_AFTER_TRAIN:
        print("\n[2/2] Testing: scoring normal and anomalous samples", flush=True)
        test_started = time.time()
        image_auc, pixel_auc, au_pro = model.evaluate(CATEGORY)
        test_elapsed = time.time() - test_started
        metrics = {
            "image_auc": image_auc["Simple3D"],
            "pixel_auc": pixel_auc["Simple3D"],
            "au_pro": au_pro["Simple3D"],
        }
    else:
        test_elapsed = 0.0
        print("\n[2/2] Testing skipped by RUN_TEST_AFTER_TRAIN=False", flush=True)

    elapsed = time.time() - started
    comparison = None
    if baseline is not None and metrics is not None:
        old_metrics = baseline["metrics"]
        old_image = float(old_metrics["image_auc"])
        old_pixel = float(old_metrics["pixel_auc"])
        new_image = float(metrics["image_auc"])
        new_pixel = float(metrics["pixel_auc"])
        comparison = {
            "baseline_path": baseline["path"],
            "image_auc_delta": round(new_image - old_image, 4),
            "pixel_auc_delta": round(new_pixel - old_pixel, 4),
            "combined_mean_before": round((old_image + old_pixel) / 2, 4),
            "combined_mean_after": round((new_image + new_pixel) / 2, 4),
            "combined_mean_delta": round((new_image + new_pixel - old_image - old_pixel) / 2, 4),
        }
    result = {
        "category": CATEGORY,
        "level": LEVEL,
        "train_samples": train_count,
        "test_samples": test_count,
        "elapsed_seconds": round(elapsed, 2),
        "phase_seconds": {
            "training_total": round(train_elapsed, 2),
            "testing_total": round(test_elapsed, 2),
            **getattr(model.method, "optimization_timing", {}),
        },
        "target_runtime_minutes": TARGET_RUNTIME_MINUTES,
        "forced_time_limit": False,
        "parameters": vars(model.args),
        "metrics": metrics,
        "comparison_to_4096_baseline": comparison,
    }
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    result_path = log_dir / f"{args.expname}_result.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 72)
    print(f"Finished {CATEGORY} in {elapsed / 60:.1f} minutes")
    if metrics is not None:
        print(f"Metrics: {metrics}")
    if comparison is not None:
        print(f"Before/after comparison: {comparison}")
    print(f"Result saved to: {result_path.resolve()}")
    print("=" * 72, flush=True)
    return result


if __name__ == "__main__":
    failures = []
    completed = []
    skipped = []
    total_categories = len(CATEGORIES)
    batch_started = time.time()
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    batch_summary_path = log_dir / f"MiniShift_{RUN_TAG}_batch_summary.json"
    print(f"V10 final batch queue: {', '.join(CATEGORIES)}", flush=True)

    for category_index, category_name in enumerate(CATEGORIES, start=1):
        CATEGORY = category_name
        print("\n" + "#" * 72)
        print(f"V10 final batch job {category_index}/{total_categories}: {CATEGORY}")
        print("#" * 72, flush=True)
        category_result_path = log_dir / f"MiniShift_{RUN_TAG}_{CATEGORY}_{LEVEL}_result.json"
        if SKIP_COMPLETED_RESULTS and category_result_path.is_file():
            try:
                previous_result = json.loads(category_result_path.read_text(encoding="utf-8"))
                if previous_result.get("metrics") is not None:
                    skipped.append(CATEGORY)
                    completed.append(previous_result)
                    print(f"Already completed; skipping: {category_result_path.resolve()}", flush=True)
                    continue
            except (OSError, json.JSONDecodeError):
                pass
        try:
            completed.append(main())
        except Exception as exc:
            failures.append((CATEGORY, str(exc)))
            print(f"\nTraining failed for {CATEGORY}: {exc}", file=sys.stderr, flush=True)
            traceback.print_exc()
            print("Continuing with the next configured category.", flush=True)
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    batch_result = {
        "run_tag": RUN_TAG,
        "categories": list(CATEGORIES),
        "completed_categories": [item["category"] for item in completed],
        "skipped_existing_categories": skipped,
        "failures": [
            {"category": failed_category, "error": error_message}
            for failed_category, error_message in failures
        ],
        "elapsed_seconds": round(time.time() - batch_started, 2),
        "results": [
            {
                "category": item["category"],
                "metrics": item.get("metrics"),
                "elapsed_seconds": item.get("elapsed_seconds"),
                "strategy_name": item.get("parameters", {}).get("strategy_name"),
                "pipeline_route": item.get("parameters", {}).get("pipeline_route"),
            }
            for item in completed
        ],
    }
    batch_summary_path.write_text(
        json.dumps(batch_result, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("\n" + "#" * 72)
    print(f"V10 final batch finished: {len(completed)}/{total_categories} completed")
    if skipped:
        print(f"Resumed from existing results: {', '.join(skipped)}")
    if failures:
        for failed_category, error_message in failures:
            print(f"FAILED {failed_category}: {error_message}")
        print(f"Batch summary saved to: {batch_summary_path.resolve()}")
        raise RuntimeError(f"V10 batch completed with {len(failures)} failed category/categories")
    print("All configured categories completed successfully.")
    print(f"Batch summary saved to: {batch_summary_path.resolve()}")
    print("#" * 72, flush=True)
