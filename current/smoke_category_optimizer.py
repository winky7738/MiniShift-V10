"""Seconds-only runtime check for the category score adapter."""

import torch

import train_single_category as train


def main():
    model = train.PatchCore(args=train.make_args())
    model._prepare_for_class("nut")
    score_map = torch.linspace(0, 1, 1000)
    score, diagnostic = model.method._compute_object_score(
        score_map,
        score_map,
        score_map,
        torch.zeros(1, 1, 3),
    )
    print("PROFILE", model.args.category_profile_name, model.args.object_tail_weights)
    print("SCORE_OK", float(score), sorted(diagnostic))

    # Exercise the new regional memory and rank-fusion path using small,
    # deterministic synthetic descriptors. This does not touch the dataset.
    method = model.method
    method.args.coreset_ratio = 0.5
    method.args.region_min_bank = 8
    method.args.scale_calibration_samples = 128
    method.args.normal_calibration_samples = 2
    method.args.p_map_mode = "dual_center"
    generator = torch.Generator().manual_seed(7)
    for sample_index in range(3):
        features = torch.rand(64, 99, generator=generator)
        centers = torch.randn(1, 64, 3, generator=generator)
        method.patch_lib.append(features)
        method.train_samples.append(
            {"features": features, "center": centers, "path": f"synthetic-{sample_index}"}
        )
    method.run_coreset()
    assert len(method.train_samples) == 3
    assert torch.isfinite(method.normal_score_stats["raw_q99"])
    query = torch.rand(32, 99, generator=generator).to(method.device)
    query_centers = torch.randn(1, 32, 3, generator=generator).to(method.device)
    method._active_query_regions = method._region_ids(query_centers)
    regional_scores, nearest = method._min_cdist_chunked(query, method.patch_lib)
    method._active_query_regions = None
    assert regional_scores.shape == (32,)
    assert nearest.shape == (32,)
    assert torch.isfinite(regional_scores).all()
    print(
        "REGIONAL_RANK_OK",
        f"regions={torch.unique(method.bank_region_ids).numel()}",
        f"scale_refs={len(method.scale_score_references)}",
        f"score_mean={regional_scores.mean().item():.6f}",
    )


if __name__ == "__main__":
    main()
