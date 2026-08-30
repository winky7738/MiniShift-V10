"""Fast, fixed-subset A/B check for competition feature changes.

This is an internal validation helper.  The official launch remains
``python train_single_category.py``.
"""

import argparse
from torch.utils.data import DataLoader, Subset

import train_single_category as train


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--norm", choices=("none", "hellinger_per_scale"), required=True)
    parser.add_argument("--num-nn", type=int, default=1)
    parser.add_argument("--num-group", type=int, default=256)
    parser.add_argument("--group-size", type=int, default=64)
    parser.add_argument("--train-limit", type=int, default=12)
    parser.add_argument("--test-limit", type=int, default=30)
    parser.add_argument("--coreset-ratio", type=float, default=0.05)
    parser.add_argument("--preselect-factor", type=int, default=0)
    parser.add_argument("--use-cache", action="store_true")
    args_cli = parser.parse_args()

    args = train.make_args()
    args.expname = f"AB_V4_{args_cli.norm}_{args_cli.num_nn}nn"
    args.num_group = args_cli.num_group
    args.group_size = args_cli.group_size
    args.coreset_ratio = args_cli.coreset_ratio
    args.fpfh_normalization = args_cli.norm
    args.anomaly_num_nn = args_cli.num_nn
    args.coreset_preselect_factor = args_cli.preselect_factor
    args.cache_features = args_cli.use_cache
    args.use_position_features = False
    args.use_category_profiles = False
    args.query_chunk = 128
    args.bank_chunk = 1024

    model = train.PatchCore(args=args)
    original_get_dataloader = model.get_dataloader

    def subset_loader(dataset_name, split, class_name, level="ALL"):
        loader = original_get_dataloader(dataset_name, split, class_name, level)
        limit = args_cli.train_limit if split == "train" else args_cli.test_limit
        return DataLoader(
            Subset(loader.dataset, range(min(limit, len(loader.dataset)))),
            batch_size=1,
            shuffle=False,
            num_workers=0,
        )

    model.get_dataloader = subset_loader
    model.fit("nut")
    if args_cli.test_limit <= 0:
        print(
            f"AB_RESULT train_only num_group={args_cli.num_group} "
            f"group_size={args_cli.group_size}"
        )
        return
    image_auc, pixel_auc, _ = model.evaluate("nut")
    print(
        f"AB_RESULT norm={args_cli.norm} num_nn={args_cli.num_nn} "
        f"image_auc={image_auc['Simple3D']:.3f} "
        f"pixel_auc={pixel_auc['Simple3D']:.3f}"
    )


if __name__ == "__main__":
    main()
