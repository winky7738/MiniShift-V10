import argparse
import os

import pandas as pd
import torchvision

from patchcore_runner import PatchCore
from data.mvtec3d import mvtec3d_classes
from data.real3d import real3d_classes
from data.anomalyshape import shapenet3d_classes
from data.MulSen import mulsen_classes
from data.MiniShiftAD import minishiftAD_classes

try:
    from data.quan import quan_classes
except ImportError:
    quan_classes = None


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def add_bool_arg(parser, name, default=False, help_text=None):
    parser.add_argument(
        name,
        nargs="?",
        const=True,
        default=default,
        type=str2bool,
        help=help_text,
    )


def load_yaml_config(config_path):
    if config_path is None:
        return {}

    try:
        import yaml
    except ImportError as exc:
        raise ImportError("PyYAML is required when using --config.") from exc

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError("Config file must define a YAML mapping at the top level.")

    return data


def apply_legacy_aliases(args):
    if getattr(args, "legacy_duplicate_scale", False) and args.scale_fusion == "legacy_equivalent":
        args.scale_fusion = "legacy_duplicate"

    if getattr(args, "use_geom4d", False) and not getattr(args, "use_geom4d_for_p", False):
        args.use_geom4d_for_p = True

    if getattr(args, "prototype_density_norm", False):
        if not getattr(args, "prototype_density_norm_for_p", False) and not getattr(args, "prototype_density_norm_for_o", False):
            args.prototype_density_norm_for_o = True

    return args


def write_experiment_log(expname, strs):
    if not (os.path.isdir("./logs") or os.path.islink("./logs")):
        os.makedirs("./logs", exist_ok=True)
    with open(f"./logs/{expname}.txt", 'a', encoding='utf-8') as f:
        f.write(strs)


def init_experiment_log(args):
    if not (os.path.isdir("./logs") or os.path.islink("./logs")):
        os.makedirs("./logs", exist_ok=True)
    log_path = f"./logs/{args.expname}.txt"
    with open(log_path, 'w', encoding='utf-8') as f:
        f.write("Experiment Parameters\n")
        f.write("=====================\n")
        for key, value in sorted(vars(args).items()):
            f.write(f"{key}: {value}\n")
        f.write("\n")



def run_3d_ads(args):
    init_experiment_log(args)
    classes = None
    if args.dataset == 'mvtec':
        classes = mvtec3d_classes()
    elif args.dataset == 'real':
        classes = real3d_classes()
    elif args.dataset == 'shapenet':
        classes = shapenet3d_classes()
    elif args.dataset == 'mulsen':
        classes = mulsen_classes() 
    elif args.dataset == 'minishift':
        classes = minishiftAD_classes() 
    elif args.dataset == 'quan':
        if quan_classes is None:
            raise ImportError("dataset 'quan' is referenced, but data/quan.py does not exist in this repository.")
        classes = quan_classes()
    elif args.dataset == 'eyecandies':
        raise NotImplementedError("dataset 'eyecandies' is referenced, but its loader is not included in this repository.")
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    METHOD_NAMES = [
        "Simple3D",
        ]

    image_rocaucs_df = pd.DataFrame(METHOD_NAMES, columns=['Method'])
    pixel_rocaucs_df = pd.DataFrame(METHOD_NAMES, columns=['Method'])
    au_pros_df = pd.DataFrame(METHOD_NAMES, columns=['Method'])
    for category in classes:
        patchcore = PatchCore(args=args)
        patchcore.fit(category)
        cls = category
        print(f"\nRunning on class {cls}\n")
        write_experiment_log(args.expname,f"\nRunning on class {cls}\n")
        image_rocaucs, pixel_rocaucs, au_pros = patchcore.evaluate(cls)
        image_rocaucs_df[cls.title()] = image_rocaucs_df['Method'].map(image_rocaucs)
        pixel_rocaucs_df[cls.title()] = pixel_rocaucs_df['Method'].map(pixel_rocaucs)
        au_pros_df[cls.title()] = au_pros_df['Method'].map(au_pros)

        print(f"\nFinished running on class {cls}\n")
        write_experiment_log(args.expname,f"\nFinished running on class {cls}\n")
        print("################################################################################\n\n")

    image_rocaucs_df['Mean'] = round(image_rocaucs_df.iloc[:, 1:].mean(axis=1),3)
    pixel_rocaucs_df['Mean'] = round(pixel_rocaucs_df.iloc[:, 1:].mean(axis=1),3)
    au_pros_df['Mean'] = round(au_pros_df.iloc[:, 1:].mean(axis=1),3)

    print("\n\n################################################################################")
    print("############################# Image ROCAUC Results #############################")
    print("################################################################################\n")
    print(image_rocaucs_df.to_markdown(index=False))
    write_experiment_log(args.expname,image_rocaucs_df.to_markdown(index=False))
    write_experiment_log(args.expname,f'\n')

    print("\n\n################################################################################")
    print("############################# Pixel ROCAUC Results #############################")
    print("################################################################################\n")
    print(pixel_rocaucs_df.to_markdown(index=False))
    write_experiment_log(args.expname,pixel_rocaucs_df.to_markdown(index=False))
    write_experiment_log(args.expname,f'\n')

    print("\n\n##########################################################################")
    print("############################# AU PRO Results #############################")
    print("##########################################################################\n")
    print(au_pros_df.to_markdown(index=False))
    write_experiment_log(args.expname,au_pros_df.to_markdown(index=False))
    write_experiment_log(args.expname,f'\n')


if __name__ == '__main__':
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument('--config', type=str, default=None, help='path to YAML config')
    pre_args, _ = pre_parser.parse_known_args()
    config_data = load_yaml_config(pre_args.config)

    parser = argparse.ArgumentParser(description='Process some integers.', parents=[pre_parser])
    parser.add_argument('--expname', type=str, default='None', help='expname')
    parser.add_argument('--device', type=str, default='cuda:0', help='expname')
    parser.add_argument('--dataset', type=str, default='shapenet', help='dataset name')
    parser.add_argument('--max_nn', type=int, default=100, help='max_nn')
    parser.add_argument('--normal_max_nn', type=int, default=10, help='normal estimation max_nn')
    parser.add_argument('--num_group', type=int, default=2048, help='num_group')
    parser.add_argument('--group_size', type=int, default=128, help='group_size')
    add_bool_arg(parser, '--use_MSND', default=False, help_text='enable multi-scale FPFH')
    add_bool_arg(parser, '--use_LFSA', default=False, help_text='enable LFSA aggregation')
    add_bool_arg(parser, '--vis_save', default=False, help_text='save visualization txt files')
    parser.add_argument('--num_MSND', type=int, default=2)
    parser.add_argument('--feature', type=str, default='FPFH')
    parser.add_argument('--level', type=str, default='ALL')
    add_bool_arg(parser, '--legacy_duplicate_scale', default=False, help_text='keep old duplicated middle-scale fusion')
    parser.add_argument('--scale_fusion', type=str, default='legacy_equivalent',
                        choices=['legacy_duplicate', 'legacy_equivalent', 'equal_3scale'])
    parser.add_argument('--post_smooth_mode', type=str, default='none', choices=['none', 'legacy', 'knn_mean'])
    parser.add_argument('--post_smooth_k', type=int, default=12)
    parser.add_argument('--post_smooth_centers', type=int, default=1024)
    parser.add_argument('--interp_chunk_size', type=int, default=10000)
    parser.add_argument('--p_map_mode', type=str, default='legacy', choices=['legacy', 'dual_center'])
    parser.add_argument('--smooth_k', type=int, default=8)
    parser.add_argument('--p_raw_weight', type=float, default=0.65)
    parser.add_argument('--p_smooth_weight', type=float, default=0.35)
    parser.add_argument('--object_top_ratio', type=float, default=0.001)
    parser.add_argument('--object_score_min_topk', type=int, default=80)
    parser.add_argument('--object_score_max_topk', type=int, default=2048)
    add_bool_arg(parser, '--use_robust_object_score', default=True, help_text='use ratio-based object score')
    add_bool_arg(parser, '--prototype_density_norm', default=False, help_text='legacy alias for O-branch density normalization')
    add_bool_arg(parser, '--prototype_density_norm_for_p', default=False, help_text='enable density normalization for P branch')
    add_bool_arg(parser, '--prototype_density_norm_for_o', default=False, help_text='enable density normalization for O branch')
    parser.add_argument('--prototype_density_k', type=int, default=5)
    parser.add_argument('--prototype_density_clip_low', type=float, default=0.01)
    parser.add_argument('--prototype_density_clip_high', type=float, default=0.99)
    add_bool_arg(parser, '--use_geom4d', default=False, help_text='legacy alias for using geom4d in P branch')
    add_bool_arg(parser, '--use_geom4d_for_p', default=False, help_text='append geom4d features in P branch')
    parser.add_argument('--geom_weight', type=float, default=0.05)
    parser.add_argument('--query_chunk', type=int, default=1024)
    parser.add_argument('--bank_chunk', type=int, default=8192)
    add_bool_arg(parser, '--use_category_profiles', default=True, help_text='enable per-category overrides')
    parser.add_argument('--object_score_mode', type=str, default='legacy', choices=['legacy', 'normal_tail_coherence'])
    parser.add_argument('--o_coherence_k', type=int, default=12)
    parser.add_argument('--normal_calibration_folds', type=int, default=0)
    add_bool_arg(parser, '--cache_features', default=False, help_text='reserved flag for future feature caching')
    parser.add_argument('--diagnostic_dir', type=str, default='./logs', help='directory for diagnostic jsonl files')
    parser.set_defaults(
        category_profiles=config_data.get('categories', {}),
        geometry_profiles=config_data.get('geometry_profiles', {}),
        category_to_geometry=config_data.get('category_to_geometry', {}),
    )
    parser.set_defaults(**{k: v for k, v in config_data.items() if k not in {'categories', 'geometry_profiles', 'category_to_geometry'}})
    args = parser.parse_args()
    args = apply_legacy_aliases(args)
    print(args)

    run_3d_ads(args)
