from data.mvtec3d import get_mvtec_loader,mvtec3d_classes
from data.real3d import get_real_loader,real3d_classes
from data.anomalyshape import get_shapenet_loader,shapenet3d_classes
from data.MulSen import get_mulsen_loader,mulsen_classes
from data.MiniShiftAD import get_minishift_loader,minishiftAD_classes
import torch
from tqdm import tqdm
from feature_extractors.FPFH import FPFHFeatures
import numpy as np
import os
from feature_extractors.pointnet2_utils import *
from copy import deepcopy

try:
    from data.quan import get_quan_loader, quan_classes
except ImportError:
    get_quan_loader = None
    quan_classes = None


def append_experiment_log(expname, text):
    if not (os.path.isdir("./logs") or os.path.islink("./logs")):
        os.makedirs("./logs", exist_ok=True)
    with open(f"./logs/{expname}.txt", "a", encoding="utf-8") as f:
        f.write(text)


class PatchCore():
    def __init__(self, ckp = '', image_size=224, args=None):
        self.base_args = deepcopy(args)
        self.args = deepcopy(args)
        self.method  = FPFHFeatures(args=self.args)
        self.dataset_name = self.args.dataset
        self.level = self.args.level

    def _resolve_args_for_class(self, class_name):
        resolved_args = deepcopy(self.base_args)
        if getattr(resolved_args, "use_category_profiles", False):
            for key, value in getattr(resolved_args, "category_profiles", {}).get(class_name, {}).items():
                setattr(resolved_args, key, value)
            geometry_name = getattr(resolved_args, "category_to_geometry", {}).get(class_name)
            if geometry_name:
                for key, value in getattr(resolved_args, "geometry_profiles", {}).get(geometry_name, {}).items():
                    setattr(resolved_args, key, value)
        return resolved_args

    def _prepare_for_class(self, class_name):
        self.args = self._resolve_args_for_class(class_name)
        self.dataset_name = self.args.dataset
        self.level = self.args.level
        self.method = FPFHFeatures(args=self.args)
        self.method.current_class_name = class_name
        log_lines = [f"[EffectiveConfig][{class_name}]\n"]
        for key, value in sorted(vars(self.args).items()):
            log_lines.append(f"{key}: {value}\n")
        log_lines.append("\n")
        print("".join(log_lines).rstrip())
        append_experiment_log(self.args.expname, "".join(log_lines))

    def get_dataloader(self,dataset_name,split,class_name,level='ALL'):
        if dataset_name == 'mvtec':
            return get_mvtec_loader(split, class_name=class_name)
        if dataset_name == 'real':
            return get_real_loader(split, class_name=class_name)
        if dataset_name == 'shapenet':
            return get_shapenet_loader(split, class_name=class_name)
        if dataset_name == 'mulsen':
            return get_mulsen_loader(split, class_name=class_name)
        if dataset_name == 'minishift':
            return get_minishift_loader(split, class_name=class_name,level=level)  
        if dataset_name == 'quan':
            if get_quan_loader is None:
                raise ImportError("dataset 'quan' is referenced, but data/quan.py does not exist in this repository.")
            return get_quan_loader(split, class_name=class_name)
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    def fit(self, class_name):
        self._prepare_for_class(class_name)
        train_loader = self.get_dataloader(self.dataset_name,'train',class_name,level=self.level)
        for pc, _, _, path in tqdm(train_loader, desc=f'Extracting train features for class {class_name}'):
            self.method.collect_features(pc, path=path)
            self.method.name_list.append(path)
        print(f'\n\nRunning coreset on class {class_name}...')
        self.method.run_coreset()


    def evaluate(self, class_name):
        image_rocaucs = dict()
        pixel_rocaucs = dict()
        au_pros = dict()
        test_loader = self.get_dataloader(self.dataset_name,'test',class_name,level=self.level)
        with torch.no_grad():
            self.method.init_para()
            self.method.name_list = []
            self.method.test_patch_lib = []
        
            for pc, mask, label, path in tqdm(test_loader, desc=f'Extracting test features for class {class_name}'):
                self.method.predict(pc, mask, label,path)
        method_name = "Simple3D"
        self.method.calculate_metrics()
        image_rocaucs[method_name] = round(self.method.image_rocauc, 3)
        pixel_rocaucs[method_name] = round(self.method.pixel_rocauc, 3)
        au_pros[method_name] = round(self.method.au_pro, 3)
        print(
            f'Class: {class_name}, {method_name} Image ROCAUC: {self.method.image_rocauc:.3f}, {method_name} Pixel ROCAUC: {self.method.pixel_rocauc:.3f}, {method_name} AU-PRO: {self.method.au_pro:.3f}')
        return image_rocaucs, pixel_rocaucs, au_pros
