"""Small end-to-end MiniShiftAD environment smoke test."""

import os
from types import SimpleNamespace

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


DATA_ROOT = os.environ.get("MINISHIFT_DATASET_PATH", "/mnt/d/BaiduNetdiskDownload")
os.environ["MINISHIFT_DATASET_PATH"] = DATA_ROOT

from data.MiniShiftAD import MiniShiftADTest, MiniShiftADTrain  # noqa: E402
from patchcore_runner import PatchCore  # noqa: E402


class SampledDataset(Dataset):
    def __init__(self, source, indices, point_count=8192):
        self.source = source
        self.indices = indices
        self.point_count = point_count

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        pc, target, label, path = self.source[self.indices[item]]
        sample_idx = np.linspace(0, pc.shape[0] - 1, self.point_count, dtype=np.int64)
        pc = pc[sample_idx]
        if torch.is_tensor(target) and target.ndim > 0:
            target = target[:, sample_idx]
        return pc, target, label, path


def loader(dataset):
    return DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)


def main():
    category = os.environ.get("MINISHIFT_SMOKE_CLASS", "flat_pad")
    train_source = MiniShiftADTrain(category)
    test_source = MiniShiftADTest(category, level="easy")
    good_idx = next(i for i, label in enumerate(test_source.labels) if label == 0)
    anomaly_idx = next(i for i, label in enumerate(test_source.labels) if label == 1)

    train_loader = loader(SampledDataset(train_source, [0, 1]))
    test_loader = loader(SampledDataset(test_source, [good_idx, anomaly_idx]))

    args = SimpleNamespace(
        expname="smoke_test",
        device="cuda:0",
        dataset="minishift",
        max_nn=20,
        num_group=128,
        group_size=16,
        use_MSND=False,
        use_LFSA=True,
        vis_save=False,
        num_MSND=2,
        feature="FPFH",
        level="easy",
    )

    print(f"DATA_ROOT={DATA_ROOT}")
    print(f"CATEGORY={category} TRAIN_SAMPLES=2 TEST_SAMPLES=2 POINTS_PER_SAMPLE=8192")
    print(f"CUDA={torch.cuda.is_available()} GPU={torch.cuda.get_device_name(0)}")

    model = PatchCore(args=args)
    model.get_dataloader = lambda dataset_name, split, class_name, level="ALL": (
        train_loader if split == "train" else test_loader
    )
    model.fit(category)
    image_auc, pixel_auc, au_pro = model.evaluate(category)
    print(f"SMOKE_RESULT image_auc={image_auc} pixel_auc={pixel_auc} au_pro={au_pro}")
    print("SMOKE_TEST_OK")


if __name__ == "__main__":
    main()
