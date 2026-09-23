"""CPU regressions for the preparation fixes, without importing CUDA packages."""
import ast
import importlib.util
import multiprocessing
import os
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def load_file(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def function_only(relative, name, globals_dict):
    """Run actual function source with explicit boundary doubles for CUDA/Waymo."""
    path = ROOT / relative
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == name)
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), globals_dict)
    return globals_dict[name]


class PreparationTests(unittest.TestCase):
    def test_tfrecord_path_and_cached_rerun(self):
        calls = []

        def reader(sequence_file, **kwargs):
            calls.append(sequence_file)
            return [{"sample_idx": 0}], [np.zeros((2, 6), dtype=np.float32)]

        run = function_only("detection/detzero_det/datasets/waymo/waymo_utils.py",
                            "process_single_sequence_and_save",
                            dict(os=os, Path=Path, multiprocessing=multiprocessing,
                                 pickle=pickle, np=np, process_single_tfrecord_multiprocessing=reader))
        with tempfile.TemporaryDirectory() as tmp:
            for filename in ("segment-demo.tfrecord", "segment-demo_with_camera_labels.tfrecord"):
                raw = Path(tmp) / filename
                raw.touch()
                result = run(str(raw), str(Path(tmp) / "processed"))
                self.assertEqual(calls[-1], str(raw))
                self.assertTrue(Path(result[0]["lidar_path"]).is_file())
                count = len(calls)
                self.assertEqual(run(str(raw), str(Path(tmp) / "processed")), result)
                self.assertEqual(len(calls), count)

    def test_crm_padding_and_ambiguous_iou_are_ignored(self):
        module = load_file("crm_targets", "refining/detzero_refine/models/modules/target_assign.py")
        assigner = module.TargetAssigner(mode="confidence", score_thresh=[0.35, 0.7])
        result = assigner.encode_torch({"iou": torch.tensor([[0.82, 0.2, 0.5, -1., float('nan')]])})
        self.assertEqual(result["mask"].tolist(), [True, True, False, False, False])
        self.assertEqual(result["score_gt"][:2].tolist(), [1., 0.])

    def test_empty_crm_mask_has_finite_differentiable_loss(self):
        # Extract the real class method without importing its CUDA-dependent package.
        path = ROOT / "refining/detzero_refine/models/modules/confidence_pointnet.py"
        cls = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef))
        fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "get_loss")
        namespace = {}
        exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), "exec"), namespace)
        from types import SimpleNamespace
        score = torch.tensor([[[0.4], [0.7]]], requires_grad=True)
        iou = torch.tensor([[[0.3], [0.6]]], requires_grad=True)
        obj = SimpleNamespace(
            preds_dict={"score_reg": score, "iou_reg": iou},
            targets_dict={"mask": torch.zeros(2, dtype=torch.bool),
                          "score_gt": torch.zeros(2), "iou_gt": torch.zeros(2)},
            bce_loss=torch.nn.BCELoss(reduction="none"), loss_weight=[1., 1.])
        loss, _ = namespace["get_loss"](obj)
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(loss.item(), 0.)
        loss.backward()
        self.assertTrue(torch.isfinite(score.grad).all())
        self.assertTrue(torch.isfinite(iou.grad).all())

    def test_database_rerun_preserves_metadata(self):
        from types import SimpleNamespace

        class TensorDouble:
            def __init__(self, value):
                self.value = value
            def unsqueeze(self, **kwargs): return self
            def float(self): return self
            def cuda(self): return self

        ops = SimpleNamespace(points_in_boxes_gpu=lambda points, boxes: torch.tensor([[0, 0]]))
        run = function_only("detection/detzero_det/datasets/waymo/waymo_preprocess.py",
                            "create_groundtruth_database",
                            dict(os=os, pickle=pickle, Path=Path, np=np,
                                 torch=SimpleNamespace(from_numpy=TensorDouble), roiaware_pool3d_utils=ops))
        with tempfile.TemporaryDirectory() as tmp:
            infos = [{"sequence_name": "demo", "sample_idx": 0,
                      "annos": {"name": np.array(["Vehicle"]), "difficulty": np.array([0]),
                                "gt_boxes_lidar": np.array([[0., 0., 0., 4., 2., 1., 0.]])}}]
            info_path = Path(tmp) / "infos.pkl"
            info_path.write_bytes(pickle.dumps(infos))
            dataset = SimpleNamespace(root_path=tmp,
                get_sweep_idxs=lambda *args: [0], get_infos_and_points=lambda *args: (infos, []),
                merge_sweeps=lambda *args: np.zeros((2, 6), dtype=np.float32))
            for _ in range(2):
                run(dataset, str(info_path), tmp)
                with open(Path(tmp) / "waymo_dbinfos_train_sampled_1_sweep_1.pkl", "rb") as stream:
                    metadata = pickle.load(stream)
                self.assertEqual(len(metadata["Vehicle"]), 1)
                self.assertEqual(metadata["Vehicle"][0]["num_points_in_gt"], 2)


if __name__ == "__main__":
    unittest.main()
