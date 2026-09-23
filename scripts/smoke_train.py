#!/usr/bin/env python3
"""Bounded, real-data CUDA training check; not a model-quality benchmark."""
import argparse
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import traceback


ROOT = Path(__file__).resolve().parents[1]


def environment():
    report = {"python": sys.version, "platform": platform.platform()}
    for package in ("torch", "numpy", "tensorflow", "numba", "easydict",
                    "spconv-cu111", "spconv-cu117", "spconv-cu120"):
        try:
            report[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report[package] = None
    for command in (["nvidia-smi"], ["nvcc", "--version"]):
        try:
            result = subprocess.run(command, capture_output=True, text=True, timeout=20)
            report[command[0]] = result.stdout + result.stderr
        except (OSError, subprocess.TimeoutExpired) as exc:
            report[command[0]] = str(exc)
    try:
        import torch
        report["cuda_available"] = torch.cuda.is_available()
        report["torch_cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            report["gpu"] = torch.cuda.get_device_name(0)
    except ImportError as exc:
        report["torch_import_error"] = str(exc)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--module", choices=("detection", "refining"))
    parser.add_argument("--cfg_file", help="Relative to the selected module's tools directory")
    parser.add_argument("--data-root", help="Prepared Waymo subset, including ImageSets")
    parser.add_argument("--output", help="New output directory; existing paths are rejected")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--pretrained-model", help="Optional compatible DetZero checkpoint")
    args = parser.parse_args()
    info = environment()
    print(json.dumps(info, indent=2), flush=True)
    if args.preflight:
        return
    if not all((args.module, args.cfg_file, args.data_root, args.output)):
        parser.error("Training requires --module, --cfg_file, --data-root and --output")
    if args.steps < 1 or args.batch_size < 2 or args.samples < args.batch_size or args.workers < 0:
        parser.error("Use steps >= 1, batch-size >= 2, samples >= batch-size, workers >= 0")
    if not info.get("cuda_available"):
        raise RuntimeError("A working CUDA PyTorch environment is required; no CPU fallback")

    import torch
    data_root = Path(args.data_root).resolve()
    output = Path(args.output).resolve()
    pretrained = str(Path(args.pretrained_model).resolve()) if args.pretrained_model else None
    output.mkdir(parents=True, exist_ok=False)
    report = {"status": "running", "environment": info, "arguments": vars(args)}
    (output / "environment.json").write_text(json.dumps(info, indent=2))
    freeze = subprocess.run([sys.executable, "-m", "pip", "freeze"],
                            capture_output=True, text=True)
    (output / "pip-freeze.txt").write_text(freeze.stdout)
    try:
        for package in ("utils", "detection", "tracking", "refining"):
            sys.path.insert(0, str(ROOT / package))
        tools_dir = ROOT / args.module / "tools"
        sys.path.insert(0, str(tools_dir))
        os.chdir(tools_dir)  # Existing YAML base paths are relative to tools/.
        from detzero_utils.config_utils import cfg, cfg_from_yaml_file
        from detzero_utils.common_utils import create_logger, set_random_seed
        from detzero_utils.model_utils import load_params_from_file, load_params_with_optimizer
        from detzero_utils.optimize_utils import build_optimizer, build_scheduler
        from train_utils import checkpoint_state, save_checkpoint

        set_random_seed(666)
        cfg_from_yaml_file(args.cfg_file, cfg)
        cfg.DATA_CONFIG.DATA_PATH = str(data_root)
        cfg.DATA_CONFIG.save_to_file = True
        cfg.MODEL.POST_PROCESSING.GENERATE_RECALL = False
        if cfg.MODEL.NAME == "CenterPoint" and cfg.MODEL.SECOND_STAGE and not pretrained:
            raise ValueError("For the PDV check supply a compatible first-stage DetZero checkpoint")
        logger = create_logger(str(output / "smoke.log"))
        (output / "resolved-config.json").write_text(json.dumps(cfg, indent=2, default=str))
        prefix = "detzero_det" if args.module == "detection" else "detzero_refine"
        datasets = importlib.import_module(prefix + ".datasets")
        models = importlib.import_module(prefix + ".models")
        dataset, loader, _ = datasets.build_dataloader(
            cfg.DATA_CONFIG, cfg.CLASS_NAMES, args.batch_size, dist=False,
            workers=args.workers, logger=logger, training=True, length=args.samples)
        if len(dataset) < args.batch_size:
            raise ValueError("Subset too small for a full batch; add labeled sequences/tracks")
        if cfg.MODEL.NAME == "ConfidenceRefineModel":
            if not dataset.pos_tk_infos or not dataset.neg_tk_infos:
                raise ValueError("CRM requires both matched and unmatched tracks; enlarge the subset")
        kwargs = {"num_class": len(cfg.CLASS_NAMES)} if args.module == "detection" else {}
        model = models.build_network(cfg.MODEL, dataset=dataset, **kwargs).cuda()
        if pretrained:
            load_params_from_file(model, pretrained, logger=logger)
        optimizer = build_optimizer(model, cfg.OPTIMIZATION)
        # Keep original epoch schedule scale; only bound how many updates we execute.
        scheduler, _ = build_scheduler(optimizer, max(len(loader), 2),
                                       cfg.OPTIMIZATION.NUM_EPOCHS, -1, cfg.OPTIMIZATION)
        model_func = models.model_fn_decorator()
        iterator = iter(loader)
        losses = []
        torch.cuda.reset_peak_memory_stats()

        def next_batch():
            nonlocal iterator
            try:
                return next(iterator)
            except StopIteration:
                iterator = iter(loader)
                return next(iterator)

        def update():
            model.train()
            optimizer.zero_grad()
            batch = next_batch()
            batch.update(cur_epoch=0, accumulated_iter=len(losses) + 1)
            loss, _, _ = model_func(model, batch)
            if not torch.isfinite(loss).all():
                raise RuntimeError("Non-finite training loss")
            loss.backward()
            active = None
            for param in model.parameters():
                if param.grad is not None:
                    if not torch.isfinite(param.grad).all():
                        raise RuntimeError("Non-finite gradient")
                    if active is None and torch.count_nonzero(param.grad).item():
                        active = param
            if active is None:
                raise RuntimeError("No nonzero gradient; cannot confirm learning")
            before = active.detach().clone()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.OPTIMIZATION.GRAD_NORM_CLIP)
            optimizer.step()
            scheduler.step()
            if torch.equal(before, active.detach()):
                raise RuntimeError("Observed trainable parameter did not change")
            value = loss.item()
            losses.append(value)
            logger.info("Step %d: loss=%.6f", len(losses), value)

        for _ in range(args.steps):
            update()
        checkpoint = output / "smoke_checkpoint"
        save_checkpoint(checkpoint_state(model, optimizer, epoch=0, it=args.steps), str(checkpoint))
        expected = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        # Reload into the same architecture after clearing weights, using the real loader.
        with torch.no_grad():
            for param in model.parameters():
                param.zero_()
        optimizer = build_optimizer(model, cfg.OPTIMIZATION)
        restored_it, _ = load_params_with_optimizer(
            model, str(checkpoint) + ".pth", optimizer=optimizer, logger=logger)
        if restored_it != args.steps:
            raise RuntimeError("Checkpoint iteration did not round-trip")
        for name, value in model.state_dict().items():
            if not torch.equal(value.detach().cpu(), expected[name]):
                raise RuntimeError("Checkpoint weight mismatch: " + name)
        del expected
        scheduler, _ = build_scheduler(optimizer, max(len(loader), 2),
                                       cfg.OPTIMIZATION.NUM_EPOCHS, -1, cfg.OPTIMIZATION)
        update()  # Checks another backward/update with the restored optimizer.
        report.update(status="passed", losses=losses, checkpoint=str(checkpoint) + ".pth",
                      peak_allocated_gib=torch.cuda.max_memory_allocated() / 1024 ** 3,
                      scope="Training updates and checkpoint round-trip only; validation/inference not tested")
    except Exception:
        report.update(status="failed", traceback=traceback.format_exc())
        raise
    finally:
        (output / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
