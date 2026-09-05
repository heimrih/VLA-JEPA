#!/usr/bin/env python3
"""Open-loop action-chunk evaluation for a VLA-JEPA checkpoint.

The evaluator conditions on the current multi-view image, current state, and
instruction, samples a complete action chunk from the policy, converts both the
prediction and target back to robot units, and compares them there.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from types import MethodType
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

# Prefer the source tree containing this script over a stale editable install.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from starVLA.dataloader.gr00t_lerobot.mixtures import DATASET_NAMED_MIXTURES
from starVLA.dataloader.lerobot_datasets import (
    collate_fn,
    get_vla_dataset,
    make_LeRobotSingleDataset,
)
from starVLA.model.framework import build_framework


DEFAULT_CHECKPOINT = (
    REPO_ROOT / "checkpoints/real_widowxai_ft/final_model/pytorch_model.pt"
)
DEFAULT_TRAIN_DATA_ROOT = (
    REPO_ROOT
    / "checkpoints/vla-jepa-pretrain/vla-jepa/real_widowxai_success_only"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_TRAIN_DATA_ROOT)
    parser.add_argument("--dataset-name", default="real_widowxai_success_only")
    parser.add_argument(
        "--normalization-dataset-root",
        type=Path,
        default=DEFAULT_TRAIN_DATA_ROOT,
        help="Root containing the dataset whose statistics were used for training.",
    )
    parser.add_argument(
        "--normalization-dataset-name",
        default="real_widowxai_success_only",
    )
    parser.add_argument(
        "--qwen-path", type=Path, default=REPO_ROOT / "model/Qwen3-VL-2B-Instruct"
    )
    parser.add_argument(
        "--vjepa-path",
        type=Path,
        default=REPO_ROOT / "model/vjepa2-vitl-fpc64-256",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16"
    )
    parser.add_argument("--attn-implementation", default="sdpa")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--num-examples",
        type=int,
        default=16,
        help="Number of unique valid frames to evaluate; 0 evaluates the full set.",
    )
    parser.add_argument(
        "--samples-per-example",
        type=int,
        default=3,
        help="Independent action samples drawn from different initial noise.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "results/real_widowxai_open_loop.json",
    )
    parser.add_argument(
        "--save-visualizations",
        action="store_true",
        help="Save current camera views, action plots, and prediction arrays.",
    )
    parser.add_argument(
        "--visualization-dir",
        type=Path,
        default=REPO_ROOT / "results/real_widowxai_open_loop_visualizations",
    )
    parser.add_argument(
        "--max-visualizations",
        type=int,
        default=8,
        help="Maximum number of evaluated examples to visualize.",
    )
    parser.add_argument(
        "--data-only",
        action="store_true",
        help="Load one example and validate shapes without loading the model.",
    )
    return parser.parse_args()


def require_path(path: Path, description: str) -> Path:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Missing {description}: {path}")
    return path


def checkpoint_run_dir(checkpoint: Path) -> Path:
    # Supported layouts are <run>/checkpoints/model.pt and
    # <run>/final_model/pytorch_model.pt.
    run_dir = checkpoint.parent.parent
    require_path(run_dir / "config.yaml", "checkpoint config.yaml")
    require_path(run_dir / "dataset_statistics.json", "dataset statistics")
    return run_dir


def build_eval_dataset(args: argparse.Namespace, cfg: Any):
    dataset_root = require_path(args.dataset_root, "evaluation dataset root")
    dataset_dir = require_path(dataset_root / args.dataset_name, "evaluation dataset")
    normalization_root = require_path(
        args.normalization_dataset_root, "normalization dataset root"
    )
    normalization_dir = require_path(
        normalization_root / args.normalization_dataset_name,
        "normalization dataset",
    )

    mixture_name = "__real_widowxai_open_loop_eval__"
    DATASET_NAMED_MIXTURES[mixture_name] = [
        (args.dataset_name, 1.0, "real_widowxai")
    ]
    data_cfg = OmegaConf.create(OmegaConf.to_container(cfg.datasets.vla_data, resolve=True))
    data_cfg.data_root_dir = str(dataset_root)
    data_cfg.data_mix = mixture_name

    # predict_action consumes only the current frame. A video horizon of one
    # avoids decoding future frames needed only by the world-model loss.
    dataset = get_vla_dataset(
        data_cfg=data_cfg,
        mode="eval",
        action_horizon=int(cfg.framework.action_model.action_horizon),
        video_horizon=1,
    )

    if dataset_dir != normalization_dir:
        reference = make_LeRobotSingleDataset(
            data_root_dir=normalization_root,
            data_name=args.normalization_dataset_name,
            robot_type="real_widowxai",
            delete_pause_frame=bool(data_cfg.get("delete_pause_frame", False)),
            action_horizon=int(cfg.framework.action_model.action_horizon),
            video_horizon=1,
            video_backend=data_cfg.get("video_backend", "torchvision_av"),
        )
        for component_dataset in dataset.datasets:
            component_dataset.set_transforms_metadata(reference.metadata)
        print(
            "Using normalization statistics from "
            f"{normalization_dir} for evaluation data {dataset_dir}"
        )

    # The generic mixture wrapper samples steps with replacement, even outside
    # training. Open-loop evaluation instead needs each valid frame at most
    # once. Exclude episode tails that cannot supply a complete action chunk.
    if len(dataset.datasets) != 1:
        raise ValueError("Exhaustive open-loop evaluation requires one dataset")
    component_dataset = dataset.datasets[0]
    action_horizon = int(cfg.framework.action_model.action_horizon)
    trajectory_lengths = dict(
        zip(component_dataset.trajectory_ids, component_dataset.trajectory_lengths)
    )
    valid_steps = [
        (trajectory_id, base_index)
        for trajectory_id, base_index in component_dataset.all_steps
        if base_index + action_horizon <= trajectory_lengths[trajectory_id]
    ]
    excluded_steps = len(component_dataset.all_steps) - len(valid_steps)
    permutation = np.random.default_rng(args.seed).permutation(len(valid_steps))
    evaluation_steps = [valid_steps[index] for index in permutation]

    def sample_unique_valid_step(self, index: int):
        trajectory_id, base_index = evaluation_steps[index]
        return self.datasets[0], trajectory_id, base_index

    dataset.sample_step = MethodType(sample_unique_valid_step, dataset)
    dataset._dataset_lengths = np.asarray([len(evaluation_steps)], dtype=np.int64)
    dataset.open_loop_sampling = {
        "mode": "unique_valid_frames_without_replacement",
        "valid_steps": len(evaluation_steps),
        "excluded_incomplete_tail_steps": excluded_steps,
    }
    print(
        f"Open-loop sampling: {len(evaluation_steps)} unique valid frames; "
        f"excluded {excluded_steps} incomplete episode-tail frames"
    )

    return dataset


def load_model(args: argparse.Namespace, cfg: Any, checkpoint: Path):
    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError(
            "VLA_JEPA.predict_action requires a CUDA device, but CUDA is not available."
        )

    qwen_path = require_path(args.qwen_path, "local Qwen model")
    vjepa_path = require_path(args.vjepa_path, "local V-JEPA model")
    cfg.framework.qwenvl.base_vlm = str(qwen_path)
    cfg.framework.qwenvl.attn_implementation = args.attn_implementation
    cfg.framework.vj2_model.base_encoder = str(vjepa_path)
    cfg.trainer.pretrained_checkpoint = None

    model = build_framework(cfg)
    try:
        state_dict = torch.load(
            checkpoint, map_location="cpu", mmap=True, weights_only=True
        )
    except TypeError:
        state_dict = torch.load(checkpoint, map_location="cpu")
    model.load_state_dict(state_dict, strict=True)
    del state_dict

    with (checkpoint_run_dir(checkpoint) / "dataset_statistics.json").open() as handle:
        model.norm_stats = json.load(handle)

    dtype = getattr(torch, args.dtype)
    return model.to(device=torch.device(args.device), dtype=dtype).eval()


def to_jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def unnormalize_flattened(
    values: np.ndarray,
    component_dataset: Any,
    modality: str,
) -> np.ndarray:
    """Invert dataloader normalization for a concatenated state/action array."""
    keys = component_dataset.modality_keys[modality]
    metadata = getattr(component_dataset.metadata.modalities, modality)
    local_keys = [key.split(".", 1)[1] for key in keys]
    widths = [int(np.prod(metadata[key].shape)) for key in local_keys]
    if sum(widths) != values.shape[-1]:
        raise ValueError(
            f"Cannot split {modality} width {values.shape[-1]} into {widths}"
        )

    boundaries = np.cumsum(widths)[:-1]
    structured = {
        key: torch.from_numpy(part.copy())
        for key, part in zip(keys, np.split(values, boundaries, axis=-1))
    }
    raw = component_dataset.transforms.unapply(structured)
    return np.concatenate([np.asarray(raw[key]) for key in keys], axis=-1).astype(
        np.float32
    )


ACTION_DIMENSION_NAMES = [
    *(f"left_joint_{index}" for index in range(6)),
    "left_gripper",
    *(f"right_joint_{index}" for index in range(6)),
    "right_gripper",
]


def save_prediction_visualization(
    output_root: Path,
    example_index: int,
    example: dict[str, Any],
    target: np.ndarray,
    prediction_samples: np.ndarray,
    component_dataset: Any,
) -> Path:
    """Save current RGB observations and raw-unit action predictions."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    example_dir = output_root / f"example_{example_index:05d}"
    example_dir.mkdir(parents=True, exist_ok=True)

    current_views = []
    for view_index, image in enumerate(example["image"]):
        image = image.convert("RGB")
        image.save(example_dir / f"current_view_{view_index}.png")
        current_views.append(image)

    if current_views:
        width = sum(image.width for image in current_views)
        height = max(image.height for image in current_views)
        montage = Image.new("RGB", (width, height))
        offset = 0
        for image in current_views:
            montage.paste(image, (offset, 0))
            offset += image.width
        montage.save(example_dir / "current_views.png")

    mean_prediction = prediction_samples.mean(axis=0)
    std_prediction = prediction_samples.std(axis=0)
    np.savez_compressed(
        example_dir / "action_prediction.npz",
        prediction_samples=prediction_samples,
        mean_prediction=mean_prediction,
        ground_truth=target,
        raw_state=unnormalize_flattened(
            np.asarray(example["state"], dtype=np.float32),
            component_dataset,
            "state",
        ),
    )
    with (example_dir / "metadata.json").open("w") as handle:
        json.dump(
            {
                "instruction": example["lang"],
                "values_are_normalized": False,
                "value_space": "raw_robot_units",
                "action_dimension_names": ACTION_DIMENSION_NAMES,
            },
            handle,
            indent=2,
        )
        handle.write("\n")

    horizon = np.arange(target.shape[0])
    rows, columns = 4, 4
    figure, axes = plt.subplots(rows, columns, figsize=(16, 12), sharex=True)
    for dimension, axis in enumerate(axes.flat):
        if dimension >= target.shape[1]:
            axis.axis("off")
            continue
        axis.plot(horizon, target[:, dimension], "o-", label="ground truth")
        axis.plot(horizon, mean_prediction[:, dimension], "o-", label="prediction")
        if prediction_samples.shape[0] > 1:
            axis.fill_between(
                horizon,
                mean_prediction[:, dimension] - std_prediction[:, dimension],
                mean_prediction[:, dimension] + std_prediction[:, dimension],
                alpha=0.2,
                label="prediction ±1 std",
            )
        name = (
            ACTION_DIMENSION_NAMES[dimension]
            if dimension < len(ACTION_DIMENSION_NAMES)
            else f"action_{dimension}"
        )
        axis.set_title(name)
        axis.grid(alpha=0.25)
    axes.flat[0].legend(fontsize="small")
    figure.supxlabel("Action-chunk horizon")
    figure.supylabel("Action (raw robot units)")
    figure.suptitle(example["lang"])
    figure.tight_layout()
    figure.savefig(example_dir / "predicted_vs_ground_truth_actions.png", dpi=150)
    plt.close(figure)
    return example_dir


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if args.num_examples < 0 or args.samples_per_example <= 0:
        raise ValueError("num-examples must be non-negative and samples-per-example positive")
    if args.max_visualizations < 0:
        raise ValueError("max-visualizations must be non-negative")

    checkpoint = require_path(args.checkpoint, "checkpoint")
    run_dir = checkpoint_run_dir(checkpoint)
    cfg = OmegaConf.load(run_dir / "config.yaml")
    dataset = build_eval_dataset(args, cfg)
    component_dataset = dataset.datasets[0]
    evaluation_limit = len(dataset) if args.num_examples == 0 else min(
        args.num_examples, len(dataset)
    )

    first = dataset[0]
    print(
        "Data contract: "
        f"views={len(first['image'])}, state={np.asarray(first['state']).shape}, "
        f"action={np.asarray(first['action']).shape}, instruction={first['lang']!r}"
    )
    if args.data_only:
        return {
            "data_only": True,
            "dataset_size": len(dataset),
            "sampling": dataset.open_loop_sampling,
            "image_views": len(first["image"]),
            "state_shape": list(np.asarray(first["state"]).shape),
            "action_shape": list(np.asarray(first["action"]).shape),
            "instruction": first["lang"],
        }

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    model = load_model(args, cfg, checkpoint)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    horizon = int(cfg.framework.action_model.action_horizon)
    action_dim = int(cfg.framework.action_model.action_dim)
    abs_sum = 0.0
    squared_sum = 0.0
    value_count = 0
    mean_abs_sum = 0.0
    mean_squared_sum = 0.0
    mean_value_count = 0
    per_horizon_abs = np.zeros(horizon, dtype=np.float64)
    per_horizon_squared = np.zeros(horizon, dtype=np.float64)
    per_horizon_count = np.zeros(horizon, dtype=np.int64)
    per_dimension_abs = np.zeros(action_dim, dtype=np.float64)
    per_dimension_squared = np.zeros(action_dim, dtype=np.float64)
    per_dimension_count = np.zeros(action_dim, dtype=np.int64)
    diversity_sum = 0.0
    diversity_count = 0
    zero_abs_sum = 0.0
    zero_squared_sum = 0.0
    persistence_abs_sum = 0.0
    persistence_squared_sum = 0.0
    baseline_value_count = 0
    persistence_per_horizon_abs = np.zeros(horizon, dtype=np.float64)
    persistence_per_horizon_count = np.zeros(horizon, dtype=np.int64)
    evaluated_examples = 0
    visualization_paths: list[Path] = []
    visualization_root = args.visualization_dir.expanduser().resolve()

    for examples in loader:
        remaining = evaluation_limit - evaluated_examples
        if remaining <= 0:
            break
        examples = examples[:remaining]
        normalized_targets = np.asarray(
            [example["action"] for example in examples], dtype=np.float32
        )
        targets = unnormalize_flattened(
            normalized_targets, component_dataset, "action"
        )
        images = [example["image"] for example in examples]
        instructions = [example["lang"] for example in examples]
        states = [example["state"] for example in examples]

        zero_error = -targets
        normalized_states = np.asarray(states, dtype=np.float32)
        raw_states = unnormalize_flattened(
            normalized_states, component_dataset, "state"
        )
        persistence_prediction = np.repeat(raw_states, horizon, axis=1)
        if persistence_prediction.shape != targets.shape:
            raise ValueError(
                "State/action shape mismatch prevents the absolute-action persistence "
                f"baseline: {persistence_prediction.shape} vs {targets.shape}"
            )
        persistence_error = persistence_prediction - targets
        zero_abs_sum += np.abs(zero_error).sum()
        zero_squared_sum += np.square(zero_error).sum()
        persistence_abs_sum += np.abs(persistence_error).sum()
        persistence_squared_sum += np.square(persistence_error).sum()
        baseline_value_count += targets.size
        persistence_per_horizon_abs += np.abs(persistence_error).sum(axis=(0, 2))
        persistence_per_horizon_count += targets.shape[0] * targets.shape[2]

        samples = []
        for _ in range(args.samples_per_example):
            normalized_predictions = model.predict_action(
                batch_images=images,
                instructions=instructions,
                state=states,
            )["normalized_actions"].astype(np.float32)
            predictions = unnormalize_flattened(
                normalized_predictions, component_dataset, "action"
            )
            if predictions.shape != targets.shape:
                raise ValueError(
                    f"Prediction/target shape mismatch: {predictions.shape} vs {targets.shape}"
                )
            samples.append(predictions)
            error = predictions - targets
            abs_error = np.abs(error)
            squared_error = np.square(error)
            abs_sum += abs_error.sum()
            squared_sum += squared_error.sum()
            value_count += error.size
            per_horizon_abs += abs_error.sum(axis=(0, 2))
            per_horizon_squared += squared_error.sum(axis=(0, 2))
            per_horizon_count += error.shape[0] * error.shape[2]
            per_dimension_abs += abs_error.sum(axis=(0, 1))
            per_dimension_squared += squared_error.sum(axis=(0, 1))
            per_dimension_count += error.shape[0] * error.shape[1]

        stacked_samples = np.stack(samples, axis=0)
        mean_prediction = stacked_samples.mean(axis=0)
        mean_error = mean_prediction - targets
        mean_abs_sum += np.abs(mean_error).sum()
        mean_squared_sum += np.square(mean_error).sum()
        mean_value_count += mean_error.size
        if args.samples_per_example > 1:
            diversity_sum += stacked_samples.std(axis=0).sum()
            diversity_count += mean_error.size

        if args.save_visualizations:
            remaining_visualizations = args.max_visualizations - len(
                visualization_paths
            )
            for batch_index, example in enumerate(examples[:remaining_visualizations]):
                visualization_paths.append(
                    save_prediction_visualization(
                        visualization_root,
                        evaluated_examples + batch_index,
                        example,
                        targets[batch_index],
                        stacked_samples[:, batch_index],
                        component_dataset,
                    )
                )

        evaluated_examples += len(examples)
        print(f"Evaluated {evaluated_examples}/{evaluation_limit} examples", flush=True)

    expected_mse = squared_sum / value_count
    mean_prediction_mse = mean_squared_sum / mean_value_count
    expected_mae = abs_sum / value_count
    persistence_mae = persistence_abs_sum / baseline_value_count
    zero_mse = zero_squared_sum / baseline_value_count
    persistence_mse = persistence_squared_sum / baseline_value_count
    result = {
        "checkpoint": checkpoint,
        "evaluation_dataset": args.dataset_root / args.dataset_name,
        "normalization_dataset": (
            args.normalization_dataset_root / args.normalization_dataset_name
        ),
        "dataset_size": len(dataset),
        "sampling": dataset.open_loop_sampling,
        "evaluated_examples": evaluated_examples,
        "samples_per_example": args.samples_per_example,
        "seed": args.seed,
        "action_shape": [horizon, action_dim],
        "metric_space": "raw_robot_units",
        "visualization_directories": visualization_paths,
        "raw_unit_metrics": {
            "expected_sample_mae": expected_mae,
            "expected_sample_mse": expected_mse,
            "expected_sample_rmse": np.sqrt(expected_mse),
            "mean_prediction_mae": mean_abs_sum / mean_value_count,
            "mean_prediction_mse": mean_prediction_mse,
            "mean_prediction_rmse": np.sqrt(mean_prediction_mse),
            "mean_sample_std": (
                diversity_sum / diversity_count if diversity_count else 0.0
            ),
            "per_horizon_mae": per_horizon_abs / per_horizon_count,
            "per_horizon_rmse": np.sqrt(per_horizon_squared / per_horizon_count),
            "per_dimension_mae": per_dimension_abs / per_dimension_count,
            "per_dimension_rmse": np.sqrt(
                per_dimension_squared / per_dimension_count
            ),
        },
        "raw_unit_baselines": {
            "zero": {
                "mae": zero_abs_sum / baseline_value_count,
                "mse": zero_mse,
                "rmse": np.sqrt(zero_mse),
            },
            "current_state_persistence": {
                "mae": persistence_mae,
                "mse": persistence_mse,
                "rmse": np.sqrt(persistence_mse),
                "per_horizon_mae": (
                    persistence_per_horizon_abs / persistence_per_horizon_count
                ),
            },
        },
        "relative_to_persistence": {
            "mae_improvement_fraction": (
                persistence_mae - expected_mae
            ) / persistence_mae,
            "interpretation": (
                "positive means the policy beats persistence; negative means it is worse"
            ),
        },
    }
    return result


def main() -> None:
    args = parse_args()
    result = evaluate(args)
    print(json.dumps(result, indent=2, default=to_jsonable))
    if not args.data_only:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w") as handle:
            json.dump(result, handle, indent=2, default=to_jsonable)
            handle.write("\n")
        print(f"Saved metrics to {args.output.resolve()}")


if __name__ == "__main__":
    main()
