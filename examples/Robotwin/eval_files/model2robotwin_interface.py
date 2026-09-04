from collections import deque
import json
from pathlib import Path
import sys
from typing import Dict, Optional

import cv2 as cv
import numpy as np
import yaml

try:
    from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
except ModuleNotFoundError as exc:
    if exc.name != "websockets":
        raise
    vlajepa_site_packages = Path("/home/heimrih/miniconda3/envs/VLA_JEPA/lib/python3.10/site-packages")
    if vlajepa_site_packages.exists():
        sys.path.append(str(vlajepa_site_packages))
        from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy
    else:
        raise

try:
    from examples.SimplerEnv.eval_files.adaptive_ensemble import AdaptiveEnsembler
except ImportError:
    AdaptiveEnsembler = None


class ModelClient:
    def __init__(
        self,
        policy_ckpt_path,
        unnorm_key: Optional[str] = None,
        policy_setup: str = "robotwin",
        horizon: int = 0,
        action_ensemble=False,
        action_ensemble_horizon: Optional[int] = 3,
        image_size: list[int] = [224, 224],
        use_ddim: bool = True,
        num_ddim_steps: int = 10,
        adaptive_ensemble_alpha=0.1,
        host="127.0.0.1",
        port=5694,
        action_mode: str = "abs",
        normalization_mode: str = "min_max",
    ) -> None:
        self.client = WebsocketClientPolicy(host, port)
        self.policy_setup = policy_setup
        self.unnorm_key = unnorm_key
        self.use_ddim = use_ddim
        self.num_ddim_steps = num_ddim_steps
        self.image_size = image_size
        self.horizon = horizon
        self.action_ensemble = action_ensemble and (AdaptiveEnsembler is not None)
        self.adaptive_ensemble_alpha = adaptive_ensemble_alpha
        self.action_ensemble_horizon = action_ensemble_horizon
        self.normalization_mode = normalization_mode
        self.action_mode = action_mode

        self.initial_state = None
        self.prev_action = None
        self.task_description = None
        self.image_history = deque(maxlen=self.horizon)
        self.action_ensembler = (
            AdaptiveEnsembler(self.action_ensemble_horizon, self.adaptive_ensemble_alpha)
            if self.action_ensemble
            else None
        )
        self.num_image_history = 0
        self.raw_actions = None

        self.action_norm_stats = self.get_action_stats(self.unnorm_key, policy_ckpt_path)
        self.action_chunk_size = self.get_action_chunk_size(policy_ckpt_path)
        server_meta = self.client.get_server_metadata()
        self.action_chunk_size = int(server_meta.get("action_chunk_size", self.action_chunk_size))

        print(
            f"*** policy_setup: {policy_setup}, unnorm_key: {unnorm_key}, "
            f"action_mode: {action_mode}, normalization_mode: {normalization_mode}, "
            f"action_chunk_size: {self.action_chunk_size}, server_meta: {server_meta} ***"
        )

    def reset(self, task_description: str) -> None:
        self.task_description = task_description
        self.image_history.clear()
        if self.action_ensembler:
            self.action_ensembler.reset()
        self.num_image_history = 0
        self.raw_actions = None
        self.initial_state = None
        self.prev_action = None

    def step(self, example: dict, step: int = 0) -> np.ndarray:
        state = example.get("state", None)
        if self.action_mode in ["delta", "rel"] and self.initial_state is None:
            if state is None:
                raise ValueError(f"action_mode='{self.action_mode}' requires state")
            self.initial_state = np.array(state).copy()

        task_description = example.get("lang", None)
        if task_description != self.task_description:
            self.reset(task_description)
            if self.action_mode in ["delta", "rel"] and state is not None:
                self.initial_state = np.array(state).copy()

        images = [self._resize_image(image) for image in example["image"]]
        vla_input = {
            "batch_images": [images],
            "instructions": [self.task_description],
            "unnorm_key": self.unnorm_key,
            "do_sample": False,
            "use_ddim": self.use_ddim,
            "num_ddim_steps": self.num_ddim_steps,
        }

        if step % self.action_chunk_size == 0 or self.raw_actions is None:
            response = self.client.infer(vla_input)
            if not response.get("ok", True):
                raise RuntimeError(response.get("error", response))
            normalized_actions = np.array(response["data"]["normalized_actions"][0])
            raw_actions = self.unnormalize_actions(normalized_actions, self.action_norm_stats)

            if self.action_mode == "delta":
                self.raw_actions = self._delta_to_absolute(raw_actions, state)
            elif self.action_mode == "rel":
                self.raw_actions = self._rel_to_absolute(raw_actions)
            else:
                self.raw_actions = raw_actions

        action_idx = min(step % self.action_chunk_size, len(self.raw_actions) - 1)
        current_action = self.raw_actions[action_idx]
        if self.action_mode == "delta":
            self.prev_action = current_action.copy()

        return current_action[[0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13]]

    @staticmethod
    def unnormalize_actions(normalized_actions: np.ndarray, action_norm_stats: Dict[str, np.ndarray]) -> np.ndarray:
        mask = np.array(action_norm_stats.get("mask", np.ones_like(action_norm_stats["min"], dtype=bool)))
        action_high = np.array(action_norm_stats["max"])
        action_low = np.array(action_norm_stats["min"])
        normalized_actions = np.clip(normalized_actions, -1, 1)
        actions = np.where(
            mask,
            0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
            normalized_actions,
        )
        return actions

    @staticmethod
    def get_action_stats(unnorm_key: str, policy_ckpt_path) -> dict:
        _, norm_stats = ModelClient.read_model_config(Path(policy_ckpt_path))
        unnorm_key = ModelClient._check_unnorm_key(norm_stats, unnorm_key)
        return norm_stats[unnorm_key]["action"]

    @staticmethod
    def get_action_chunk_size(policy_ckpt_path) -> int:
        model_config, _ = ModelClient.read_model_config(Path(policy_ckpt_path))
        return int(model_config["framework"]["action_model"]["future_action_window_size"]) + 1

    @staticmethod
    def read_model_config(policy_ckpt_path: Path) -> tuple[dict, dict]:
        run_dir = policy_ckpt_path.parents[1]
        config_yaml = run_dir / "config.yaml"
        stats_json = run_dir / "dataset_statistics.json"
        if not config_yaml.exists():
            raise FileNotFoundError(f"Missing config.yaml for checkpoint run dir: {run_dir}")
        if not stats_json.exists():
            raise FileNotFoundError(f"Missing dataset_statistics.json for checkpoint run dir: {run_dir}")
        with open(config_yaml, "r", encoding="utf-8") as f:
            model_config = yaml.safe_load(f)
        with open(stats_json, "r", encoding="utf-8") as f:
            norm_stats = json.load(f)
        return model_config, norm_stats

    @staticmethod
    def _check_unnorm_key(norm_stats, unnorm_key):
        if unnorm_key is None:
            assert len(norm_stats) == 1, (
                f"Your model was trained on more than one dataset, choose one of: {norm_stats.keys()}"
            )
            unnorm_key = next(iter(norm_stats.keys()))
        assert unnorm_key in norm_stats, f"Unknown unnorm_key={unnorm_key}; choose from {norm_stats.keys()}"
        return unnorm_key

    def _delta_to_absolute(self, delta_actions: np.ndarray, current_state: np.ndarray) -> np.ndarray:
        abs_actions = np.zeros_like(delta_actions)
        base = self.prev_action if self.prev_action is not None else self.initial_state
        for idx in range(len(delta_actions)):
            abs_actions[idx] = delta_actions[idx] + base
            base = abs_actions[idx]
        return abs_actions

    def _rel_to_absolute(self, rel_actions: np.ndarray) -> np.ndarray:
        return rel_actions + self.initial_state

    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        return cv.resize(image, tuple(self.image_size), interpolation=cv.INTER_AREA)


def get_model(usr_args):
    policy_ckpt_path = usr_args.get("policy_ckpt_path")
    if policy_ckpt_path is None:
        raise ValueError("policy_ckpt_path must be provided")
    return ModelClient(
        policy_ckpt_path=policy_ckpt_path,
        host=usr_args.get("host", "127.0.0.1"),
        port=usr_args.get("port", 5694),
        unnorm_key=usr_args.get("unnorm_key", None),
        action_mode=usr_args.get("action_mode", "abs"),
        normalization_mode=usr_args.get("normalization_mode", "min_max"),
    )


def reset_model(model):
    model.reset(task_description="")


def eval(TASK_ENV, model, observation):
    instruction = TASK_ENV.get_instruction()
    head_img = observation["observation"]["head_camera"]["rgb"]
    left_img = observation["observation"]["left_camera"]["rgb"]
    right_img = observation["observation"]["right_camera"]["rgb"]
    state = observation["joint_action"]["vector"]

    action = model.step(
        {
            "lang": str(instruction),
            "image": [head_img, left_img, right_img],
            "state": state,
        },
        step=TASK_ENV.take_action_cnt,
    )
    TASK_ENV.take_action(action)
