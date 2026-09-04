"""RoboTwin evaluation adapter for state-conditioned VLA-JEPA DiT policies."""

from pathlib import Path

import numpy as np

from model2robotwin_interface import ModelClient, eval, reset_model


class DiTModelClient(ModelClient):
    """Send proprioception in the state layout used by the training dataset."""

    # Environment: left arm, left gripper, right arm, right gripper.
    # Training:    left arm, right arm, left gripper, right gripper.
    ENV_TO_TRAIN_STATE = np.array(
        [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12, 6, 13], dtype=np.int64
    )

    def __init__(self, policy_ckpt_path, **kwargs) -> None:
        model_config, norm_stats = self.read_model_config(Path(policy_ckpt_path))
        framework = model_config.get("framework", {})
        action_config = framework.get("action_model", {})
        dataset_config = model_config.get("datasets", {}).get("vla_data", {})

        framework_name = framework.get("name")
        action_model_type = str(action_config.get("action_model_type", ""))
        if framework_name != "VLA_JEPA" or not action_model_type.startswith("DiT"):
            raise ValueError(
                "The DiT RoboTwin evaluator requires framework.name='VLA_JEPA' and a "
                f"DiT action model; got framework={framework_name!r}, "
                f"action_model_type={action_model_type!r}."
            )
        if not dataset_config.get("with_state", False):
            raise ValueError(
                "This evaluator is for state-conditioned checkpoints, but the saved config "
                "has datasets.vla_data.with_state disabled."
            )

        self.state_dim = int(action_config.get("state_dim", 0))
        if self.state_dim != len(self.ENV_TO_TRAIN_STATE):
            raise ValueError(
                "The RoboTwin DiT state adapter currently expects state_dim=14; "
                f"the checkpoint declares state_dim={self.state_dim}."
            )
        norm_key = self._check_unnorm_key(norm_stats, kwargs.get("unnorm_key"))
        state_stats = norm_stats[norm_key].get("state", {})
        self.state_min = np.asarray(state_stats.get("min", []), dtype=np.float32)
        self.state_max = np.asarray(state_stats.get("max", []), dtype=np.float32)
        if self.state_min.size != self.state_dim or self.state_max.size != self.state_dim:
            raise ValueError(
                f"Expected {self.state_dim}-D state min/max statistics for {norm_key!r}; "
                f"got min={self.state_min.shape}, max={self.state_max.shape}."
            )
        super().__init__(policy_ckpt_path=policy_ckpt_path, **kwargs)

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Apply the RobotwinAgilexDataConfig training-time state transforms."""
        normalized = np.zeros_like(state, dtype=np.float32)

        # The 12 arm joints use min-max normalization to [-1, 1].
        joint_count = self.state_dim - 2
        denominator = self.state_max[:joint_count] - self.state_min[:joint_count]
        nonconstant = denominator != 0
        normalized[:joint_count][nonconstant] = (
            2.0
            * (state[:joint_count][nonconstant] - self.state_min[:joint_count][nonconstant])
            / denominator[nonconstant]
            - 1.0
        )

        # Both grippers use the dataset's binary transform.
        normalized[joint_count:] = (state[joint_count:] > 0.5).astype(np.float32)
        return normalized

    def step(self, example: dict, step: int = 0) -> np.ndarray:
        state = example.get("state")
        if state is None:
            raise ValueError("The state-conditioned DiT policy requires a RoboTwin joint state.")

        task_description = example.get("lang")
        if task_description != self.task_description:
            self.reset(task_description)

        state_array = np.asarray(state, dtype=np.float32).reshape(-1)
        if state_array.size != self.state_dim:
            raise ValueError(
                f"Expected a {self.state_dim}-D RoboTwin state, got shape {np.asarray(state).shape}."
            )
        state_array = state_array[self.ENV_TO_TRAIN_STATE]
        state_array = self._normalize_state(state_array)

        images = [self._resize_image(image) for image in example["image"]]
        vla_input = {
            "batch_images": [images],
            "instructions": [self.task_description],
            # Training supplies normalized state as [batch, time, state_dim].
            "state": state_array.reshape(1, 1, self.state_dim),
            "unnorm_key": self.unnorm_key,
            "do_sample": False,
            "use_ddim": self.use_ddim,
            "num_ddim_steps": self.num_ddim_steps,
        }

        if step % self.action_chunk_size == 0 or self.raw_actions is None:
            response = self.client.infer(vla_input)
            if not response.get("ok", True):
                raise RuntimeError(response.get("error", response))
            normalized_actions = np.asarray(response["data"]["normalized_actions"][0])
            self.raw_actions = self.unnormalize_actions(normalized_actions, self.action_norm_stats)

        action_idx = min(step % self.action_chunk_size, len(self.raw_actions) - 1)
        current_action = self.raw_actions[action_idx]

        # Training: left arm, right arm, left gripper, right gripper.
        # Environment: left arm, left gripper, right arm, right gripper.
        return current_action[[0, 1, 2, 3, 4, 5, 12, 6, 7, 8, 9, 10, 11, 13]]


def get_model(usr_args):
    policy_ckpt_path = usr_args.get("policy_ckpt_path")
    if policy_ckpt_path is None:
        raise ValueError("policy_ckpt_path must be provided")
    return DiTModelClient(
        policy_ckpt_path=policy_ckpt_path,
        host=usr_args.get("host", "127.0.0.1"),
        port=usr_args.get("port", 5694),
        unnorm_key=usr_args.get("unnorm_key"),
        action_mode=usr_args.get("action_mode", "abs"),
        normalization_mode=usr_args.get("normalization_mode", "min_max"),
    )
