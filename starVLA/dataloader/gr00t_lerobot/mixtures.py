"""
mixtures.py

Defines a registry of dataset mixtures and weights for the Open-X Embodiment Datasets. Each dataset is associated with
a float "sampling weight"
"""

from typing import Dict, List, Tuple

_ROBOTWIN_TASKS = [
    "adjust_bottle",
    "beat_block_hammer",
    "blocks_ranking_rgb",
    "blocks_ranking_size",
    "click_alarmclock",
    "click_bell",
    "dump_bin_bigbin",
    "grab_roller",
    "handover_block",
    "handover_mic",
    "hanging_mug",
    "lift_pot",
    "move_can_pot",
    "move_pillbottle_pad",
    "move_playingcard_away",
    "move_stapler_pad",
    "open_laptop",
    "open_microwave",
    "pick_diverse_bottles",
    "pick_dual_bottles",
    "place_a2b_left",
    "place_a2b_right",
    "place_bread_basket",
    "place_bread_skillet",
    "place_burger_fries",
    "place_can_basket",
    "place_cans_plasticbox",
    "place_container_plate",
    "place_dual_shoes",
    "place_empty_cup",
    "place_fan",
    "place_mouse_pad",
    "place_object_basket",
    "place_object_scale",
    "place_object_stand",
    "place_phone_stand",
    "place_shoe",
    "press_stapler",
    "put_bottles_dustbin",
    "put_object_cabinet",
    "rotate_qrcode",
    "scan_object",
    "shake_bottle",
    "shake_bottle_horizontally",
    "stack_blocks_three",
    "stack_blocks_two",
    "stack_bowls_three",
    "stack_bowls_two",
    "stamp_seal",
    "turn_switch",
]


# Dataset mixture name mapped to a list of tuples containing:
## {nakename: [(data_name, sampling_weight, robot_type)] }
DATASET_NAMED_MIXTURES = {

    "libero_all": [
        ("libero_object_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
        ("libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
                # ("libero_90_no_noops_lerobot", 1.0, "libero_franka"),
    ],

    "droid": [
        ("", 1.0, "libero_franka"),
    ],

    "fr3_realworld": [
        ("", 1.0, "fr3_real_world"),
    ],

    "libero_goal": [
        ("libero_goal_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    "libero_object": [
        ("libero_object_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    "libero_spatial": [
        ("libero_spatial_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    "libero_10": [
        ("libero_10_no_noops_1.0.0_lerobot", 1.0, "libero_franka"),
    ],
    "libero_90": [
        ("libero_90_no_noops_lerobot", 1.0, "libero_franka"),
        # ("libero_90_no_noops_lerobot", 1.0, "libero_ur5"),
    ],

    "bridge": [
        ("bridge_orig_1.0.0_lerobot", 1.0, "oxe_bridge"),
    ],
    "bridge_rt_1": [
        ("bridge_orig_1.0.0_lerobot", 1.0, "oxe_bridge"),
        ("fractal20220817_data_0.1.0_lerobot", 1.0, "oxe_rt1"),
    ],

    "demo_sim_pick_place": [
        ("sim_pick_place", 1.0, "demo_sim_franka_delta_joints"),
    ],

    "custom_dataset": [
        ("custom_dataset_name", 1.0, "custom_robot_config"),
    ],
    "custom_dataset_2": [
        ("custom_dataset_name_1", 1.0, "custom_robot_config"),
        ("custom_dataset_name_2", 1.0, "custom_robot_config"),
    ],

    "BEHAVIOR_challenge": [
        ("BEHAVIOR_challenge", 1.0, "R1Pro"),
    ],

    "robotwin": [
        (task, 1.0, "robotwin") for task in _ROBOTWIN_TASKS
    ],
    "robotwin_task1": [
        ("adjust_bottle", 1.0, "robotwin"),
    ],
    "robotwin_task2": [
        ("place_a2b_left", 1.0, "robotwin"),
        ("place_a2b_right", 1.0, "robotwin"),
    ],
    "robotwin_all": [
        (f"{domain}/{task}", 1.0, "robotwin")
        for task in _ROBOTWIN_TASKS
        for domain in ("Clean", "Randomized")
    ],
    "robotwin_all_50": [
        (f"{domain}/{task}", 1.0, "robotwin50")
        for task in _ROBOTWIN_TASKS
        for domain in ("Clean", "Randomized")
    ],
    "real_widowxai_success_only": [
        ("real_widowxai_success_only", 1.0, "real_widowxai"),
    ],

}
