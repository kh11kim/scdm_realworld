from pathlib import Path

ALLEGRO_LEFT_URDF_PATH = Path(
    "/root/allegro_ws/src/allegro_hand_ros2_v5/src/allegro_hand_controllers/urdf/allegro_hand_description_left_B.urdf"
)
FINGERS = {
    "index":  ["link_0_0", "link_1_0", "link_2_0", "link_3_0", "link_3_0_tip"],
    "middle": ["link_4_0", "link_5_0", "link_6_0", "link_7_0", "link_7_0_tip"],
    "pinky":  ["link_8_0", "link_9_0", "link_10_0", "link_11_0", "link_11_0_tip"],
    "thumb":  ["link_12_0", "link_13_0", "link_14_0", "link_15_0", "link_15_0_tip"],
}
COLOR = {
    "index":  (1.0, 0.2, 0.2),
    "middle": (0.2, 1.0, 0.2),
    "pinky":  (0.2, 0.2, 1.0),
    "thumb":  (1.0, 0.7, 0.2),
}

FINGER_ORDER = ["index", "middle", "pinky", "thumb"]