import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Int32MultiArray          # === NEW: tactile msg ===
import numpy as np
from pathlib import Path
import viser
import yourdfpy
from functools import partial
from .config import *

class AllegroSimpleController(Node):
    def __init__(self):
        super().__init__('allegro_simple_controller')

        self.initialized = False
        self.ready = False                    # 원래 있던 ready 게이트 유지
        self.q_robot = None                  # 최신 joint state

        # === NEW: tactile buffer ===
        self.tactile_raw = np.zeros(4, dtype=np.int32)

        self.NUM = 0
        self.cmd_topic = f'/allegroHand_{self.NUM}/joint_cmd'
        self.state_topic = f'/allegroHand_{self.NUM}/joint_states'
        self.tactile_topic = f'/allegroHand_{self.NUM}/tactile_sensors'   # 예시

        # ROS pub/sub
        self.publisher_ = self.create_publisher(JointState, self.cmd_topic, 10)
        self.joint_state_sub = self.create_subscription(
            JointState,
            self.state_topic,
            self.joint_state_callback,
            10
        )
        self.tactile_sub = self.create_subscription(
            Int32MultiArray,
            self.tactile_topic,
            self._sensor_callback,
            10
        )

        self.init_timer = self.create_timer(0.1, self.try_initial_sync)

        # viser + URDF
        self.server = viser.ViserServer()
        self.urdf = yourdfpy.URDF.load(
            ALLEGRO_LEFT_URDF_PATH,
            filename_handler=partial(
                yourdfpy.filename_handler_magic,
                dir=ALLEGRO_LEFT_URDF_PATH.parent,
            ),
        )

        self.joint_names = self.urdf.actuated_joint_names

        self.sliders = {}
        with self.server.gui.add_folder("Joint position control"):
            for jname in self.joint_names:
                joint: yourdfpy.Joint = self.urdf.joint_map[jname]
                central = (joint.limit.lower + joint.limit.upper)/2.0
                slider = self.server.gui.add_slider(
                    label=jname,
                    min=joint.limit.lower,
                    max=joint.limit.upper,
                    step=1e-3,
                    initial_value=central,
                )
                self.sliders[jname] = slider
                slider.on_update(
                    lambda _: self.on_slider_update_and_publish()
                )

        self.reset_button = self.server.gui.add_button("Reset")
        @self.reset_button.on_click
        def _(_):
            if self.q_robot is None:
                return
            for jname, val in zip(self.joint_names, self.q_robot):
                self.sliders[jname].value = float(val)
            self.draw_link_points(self.q_robot)

    # ---- property & ROS callbacks ----

    @property
    def q_from_gui(self):
        q_list = [self.sliders[jname].value for jname in self.joint_names]
        return np.array(q_list, dtype=float)

    def joint_state_callback(self, msg: JointState):
        try:
            name_to_pos = dict(zip(msg.name, msg.position))
            q = np.array([name_to_pos[j] for j in self.joint_names], dtype=float)
            self.q_robot = q
        except Exception as e:
            self.get_logger().error(f"joint_state_callback error: {e}")

    def _sensor_callback(self, msg: Int32MultiArray):
        try:
            data = np.asarray(msg.data, dtype=np.int32)
            if data.size < 4:
                self.get_logger().warn(
                    f"tactile msg too short: expected 4, got {data.size}"
                )
                return
            self.tactile_raw[:] = data[:4]
            if self.q_robot is None:
                self.draw_link_points(self.q_from_gui)
            else:
                self.draw_link_points(self.q_robot)
        except Exception as e:
            self.get_logger().error(f"sensor error: {e}")


    def try_initial_sync(self):
        if self.q_robot is None:
            return
        if self.initialized:
            return

        self.get_logger().info("Initializing sliders with real robot states...")

        for jname, val in zip(self.joint_names, self.q_robot):
            self.sliders[jname].value = float(val)

        self.draw_link_points(self.q_robot)

        self.initialized = True
        self.ready = True
        self.init_timer.cancel()
        print("ready")

    # ---- Visualization ----
    def draw_link_points(self, q):
        self.urdf.update_cfg(q)
        points = {}

        for jname, joint in self.urdf.joint_map.items():
            lname = joint.child
            T_parent_child = self.urdf.get_transform(joint.child)
            points[lname] = T_parent_child[:3, -1]

        pcd = np.array(list(points.values()))
        self.server.scene.add_point_cloud(
            "points", pcd, (0.0, 0.0, 0.0), point_size=0.006,
            point_shape='circle'
        )

        max_val = 1000. if np.max(self.tactile_raw) > 0 else 1.0
        norm_tactile = np.clip(self.tactile_raw.astype(float) / max_val, 0.0, 1.0)

        for fi, fname in enumerate(FINGER_ORDER):
            links = FINGERS[fname]
            pts = np.array([points[l] for l in links])
            segments = np.stack([pts[:-1], pts[1:]], axis=1)

            base_color = np.array(COLOR[fname], dtype=float)
            # 0일 때도 너무 어둡지 않게 offset 줌
            w = 0.3 + 0.7 * norm_tactile[fi]
            color = tuple(np.clip(base_color * w, 0.0, 1.0))

            self.server.scene.add_line_segments(
                f"finger_{fname}",
                points=segments,
                colors=color,
                line_width=30.0,
            )

    # ---- GUI → ROS ----
    def on_slider_update_and_publish(self):
        if not self.ready:
            return

        q = self.q_from_gui
        self.publish_cmd(q)
        self.draw_link_points(q)

    def publish_cmd(self, q):
        msg = JointState()
        msg.name = self.joint_names
        msg.position = list(q)
        msg.velocity = []
        msg.effort = []
        self.publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = AllegroSimpleController()

    node.draw_link_points(node.q_from_gui)

    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()