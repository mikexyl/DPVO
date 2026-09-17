"""One disposable process owns the RealSense, DPVO, and distributed frontend."""
import rclpy
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor


def main(args=None):
    # GPU and SDK imports occur only in this disposable worker.
    from .node import MultiRobotDpvoNode
    from .online_camera import OnlineCamera

    rclpy.init(args=args)
    executor = MultiThreadedExecutor(num_threads=4)
    tracker = camera = None
    try:
        tracker = MultiRobotDpvoNode()
        executor.add_node(tracker)
        camera = OnlineCamera()
        executor.add_node(camera)
        executor.spin()
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        if camera is not None:
            camera.close()
        executor.shutdown(timeout_sec=5)
        if tracker is not None:
            tracker.close()
            tracker.destroy_node()
        if camera is not None:
            camera.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()
