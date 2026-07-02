#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import time

from aimdk_msgs.srv import SetMcPresetMotion, PlayTts
from aimdk_msgs.msg import (
    RequestHeader, McPresetMotion, McControlArea, CommonState,
    TtsPriorityLevel
)

class ActionSequenceDemo(Node):
    def __init__(self):
        super().__init__('action_sequence_demo')
        
        self.declare_parameter('wait_duration', 3.0)
        self.declare_parameter('wave_count', 3)

        # Create service clients
        self.set_preset_motion_client = self.create_client(
            SetMcPresetMotion, '/aimdk_5Fmsgs/srv/SetMcPresetMotion')
        self.play_tts_client = self.create_client(
            PlayTts, '/aimdk_5Fmsgs/srv/PlayTts')
    
        self.get_logger().info('Action Sequence Demo created')
    
        # Wait for services to become available
        while not self.set_preset_motion_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for preset motion service...')
        while not self.play_tts_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Waiting for TTS service...')

    def perform_preset_motion(self, area_id, motion_id):
        """Execute a preset motion"""
        request = SetMcPresetMotion.Request()
        request.header = RequestHeader()
        request.header.stamp = self.get_clock().now().to_msg()
    
        motion = McPresetMotion()
        motion.value = motion_id
        area = McControlArea()
        area.value = area_id
    
        request.motion = motion
        request.area = area
        request.interrupt = False
    
        self.get_logger().info(f'Executing preset motion: area={area_id}, motion={motion_id}')
        future = self.set_preset_motion_client.call_async(request)
        rclpy.spin_until_future_complete(self, future)
    
        if future.result() is not None:
            response = future.result()
            if response.response.header.code == 0:
                self.get_logger().info('Preset motion executed successfully')
                return True
            else:
                self.get_logger().error('Preset motion execution failed')
                return False
        else:
            self.get_logger().error('Failed to call preset motion service')
            return False

    def speak(self, text):
        """Trigger TTS speech output"""
        if not self.play_tts_client.wait_for_service(timeout_sec=5.0):
            self.get_logger().error('TTS service unavailable')
            return False
    
        request = PlayTts.Request()
        request.header.header.stamp = self.get_clock().now().to_msg()
    
        # Configure TTS request
        request.tts_req.text = text
        request.tts_req.domain = 'action_sequence_demo'  # Caller identifier
        request.tts_req.trace_id = 'sequence'            # Request ID
        request.tts_req.is_interrupted = True            # Allow interrupting same-priority speech
        request.tts_req.priority_weight = 0
        request.tts_req.priority_level = TtsPriorityLevel()
        request.tts_req.priority_level.value = 6         # Priority level
    
        self.get_logger().info(f'TTS speak: {text}')
        future = self.play_tts_client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=10.0)
    
        if future.result() is not None:
            response = future.result()
            if response.tts_resp.is_success:
                self.get_logger().info('Speech playback succeeded')
                return True
            else:
                self.get_logger().error('Speech playback failed')
                return False
        else:
            self.get_logger().error('Failed to call TTS service')
            return False

    def wave_hand(self):
        """Control robot to wave hand"""
        # Use preset motion: right-hand wave (area=2, motion=1002)
        return self.perform_preset_motion(2, 1002)

    def shake_hand(self):
        """Control robot to perform handshake"""
        # Use preset motion: right-hand handshake (area=2, motion=1003)
        return self.perform_preset_motion(2, 1003)

    def perform_action_sequence(self):
        """Execute the complete action sequence"""
        self.get_logger().info('Starting action sequence...')
        self.get_logger().info('Prerequisite: robot must be standing')
        
        wait_duration = self.get_parameter('wait_duration').value

        # 1. First action: Wave hand
        if not self.wave_hand():
            self.get_logger().error('Wave hand failed')
            return False
        time.sleep(wait_duration)
    
        # 2. TTS speech
        if not self.speak('Hello! I am AgiBot X2. Now demonstrating a hand action sequence!'):
            self.get_logger().error('TTS failed')
            return False
        time.sleep(wait_duration)
    
        # 3. Second action: Handshake
        if not self.shake_hand():
            self.get_logger().error('Handshake failed')
            return False
        time.sleep(wait_duration)
    
        self.get_logger().info('Action sequence completed')
        return True

def main(args=None):
    rclpy.init(args=args)
    demo = ActionSequenceDemo()

    # Execute the action sequence
    demo.perform_action_sequence()

    # Shut down the ROS node
    demo.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
