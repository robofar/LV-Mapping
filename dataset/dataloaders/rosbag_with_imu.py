# MIT License
#
# Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
# Stachniss.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import glob
import os
import sys
import numpy as np
from pathlib import Path
from typing import Sequence

import natsort


class RosbagIMUDataset:
    def __init__(self, data_dir: Path, topic: str, imu_topic: str, *_, **__):
        """ROS1 / ROS2 bagfile dataloader.

        It can take either one ROS2 bag file or one or more ROS1 bag files belonging to a split bag.
        The reader will replay ROS1 split bags in correct timestamp order.

        TODO: Merge mcap and rosbag dataloaders into 1
        """
        try:
            from rosbags.highlevel import AnyReader
        except ModuleNotFoundError:
            print('rosbags library not installed, run "pip install -U rosbags"')
            sys.exit(1)

        from utils.point_cloud2 import read_point_cloud

        self.read_point_cloud = read_point_cloud
        if data_dir.is_file():
            self.sequence_id = os.path.basename(data_dir).split(".")[0]
            self.bag = AnyReader([data_dir])
        else:
            bagfiles = [Path(path) for path in glob.glob(os.path.join(data_dir, "*.bag"))]
            if len(bagfiles) > 0: # combine all the bags
                self.sequence_id = os.path.basename(bagfiles[0]).split(".")[0]
                self.bag = AnyReader(bagfiles)
            else:
                self.sequence_id = os.path.basename(data_dir).split(".")[0]
                self.bag = AnyReader([data_dir])

        if len(self.bag.paths) > 1:
            print("Reading multiple .bag files in directory:")
            print("\n".join(natsort.natsorted([path.name for path in self.bag.paths])))

        # print(topic)

        self.bag.open()
        self.pc_topic = self.check_topic(topic)
        self.imu_topic = self.check_imu_topic(imu_topic)

        self.n_scans = self.bag.topics[self.pc_topic].msgcount

        self.n_imus = self.bag.topics[self.imu_topic].msgcount
        if self.n_imus > 0:
            self.imu_on = True
        # print('imu count:', self.n_imus)

        # limit connections to selected topic
        pc_connections = [x for x in self.bag.connections if x.topic == self.pc_topic] # connections just mean topics
        self.pc_msgs = self.bag.messages(connections=pc_connections) # get all point cloud messages

        imu_connections = [x for x in self.bag.connections if x.topic == self.imu_topic]
        self.imu_msgs = self.bag.messages(connections=imu_connections)

        # print(self.imu_msgs)

        self.timestamps = [] # framewise timestamp
        self.first_pc_timestamp_ns = 0

        self.imu_buffer = []
        self.imu_buffer_ts = []
        self.cur_ts_imu = -1e9 # unit: ns
        self.last_ts_imu = 0
        self.imu_read_count = 0

    def __del__(self):
        if hasattr(self, "bag"):
            self.bag.close()

    def __len__(self):
        return self.n_scans

    def __getitem__(self, idx):
        pc_connection, pc_timestamp_ns, pc_data = next(self.pc_msgs)

        if idx==0:
            self.first_pc_timestamp_ns = pc_timestamp_ns

        pc_timestamp_ns -= self.first_pc_timestamp_ns  # minus the beginning reference  

        # print(pc_timestamp_ns)
        self.timestamps.append(pc_timestamp_ns / 1e9) 
        pc_msg = self.bag.deserialize(pc_data, pc_connection.msgtype)


        points, point_ts = self.read_point_cloud(pc_msg)
        # print(point_ts)

        # lidar_timestamp = dt.datetime.fromtimestamp(int(lidar_timestamp_s))
        # lidar_timestamp += dt.timedelta(microseconds=(timestamp%1e9) / 1000) 
        # print(lidar_timestamp)

        if point_ts is not None:
            pc_min_ts = np.min(point_ts)
            pc_max_ts = np.max(point_ts)
            point_ts_delta = pc_max_ts - pc_min_ts
            if point_ts_delta < 1.0: # unit would be s instead of ns
                # then convert to ns
                point_ts *= 1e9
                pc_min_ts *= 1e9
                pc_max_ts *= 1e9
            
            point_ts = pc_timestamp_ns + point_ts # pointwise timestamp, unit: ns
            pc_min_ts += pc_timestamp_ns
            pc_max_ts += pc_timestamp_ns
        else:
            pc_min_ts = pc_timestamp_ns
            pc_max_ts = pc_min_ts + 1e8 # pointwise timestamp, unit: ns

        linear_acc = []
        angular_velo = []
        imu_ts = []
        imu_dt = []

        while self.cur_ts_imu < pc_max_ts and self.imu_read_count < self.n_imus: # ns
            imu_connection, cur_ts_imu, imu_data = next(self.imu_msgs) # ns
            imu_msg = self.bag.deserialize(imu_data, imu_connection.msgtype)
            cur_ts_imu -= self.first_pc_timestamp_ns  # minus the beginning reference  
            self.cur_ts_imu = cur_ts_imu
            self.imu_buffer.append(imu_msg)
            self.imu_buffer_ts.append(cur_ts_imu) # ns
            self.imu_read_count += 1

        imu_buffer_ts_np = np.array(self.imu_buffer_ts) # ns

        frame_imu_begin_idx = closest_ts_idx(pc_min_ts, imu_buffer_ts_np)
        frame_imu_end_idx = closest_ts_idx(pc_max_ts, imu_buffer_ts_np)

        frame_imu_data = None
        if frame_imu_end_idx - frame_imu_begin_idx > 3: # need to have enough IMU measurements in between
        
            # frame_imu_idx_range = range(frame_imu_begin_idx, frame_imu_end_idx+1)

            # skip the first imu data in a frame
            frame_imu_idx_range = range(frame_imu_begin_idx+1, frame_imu_end_idx+1)

            for idx in frame_imu_idx_range:
                imu_msg = self.imu_buffer[idx]

                linear_acc.append([imu_msg.linear_acceleration.x,
                                imu_msg.linear_acceleration.y,
                                imu_msg.linear_acceleration.z])
                angular_velo.append([imu_msg.angular_velocity.x,
                                    imu_msg.angular_velocity.y,
                                    imu_msg.angular_velocity.z])
                imu_ts.append(self.imu_buffer_ts[idx]/1e9) # s
                if idx > 0:
                    dt = (self.imu_buffer_ts[idx] - self.imu_buffer_ts[idx-1])/1e9 # s
                else:
                    dt = 1e-9
                imu_dt.append(dt) # unit: s

            frame_imu_data = {
                'ts': np.array(imu_ts),
                'dt': np.array(imu_dt),
                'acc': np.array(linear_acc),
                'gyro': np.array(angular_velo)
            }
        
            # Remove used IMU messages from the buffer
            buffer_left = 10

            self.imu_buffer = [imu_msg for i, imu_msg in enumerate(self.imu_buffer) if i >= frame_imu_end_idx-buffer_left]

            self.imu_buffer_ts = [imu_ts for i, imu_ts in enumerate(self.imu_buffer_ts) if i >= frame_imu_end_idx-buffer_left]

        if point_ts is not None:
            point_ts /= 1e9 # unit: s

        frame_data = {"points": points, "point_ts": point_ts, "imus": frame_imu_data}

        return frame_data

    @staticmethod
    def to_sec(nsec: int):
        return (nsec) / 1e9

    def get_frames_timestamps(self) -> list:
        return self.timestamps

    def check_topic(self, topic: str) -> str:
        # Extract all PointCloud2 msg topics from the bagfile
        point_cloud_topics = [
            topic[0]
            for topic in self.bag.topics.items()
            if topic[1].msgtype == "sensor_msgs/msg/PointCloud2"
        ]

        def print_available_topics_and_exit():
            print("Select from the following topics:")
            print(50 * "-")
            for t in point_cloud_topics:
                print(f"{t}")
            print(50 * "-")
            sys.exit(1)

        if topic and topic in point_cloud_topics:
            return topic
        # when user specified the topic check that exists
        if topic and topic not in point_cloud_topics:
            print(
                f'[ERROR] Dataset does not containg any point cloud msg with the topic name "{topic}". '
                "Specify the correct topic name by python pin_slam.py path/to/config/file.yaml rosbag_with_imu your/pc_topic your/imu_topic ... ..."
            )
            print_available_topics_and_exit()
        if len(point_cloud_topics) > 1:
            print(
                "Multiple sensor_msgs/msg/PointCloud2 topics available."
                "Specify the correct topic name by python pin_slam.py path/to/config/file.yaml rosbag_with_imu your/pc_topic your/imu_topic ... ..."
            )
            print_available_topics_and_exit()

        if len(point_cloud_topics) == 0:
            print("[ERROR] Your dataset does not contain any sensor_msgs/msg/PointCloud2 topic")
        if len(point_cloud_topics) == 1:
            return point_cloud_topics[0]
    
    # TODO: not support currently
    def check_imu_topic(self, topic: str) -> str:
        # Extract all IMU msg topics from the bagfile
        imu_topics = [
            topic[0]
            for topic in self.bag.topics.items()
            if topic[1].msgtype == "sensor_msgs/msg/Imu"
        ]

        def print_available_topics_and_exit():
            print("Select from the following topics:")
            print(50 * "-")
            for t in imu_topics:
                print(f"{t}")
            print(50 * "-")
            sys.exit(1)

        if topic and topic in imu_topics:
            return topic
        # when user specified the topic check that exists
        if topic and topic not in imu_topics:
            print(
                f'[ERROR] Dataset does not containg any IMU msg with the topic name "{topic}". '
                "Specify the correct topic name by python pin_slam.py path/to/config/file.yaml rosbag_with_imu your/pc_topic your/imu_topic ... ..."
            )
            print_available_topics_and_exit()
        if len(imu_topics) > 1:
            print(
                "Multiple sensor_msgs/msg/Imu topics available."
                "Specify the correct topic name by python pin_slam.py path/to/config/file.yaml rosbag_with_imu your/pc_topic your/imu_topic ... ..."
            )
            print_available_topics_and_exit()

        if len(imu_topics) == 0:
            print("[ERROR] Your dataset does not contain any sensor_msgs/msg/Imu topic")
        if len(imu_topics) == 1:
            return imu_topics[0]
        


def closest_ts_idx(ts_ref, ts_for_sort):
    delta_ts = np.abs(ts_for_sort - ts_ref)
    idx = np.argmin(delta_ts)
    return idx
