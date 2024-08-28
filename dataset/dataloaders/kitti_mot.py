# MIT License
#
# Copyright (c) 2022 Ignacio Vizzo, Tiziano Guadagnino, Benedikt Mersch, Cyrill
# Stachniss.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to mse, copy, modify, merge, publish, distribute, sublicense, and/or sell
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
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE msE OR OTHER DEALINGS IN THE
# SOFTWARE.
import glob
import importlib
import os

import cv2
import numpy as np
import open3d as o3d

# https://www.cvlibs.net/datasets/kitti/setup.php
# https://www.cvlibs.net/datasets/kitti/eval_tracking.php # KITTI MOT dataset

# TODO

class KITTIMOTDataset:
    def __init__(self, data_dir, sequence: str, *_, **__):
        
        self.sequence_id = str(sequence).zfill(4)
        # include the data dir as kitti_mot/training/
        # self.kitti_sequence_dir = os.path.join(data_dir, "sequences", self.sequence_id)
        
        self.velodyne_dir = os.path.join(data_dir, "velodyne", self.sequence_id) 
        self.scan_files = sorted(glob.glob(self.velodyne_dir + "*.bin"))
        scan_count = len(self.scan_files)

        # img related
        self.load_img = False # default
        self.use_only_colorized_points = True

        # cam 2 (color)
        self.img2_dir = os.path.join(data_dir, "image_02", self.sequence_id) 
        self.img2_files = sorted(glob.glob(self.img2_dir + "*.png"))
        img2_count = len(self.img2_files)
        if img2_count == scan_count:
            self.image_available = True
        else:
            self.image_available = False

        # cam 3 (color)
        self.img3_dir = os.path.join(data_dir, "image_03", self.sequence_id) 
        self.img3_files = sorted(glob.glob(self.img3_dir + "*.png"))
        img3_count = len(self.img3_files)

        # cam 2 sky mask
        # cam 3 sky mask

        calib_file_path = os.path.join(data_dir, "calib", self.sequence_id+".txt")
        self.calibration = self.read_calib_file(calib_file_path) 

        calib_data = self._load_calib() # load all calib first

        self.main_cam_name = "cam2" # cam2 as main cam

        if self.image_available: # now we use cam2 (left color)
            self.T_c_l_mats = {self.main_cam_name: calib_data['T_cam2_velo']}
            self.K_mats = {self.main_cam_name: calib_data["K_cam2"]}

            self.intrinsic = o3d.camera.PinholeCameraIntrinsic()
            self.intrinsic.set_intrinsics(
                                        height=375,
                                        width=1242,
                                        fx=calib_data["K_cam2"][0,0],
                                        fy=calib_data["K_cam2"][1,1],
                                        cx=calib_data["K_cam2"][0,2],
                                        cy=calib_data["K_cam2"][1,2])

            self.extrinsic = calib_data['T_cam2_velo']

        # FIXME: mono_depth rgbd version

        # self.K_mats = {self.left_cam_name: calib_data["K_cam2"]}
        # self.T_l_c = np.eye(4)
        # self.T_c_l = np.linalg.inv(self.T_l_c)
        # self.T_c_l_mats = {self.left_cam_name: self.T_c_l}
        
        # print(calib_data["K_cam2"])
        ###

        oxts_file_path = os.path.join(data_dir, "oxts", self.sequence_id+".txt")

        self.oxts, self.imu_poses = self.load_oxts_packets_and_poses(oxts_file_path) # gt poses in IMU frame

        # GT poses in LiDAR frame
        self.gt_poses = self.Tr_lidar_imu @ self.imu_poses @ self.Tr_imu_lidar 

    def __getitem__(self, idx):
        
        points = self.scans(idx)
        point_ts = self.get_timestamps(points)

        if self.load_img and self.image_available:
            img = self.read_img(self.img2_files[idx]) # just for vis here
        
            points_rgb = np.ones_like(points)

            # project to the image plane to get the corresponding color
            points_rgb, depth_map = self.project_points_to_cam(points, points_rgb, img, self.T_c_l_mats[self.left_cam_name], self.K_mats[self.left_cam_name])

            if self.use_only_colorized_points:
                with_rgb_mask = (points_rgb[:, 3] == 0)
                points = points[with_rgb_mask]
                points_rgb = points_rgb[with_rgb_mask]

            # we skip the intensity here for now (and also the color mask)
            points = np.hstack((points[:,:3], points_rgb[:,:3]))

            img = np.concatenate((img, np.expand_dims(depth_map, axis=-1)), axis=-1) # 4 channels
            img_dict = {self.left_cam_name: img}

            frame_data = {"points": points, "point_ts": point_ts, "img": img_dict}
        else:
            frame_data = {"points": points, "point_ts": point_ts}

        return frame_data

    def __len__(self):
        return len(self.scan_files)

    def scans(self, idx):
        return self.read_point_cloud(self.scan_files[idx])

    def apply_calibration(self, poses: np.ndarray) -> np.ndarray:
        """Converts from Velodyne to Camera Frame"""
        Tr = np.eye(4, dtype=np.float64)
        Tr[:3, :4] = self.calibration["Tr"].reshape(3, 4)
        return Tr @ poses @ np.linalg.inv(Tr)

    def read_point_cloud(self, scan_file: str):
        points = np.fromfile(scan_file, dtype=np.float32).reshape((-1, 4))[:, :4].astype(np.float64)
        return points # N, 4
    
    def read_img(self, img_file: str):
        img = cv2.imread(img_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return img
    
    # velodyne lidar
    @staticmethod
    def get_timestamps(points):
        x = points[:, 0]
        y = points[:, 1]
        yaw = -np.arctan2(y, x)
        timestamps = 0.5 * (yaw / np.pi + 1.0)
        return timestamps

    def load_poses(self, poses_file):
        def _lidar_pose_gt(poses_gt):
            _tr = self.calibration["Tr"].reshape(3, 4)
            tr = np.eye(4, dtype=np.float64)
            tr[:3, :4] = _tr
            left = np.einsum("...ij,...jk->...ik", np.linalg.inv(tr), poses_gt)
            right = np.einsum("...ij,...jk->...ik", left, tr)
            return right

        poses = np.loadtxt(poses_file, delimiter=" ")
        n = poses.shape[0]
        poses = np.concatenate(
            (poses, np.zeros((n, 3), dtype=np.float32), np.ones((n, 1), dtype=np.float32)), axis=1
        )
        poses = poses.reshape((n, 4, 4))  # [N, 4, 4]
        return _lidar_pose_gt(poses)

    def get_frames_timestamps(self) -> np.ndarray:
        timestamps = np.loadtxt(os.path.join(self.kitti_sequence_dir, "times.txt")).reshape(-1, 1)
        return timestamps

    @staticmethod
    def read_calib_file(file_path: str) -> dict:
        calib_dict = {}
        with open(file_path, "r") as calib_file:
            for line in calib_file.readlines():
                tokens = line.split(" ")
                if tokens[0] == "calib_time:":
                    continue
                # Only read with float data
                if len(tokens) > 0:
                    values = [float(token) for token in tokens[1:]]
                    values = np.array(values, dtype=np.float32)

                    # The format in KITTI's file is <key>: <f1> <f2> <f3> ...\n -> Remove the ':'
                    key = tokens[0][:-1]
                    calib_dict[key] = values
        return calib_dict

    def tracking_calib_from_txt(calibration_path):
        # borrow from https://github.com/fudan-zvg/PVG/blob/main/scene/kittimot_loader.py
        """
        Extract tracking calibration information from a KITTI tracking calibration file.

        This function reads a KITTI tracking calibration file and extracts the relevant
        calibration information, including projection matrices and transformation matrices
        for camera, LiDAR, and IMU coordinate systems.

        Args:
            calibration_path (str): Path to the KITTI tracking calibration file.

        Returns:
            dict: A dictionary containing the following calibration information:
                P0, P1, P2, P3 (np.array): 3x4 projection matrices for the cameras.
                Tr_cam2camrect (np.array): 4x4 transformation matrix from camera to rectified camera coordinates.
                Tr_velo2cam (np.array): 4x4 transformation matrix from LiDAR to camera coordinates.
                Tr_imu2velo (np.array): 4x4 transformation matrix from IMU to LiDAR coordinates.
        """
        # Read the calibration file
        f = open(calibration_path)
        calib_str = f.read().splitlines()

        # Process the calibration data
        calibs = []
        for calibration in calib_str:
            calibs.append(np.array([kitti_string_to_float(val) for val in calibration.split()[1:]]))

        # Extract the projection matrices
        P0 = np.reshape(calibs[0], [3, 4])
        P1 = np.reshape(calibs[1], [3, 4])
        P2 = np.reshape(calibs[2], [3, 4])
        P3 = np.reshape(calibs[3], [3, 4])

        # Extract the transformation matrix for camera to rectified camera coordinates
        Tr_cam2camrect = np.eye(4)
        R_rect = np.reshape(calibs[4], [3, 3])
        Tr_cam2camrect[:3, :3] = R_rect

        # Extract the transformation matrices for LiDAR to camera and IMU to LiDAR coordinates
        Tr_velo2cam = np.concatenate([np.reshape(calibs[5], [3, 4]), np.array([[0.0, 0.0, 0.0, 1.0]])], axis=0)
        Tr_imu2velo = np.concatenate([np.reshape(calibs[6], [3, 4]), np.array([[0.0, 0.0, 0.0, 1.0]])], axis=0)

        return {
            "P0": P0,
            "P1": P1,
            "P2": P2,
            "P3": P3,
            "Tr_cam2camrect": Tr_cam2camrect,
            "Tr_velo2cam": Tr_velo2cam,
            "Tr_imu2velo": Tr_imu2velo,
        }
    
    def project_points_to_cam(self, points, points_rgb, img, T_c_l, K_mat):
        
        # points as np.numpy (N,4)
        points[:,3] = 1 # homo coordinate

        # transfrom velodyne points to camera coordinate
        points_cam = np.matmul(T_c_l, points.T).T # N, 4
        points_cam = points_cam[:,:3] # N, 3

        # project to image space
        u, v, depth= self.persepective_cam2image(points_cam.T, K_mat) 
        u = u.astype(np.int32)
        v = v.astype(np.int32)

        img_height, img_width, _ = np.shape(img)

        # prepare depth map for visualization
        depth_map = np.zeros((img_height, img_width))
        depth_img = np.zeros((img_height, img_width, 3))
        mask = np.logical_and(np.logical_and(np.logical_and(u>=0, u<img_width), v>=0), v<img_height)
        
        # visualize points within 30 meters
        min_depth = 1.0
        max_depth = 100.0
        mask = np.logical_and(np.logical_and(mask, depth>min_depth), depth<max_depth)
        
        v_valid = v[mask]
        u_valid = u[mask]

        depth_map[v_valid,u_valid] = depth[mask]

        # print(np.shape(points_rgb))

        points_rgb[mask, :3] = img[v_valid,u_valid].astype(np.float64)/255.0 # 0-1
        points_rgb[mask, 3] = 0 # has color

        return points_rgb, depth_map
    
    def persepective_cam2image(self, points, K_mat):
        ndim = points.ndim
        if ndim == 2:
            points = np.expand_dims(points, 0)
        points_proj = np.matmul(K_mat[:3,:3].reshape([1,3,3]), points)
        depth = points_proj[:,2,:]
        depth[depth==0] = -1e-6
        u = np.round(points_proj[:,0,:]/np.abs(depth)).astype(int)
        v = np.round(points_proj[:,1,:]/np.abs(depth)).astype(int)

        if ndim==2:
            u = u[0]; v=v[0]; depth=depth[0]
        return u, v, depth
    
    # from pykitti
    def _load_calib(self):
        """Load and compute intrinsic and extrinsic calibration parameters."""
        # We'll build the calibration parameters as a dictionary, then
        # convert it to a namedtuple to prevent it from being modified later
        data = {}

        filedata = self.calibration

        # Create 3x4 projection matrices
        P_rect_00 = np.reshape(filedata['P0'], (3, 4))
        P_rect_10 = np.reshape(filedata['P1'], (3, 4))
        P_rect_20 = np.reshape(filedata['P2'], (3, 4))
        P_rect_30 = np.reshape(filedata['P3'], (3, 4))

        data['P_rect_00'] = P_rect_00
        data['P_rect_10'] = P_rect_10
        data['P_rect_20'] = P_rect_20
        data['P_rect_30'] = P_rect_30

        # Compute the rectified extrinsics from cam0 to camN
        T1 = np.eye(4)
        T1[0, 3] = P_rect_10[0, 3] / P_rect_10[0, 0]
        T2 = np.eye(4)
        T2[0, 3] = P_rect_20[0, 3] / P_rect_20[0, 0]
        T3 = np.eye(4)
        T3[0, 3] = P_rect_30[0, 3] / P_rect_30[0, 0]

        # Compute the velodyne to rectified camera coordinate transforms
        data['T_cam0_velo'] = np.reshape(filedata['Tr'], (3, 4))
        data['T_cam0_velo'] = np.vstack([data['T_cam0_velo'], [0, 0, 0, 1]])
        data['T_cam1_velo'] = T1.dot(data['T_cam0_velo'])
        data['T_cam2_velo'] = T2.dot(data['T_cam0_velo'])
        data['T_cam3_velo'] = T3.dot(data['T_cam0_velo'])

        # Compute the camera intrinsics
        data['K_cam0'] = P_rect_00[0:3, 0:3]
        data['K_cam1'] = P_rect_10[0:3, 0:3]
        data['K_cam2'] = P_rect_20[0:3, 0:3]
        data['K_cam3'] = P_rect_30[0:3, 0:3]

        # Compute the stereo baselines in meters by projecting the origin of
        # each camera frame into the velodyne frame and computing the distances
        # between them
        p_cam = np.array([0, 0, 0, 1])
        p_velo0 = np.linalg.inv(data['T_cam0_velo']).dot(p_cam)
        p_velo1 = np.linalg.inv(data['T_cam1_velo']).dot(p_cam)
        p_velo2 = np.linalg.inv(data['T_cam2_velo']).dot(p_cam)
        p_velo3 = np.linalg.inv(data['T_cam3_velo']).dot(p_cam)

        data['b_gray'] = np.linalg.norm(p_velo1 - p_velo0)  # gray baseline
        data['b_rgb'] = np.linalg.norm(p_velo3 - p_velo2)   # rgb baseline

        return data
    
     ### FROM THIS POINT EVERYTHING IS COPY PASTED FROM PYKITTI
    @staticmethod
    def transform_from_rot_trans(R, t):
        """Transforation matrix from rotation matrix and translation vector."""
        R = R.reshape(3, 3)
        t = t.reshape(3, 1)
        return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))
    
    @staticmethod
    def pose_from_oxts_packet(packet, scale):
        """Helper method to compute a SE(3) pose matrix from an OXTS packet."""

        def rotx(t):
            """Rotation about the x-axis."""
            c = np.cos(t)
            s = np.sin(t)
            return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])

        def roty(t):
            """Rotation about the y-axis."""
            c = np.cos(t)
            s = np.sin(t)
            return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])

        def rotz(t):
            """Rotation about the z-axis."""
            c = np.cos(t)
            s = np.sin(t)
            return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

        er = 6378137.0  # earth radius (approx.) in meters

        # Use a Mercator projection to get the translation vector
        tx = scale * packet.lon * np.pi * er / 180.0
        ty = scale * er * np.log(np.tan((90.0 + packet.lat) * np.pi / 360.0))
        tz = packet.alt
        t = np.array([tx, ty, tz])

        # Use the Euler angles to get the rotation matrix
        Rx = rotx(packet.roll)
        Ry = roty(packet.pitch)
        Rz = rotz(packet.yaw)
        R = Rz.dot(Ry.dot(Rx))

        # Combine the translation and rotation into a homogeneous transform
        return R, t
    
    def postprocess_oxts_poses(poses_in):
        """ convert coordinate system from
        #   x=forward, y=right, z=down 
        # to
        #   x=forward, y=left, z=up
        """

        R = np.array([[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]])
        
        poses  = []
        
        for i in range(len(poses_in)):
            # if there is no data => no pose
            if not len(poses_in[i]):
                poses.append([])
                continue
            P = poses_in[i]
            poses.append(np.matmul(R, P.T).T )
        
        return poses

    def load_oxts_packets_and_poses(self, oxts_files):
        """Generator to read OXTS ground truth data.

        Poses are given in an East-North-Up coordinate system
        whose origin is the first GPS position.

        GPS/IMU 3D localization unit
        ============================

        The GPS/IMU information is given in a single small text file which is
        written for each synchronized frame. Each text file contains 30 values
        which are:

          - lat:     latitude of the oxts-unit (deg)
          - lon:     longitude of the oxts-unit (deg)
          - alt:     altitude of the oxts-unit (m)
          - roll:    roll angle (rad),  0 = level, positive = left side up (-pi..pi)
          - pitch:   pitch angle (rad), 0 = level, positive = front down (-pi/2..pi/2)
          - yaw:     heading (rad),     0 = east,  positive = counter clockwise (-pi..pi)
          - vn:      velocity towards north (m/s)
          - ve:      velocity towards east (m/s)
          - vf:      forward velocity, i.e. parallel to earth-surface (m/s)
          - vl:      leftward velocity, i.e. parallel to earth-surface (m/s)
          - vu:      upward velocity, i.e. perpendicular to earth-surface (m/s)
          - ax:      acceleration in x, i.e. in direction of vehicle front (m/s^2)
          - ay:      acceleration in y, i.e. in direction of vehicle left (m/s^2)
          - az:      acceleration in z, i.e. in direction of vehicle top (m/s^2)
          - af:      forward acceleration (m/s^2)
          - al:      leftward acceleration (m/s^2)
          - au:      upward acceleration (m/s^2)
          - wx:      angular rate around x (rad/s)
          - wy:      angular rate around y (rad/s)
          - wz:      angular rate around z (rad/s)
          - wf:      angular rate around forward axis (rad/s)
          - wl:      angular rate around leftward axis (rad/s)
          - wu:      angular rate around upward axis (rad/s)
          - posacc:  velocity accuracy (north/east in m)
          - velacc:  velocity accuracy (north/east in m/s)
          - navstat: navigation status
          - numsats: number of satellites tracked by primary GPS receiver
          - posmode: position mode of primary GPS receiver
          - velmode: velocity mode of primary GPS receiver
          - orimode: orientation mode of primary GPS receiver

        To read the text file and interpret them properly an example is given in
        the matlab folder: First, use oxts = loadOxtsliteData('2011_xx_xx_drive_xxxx')
        to read in the GPS/IMU data. Next, use pose = convertOxtsToPose(oxts) to
        transform the oxts data into local euclidean poses, specified by 4x4 rigid
        transformation matrices. For more details see the comments in those files.

        """
        # Per dataformat.txt
        OxtsPacket = namedtuple(
            "OxtsPacket",
            "lat, lon, alt, "
            + "roll, pitch, yaw, "
            + "vn, ve, vf, vl, vu, "
            + "ax, ay, az, af, al, au, "
            + "wx, wy, wz, wf, wl, wu, "
            + "pos_accuracy, vel_accuracy, "
            + "navstat, numsats, "
            + "posmode, velmode, orimode",
        )

        # Bundle into an easy-to-access structure
        OxtsData = namedtuple("OxtsData", "packet, T_w_imu")
        # Scale for Mercator projection (from first lat value)
        scale = None
        # Origin of the global coordinate system (first GPS position)
        origin = None

        oxts = []
        T_w_imu_poses = []

        for filename in oxts_files:
            with open(filename, "r") as f:
                for line in f.readlines():
                    line = line.split()
                    # Last five entries are flags and counts
                    line[:-5] = [float(x) for x in line[:-5]]
                    line[-5:] = [int(float(x)) for x in line[-5:]]

                    packet = OxtsPacket(*line)

                    if scale is None:
                        scale = np.cos(packet.lat * np.pi / 180.0)

                    R, t = self.pose_from_oxts_packet(packet, scale)

                    if origin is None:
                        origin = t

                    T_w_imu = self.transform_from_rot_trans(R, t)
                    T_w_imu_poses.append(T_w_imu)

                    # print(T_w_imu)

                    oxts.append(OxtsData(packet, T_w_imu))

        # imu frame definition is different from original KITTI_raw
        # convert coordinate system from
        #   x=forward, y=right, z=down 
        # to
        #   x=forward, y=left, z=up
        tran_mat = np.array([[1,0,0,0], [0,-1,0,0], [0,0,-1,0], [0,0,0,1]])
        T_w_imu_poses = T_w_imu_poses @ tran_mat

        # # Start from identity
        first_pose = T_w_imu_poses[0]
        T_w_imu_poses = np.linalg.inv(first_pose) @ T_w_imu_poses
        
        return oxts, T_w_imu_poses
    
