
# PINGS

For development, please create your own branch (for example, `dev/faris`). 

For questions and proposals, please raise issues or contact Yue and Starry via WhatsApp.

## Requirements

1. follow the installation instruction of PIN-SLAM

2. in PIN-SLAM's conda environment, try to install the other dependencies needed for [2DGS](https://github.com/hbb1/2d-gaussian-splatting), mainly `submodules/diff-surfel-rasterization`. Please follow 2D-GS's installation instruction:

```
cd ..
git clone https://github.com/hbb1/2d-gaussian-splatting.git --recursive
cd 2d-gaussian-splatting
pip install submodules/diff-surfel-rasterization
```

We may change this to [gaussian-surfels](https://github.com/turandai/gaussian_surfels) later.

3. Prepare the KITTI dataset, KITTI-360 dataset and Nuscenes dataset. You may directly contact Yue to get the dataset in the required structure.

### Visualization

you can visualize the neural gaussians as point cloud in the visualizer by pressing `P`. For the other instructions for visualizer, please refer to the readme file


## Run on KITTI dataset

we are using the kitti odometry dataset here, you need `image_2` and `velodyne` folder in your sequence base folder

```
python pin_slam.py ./config/lidar_slam/run_kitti_gs.yaml kitti 00 -i ./data/kitti/ -dvl

python pin_slam.py ./config/lidar_slam/run_kitti_gs.yaml kitti 04 -i ./data/kitti/ -dvl
```


## Run on KITTI 360 dataset

```
python pin_slam.py ./config/lidar_slam/run_kitti360.yaml kitti360 00 -i ./data/kitti360/ -dvl

python pin_slam.py ./config/lidar_slam/run_kitti360.yaml kitti360 03 -i ./data/kitti360/ -dvl
```


## Run on Nuscenes dataset (under development)

```
python pin_slam.py ./config/lidar_slam/run_fast.yaml nuscenes 0061 -i ./data/nuscenes/v1.0-mini/ -dvl

python pin_slam.py ./config/lidar_slam/run_fast.yaml nuscenes 0655 -i ./data/nuscenes/v1.0-mini/ -dvl
```

### available scenes for Nuscenes

for the `v1.0-mini` split

```
scene-0061, Parked truck, construction, intersectio... [18-07-24 03:28:47]   19s, singapore-onenorth, #anns:4622
scene-0103, Many peds right, wait for turning car, ... [18-08-01 19:26:43]   19s, boston-seaport, #anns:2046
scene-0655, Parking lot, parked cars, jaywalker, be... [18-08-27 15:51:32]   20s, boston-seaport, #anns:2332
scene-0553, Wait at intersection, bicycle, large tr... [18-08-28 20:48:16]   20s, boston-seaport, #anns:1950
scene-0757, Arrive at busy intersection, bus, wait ... [18-08-30 19:25:08]   20s, boston-seaport, #anns:592
scene-0796, Scooter, peds on sidewalk, bus, cars, t... [18-10-02 02:52:24]   20s, singapore-queensto, #anns:708
scene-0916, Parking lot, bicycle rack, parked bicyc... [18-10-08 07:37:13]   20s, singapore-queensto, #anns:2387
scene-1077, Night, big street, bus stop, high speed... [18-11-21 11:39:27]   20s, singapore-hollandv, #anns:890
scene-1094, Night, after rain, many peds, PMD, ped ... [18-11-21 11:47:27]   19s, singapore-hollandv, #anns:1762
scene-1100, Night, peds in sidewalk, peds cross cro... [18-11-21 11:49:47]   19s, singapore-hollandv, #anns:935
```

### notes on nuscenes dataset

+ only the sample keyframes (per 0.5s / 2Hz) have the anotation
+ images have different frequency as the Lidar frames, has lower frequency and is not as constant, we need to somehow associate the imgs to the Lidar frames
+ for the keyframes, all the sensor data are available
+ only 20 seconds for a scene (40 keyframes, 400 lidar frames, fewer img frames)
+ timestamps and pose are available for each measurement of each sensor

## Run on Waymo dataset (under development)


## Run on VBR dataset

```
python pin_slam.py ./config/lidar_slam/run_vbr_gs.yaml vbr -i ./data/vbr/vbr_slam/spagna/spagna_train0/kitti_format/ -dvl
```

## Run on TUM RGBD dataset

```
python pin_slam.py ./config/rgbd_slam/run_tum_gs.yaml tum rgbd_dataset_freiburg1_desk -i ./data/TUM/ -dvl
```

## Run on Replica RGBD dataset

```
python pin_slam.py ./config/rgbd_slam/run_replica_gs.yaml replica -i ./data/Replica/room0 -dvl

python pin_slam.py ./config/rgbd_slam/run_replica_gs.yaml replica -i ./data/Replica/room0 -dvl --range 0 2000 5

```

... ...

## TODO List

- [ ] Add pruning
- [ ] Keyframe strategy
- [ ] Add SDF / SDF gradient consistency loss
- [ ] Support multi-cam datasets
- [ ] Deal with dynamic objects
- [ ] Evaluate Chamfer distance and PSNR


## Useful links
+ [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting)

+ [2D-GS](https://github.com/hbb1/2d-gaussian-splatting)

+ [Gaussian-Surfels](https://github.com/turandai/gaussian_surfels)

+ [MonoGS](https://github.com/muskie82/MonoGS)

+ [GS-ICP-SLAM](https://github.com/Lab-of-AI-and-Robotics/GS_ICP_SLAM)

+ [Street-Gaussians](https://github.com/LightwheelAI/street-gaussians-ns)

+ [Scaffold-GS](https://github.com/city-super/Scaffold-GS)

+ [PIN-SLAM](https://github.com/PRBonn/PIN_SLAM)

+ [SHINE-Mapping](https://github.com/PRBonn/SHINE_mapping)

Please add more links here