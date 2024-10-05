
# PINGS

For development, please create your own branch (for example, `dev/faris`). 

For questions and proposals, please raise issues or contact Yue and Starry via WhatsApp.

## Install

Just Follow the installation instruction of PIN-SLAM in `readme.md`.

## TO Use
Prepare the KITTI dataset, KITTI-360 dataset and Nuscenes dataset. You may directly contact Yue to get the dataset in the required structure.

Add the `-s` flag to saved the resulting GS map in `.ply` format for offline inspection and evaluation 

### Visualization

you can visualize the neural gaussians as point cloud in the visualizer by pressing `P`. For the other instructions for visualizer, please refer to the readme file


## Run on KITTI dataset

we are using the kitti odometry dataset here, you need `image_2` and `velodyne` folder in your sequence base folder

```
python pin_slam.py ./config/lidar_slam/run_kitti_gs.yaml kitti 00 -i ./data/kitti/ -vl

python pin_slam.py ./config/lidar_slam/run_kitti_gs.yaml kitti 04 -i ./data/kitti/ -vl

python pin_slam.py ./config/lidar_slam/run_kitti_gs.yaml kitti 07 -i ./data/kitti/ -vl
```

**pass**


## Run on KITTI 360 dataset

```
python pin_slam.py ./config/lidar_slam/run_kitti_gs.yaml kitti360 00 -i ./data/kitti360/ -dvl --deskew

python pin_slam.py ./config/lidar_slam/run_kitti360_gs.yaml kitti360 00 -i ./data/kitti360/ -dvl

python pin_slam.py ./config/lidar_slam/run_kitti360_gs.yaml kitti360 03 -i ./data/kitti360/ -dvl
```

**pass**

## Run on KITTI MOT dataset

```
python pin_slam.py ./config/lidar_slam/run_kitti_mot_gs.yaml kitti_mot 02 -i ./data/kitti/kitti_mot/training -vl

```

Used by PVG, frame 380 to 431 on sequence 01

**pass**

## Run on VKITTI synthetic dataset

```
python pin_slam.py ./config/lidar_slam/run_vkitti_gs.yaml vkitti 01 -i ./data/vkitti2 -dvl

```


**pass**


## Run on Nuscenes dataset
```
python pin_slam.py ./config/lidar_slam/run_nuscenes_gs.yaml nuscenes 0061 -i ./data/nuscenes/v1.0-mini/ -dvl


python pin_slam.py ./config/lidar_slam/run_nuscenes_gs.yaml nuscenes 0061 -i ./data/nuscenes/v1.0-mini/ -dvl

```

**pass**

<details>
  <summary>[Details (click to expand)]</summary>

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

</details>

## Run on Waymo dataset

[Dataset download link](https://console.cloud.google.com/storage/browser/waymo_open_dataset_v_1_4_0;tab=objects?pli=1&prefix=&forceOnObjectsSortingFiltering=false)

[Preproceesed download link for 4 sequences used by street-gs](https://github.com/LightwheelAI/street-gaussians-ns)


```
python pin_slam.py ./config/lidar_slam/run_waymo_gs.yaml waymo -i ./data/waymo/waymo_10588771936253546636_2300_000_2320_000/10588771936253546636_2300_000_2320_000/ -vl

python pin_slam.py ./config/lidar_slam/run_waymo_gs.yaml waymo -i ./data/waymo/waymo_8398516118967750070_3958_000_3978_000/8398516118967750070_3958_000_3978_000/ -vl

python pin_slam.py ./config/lidar_slam/run_waymo_gs.yaml waymo -i ./data/waymo/waymo_10448102132863604198_472_000_492_000/10448102132863604198_472_000_492_000/ -vl

python pin_slam.py ./config/lidar_slam/run_waymo_gs.yaml waymo -i ./data/waymo/waymo_2094681306939952000_2972_300_2992_300/2094681306939952000_2972_300_2992_300/ -vl
```

**pass**

## Run on IPB car dataset

```
python pin_slam.py ./config/lidar_slam/run_ipbcar_gs.yaml ipb_car2 -i ./data/ipb_car/2023-06-13-new_setup_long_recording/kitti_format/ -vl

python pin_slam.py ./config/lidar_slam/run_ipbcar_gs.yaml ipb_car2 -i ./data/ipb_car/2023-06-13-new_setup_long_recording/kitti_format/ -vl --range 600 1000 1

python pin_slam.py ./config/lidar_slam/run_ipbcar_gs.yaml ipb_car -i ./data/ipb_car/pings_test_seq/ -vl --range 50 150 1
```

The light across different cameras seem to be not identical, figure out a way to do the correction

**pass**

## Run on R3Live dataset

```
python pin_slam.py ./config/lidar_slam/run_r3live_gs.yaml r3live -i ./data/r3live/hku_campus_seq_00_kitti_format/ -dvls
```

**pass**

## Run on VBR dataset

```
python pin_slam.py ./config/lidar_slam/run_vbr_gs.yaml vbr -i ./data/vbr/vbr_slam/spagna/spagna_train0/kitti_format/ -dvl
```
**pass**

## Run on TUM RGBD dataset

```
python pin_slam.py ./config/rgbd_slam/run_tum_gs.yaml tum rgbd_dataset_freiburg1_desk -i ./data/TUM/ -dvl
```

**pass**

## Run on Bonn RGBD dataset
```
python pin_slam.py ./config/rgbd_slam/run_bonn_gs.yaml bonn -i ./data/bonn_rgbd/rgbd_bonn_static/ -dvl
```

**pass**

## Run on Replica RGBD dataset

```
python pin_slam.py ./config/rgbd_slam/run_replica_gs.yaml replica -i ./data/Replica/room0 -dvl

python pin_slam.py ./config/rgbd_slam/run_replica_gs.yaml replica -i ./data/Replica/room0 -dvl --range 0 2000 5

```
**pass**

## Run on Neural RGBD dataset

```
python pin_slam.py ./config/rgbd_slam/run_neuralrgbd_gs.yaml neuralrgbd -i ./data/neural_rgbd_data/breakfast_room/ -dvl 
```

**pass**

## Run on Azure dataset 

provided by RTG-SLAM

```
python pin_slam.py ./config/rgbd_slam/run_azure_gs.yaml azure -i ./data/azure/hotel -dvl
```

## Run on CKA greenhouse dataset
```
python pin_slam.py ./config/rgbd_slam/run_cka_pepper.yaml cka -i ./data/CKA/shape_completion_challenge/test/p1/input/ -dvl

python pin_slam.py ./config/rgbd_slam/run_cka_pepper.yaml cka -i ./data/CKA/CKA_fruit/processed/2022_09_05/row1/before/realsense/ -dvl
```

**pass**
... ...


## Mono depth prediction by depth anything

```
git clone https://github.com/DepthAnything/Depth-Anything-V2
cd Depth-Anything-V2/metric_depth

# download the model checkpoints

# example for kitti dataset

# rgb2depth
python run.py \
--encoder vits \
--load-from ./checkpoints/depth_anything_v2_metric_vkitti_vits.pth \
--max-depth 80 \
--img-path ../../PINGS/data/kitti/sequences/04/image_2/ \
--outdir ../../PINGS/data/kitti/sequences/04/image_2_monodepth/ \
--save-numpy \
--pred-only


# img2ply
python depth_to_pointcloud.py \
--encoder vits \
--load-from ./checkpoints/depth_anything_v2_metric_vkitti_vits.pth \
--max-depth 80 \
--img-path ../../PINGS/data/kitti/sequences/04/image_2/ \
--outdir ../../PINGS/data/kitti/sequences/04/image_2_ply/ \
--focal-length-x 718.856 \
--focal-length-y 718.856 

```

Now we switch to metric3D, actually works pretty well

### Sample more freespace point and then we can build static map
Then you can actually use this static map to prune those neural points (gaussians) in the freespace
Prune non-free neural points with too large SDF value 

Or maybe we can use MOS results, how to save the MOS results using Ben's MapMOS?

### Place recognition

Now we can use the visual (image) data for place recognition. We can even use those rendered views.

We may consider some visual place recognition methods such as NetVLAD or pyBoW. Try these later.

### Notes
Gaussian surfels seems to be 1.5x faster than 2DGS. However, they do not have the distortion loss directlt implemented.

How to run even faster?

1. enable GPU performance mode (prefer maximum perfromance)
2. export CUDA_LAUNCH_BLOCKING=0
3. export TORCH_USE_CUDA_MEMORY_POOL=0
4. export TORCH_CUDA_DEBUG=0
5. export CUDA_ASSERT=0
6. export CUDA_MEMCHECK=0


*fix the CUDA illegal access issue*
NaN problem in rotation

```
# correct the potenitial nan numbers containing in rotation during the initialization
new_rots = torch.nan_to_num(new_rots, 0, 0)
```

### Set used GPU on server
```
export CUDA_VISIBLE_DEVICES=2
```

## GS Visualizer

[Three.js Viewer](https://projects.markkellogg.org/threejs/demo_gaussian_splats_3d.php)

[WebGL Viewer](https://antimatter15.com/splat/)

[PolyCam Viewer](https://poly.cam/tools/gaussian-splatting)

[SuperSplat Viewer](https://playcanvas.com/supersplat/editor)

## TODO List

- [ ] Add pruning (together with the moving objects)
- [ ] Keyframe strategy
- [x] Figure out the issue of the normal loss, does not work properly
- [ ] Use the dense rendered depth from the PIN map (mesh or SDF) as the depth supervision instead of the raw LiDAR measurement, try this, almost your final chance
- [ ] Gaussians update after PGO pose correction
- [x] Make use of the mono normal clue in training, also use mono normal as the initialization of the gaussians
- [x] Figure out how is the rendered normal calculated (and is it proper to directly optimize the surfel normals in 3D?), normal at the accumulated alpha = 0.5
- [x] Batch mode
- [x] Add SDF / SDF gradient consistency loss, Gaussian, PIN jointly optimization
- [x] Depth rendering loss (optional)
- [x] Support multi-cam datasets
- [x] Add mono depth estimation
- [x] Use depth map to do TSDF fusion to generate the refined mesh
- [x] Check RTG-SLAM (opacity in RTG-SLAM are fixed as either 0.99 or 0.1)
- [ ] Allow the freee gaussians to move freely with a larger learning rate, but then you need to have two parameter lists
- [x] Add online gaussian visualizer, check MonoGS
- [x] Merge the two visualizers into one
- [x] Figure out what the intrinsic and extrinsic of the R3Live dataset, add dataloader
- [ ] Figure out what the intrinsic and extrinsic of the BotanicGarden dataset, add dataloader
- [x] New KITTI-MOT dataloader
- [ ] New Waymo dataloader
- [ ] KITTI-360 NVS benchmark
- [ ] Deal with dynamic objects (tracking, filtering or 4DGS)
- [x] Sky (out-of-lidar-fov) masking
- [ ] Level of details, especially for the closer range
- [ ] Evaluate Chamfer distance and PSNR
- [x] Think about better way to predict more gaussians from the neural point in a memory efficient way (like ScaffoldGS) TODO, DO IT, MLP predict gaussian parameters, for color, also learn a residual
- [ ] Opacity regularization loss
- [ ] Only spawn the points in the frustum
- [x] Prune dynamic Gaussians using the PIN Map
- [ ] Add BoW python for image place recognition (loop detection)
- [ ] Add IMU, make the pose estimation more robust, use Pinochio, but add the new IMU data loader
- [ ] Think about if we can do something to refine the monodepth estimation using the already built map (like Double Take)
- [ ] Add the opacity entropy loss, either 0 or 1
- [ ] Normal smoothness loss
- [x] Stop visualizer operation with a checkbox (not let the training get so slow)
- [ ] Memory mechanism to tackle forgetting
- [ ] Maybe have sperate color and geometry feature
- [x] Try GaussianSurfel instead of 2D GS
- [ ] Exposion compensation (refer to R3Live, FastLIVO, etc.), would be very useful and important
- [ ] The goal is RAL or ISPRS Journal

## Useful links
+ [3D-GS](https://github.com/graphdeco-inria/gaussian-splatting)

+ [2D-GS](https://github.com/hbb1/2d-gaussian-splatting)

+ [Gaussian-Surfels](https://github.com/turandai/gaussian_surfels)

+ [2.5D-GS](https://github.com/hugoycj/2.5d-gaussian-splatting)

+ [MonoGS](https://github.com/muskie82/MonoGS)

+ [RTG-SLAM](https://github.com/MisEty/RTG-SLAM)

+ [GS-ICP-SLAM](https://github.com/Lab-of-AI-and-Robotics/GS_ICP_SLAM)

+ [Street-Gaussians](https://github.com/zju3dv/street_gaussians)

+ [Street-Gaussians-NerfStudio](https://github.com/LightwheelAI/street-gaussians-ns)

+ [Scaffold-GS](https://github.com/city-super/Scaffold-GS)

+ [Driving Gaussians](https://github.com/YuePanEdward/DrivingGS)

+ [GaussianPro](https://github.com/kcheng1021/GaussianPro)

+ [Hierarchical 3DGS](https://github.com/graphdeco-inria/hierarchical-3d-gaussians)

+ [GSDF](https://github.com/city-super/GSDF)

+ [GOF](https://github.com/autonomousvision/gaussian-opacity-fields)

+ [PGSR](https://github.com/zju3dv/PGSR)

+ [NeuSG](https://arxiv.org/pdf/2312.00846)

+ [RaDe-GS](https://baowenz.github.io/radegs/)

+ [3D GS with depth rendering](https://github.com/ingra14m/depth-diff-gaussian-rasterization)

+ [MipGS](https://github.com/autonomousvision/mip-splatting)

+ [GauStudio](https://github.com/GAP-LAB-CUHK-SZ/gaustudio)

+ [NerfStudio-Splatfacto](https://docs.nerf.studio/nerfology/methods/splat.html)

+ [LargeScale_3DGS](https://github.com/DeepLabc/LargeScale_3DGS)

+ [MARS: Nerf-based Simulation](https://github.com/OPEN-AIR-SUN/mars)

+ [NerualSim](https://github.com/PJLab-ADG/neuralsim)

+ [PVG](https://github.com/fudan-zvg/PVG/)

+ [OmniRe Driving Studio](https://github.com/ziyc/drivestudio)

+ [GS4Robo](https://github.com/dtc111111/awesome-3dgs-for-robotics)

+ [Grounded-SAM2](https://github.com/IDEA-Research/Grounded-SAM-2)

+ [R3Live](https://github.com/hku-mars/r3live)

+ [ImMesh](https://github.com/hku-mars/ImMesh)

+ [BotanicGardenDataset](https://github.com/robot-pesg/BotanicGarden/tree/main/leaderboard)

+ [PIN-SLAM](https://github.com/PRBonn/PIN_SLAM)

+ [SHINE-Mapping](https://github.com/PRBonn/SHINE_mapping)

Please add more links here