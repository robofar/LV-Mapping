## How to do the experiments in simulation

1. Launch the [simulation environment](https://www.cmu-exploration.com/)

```
cd ~/autonomous_exploration_development_environment
source devel/setup.sh
roslaunch vehicle_simulator system_garage.launch
... ...
roslaunch vehicle_simulator system_[xxx_environment].launch
```

2. Launch PIN-SLAM

```
cd /path/to/pin-slam

python3 pin_slam_ros.py ./config/lidar_slam/run_explore.yaml /velodyne_points

```

In another terminal, turn on Rviz for PIN

```
rviz -d ./config/pin_slam_ros.rviz 
```

3. Lauch the explorer ([TARE planner](https://www.cmu-exploration.com/tare-planner))
```
cd ~/tare_planner
source devel/setup.sh
roslaunch tare_planner explore_garage.launch
... ... 
roslaunch tare_planner explore_[xxx_environment].launch
```

4. More links to CMU-Explorer



## TODO list

- [ ] Let the simulator and planner to subscribe to the pose estimated by PIN-SLAM
- [ ] Local planning and obstacle avoidance based on PIN SDF (may consider to use with this [planner](https://github.com/ethz-asl/mav_voxblox_planning))
- [ ] Merge Rviz for PIN-SLAM and the planner, add local mesh visualization for PIN
- [ ] Implement free-space neural points
- [ ] Support other simulators