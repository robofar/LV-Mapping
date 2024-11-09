python pings.py ./config/lidar_slam/run_ipbcar_gs_debug.yaml ipb_car -i ./data/ipb_car/2024-04-30_cheap_car/extracted/ -sg --range 2030 2050 1 --tag sanity_1109
python pings.py ./config/lidar_slam/run_ipbcar_gs.yaml ipb_car -i ./data/ipb_car/2024-04-30_cheap_car/extracted/ -sg --range 2030 2180 1 --tag church_1109
python pings.py ./config/lidar_slam/run_ipbcar_gs_no_consistency.yaml ipb_car -i ./data/ipb_car/2024-04-30_cheap_car/extracted/ -sg --range 2030 2180 1 --tag church_1109
python pings.py ./config/lidar_slam/run_ipbcar_gs.yaml ipb_car -i ./data/ipb_car/2024-04-30_cheap_car/extracted/ -sg --range 1850 2000 1 --tag roundabout_1109
python pings.py ./config/lidar_slam/run_ipbcar_gs_no_consistency.yaml ipb_car -i ./data/ipb_car/2024-04-30_cheap_car/extracted/ -sg --range 1850 2000 1 --tag roundabout_1109
python pings.py ./config/lidar_slam/run_ipbcar_gs.yaml ipb_car -i ./data/ipb_car/2024-04-30_cheap_car/extracted/ -sg --range 550 700 1 --tag neighbor_1109
python pings.py ./config/lidar_slam/run_ipbcar_gs_no_consistency.yaml ipb_car -i ./data/ipb_car/2024-04-30_cheap_car/extracted/ -sg --range 550 700 1 --tag neighbor_1109
python pings.py ./config/lidar_slam/run_ipbcar_gs.yaml ipb_car -i ./data/ipb_car/2024-04-30_cheap_car/extracted/ -sg --range 100 250 1 --tag math_1109
python pings.py ./config/lidar_slam/run_ipbcar_gs_no_consistency.yaml ipb_car -i ./data/ipb_car/2024-04-30_cheap_car/extracted/ -sg --range 100 250 1 --tag math_1109
python pings.py ./config/lidar_slam/run_ipbcar_gs.yaml ipb_car -i ./data/ipb_car/2024-04-30_cheap_car/extracted/ -sg --range 2650 2800 1 --tag seba_1109
python pings.py ./config/lidar_slam/run_ipbcar_gs_no_consistency.yaml ipb_car -i ./data/ipb_car/2024-04-30_cheap_car/extracted/ -sg --range 2650 2800 1 --tag seba_1109
# python pings.py ./config/lidar_slam/run_ipbcar_gs.yaml ipb_car -i ./data/ipb_car/2024-04-30_cheap_car/extracted/ -sg --range 8350 8500 1
# python pings.py ./config/lidar_slam/run_ipbcar_gs_no_consistency.yaml ipb_car -i ./data/ipb_car/2024-04-30_cheap_car/extracted/ -sg --range 8350 8500 1