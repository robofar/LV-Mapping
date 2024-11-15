# exp_tag=1109
exp_tag=1113_4

# echo "wait 5 hours ..."

# # TODO: delete
# sleep 5h

# # Your commands after the wait
# echo "5 hours have passed. Continuing script..."

# echo "Begin with a quick sanity test"
# python pings.py ./config/lidar_slam/run_ipbcar_gs_debug.yaml ipb_car -i ./data/ipb_car/2024-04-30_cheap_car/extracted/ -sg --range 2030 2045 1 --tag sanity_${exp_tag}

# echo "Begin the mapping on 5 IPB car sequences"

# # python pings.py ./config/lidar_slam/run_ipbcar_gs_full_res.yaml ipb_car -i ./data/ipb_car/2024-04-30_cheap_car/extracted/ -sgm --range 2030 2180 1 --tag church_${exp_tag}
# python pings.py ./config/lidar_slam/run_ipbcar_gs.yaml ipb_car -i ./data/ipb_car/2024-04-30_cheap_car/extracted/ -sg --range 2030 2180 1 --tag church_${exp_tag}
# # python pings.py ./config/lidar_slam/run_ipbcar_gs_2.yaml ipb_car -i ./data/ipb_car/2024-04-30_cheap_car/extracted/ -sgm --range 2030 2180 1 --tag church_${exp_tag}
# # python pings.py ./config/lidar_slam/run_ipbcar_gs_3dgs_cons.yaml ipb_car -i ./data/ipb_car/2024-04-30_cheap_car/extracted/ -sgm --range 2030 2180 1 --tag church_${exp_tag}
# # # python pings.py ./config/lidar_slam/run_ipbcar_gs_no_consistency.yaml ipb_car -i ./data/ipb_car/2024-04-30_cheap_car/extracted/ -sg --range 2030 2180 1 --tag church_${exp_tag}
# # python pings.py ./config/lidar_slam/run_ipbcar_gs_3dgs.yaml ipb_car -i ./data/ipb_car/2024-04-30_cheap_car/extracted/ -sg --range 2030 2180 1 --tag church_${exp_tag}

# # python pings.py ./config/lidar_slam/run_ipbcar_gs_full_res.yaml ipb_car -i ./data/ipb_car/2024-04-30_cheap_car/extracted/ -sgm --range 1850 2000 1 --tag roundabout_${exp_tag}
# python pings.py ./config/lidar_slam/run_ipbcar_gs.yaml ipb_car -i ./data/ipb_car/2024-04-30_cheap_car/extracted/ -sg --range 1850 2000 1 --tag roundabout_${exp_tag}
# # python pings.py ./config/lidar_slam/run_ipbcar_gs_2.yaml ipb_car -i ./data/ipb_car/2024-04-30_cheap_car/extracted/ -sgm --range 1850 2000 1 --tag roundabout_${exp_tag}
# # python pings.py ./config/lidar_slam/run_ipbcar_gs_3dgs_cons.yaml ipb_car -i ./data/ipb_car/2024-04-30_cheap_car/extracted/ -sgm --range 1850 2000 1 --tag roundabout_${exp_tag}
# # # # python pings.py ./config/lidar_slam/run_ipbcar_gs_no_consistency.yaml ipb_car -i ./data/ipb_car/2024-04-30_cheap_car/extracted/ -sg --range 1850 2000 1 --tag roundabout_${exp_tag}
# # # # python pings.py ./config/lidar_slam/run_ipbcar_gs_3dgs.yaml ipb_car -i ./data/ipb_car/2024-04-30_cheap_car/extracted/ -sg --range 1850 2000 1 --tag roundabout_${exp_tag}

# # python pings.py ./config/lidar_slam/run_ipbcar_gs_full_res.yaml ipb_car -i ./data/ipb_car/2024-04-30_cheap_car/extracted/ -sgm --range 550 700 1 --tag neighbor_${exp_tag}
# python pings.py ./config/lidar_slam/run_ipbcar_gs.yaml ipb_car -i ./data/ipb_car/2024-04-30_cheap_car/extracted/ -sg --range 550 700 1 --tag neighbor_${exp_tag}
# # python pings.py ./config/lidar_slam/run_ipbcar_gs_2.yaml ipb_car -i ./data/ipb_car/2024-04-30_cheap_car/extracted/ -sgm --range 550 700 1 --tag neighbor_${exp_tag}
# # python pings.py ./config/lidar_slam/run_ipbcar_gs_3dgs_cons.yaml ipb_car -i ./data/ipb_car/2024-04-30_cheap_car/extracted/ -sgm --range 550 700 1 --tag neighbor_${exp_tag}
# # # # python pings.py ./config/lidar_slam/run_ipbcar_gs_no_consistency.yaml ipb_car -i ./data/ipb_car/2024-04-30_cheap_car/extracted/ -sg --range 550 700 1 --tag neighbor_${exp_tag}
# # # # python pings.py ./config/lidar_slam/run_ipbcar_gs_3dgs.yaml ipb_car -i ./data/ipb_car/2024-04-30_cheap_car/extracted/ -sg --range 550 700 1 --tag neighbor_${exp_tag}

# # python pings.py ./config/lidar_slam/run_ipbcar_gs_full_res.yaml ipb_car -i ./data/ipb_car/2024-04-30_cheap_car/extracted/ -sg --range 100 250 1 --tag math_${exp_tag}
# python pings.py ./config/lidar_slam/run_ipbcar_gs.yaml ipb_car -i ./data/ipb_car/2024-04-30_cheap_car/extracted/ -sg --range 100 250 1 --tag math_${exp_tag}
# # python pings.py ./config/lidar_slam/run_ipbcar_gs_2.yaml ipb_car -i ./data/ipb_car/2024-04-30_cheap_car/extracted/ -sgm --range 100 250 1 --tag math_${exp_tag}
# # python pings.py ./config/lidar_slam/run_ipbcar_gs_3dgs_cons.yaml ipb_car -i ./data/ipb_car/2024-04-30_cheap_car/extracted/ -sgm --range 100 250 1 --tag math_${exp_tag}
# # # # python pings.py ./config/lidar_slam/run_ipbcar_gs_no_consistency.yaml ipb_car -i ./data/ipb_car/2024-04-30_cheap_car/extracted/ -sg --range 100 250 1 --tag math_${exp_tag}
# # # # python pings.py ./config/lidar_slam/run_ipbcar_gs_3dgs.yaml ipb_car -i ./data/ipb_car/2024-04-30_cheap_car/extracted/ -sg --range 100 250 1 --tag math_${exp_tag}

# # python pings.py ./config/lidar_slam/run_ipbcar_gs_full_res.yaml ipb_car -i ./data/ipb_car/2024-04-30_cheap_car/extracted/ -sg --range 2650 2800 1 --tag seba_${exp_tag}
# python pings.py ./config/lidar_slam/run_ipbcar_gs.yaml ipb_car -i ./data/ipb_car/2024-04-30_cheap_car/extracted/ -sg --range 2650 2800 1 --tag seba_${exp_tag}
# # python pings.py ./config/lidar_slam/run_ipbcar_gs_2.yaml ipb_car -i ./data/ipb_car/2024-04-30_cheap_car/extracted/ -sgm --range 2650 2800 1 --tag seba_${exp_tag}
# # python pings.py ./config/lidar_slam/run_ipbcar_gs_3dgs_cons.yaml ipb_car -i ./data/ipb_car/2024-04-30_cheap_car/extracted/ -sgm --range 2650 2800 1 --tag seba_${exp_tag}
# # # python pings.py ./config/lidar_slam/run_ipbcar_gs_no_consistency.yaml ipb_car -i ./data/ipb_car/2024-04-30_cheap_car/extracted/ -sg --range 2650 2800 1 --tag seba_${exp_tag}
# # # python pings.py ./config/lidar_slam/run_ipbcar_gs_3dgs.yaml ipb_car -i ./data/ipb_car/2024-04-30_cheap_car/extracted/ -sg --range 2650 2800 1 --tag seba_${exp_tag}

# not used
# python pings.py ./config/lidar_slam/run_ipbcar_gs.yaml ipb_car -i ./data/ipb_car/2024-04-30_cheap_car/extracted/ -sg --range 8350 8500 1 --tag popposdolf_${exp_tag}
# python pings.py ./config/lidar_slam/run_ipbcar_gs_no_consistency.yaml ipb_car -i ./data/ipb_car/2024-04-30_cheap_car/extracted/ -sg --range 8350 8500 1 --tag popposdolf_${exp_tag}

echo "Begin to eval on the (reverse drive) test sequences"

# python inspect_pings.py  ./pings_experiments/church_${exp_tag}_test_ipbcar_gs_full*/ -i ./data/ipb_car/2024-04-30_cheap_car/extracted_2/  --range 2660 2720 1 -e --vis_off
python inspect_pings.py  ./pings_experiments/church_${exp_tag}_test_ipbcar_gs_ours*/ -i ./data/ipb_car/2024-04-30_cheap_car/extracted_2/  --range 2660 2720 1 -e --vis_off
# python inspect_pings.py  ./pings_experiments/church_${exp_tag}_test_ipbcar_gs_newours*/ -i ./data/ipb_car/2024-04-30_cheap_car/extracted_2/  --range 2660 2720 1 -e --vis_off
# python inspect_pings.py  ./pings_experiments/church_${exp_tag}_test_ipbcar_gs_cons_3dgs*/ -i ./data/ipb_car/2024-04-30_cheap_car/extracted_2/  --range 2660 2720 1 -e --vis_off
# # python inspect_pings.py  ./pings_experiments/church_${exp_tag}_test_ipbcar_gs_without*/ -i ./data/ipb_car/2024-04-30_cheap_car/extracted_2/  --range 2660 2720 1 -e --vis_off
# # python inspect_pings.py  ./pings_experiments/church_${exp_tag}_test_ipbcar_gs_3d*/ -i ./data/ipb_car/2024-04-30_cheap_car/extracted_2/  --range 2660 2720 1 -e --vis_off

# python inspect_pings.py  ./pings_experiments/roundabout_${exp_tag}_test_ipbcar_gs_full*/ -i ./data/ipb_car/2024-04-30_cheap_car/extracted_2/  --range 2830 2890 1 -e --vis_off
python inspect_pings.py  ./pings_experiments/roundabout_${exp_tag}_test_ipbcar_gs_ours*/ -i ./data/ipb_car/2024-04-30_cheap_car/extracted_2/  --range 2830 2890 1 -e --vis_off
# python inspect_pings.py  ./pings_experiments/roundabout_${exp_tag}_test_ipbcar_gs_newours*/ -i ./data/ipb_car/2024-04-30_cheap_car/extracted_2/  --range 2830 2890 1 -e --vis_off
# python inspect_pings.py  ./pings_experiments/roundabout_${exp_tag}_test_ipbcar_gs_cons_3dgs*/ -i ./data/ipb_car/2024-04-30_cheap_car/extracted_2/  --range 2830 2890 1 -e --vis_off
# # python inspect_pings.py  ./pings_experiments/roundabout_${exp_tag}_test_ipbcar_gs_without*/ -i ./data/ipb_car/2024-04-30_cheap_car/extracted_2/  --range 2830 2890 1 -e --vis_off
# # python inspect_pings.py  ./pings_experiments/roundabout_${exp_tag}_test_ipbcar_gs_3d*/ -i ./data/ipb_car/2024-04-30_cheap_car/extracted_2/  --range 2830 2890 1 -e --vis_off

# python inspect_pings.py  ./pings_experiments/neighbor_${exp_tag}_test_ipbcar_gs_full*/ -i ./data/ipb_car/2024-04-30_cheap_car/extracted_2/  --range 4330 4390 1 -e --vis_off
python inspect_pings.py  ./pings_experiments/neighbor_${exp_tag}_test_ipbcar_gs_ours*/ -i ./data/ipb_car/2024-04-30_cheap_car/extracted_2/  --range 4330 4390 1 -e --vis_off
# python inspect_pings.py  ./pings_experiments/neighbor_${exp_tag}_test_ipbcar_gs_newours*/ -i ./data/ipb_car/2024-04-30_cheap_car/extracted_2/  --range 4330 4390 1 -e --vis_off
# python inspect_pings.py  ./pings_experiments/neighbor_${exp_tag}_test_ipbcar_gs_cons_3dgs*/ -i ./data/ipb_car/2024-04-30_cheap_car/extracted_2/  --range 4330 4390 1 -e --vis_off
# # python inspect_pings.py  ./pings_experiments/neighbor_${exp_tag}_test_ipbcar_gs_without*/ -i ./data/ipb_car/2024-04-30_cheap_car/extracted_2/  --range 4330 4390 1 -e --vis_off
# # python inspect_pings.py  ./pings_experiments/neighbor_${exp_tag}_test_ipbcar_gs_3d*/ -i ./data/ipb_car/2024-04-30_cheap_car/extracted_2/  --range 4330 4390 1 -e --vis_off

# python inspect_pings.py  ./pings_experiments/math_${exp_tag}_test_ipbcar_gs_full*/ -i ./data/ipb_car/2024-04-30_cheap_car/extracted_2/  --range 5340 5400 1 -e --vis_off
python inspect_pings.py  ./pings_experiments/math_${exp_tag}_test_ipbcar_gs_ours*/ -i ./data/ipb_car/2024-04-30_cheap_car/extracted_2/  --range 5340 5400 1 -e --vis_off
# python inspect_pings.py  ./pings_experiments/math_${exp_tag}_test_ipbcar_gs_newours*/ -i ./data/ipb_car/2024-04-30_cheap_car/extracted_2/  --range 5340 5400 1 -e --vis_off
# python inspect_pings.py  ./pings_experiments/math_${exp_tag}_test_ipbcar_gs_cons_3dgs*/ -i ./data/ipb_car/2024-04-30_cheap_car/extracted_2/  --range 5340 5400 1 -e --vis_off
# # python inspect_pings.py  ./pings_experiments/math_${exp_tag}_test_ipbcar_gs_without*/ -i ./data/ipb_car/2024-04-30_cheap_car/extracted_2/  --range 5340 5400 1 -e --vis_off
# # python inspect_pings.py  ./pings_experiments/math_${exp_tag}_test_ipbcar_gs_3d*/ -i ./data/ipb_car/2024-04-30_cheap_car/extracted_2/  --range 5340 5400 1 -e --vis_off

# python inspect_pings.py  ./pings_experiments/seba_${exp_tag}_test_ipbcar_gs_full*/ -i ./data/ipb_car/2024-04-30_cheap_car/extracted_2/  --range 2200 2260 1 -e --vis_off
python inspect_pings.py  ./pings_experiments/seba_${exp_tag}_test_ipbcar_gs_ours*/ -i ./data/ipb_car/2024-04-30_cheap_car/extracted_2/  --range 2200 2260 1 -e --vis_off
# python inspect_pings.py  ./pings_experiments/seba_${exp_tag}_test_ipbcar_gs_newours*/ -i ./data/ipb_car/2024-04-30_cheap_car/extracted_2/  --range 2200 2260 1 -e --vis_off
# python inspect_pings.py  ./pings_experiments/seba_${exp_tag}_test_ipbcar_gs_cons_3dgs*/ -i ./data/ipb_car/2024-04-30_cheap_car/extracted_2/  --range 2200 2260 1 -e --vis_off
# # python inspect_pings.py  ./pings_experiments/seba_${exp_tag}_test_ipbcar_gs_without*/ -i ./data/ipb_car/2024-04-30_cheap_car/extracted_2/  --range 2200 2260 1 -e --vis_off
# # python inspect_pings.py  ./pings_experiments/seba_${exp_tag}_test_ipbcar_gs_3d*/ -i ./data/ipb_car/2024-04-30_cheap_car/extracted_2/  --range 2200 2260 1 -e --vis_off