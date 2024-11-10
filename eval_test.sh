echo "Begin to eval on the (reverse drive) test sequences"

exp_tag=1109
python inspect_pings.py  ./pings_experiments/church_${exp_tag}_test_ipbcar_gs_ours*/ -i ./data/ipb_car/2024-04-30_cheap_car/extracted_2/  --range 2660 2720 1 -e --vis_off
python inspect_pings.py  ./pings_experiments/church_${exp_tag}_test_ipbcar_gs_without*/ -i ./data/ipb_car/2024-04-30_cheap_car/extracted_2/  --range 2660 2720 1 -e --vis_off
python inspect_pings.py  ./pings_experiments/roundabout_${exp_tag}_test_ipbcar_gs_ours*/ -i ./data/ipb_car/2024-04-30_cheap_car/extracted_2/  --range 2830 2890 1 -e --vis_off
python inspect_pings.py  ./pings_experiments/roundabout_${exp_tag}_test_ipbcar_gs_without*/ -i ./data/ipb_car/2024-04-30_cheap_car/extracted_2/  --range 2830 2890 1 -e --vis_off
python inspect_pings.py  ./pings_experiments/neighbor_${exp_tag}_test_ipbcar_gs_ours*/ -i ./data/ipb_car/2024-04-30_cheap_car/extracted_2/  --range 4330 4390 1 -e --vis_off
python inspect_pings.py  ./pings_experiments/neighbor_${exp_tag}_test_ipbcar_gs_without*/ -i ./data/ipb_car/2024-04-30_cheap_car/extracted_2/  --range 4330 4390 1 -e --vis_off
python inspect_pings.py  ./pings_experiments/math_${exp_tag}_test_ipbcar_gs_ours*/ -i ./data/ipb_car/2024-04-30_cheap_car/extracted_2/  --range 5340 5400 1 -e --vis_off
python inspect_pings.py  ./pings_experiments/math_${exp_tag}_test_ipbcar_gs_without*/ -i ./data/ipb_car/2024-04-30_cheap_car/extracted_2/  --range 5340 5400 1 -e --vis_off
python inspect_pings.py  ./pings_experiments/seba_${exp_tag}_test_ipbcar_gs_ours*/ -i ./data/ipb_car/2024-04-30_cheap_car/extracted_2/  --range 2200 2260 1 -e --vis_off
python inspect_pings.py  ./pings_experiments/seba_${exp_tag}_test_ipbcar_gs_without*/ -i ./data/ipb_car/2024-04-30_cheap_car/extracted_2/  --range 2200 2260 1 -e --vis_off