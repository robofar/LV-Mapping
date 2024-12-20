# This script is used to evaluate the mesh quality of the predicted mesh

import csv
from eval_mesh_utils import eval_mesh
import os

########################################### MaiCity Dataset ###########################################
# dataset_name = "maicity_01_"

# bath_path = "/media/yuepan/DATA/1_data/maicity/01/baseline"

# # ground truth point cloud (or mesh) file
# # (optional masked by the intersection part of all the compared method)
# gt_pcd_path = "/media/yuepan/DATA/1_data/maicity/01/baseline/gt_map_pc_mai.ply"
# #gt_pcd_path = "xxx/mai_city/gt_map_pc_mai_inter_croped.ply"

# gt_pcd_path = os.path.join(bath_path, "gt_map_pc_mai.ply")

# pred_mesh_path = "xxx/mai_shine_prediction.ply"
# method_name = "ours_xxx"

# # pred_mesh_path = "xxx/baseline/vdb_fusion_xxx.ply"
# # method_name = "vdb_fusion_xxx"

# # pred_mesh_path = "xxx/baseline/puma_xxx.ply"
# # method_name = "puma_xxx"

# pred_mesh_path = os.path.join(bath_path, "pin_slam", "mesh_20cm_v2.ply")
# method_name = "pin_slam"

# ########################################### MaiCity Dataset ###########################################


# ######################################## Newer College Dataset ########################################
# dataset_name = "ncd_quad_"

# bath_path = "/media/yuepan/DATA/1_data/ncd/02/ncd_map_baselines"

# gt_pcd_path = os.path.join(bath_path, "ncd_quad_gt_pc.ply") #"xxx/ncd_example/quad/ncd_quad_gt_pc.ply"

# pred_mesh_path = os.path.join(bath_path, "newer_college_4d.ply") # "xxx/ncd_shine_prediction.ply"
# method_name = "ours_4d"

# pred_mesh_path = os.path.join(bath_path, "vdb_fusion", "mesh_vdb_10cm_nocarving_quad_refine.ply") # "xxx/baseline/vdb_fusion_xxx.ply"
# method_name = "vdb_fusion"

# pred_mesh_path = os.path.join(bath_path, "nksr", "kiss_poses_recon_mesh_ncd_every5_normal_50_200_voxel20cm.ply")
# method_name = "nksr_20cm"

# # pred_mesh_path = os.path.join(bath_path, "shine", "shine_ncd.ply") # "xxx/baseline/vdb_fusion_xxx.ply"
# # method_name = "shine"

# # pred_mesh_path = os.path.join(bath_path, "pin_slam", "mesh_20cm.ply") # "xxx/baseline/vdb_fusion_xxx.ply"
# # method_name = "pin_slam"

# pred_mesh_path = os.path.join(bath_path, "pin_slam", "mesh_30cm.ply") # "xxx/baseline/vdb_fusion_xxx.ply"
# method_name = "pin_slam"

# pred_mesh_path = os.path.join(bath_path, "pin_slam", "mesh_2_20cm_18.ply") # "xxx/baseline/vdb_fusion_xxx.ply"
# method_name = "pin_slam"

# pred_mesh_path = "xxx/baseline/puma_xxx.ply"
# method_name = "puma_xxx"

# pred_mesh_path = os.path.join(bath_path, "slamesh", "slamesh_recon.ply") # "xxx/baseline/vdb_fusion_xxx.ply"
# method_name = "slamesh"


# dataset_name = "ncd_math_"

# bath_path = "/media/yuepan/Expansion/1_data/ncd_128/math_easy/map_compare"

# gt_pcd_path = os.path.join(bath_path, "gt-maths-institute.ply") #"xxx/ncd_example/quad/ncd_quad_gt_pc.ply"

# pred_mesh_path = os.path.join(bath_path, "vdb_fusion", "vdb_fusion_mesh_20cm_v2.ply") # "xxx/baseline/vdb_fusion_xxx.ply"
# method_name = "vdb_fusion"

# # pred_mesh_path = os.path.join(bath_path, "nksr", "kiss_poses_recon_mesh_ncd_math_every20_normal_20_100_voxel20cm.ply")
# # method_name = "nksr_20cm"

# # # pred_mesh_path = os.path.join(bath_path, "shine", "shine_20cm.ply") 
# # # method_name = "shine"

# pred_mesh_path = os.path.join(bath_path, "pin_slam", "mesh_20cm_pin_10_deskew.ply") 
# method_name = "pin_slam_deskew_used"

# # pred_mesh_path = os.path.join(bath_path, "slamesh", "ncd128_math_slamesh.ply") 
# # method_name = "slamesh"

# # pred_mesh_path = os.path.join(bath_path, "puma", "puma_recon_11.ply")
# # method_name = "puma"

######################################## Newer College Dataset ########################################


######################################## Oxford-Spires Dataset ########################################
dataset_name = "oxford_keble_"
bath_path = "./data/Oxford-Spires-Dataset/keble-college-gt-clouds"


# dataset_name = "oxford_observatory_"
# bath_path = "./data/Oxford-Spires-Dataset/observatory-quarter-gt-clouds"


# dataset_name = "oxford_observatory_"
# bath_path = "./data/Oxford-Spires-Dataset/christ-church-gt-clouds"


# dataset_name = "oxford_blenheim_"
# bath_path = "./data/Oxford-Spires-Dataset/blenheim-palace-gt-clouds"


# dataset_name = "oxford_bodleian_"
# bath_path = "./data/Oxford-Spires-Dataset/bodleian-library-gt-clouds"


gt_pcd_path = os.path.join(bath_path, "merged-cloud-1cm.pcd")

# gt_pcd_path = os.path.join(bath_path, "merged-cloud-5cm.ply")

pred_mesh_path = os.path.join(bath_path, "results", "pin_mesh_15cm.ply")
method_name = "pin_15cm"

pred_mesh_path = os.path.join(bath_path, "results", "pings_mesh_15cm.ply")
method_name = "pings_15cm"


######################################## Oxford-Spires Dataset ########################################


# # evaluation results output file
# base_output_folder = "./experiments/evaluation/"

# output_csv_path = base_output_folder + dataset_name + method_name + "_eval.csv"

# # evaluation parameters
# # For MaiCity
# down_sample_vox = 0.02
# dist_thre = 0.1
# truncation_dist_acc = 0.2 
# truncation_dist_com = 2.0

# # For NCD
# down_sample_vox = 0.02
# dist_thre = 0.2
# truncation_dist_acc = 0.4 # 0.4 (used in shine-mapping)
# truncation_dist_com = 2.0


######################################## Oxford Dataset ########################################

# evaluation results output file
base_output_folder = "./pings_experiments/mesh_evaluation/"
# create the folder if it doesn't exist
os.makedirs(base_output_folder, exist_ok=True)

output_csv_path = os.path.join(base_output_folder, dataset_name + method_name + "_eval.csv")

# evaluation parameters (unit: m)
down_sample_vox = 0.02
dist_thre = 0.2
truncation_dist_acc = 0.4 # 0.4 (used in shine-mapping)
truncation_dist_com = 2.0

# evaluation
eval_metric = eval_mesh(pred_mesh_path, gt_pcd_path, down_sample_res=down_sample_vox, threshold=dist_thre, 
                        truncation_acc = truncation_dist_acc, truncation_com = truncation_dist_com, gt_bbx_mask_on = True) 

print(method_name)
print(eval_metric)

evals = [eval_metric]

csv_columns = ['MAE_accuracy(m)', 'MAE_completeness(m)', 'Chamfer_L1(m)', 'Chamfer_L2(m)', \
        'Precision[Accuracy](%)', 'Recall[Completeness](%)', 'F-score (%)', 'Spacing(m)', \
        'Inlier_threshold(m)', 'Outlier_truncation_acc(m)', 'Outlier_truncation_com(m)']

try:
    with open(output_csv_path, 'w') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
        writer.writeheader()
        for data in evals:
            writer.writerow(data)
except IOError:
    print("I/O error")

