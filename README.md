# PINGS: Gaussian Splatting Meets Distance Field Within a Point-based Implicit Neural Map


## Demo

**SLAM process**

<p align="center">
  <img src="https://github.com/user-attachments/assets/c481cf2b-7011-4711-a0e4-70aef5e3fe74" alt="pings_demo_hku_dataset">
</p>

**Rendering from the PINGS map**

<p align="center">
  <img src="https://github.com/user-attachments/assets/7b5ebf77-93fe-43b0-8766-13e6451d8770" alt="pings_demo_ipbcar_rendering">
</p>


## Installation

### 1. Clone the repository

```
git clone git@gitlab.ipb.uni-bonn.de:yue.pan/PINGS.git --recursive
cd PINGS
```

### 2. Set up conda environment

```
conda create --name pings python=3.10
conda activate pings
```

### 3. Install the key requirement PyTorch

```
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=11.8 -c pytorch -c nvidia
```

The commands depend on your CUDA version (check it by `nvcc --version`). You may check the instructions [here](https://pytorch.org/get-started/previous-versions/).


### 4. Install other dependency

```
pip3 install -r requirements.txt
```

----

## Run PINGS

### Run on IPB Car 

**Download example data**

```
bash scripts/download_ipbcar.sh
```

**Run on a test subset**

```
# example ipb_car neighborhood 

python3 pings.py ./config/lidar_slam/run_ipbcar_gs.yaml ipb_car -i ./data/ipb_car/ipbcar_test_subset/ -vmsg --range 550 700 1 --tag neighborhood
```


### Run on Oxford Spires

```
python pings.py ./config/lidar_slam/run_oxford_gs.yaml oxford -i ./data/Oxford-Spires-Dataset/2024-03-12-keble-college-04/ -vmsg

python pings.py ./config/lidar_slam/run_oxford_gs_raw.yaml oxford_raw -i ./data/Oxford-Spires-Dataset/2024-03-18-christ-church-02/ -vmsg
```

### To run on your own data

You may add your own data loader in `dataset/dataloaders/`. You may take `ipb_car.py`, `kitti.py`, `kitti360.py`, `waymo.py`, etc. as the templates. 

Then, you can run PINGS likewise:

```
python3 pings.py ./config/[your_config].yaml [your_dataset] -i ./data/path_to_your_data/ -vmsg
```

Check `python pings.py -h` for more details for usage.


### Inspect the results afterwards

**Examples**

```
python inspect_pings.py ./pings_experiments/small_loop_test_ipbcar_gs_longer_ipb_car__2024-12-10_23-54-59/ -f 2400

python inspect_pings.py  ./pings_experiments/church_${exp_tag}_test_ipbcar_gs*/ -i ./data/ipb_car/2024-04-30_cheap_car/extracted_2/  --range 2660 2720 1 -e --vis_off
```

### For more usages, please refer to NOTE.md


## Contact
If you have any questions, please contact:

- Yue Pan {[yue.pan@igg.uni-bonn.de]()}

## Related Projects

[PIN-SLAM](https://github.com/PRBonn/PIN_SLAM): LiDAR SLAM Using a Point-Based Implicit Neural Representation for Achieving Global Map Consistency
