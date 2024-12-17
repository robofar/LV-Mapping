# PINGS: Gaussian Splatting Meets Distance Field Within a Point-based Implicit Neural Map

## Installation


### 1. Set up conda environment

```
conda create --name pin python=3.10
conda activate pin
```

### 2. Install the key requirement PyTorch

```
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.7 -c pytorch -c nvidia 
```

The commands depend on your CUDA version (check it by `nvcc --version`). You may check the instructions [here](https://pytorch.org/get-started/previous-versions/).


### 3. Install other dependency

```
pip3 install -r requirements.txt
```

----

## Run PINGS

### Clone the repository

```
git clone git@gitlab.ipb.uni-bonn.de:yue.pan/PINGS.git --recursive
cd PINGS
```


### Run on IPB Car 

```
# example ipb_car church 

python pings.py ./config/lidar_slam/run_ipbcar_gs_new_test.yaml ipb_car -i ./data/ipb_car/2024-04-30_cheap_car/extracted/ -vmgs --range 2030 2180 1 --tag church
```

### Run on Oxford Spires

```
python pings.py ./config/lidar_slam/run_oxford_gs.yaml oxford -i ./data/Oxford-Spires-Dataset/2024-03-12-keble-college-04/ -vmsg

python pings.py ./config/lidar_slam/run_oxford_gs_raw.yaml oxford_raw -i ./data/Oxford-Spires-Dataset/2024-03-18-christ-church-02/ -vmsg
```



### Inspect the results afterwards

Example
```
python inspect_pings.py ./pings_experiments/small_loop_test_ipbcar_gs_longer_ipb_car__2024-12-10_23-54-59/ -f 2400

python inspect_pings.py  ./pings_experiments/church_${exp_tag}_test_ipbcar_gs*/ -i ./data/ipb_car/2024-04-30_cheap_car/extracted_2/  --range 2660 2720 1 -e --vis_off
```

### For more details, please refer to NOTE.md


## Contact
If you have any questions, please contact:

- Yue Pan {[yue.pan@igg.uni-bonn.de]()}

