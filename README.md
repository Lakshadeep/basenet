# BaSeNet

Code repository for our paper "BaSeNet - A Learning-based Mobile Manipulator Base Pose Sequence Planning for Pickup Tasks" by Lakshadeep Naik, Sinan Kalkan, Sune L. Sørensen, Mikkel B. Kjærgaard and Norbert Kruger accepted at 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 

[Paper pre-print](https://portal.findresearcher.sdu.dk/files/265119912/Base_pose_sequence_planning_using_RL_GNN.pdf)
[Project webpage](https://lakshadeep.github.io/basenet/)

# Pre-requisites
Our code uses NVIDIA Isaac Sim for simulation. Installation instructions can be found [here](https://docs.omniverse.nvidia.com/isaacsim/latest/installation/install_workstation.html). This code has been tested with Isaac Sim version 'isaac_sim-2022.2.1'

Further, following python packages should be installed in the Isaac sim python environment:
```
omegaconf, hydra, hydra-core, tqdm, opencv-python,  mushroom-rl (local), shapely
```

'local' - local installation of the package is required


#### Installing new python packages in Isaac
```
./python.sh -m pip install {name of the package}  --global-option=build_ext --global-option=build_ext  --global-option="-I{Isaac install path}/ov/pkg/isaac_sim-2022.2.1/kit/python/include/python3.7m"
```

#### Installing local python package in Isaac (for mushoorm-rl and this package)
```
./python.sh -m pip install -e {package path}/  --global-option=build_ext --global-option=build_ext  --global-option="-I{Isaac install pathj}/ov/pkg/isaac_sim-2022.2.1/kit/python/include/python3.7m"
```

#### Downloading assets folders
Isaac environments:
```
https://drive.google.com/file/d/1M_ZJ-aGrzppw-4W2XNWWGqJjJK4n369T/view?usp=drive_link
```
UR5e assets
```
https://drive.google.com/file/d/1T60DYvzTUsxCs0e-PZlhVjGw636J9WoX/view?usp=drive_link
```
Download, unzip both the folders and place them inside the repository.

## NOTES:
- Before trying to run the code, please change relative paths in all the config files in 'conf' folder.

# To run the scripts
First open `{Isaac install path}/ov/pkg/isaac_sim-2022.2.1` in terminal and run the following command:
```
./python.sh {package path}/{script name}.py 

```

# Training
Layer 1: optimal base pose for grasping

(Ensure that `optimize_base_poses' config file is loaded in cnf/config.yml)
```
./python.sh {package path}/basenet/train/optimize_base_poses.py 

```

Layer 2: determine grasp sequence

(Ensure that `grasp_sequence' config file is loaded in cnf/config.yml)
```
./python.sh {package path}/basenet/train/grasp_sequence.py 

```

## If you found our work useful, consider citing our paper:

L. Naik, S. Kalkan, S. L. Sørensen, M. B. Kjærgaard, and N. Krüger, “Basenet: A learning-based mobile manipulator base pose sequence planning for pickup tasks,” in 2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) (Accepted), 2024.

```
@INPROCEEDINGS{naik2024basenet,
  author={Lakshadeep Naik and Sinan Kalkan and Sune L. Sørensen and Mikkel B. Kjærgaard and Norbert Krüger},
  booktitle={2024 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)}, 
  title={BaSeNet: A Learning-based Mobile Manipulator Base Pose Sequence Planning for Pickup Tasks}, 
  year={2024}
}
```
