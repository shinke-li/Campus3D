# Campus3D：A Photogrammetry Point Cloud Benchmark for Outdoor Scene Hierarchical Understanding
#### The repository contains the utilization and implementation of this [ACM MM 2020 Paper](https://3d.dataset.site). The Campus3D dataset including full version and reduced version can be donwloaded from our [official website](https://3d.dataset.site) or the [alternative](https://3d.nus.app).

### Running Environment
The repo has been tested on Python 2.7.17 and Python 3.7.3.
To implementation the model of [Pointnet2] (https://github.com/charlesq34/pointnet2), Python 2.7.17 and tensorflow 1.14 are required. 

|  Package   | Version  |
|  ----  | ----  |
|numpy|1.16.6|
|numba|0.47.0|
|open3d|0.9.0.0|
|pyyaml|5.1.2|
|sparse|0.6.0|
|h5py|2.10.0|
|torch|1.0.0|
|faiss-gpu|1.6.3| 
|tensorflow-gpu|1.13.1|


faiss-gpu is optional for GPU KNN search. 

### Training and Evaluation 
Download the [reduced version]((https://3d.dataset.site) of Campus3D and place them into `data`. The data folder should be in the following structure:
```
├── data
│   ├── data_list.yaml
│   └── area_name_1
│       └── area_name_1.pcd
|       └──  area_name_1labeX.npy
|       └──  area_name_1labeY.npy
|       └──  ...
│   └── area_name_2
│       └── area_name_2.pcd
|       └──  area_name_2labeX.npy
|       └──  area_name_2labeY.npy
|       └──  ...
│   └── ...
```
The `sphere` folder contains the front-view XYZ maps converted from `velodyne` point clouds using the script in `./preprocess/sphere_map.py`. After data preparation, readers can train VS3D from scratch by running:
```bash
cd core
python main.py --mode train --gpu GPU_ID
```
The models are saved in `./core/runs/weights` during training. Reader can refer to `./core/main.py` for other options in training.
