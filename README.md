# Campus3D：A Photogrammetry Point Cloud Benchmark for Hierarchical Understanding of Outdoor Scene
#### The repository contains the utilization and implementation of this [ACM MM 2020 Paper](https://arxiv.org/abs/2008.04968). The Campus3D dataset including full version and reduced version can be donwloaded from our [official website](https://3d.dataset.site) or the [alternative](https://3d.nus.app). A reproduce work can be found in [Campus3D-reproducibility](https://github.com/Yuqing-Liao/reproduce-campus3d), which reimplemented the hierarchical learning in PyTorch. More details of the reproduce work are presented in the [ACM MM 2021 companion paper](https://3d.dataset.site).
![](SixRegion.png)
### Running Environment
#### Python packages
The repo has been tested on Python 2.7.17 and Python 3.7.3.
To implementation the model of [Pointnet2](https://github.com/charlesq34/pointnet2), Python 2.7.17 and tensorflow 1.13 are required. 

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

#### Pointnet2 Compile
To run the hierarchical model, one has to comile the tensorflow operations of pointnet2 (`models/sem_seg/pointnet2/tf_op`).
The compiled `*.so` files in this repo was based on CUDA 10.0 and above python packages.

#### Docker

To deploy the environment easily, docker file in `docker` is provided. Below is the example of using the docker. 

First build the docker image:
```bash
cd docker
docker build -t shinkeli/campus3d:latest .
```
Run the docker (version>=19.03 with nvidia-container-toolkit) in bash:
```bash
docker run -it --gpus all -v <path_to_Campus3D>:/root/Campus3D shinkeli/campus3d:latest /bin/bash
```

### Training and Evaluation 
Download the [reduced version](https://3d.dataset.site) of Campus3D and place them into `data`. The data folder should be in the following structure:
```
├── data
│   ├── data_list.yaml
|   ├── matrix_file_list.yaml
|   └── h_matrices
|       └──lX.csv
|       └── ...
│   └── <area_name_1>
│       └── <area_name_1.pcd
|       └──  <area_name_1>labeX.npy
|       └──  <area_name_1>labeY.npy
|       └──  ...
│   └── <area_name_2>
│       └── <area_name_2>.pcd
|       └── <area_name_2>labeX.npy
|       └── <area_name_2>labeY.npy
|       └──  ...
│   └── ...
```
Each folder with <area_name> contains the point cloud and label data of one area. The `h_matrices` folders contains the hierarchical linear relationship between the label in one level and the bottom level. For other structure of data, one can modify data config file `data_list.yaml` to set customized path. In addition, the train/val/test split can be reset by the data config file.

For the setting of sampling and model, each folder in `configs` contains one version of setting. The default config folder is `configs/sem_seg_default_block`, and there are captions for arguments in the config file of this folder.

To apply training of the model:
```bash
cd Campus3D
python engine/train.py -cfg <config_dir>
```
The default `<config_dir>` is `configs/sem_seg_default_block`. The model will be saved in `log/<dir_name>`, where the `<dir_name>` is the set "OUTPUT_DIR" in the config file.


To apply evaluation of the model on the test set:
```bash
cd Campus3D
python engine/eval.py -cfg  <config_dir> -s TEST_SET -ckpt <check_point_name> -o <output_log> -gpu <gpu_id>
```
The `<check_point_name>` is the name of ckpt in `log/<dir_name>`, where the `<dir_name>` is the set "OUTPUT_DIR" in the config file. The result of IoU, Overall Accuracy and Consistency Rate wiil be written into `<output_log>`, for which the default name depends on the datetime. `<gpu_id>` is to set the gpu id for 'faiss' implementation of GPU based nearest neighbour search.
