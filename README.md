# Point-based Multi-view Stereo Network & Visibility-aware Point-based Multi-view Stereo Network

## Introduction
[PointMVSNet](http://hansf.me/projects/PMVSNet/) is a deep point-based deep framework for multi-view stereo (MVS). PointMVSNet directly processes the target scene as point clouds and predicts the depth in a coarse-to-fine manner. Our network leverages 3D geometry priors and 2D texture information jointly and effectively by fusing them into a feature-augmented point cloud, and processes the point cloud to estimate the 3D flow for each point. 

[VAPointMVSNet](https://ieeexplore.ieee.org/abstract/document/9076298) extends PointMVSNet with visibility-aware multi-view feature aggregations, which allows the network to aggregate multi-view appearance cues while taking into account occlusions.

If you find this project useful for your research, please cite: 

```
@ARTICLE{ChenVAPMVSNet2020TPAMI,
  author={Chen, Rui and Han, Songfang and Xu, Jing and Su, Hao},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence}, 
  title={Visibility-Aware Point-Based Multi-View Stereo Network}, 
  year={2020},
  volume={},
  number={},
  pages={1-1},}
```

```
@InProceedings{ChenPMVSNet2019ICCV,
    author = {Chen, Rui and Han, Songfang and Xu, Jing and Su, Hao},
    title = {Point-based Multi-view Stereo Network},
    booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
    year = {2019}
}
```

## How to use

### Environment
The environment requirements are listed as follows:
- Pytorch 1.0.1 
- CUDA 9.0 
- CUDNN 7.4.2
- GCC5

### Installation
* Check out the source code 

    ```git clone https://github.com/callmeray/PointMVSNet && cd PointMVSNet```
* Install dependencies 

    ```bash install_dependencies.sh```
* Compile CUDA extensions 

    ```bash compile.sh```

### Training
* Download the preprocessed [DTU training data](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view) from [MVSNet](https://github.com/YoYo000/MVSNet) and unzip it to ```data/dtu```.
* Train the network

    ```python pointmvsnet/train.py --cfg configs/dtu_wde3.yaml```
  
  You could change the batch size in the configuration file according to your own pc.

### Testing
* Download the [rectified images](http://roboimagedata2.compute.dtu.dk/data/MVS/Rectified.zip) from [DTU benchmark](http://roboimagedata.compute.dtu.dk/?page_id=36) and unzip it to ```data/dtu/Eval```.
* Test with your own model

    ```python pointmvsnet/test.py --cfg configs/dtu_wde3.yaml```
    
* Test with the pretrained model

    ```python pointmvsnet/test.py --cfg configs/dtu_wde3.yaml TEST.WEIGHT outputs/dtu_wde3/model_pretrained.pth```

### Depth Fusion
PointMVSNet generates per-view depth map. We need to apply depth fusion ```tools/depthfusion.py``` to get the complete point cloud. Please refer to [MVSNet](https://github.com/YoYo000/MVSNet) for more details.
    
