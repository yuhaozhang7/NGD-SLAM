<div align="center">
  <h1>NGD-SLAM: Towards Real-Time Dynamic SLAM <br>without GPU</h2>
  <strong>IROS 2025</strong>
  <br>
    <a href="https://yuhaozhang7.github.io" target="_blank">Yuhao Zhang</a><sup>1,2</sup>,
    <a href="#" target="_blank">Mihai Bujanca</a><sup>3</sup>,
    <a href="#" target="_blank">Mikel Luján</a><sup>1</sup>
  <p>
    <h45>
      <sup>1</sup>University of Manchester &nbsp;&nbsp;
      <sup>2</sup>University of Cambridge &nbsp;&nbsp;
      <br>
      <sup>3</sup>Qualcomm Technologies XR Labs &nbsp;&nbsp;
    </h5>
  </p>

  [<img src="https://img.shields.io/badge/Video-Bilibili-pink" alt="Bilibili">](https://www.bilibili.com/video/BV1XKT5eaEsT/)
  [<img src="https://img.shields.io/badge/arXiv-2405.07392-990000" alt="Arxiv">](https://arxiv.org/abs/2405.07392)

</div>

<p align="center">
  <img src="assets/bonn_crowd_small.gif" alt="GIF 1" width="240">
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="assets/bonn_mov_small.gif" alt="GIF 2" width="240">
</p>

This is a visual SLAM system designed for dynamic environments, based on the [ORB-SLAM3](https://github.com/UZ-SLAMLab/ORB_SLAM3) framework. It runs in real-time on a single laptop CPU without compromising tracking accuracy.


# 1. Prerequisites
The system is tested on **Ubuntu 20.04** and **22.04**, and it should be easy to compile in other platforms. A powerful computer will provide more stable and accurate results.

## C++11 or C++0x Compiler
It uses the new thread and chrono functionalities of C++11.

## Pangolin
It uses [Pangolin](https://github.com/stevenlovegrove/Pangolin) for visualization and user interface. Dowload and install instructions can be found at: https://github.com/stevenlovegrove/Pangolin.

## OpenCV
It uses [OpenCV](http://opencv.org) to manipulate images and features. Dowload and install instructions can be found at: http://opencv.org. **Required at leat 4.4**.

## Eigen3
Required by g2o (see below). Download and install instructions can be found at: http://eigen.tuxfamily.org. **Required at least 3.1.0**.

## YOLO (Included in Thirdparty folder)
It uses the [C++ version](https://github.com/hpc203/yolov34-cpp-opencv-dnn) of the [YOLO-fastest](https://github.com/dog-qiuqiu/Yolo-Fastest.git) model. The model configuration and pre-trained weights are included in the *Thirdparty* folder and are loaded using OpenCV.

## DBoW2 and g2o (Included in Thirdparty folder)
It uses modified versions of the [DBoW2](https://github.com/dorian3d/DBoW2) library to perform place recognition and [g2o](https://github.com/RainerKuemmerle/g2o) library to perform non-linear optimizations. Both modified libraries (which are BSD) are included in the *Thirdparty* folder.

## Python
Required to calculate the alignment of the trajectory with the ground truth. **Required Numpy module**.

* (win) http://www.python.org/downloads/windows
* (deb) `sudo apt install libpython2.7-dev`
* (mac) preinstalled with osx

# 2. Building NGD-SLAM
Clone the repository:
```
git clone https://github.com/yuhaozhang7/NGD-SLAM.git NGD-SLAM
```

It provides a script `build.sh` to build the libraries. Please make sure you have installed all required dependencies. Execute:
```
cd NGD-SLAM
chmod +x build.sh
./build.sh
```

# 3. Running TUM Examples
[TUM dataset](https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download) includes sequences that are captured using an RGB-D camera in dynamic environments. Download the desired sequence and uncompress it. Below is an example command for the *freiburg3_walking_xyz* sequence:
```
./Examples/RGB-D/rgbd_tum ./Vocabulary/ORBvoc.txt ./Examples/RGB-D/TUM3.yaml ./path/to/TUM/rgbd_dataset_freiburg3_walking_xyz ./Examples/RGB-D/associations/fr3_walk_xyz.txt
```

# 4. Citation
If you find this work to be useful in your research, please consider citing:
```bibtex
@article{zhang2024ngdslam,
  title={NGD-SLAM: Towards Real-Time Dynamic SLAM without GPU},
  author={Zhang, Yuhao and Bujanca, Mihai and Luján, Mikel},
  journal={arXiv preprint arXiv:2405.07392},
  year={2024}
}
```
