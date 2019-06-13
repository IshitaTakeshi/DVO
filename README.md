Tadataka
========

[![Build Status](https://travis-ci.org/IshitaTakeshi/Tadataka.svg?branch=develop)](https://travis-ci.org/IshitaTakeshi/Tadataka)
[![codecov](https://codecov.io/gh/IshitaTakeshi/Tadataka/branch/develop/graph/badge.svg)](https://codecov.io/gh/IshitaTakeshi/Tadataka)
  
An open-source Visual Odometry / Visual SLAM implementation

This project aims to provide a package that is easy to compare several Visual SLAM techniques

## Currently implemented

- [x] DVO (Dense Visual Odometry)

## Usage

```
pip3 install -e .
bash examples/dataset.sh
python3 examples/rgbd_desk.py
python3 tools/evaluate_ate.py datasets/rgbd_dataset_freiburg1_desk/groundtruth.txt poses.txt --verbose --plot plot.png
```

## Running tests

```
pytest tests
```

## References

### DVO

Although `ldvo` is listed below, none of code is copied to this software.

```bibtex
@inproceedings{steinbrucker2011real,
  title={Real-time visual odometry from dense RGB-D images},
  author={Steinbr{\"u}cker, Frank and Sturm, J{\"u}rgen and Cremers, Daniel},
  booktitle={2011 IEEE International Conference on Computer Vision Workshops (ICCV Workshops)},
  pages={719--722},
  year={2011},
  organization={IEEE}
}
@article{kerl2012odometry,
  title={Odometry from rgb-d cameras for autonomous quadrocopters},
  author={Kerl, Christian},
  journal={Master's Thesis, Technical University},
  year={2012},
  publisher={Citeseer}
}
@misc{ldvo,
  title = {LDVO - Lightweight Dense Visual Odometry},
  author = {Maier, Robert},
  howpublished = "\url{https://github.com/robmaier/ldvo}",
}
```
