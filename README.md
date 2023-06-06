# Infer_demo_TransFlow

This repository provides demo infer code for: Transflow: Transformer as flow learner.



## Installation
1. Clone the repository.
2. install dependencies.

```
pip install -r requirement.txt
```

# Demo visualization of flow estimation
1. Download landmark pre-trained model at [GoogleDrive](https://drive.google.com/file/d/1tDqX2nG1qATqrd2fEb4Sgs4av25d9tgN/view?usp=sharing), and put it into ```FaceLandmark/model/```
2. Run the test file

```
python visualize_flow.py
```

If you find the demo is useful, you can be cite it as:

```
@inproceedings{lu2023transflow,
  title={Transflow: Transformer as flow learner},
  author={Lu, Yawen and Wang, Qifan and Ma, Siqi and Geng, Tong and Chen, Yingjie Victor and Chen, Huaijin and Liu, Dongfang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={18063--18073},
  year={2023}
}
```
