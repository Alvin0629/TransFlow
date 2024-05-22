# TransFlow

This repository provides code for: **Transflow: Transformer as flow learner.**



## Installation
1. Clone the repository.
2. install dependencies.

```
pip install -r requirements.txt
```

## Demo of flow estimation
1. Download trained model at [GoogleDrive] https://drive.google.com/drive/folders/1XbK0gDshbqZRirEvC9eA4OHB_crn9d0z?usp=sharing), put the demo.pth into ```checkpoints/``` and put the mae_pvt.pth into the ```root/``` folder.

2. Run the inference 

```
python infer.py --keep_size
```
3. The flow estimations for ```demo_frames/``` will be saved under ```demo_viz_output/demo_frames/```

## Training of flow estimation, support multi-frame
Refer to files with suffix of '_multiframe' for arbitary frame number setting.
```
python -u train_flow.py  --name NAME --stage 'STAGE_NAME' --validation 'VAL_NAME'
```

If you find the work is useful, please cite it as:

```
@inproceedings{lu2023transflow,
  title={Transflow: Transformer as flow learner},
  author={Lu, Yawen and Wang, Qifan and Ma, Siqi and Geng, Tong and Chen, Yingjie Victor and Chen, Huaijin and Liu, Dongfang},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={18063--18073},
  year={2023}
}
```
