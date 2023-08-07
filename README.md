# TPC-NAS: Sub-Five-Minute Neural Architecture Search for Image Classification, Object-Detection, and Super-Resolution

# Pretrained Model and Result
Download and unzip save_dir in the root of TPC-NAS from the link below:
https://drive.google.com/file/d/1YoSz87M9HNoF24zzg3xMWXUFo2FcRZbX/view?usp=sharing

Change the data path in DataLoader

## run 
```
sh scripts/TPC_NAS_yolov4_flops20G.sh
sh scripts/TPC_NAS_yolov4_flops40G.sh
```

# Open Source
Some few files in this repository are modified from the following open-source implementations:
```
https://github.com/DeepVoltaire/AutoAugment/blob/master/autoaugment.py
https://github.com/VITA-Group/TENAS
https://github.com/SamsungLabs/zero-cost-nas
https://github.com/BayesWatch/nas-without-training
https://github.com/rwightman/gen-efficientnet-pytorch
https://pytorch.org/vision/0.8/_modules/torchvision/models/resnet.html
```
Most of the code thanks to the contribution of Zen-NAS
```
https://github.com/idstcv/ZenNAS
```


