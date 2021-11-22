# IFR
This repository is released for double-blind submission, which can reproduce the main results (the baseline and our proposed IFR) of the experiment on Cityscapes with 1/30 labeled samples. Experiments on other partition settings and the CamVid dataset can be easily implemented by slightly modifying the config file.

## Install & Requirements
The code has been tested on pytorch=1.8.2 and python3.8. Please refer to `requirements.txt` for detailed information.

**To Install python packages**
```
pip install -r requirements.txt
```

## Download Pretrained Weights
````bash
mkdir ./IFR/pretrained
cd ./IFR/pretrained
# download resnet18 imagenet pretrained weight
wget http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet18-imagenet.pth
# download resnet101 imagenet pretrained weight
wget http://sceneparsing.csail.mit.edu/model/pretrained_resnet/resnet101-imagenet.pth
````

## Data preparation
You need to download the [Cityscapes](https://www.cityscapes-dataset.com/) datasets.

Your directory tree should be look like this:
````bash
./IFR/data
├── cityscapes
│   ├── gtFine
│   │   ├── train
│   │   └── val
│   └── leftImg8bit_sequence
│       ├── train
│       └── val
````

## Prepare Downsample Dataset
Generated downsample dataset would be saved in ./data
````bash
cd ./IFR
python tools/data_downsample.py
````

## Stage One Training of Accel
For example, train image segmentation model on 2 GPUs. Checkpoints would saved in ./IFR/work_dirs.
````bash
# train PSP18 baseline model
cd ./IFR/exp/sup_30_res18/scripts
bash train.sh
# train PSP101 baseline model
cd ./IFR/exp/sup_30_res101/scripts
bash train.sh
# train PSP18 IFR model
cd ./IFR/exp/IFR_30_res18/scripts
bash train.sh
# train PSP101 IFR model
cd ./IFR/exp/IFR_30_res101/scripts
bash train.sh
````

## Stage Two Training of Accel
For example, train Accel18 on 2 GPUs. Checkpoints would saved in ./Accel/work_dirs.
````bash
mkdir ./Accel/work_dirs
# train Accel18 with baseline model
cd ./Accel/exp/accel18_30_sup/script
bash train.sh
# train Accel18 with IFR model
cd ./Accel/exp/accel18_30_IFR/script
bash train.sh
````