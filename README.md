# PSScreen: Partially Supervised Multiple Retinal Disease Screening

This repository is the official implementation of paper: PSScreen: Partially Supervised Multiple Retinal Disease Screening

## Environment

- python: 3.9.19
- pytorch: 2.4.0
- numpy: 1.24.3
- pandas: 2.2.3
- scikit-learn: 1.5.0
- albumentations: 1.0.1
- opencv-python

## Preliminary

- Create several folders (`scripts/exp/log`, `scripts/exp/code`, `scripts/exp/checkpoint`) to record experiment details.
- Create a folder named `pretrained_model` under the `PSScreen`, and place the [pretrained weights of the ResNet101](https://unioulu-my.sharepoint.com/:u:/g/personal/bzheng24_univ_yo_oulu_fi/EX8_ALmVwYhDqkrL5XOygBwB7vuvxbfONCTuqccne-77jw?e=TEXuNf) inside it.

## Dataset Preparations

1. Download these partially/fully labeled open-access datasets: [DDR](https://github.com/nkicsl/DDR-dataset), [ADAM](https://drive.google.com/file/d/1Uz5x0aqXb0aecjzNWQ4522oCxaRDZxBt/view), [PALM](https://drive.google.com/file/d/14XWD6kX0dVRfAyEc7FkZGKZibWEkvnyv/view), [Kaggle-CAT](https://www.kaggle.com/datasets/jr2ngb/cataractdataset),[Kaggle-HR](https://www.kaggle.com/datasets/harshwardhanfartale/hypertension-and-hypertensive-retinopathy-dataset),[REFUGE2](https://drive.google.com/file/d/1DspRzDqypeBOxZnWPQxmXprNVmJwkBRJ/view), [APTOS](https://www.kaggle.com/competitions/aptos2019-blindness-detection/data),[ORIGA<sup>light</sup>](https://pubmed.ncbi.nlm.nih.gov/21095735/),[RFMiD](https://riadd.grand-challenge.org/download-all-classes/),[HPMI](https://figshare.com/articles/dataset/HPMI_A_retinal_fundus_image_dataset_for_identification_of_high_and_pathological_myopia_based_on_deep_learning/24800232?file=49305304),[ODIR](https://odir2019.grand-challenge.org/dataset/). All data from ORIGA<sup>light</sup>, HPMI, and APTOS are used for testing, while only the test set from RFMiD is used for testing.
2. Crop the field of view region from the fundus image, and the corresponding code is in `datasets/threshold_crop.py`.
3. Pad image with zeros so that the short and long sides are of equal length, and the corresponding code is in `datasets/pad_image_with_zero.py`.
4. Organize the downloaded and preprocessed dataset according to the following structure:

 ```none
.
└── retinal_datasets/
    ├── data/
    │   ├── train/
    │   │   ├── ADAM_trainset
    │   │   ├── CAT_trainset
    │   │   ├── DDR_trainset
    │   │   ├── HR_trainset
    │   │   ├── ODIR_trainset
    │   │   ├── PALM_trainset
    │   │   └── REFUGE_trainset
    │   ├── valid/
    │   │   ├── ADAM_validset
    │   │   ├── CAT_validset
    │   │   ├── DDR_validset
    │   │   ├── HR_validset
    │   │   ├── PALM_validset
    │   │   └── REFUGE_validset
    │   └── test/
    │       ├── ADAM_testset
    │       ├── APTOS_dataset
    │       ├── CAT_testset
    │       ├── DDR_testset
    │       ├── HPMI_dataset
    │       ├── HR_testset
    │       ├── ODIR200x3_testset
    │       ├── ODIR_testset
    │       ├── ORIGA_dataset
    │       ├── PALM_testset
    │       ├── REFUGE_testset
    │       └── RFMiD_dataset
    └── labels/
        ├── train
        ├── test
        └── valid
```
## Training

```sh
bash ./scripts/run.sh
```

## Evaluation

Set `resumeModel` in `test.sh` to the path where the checkpoint is saved.

```sh
bash ./scripts/test.sh
```
## Checkpoint

The pretrained weights of the following models are available at this [link](https://unioulu-my.sharepoint.com/:u:/g/personal/bzheng24_univ_yo_oulu_fi/ETRhBmwPeIdGoeDka1DpiJ8Bd3pjiPiRM7s9D3GdZ-ZjsQ?e=L8NrkJ)




