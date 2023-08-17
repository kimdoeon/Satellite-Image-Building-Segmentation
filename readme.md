# 위성 이미지 건물 영역 분할 in Dacon
https://dacon.io/competitions/official/236092/overview/description


## Environment
* Ubuntu 22.04.2 LTS
* CUDA Version: 12.0
* RTX 3090 x 1
* python 3.11.4


## Install
```
conda create -n segment python=3.11.4 -y
conda activate segment
pip install -r requirements.txt
```
monai, segmentation_models_pytorch, transformers, tensorboard


## Data

### Move
* train2.csv, test.csv, sample_submission.csv
* train_img
* test_img

**=> data 디렉토리로 이동**


### Divide
* stride = 256
* size = 512

```
cd data
python data.py
```

**=> data/data_512 생성**


### OBA(Object-Based Augmentation)
약 25분 소요(Divide랑 같이 실행 추천)

data/oba.ipynb  
전부 실행  

[paper](https://openaccess.thecvf.com/content/ICCV2021W/ILDAV/papers/Illarionova_Object-Based_Augmentation_for_Building_Semantic_Segmentation_Ventura_and_Santa_Rosa_ICCVW_2021_paper.pdf)

**=> data/oba/result2 생성**

## Train
```
python train_smp_one.py --config train_smp_512.yaml
```

## Inference
config/predict_smp_512.yaml 수정  
tta_smp_inference.ipynb 실행  
