# Deep-dive to Transformer!
Imcommit's lowest Transformer lecture <br /> 

## 1. Deeplearning tutorial
**Data**: toy dataset의 일종인 IMDB 데이터셋으로,  <br /> 
**Model**: Transformer가 아닌 간단한 구조의 모델로,  <br /> 
**Training**: 딥러닝 모델의 학습 방식을 숫자로 간단히 확인하는  <br /> 

코드입니다. 해설: [링크](https://www.youtube.com/watch?v=cp_6W734zXI)

1) 데이터셋을 deeplearning_tutorial 폴더에 다운로드: [google drive](https://drive.google.com/file/d/1KxY5kLYkUdRybC_hwKcMIbyidKLnltQ0/view?usp=drive_link)
2) `cd deeplearning_tutorial`
3) `python main.py`

## 2. Data

데이터 실습 코드 [링크](https://www.youtube.com/watch?v=_b-PK5CWWhk)

1) `cd data_modularized`
2) `python main.py`

과제: [stanford imdb dataset](https://ai.stanford.edu/~amaas/data/sentiment/)을 다운받고 imdb.py를 수정해서 학습시키기

## 3. Model(Transformer)

모델 실습 코드 [링크](https://www.youtube.com/watch?v=dKhk_rqZmes)

1) `git checkout tfdd_model`
2) `cd src`
3) `python main.py`

과제: Transformer 내부 모듈의 입출력 구조 파악하기

## 4. Training

학습 실습 코드 [링크](https://www.youtube.com/watch?v=TgRimIlDGEc)

1) `git checkout tfdd_train`
2) `cd src`
3) `python train_accelerate.py`