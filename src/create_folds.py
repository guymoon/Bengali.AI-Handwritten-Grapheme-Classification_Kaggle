import pandas as pd
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
import numpy as np


if __name__ =='__main__':
df = pd.read_csv('/gdrive/My Drive/kaggle/bengaliai-cv19/train.csv')
# print(df.head())
df.loc[:, 'kfold'] = -1 #kfold 컬럼 추가해서 -1로 모두 넣어주기

# frac을 입력하면 전체 row에서 몇%의 데이터를 return할 것인지 정할 수 있다.
#데이터프레임에 인덱스로 들어가 있어야 할 데이터가 일반 데이터 열에 들어가 있거나 반대로 일반 데이터 열이어야 할 것이 인덱스로 되어 있을 수 있다. 이 때는 set_index
#명령이나 reset_index 명령으로 인덱스와 일반 데이터 열을 교환할 수 있다.
df = df.sample(frac = 1).reset_index(drop=True)

X = df.image_id.values
y = df[['grapheme_root', 'vowel_diacritic', 'consonant_diacritic']]

mskf = MultilabelStratifiedKFold(n_splits = 5)

print('--------------------------------')
for fold, (trn_, val_) in enumerate(mskf.split(X,y)):
    # print(fold) #0~4 ->5개
    print('Train: ', trn_, 'Val ', val_)
    df.loc[val_, 'kfold'] = fold

# print(df.kfold.value_counts()) #40168개씩 5개 - 총 200840

df.to_csv('/gdrive/My Drive/kaggle/bengaliai-cv19/train_folds.csv', index = False)


