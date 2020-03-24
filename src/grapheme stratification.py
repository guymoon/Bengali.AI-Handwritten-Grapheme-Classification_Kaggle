import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold

# get data
nfold = 8
seed = 42

train_df = pd.read_csv('/workspace/inyong/inyong_datasets/kaggle/bengaliai-cv19/train.csv')
train_df['id'] = train_df['image_id'].apply(lambda x: int(x.split('_')[1]))
le = LabelEncoder()
train_df['grapheme_enc'] = le.fit_transform(train_df['grapheme'])

X = train_df[['id', 'grapheme_enc']].values
y = train_df['grapheme_enc'].values

train_df['fold'] = np.nan

# split data
skf = StratifiedKFold(n_splits=nfold, shuffle=True, random_state=seed)
for i, (_, val_index) in enumerate(skf.split(X, y)):
    train_df.iloc[val_index, -1] = i

train_df['fold'] = train_df['fold'].astype('int')

train_df['fold'].value_counts()

#output
cols = ['image_id','grapheme_root','vowel_diacritic','consonant_diacritic','grapheme','fold']
train_df[cols].to_csv('/workspace/inyong/inyong_datasets/kaggle/bengaliai-cv19/train_with_10_fold_42.csv', index=False)