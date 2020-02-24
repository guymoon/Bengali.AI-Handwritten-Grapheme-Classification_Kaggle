import pandas as pd
import joblib
import glob
from tqdm import tqdm

if __name__ == '__main__':
#return a list of paths matching a pathname pattern
files = glob.glob('/gdrive/My Drive/kaggle/bengaliai-cv19/train_*.parquet')

    for f in files:
        #parquet 파일 읽고
        df = pd.read_parquet(f)
        #image_id의 valuese들만 따로 저장하고,image_id col 날려
        image_ids = df.image_id.values
        df = df.drop('image_id', axis = 1)
        #
        image_array = df.values
        
        #
        for j, img_id in tqdm(enumerate(image_ids), total = len(image_ids)):
            joblib.dump(image_array[j, :],
                        f'/gdrive/My Drive/kaggle/bengaliai-cv19/image_pickles/{img_id}.pkl')
