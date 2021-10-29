import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE
from imblearn.over_sampling import SVMSMOTE
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
import nltk
import re
sm = SMOTE(random_state=42)
cou=0

#fn =['cbow_Train', 'skg_Train']

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS =nltk.corpus.stopwords.words('english')

def clean_text(text):
    text = BeautifulSoup(text, "lxml").text # HTML decoding
    text = text.lower() # lowercase text
    text = REPLACE_BY_SPACE_RE.sub(' ', text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = BAD_SYMBOLS_RE.sub('', text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join(word for word in text.split() if word not in STOPWORDS) # delete stopwors from text
    return text

fn =['w2vec_train_labelled']
for i in range(0,1):
    fname='dataset/'+fn[i]+'.csv'
    df=np.genfromtxt(fname,delimiter=',')
    #df = pd.read_csv(fname,encoding='latin-1', usecols = ['tweet'])
    #datan = df
    datan=df[:,0:-1]
    out=df[:,-1]
    #out = pd.read_csv(fname,encoding='latin-1', usecols = ['label'])
    X_res, y_res = sm.fit_resample(datan,out)
    y=y_res.reshape(-1,1)
    d=np.concatenate((X_res,y),axis=1)
    fname='smote1_'+fn[i]+'.csv'
    np.savetxt(fname,d, delimiter=',', fmt='%f')







        
    
