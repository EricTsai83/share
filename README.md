### convert fullwidth to halfwidth
```python
def strQ2B(ustring):
    rstring = ''
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:
            inside_code = 32
        elif 65281 <= inside_code <= 65374:
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring
```


### pandas.DataFrame.pipe
https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.pipe.html



### kmeans for network graph
```python
import numpy as np
import pandas as pd
import pickle
from tqdm.auto import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

with open('./Data/df_brand_result_alpha.pickle', 'rb') as f:
    df = pickle.load(f)
    
df = df.drop(['Attribute_Freq15', 'Attribute_Freq15_w2v_filter', 'Attribute_diff'], axis=1)
df.columns = ['brand_name', 'brand_attr', 'brand_attr_vec']
df.head()

li = [
    'CONSTANT 康斯登', 'SEIKO 精工', 'Calvin Klein 凱文克萊', 'ONOLA', 'ORIENT 東方錶', 'CITIZEN 星辰',
    'adidas 愛迪達', 'NIKE 耐吉', 'native', 'SKECHERS', 'PUMA', 'BROOKS', 'K-SWISS', 'asics 亞瑟士',
    'T 世家', '立品茶園', '喝茶閒閒', 'xiao de tea 茶曉得', '名池茶葉', '好茶在人間', 'TEAMTE',
    '智力', '孩子國', '南一', '中華教育', '國語日報', '康軒'
]
df = df.query("brand_name in @li")
# remove brand which attr is empty
df = df[~(df['brand_attr_vec'].apply(len)==0)].reset_index(drop=True)
# compute average word vector
df['average_word2vec'] = [np.mean(row, axis=0) for row in df['brand_attr_vec']]
df.head()

def _cosine_similarity(x, y):
    return 0.5 + 0.5 * cosine_similarity(x, y)
    
X = Y = df['average_word2vec'].tolist()
cos_sim_matrix = pd.DataFrame(cos_sim_matrix, columns=col_id_li).set_index(pd.Index(col_id_li))
k = 4
x = np.stack(df['average_word2vec'], axis=0)
kmeans = KMeans(n_clusters=k, random_state=42)
clusters_pred = kmeans.fit_predict(x)

df['group'] = kmeans.labels_

def _rmse(vec1, vec2):
    vdiff = vec1 - vec2
    rmse = np.sqrt(np.mean(vdiff**2))
    return rmse
    
    
li_all = []
for v1 in tqdm(kmeans.cluster_centers_):
    li = []
    for v2 in kmeans.cluster_centers_:
        res = _rmse(v1, v2)
        li.append(res)
    li_all.append(li)
 
 
cluster_center_eclidean_dist = pd.DataFrame(li_all)

cluster_center_eclidean_dist


with open('./Data/network_graph_data.json', 'w', encoding='utf-8') as f:
    df.to_json(f, force_ascii=False, orient='records')
  
cos_sim_matrix.iloc[:5, :5]

with open('./Data/cos_sim_matrix.json', 'w', enconding='utf-8') as f:
    cos_sim_matrix.to_json(f, force_ascii=False, orient='index')
```
