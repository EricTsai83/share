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


### network graph
```python
import igraph as ig
import urllib.request, json
import plotly
import plotly.graph_objs as go
import json
import pandas as pd
import numpy as np
from tqdm.auto import tqdm

df = pd.read_json('./Data/network_graph_data.json')
df.head()
df['group'].unique()
len(df)
cos_sim_matrix = pd.read_json('./Data/cos_sim_matrix.json', orient='index')
cos_sim_matrix.head()

cos_sim_matrix = cos_sim_matrix.loc[cos+sim_matrix.index.isin(df['brand_name']), cos_sim_matrix.columns.isin(df['brand_name'])]
threshold = 0.93
remain_li = []
for col in cos_sim_matrix.columns:
    if sum(cos_sim_matrix[col]>=threshold) >= 2:
        remain_li.append(col)
        
        
cos_sim_matrix = cos_sim_matrix.loc[cos_sim_matrix.index.isin(remain_li), cos_sim_matrix.columns.isin(remain_li)]
df = df[df['brand_name'].isin(remain_li)].reset_index(drop=True)

# create index hash table
hash_table = {col: idx for idx, col in enumerate(cos_sim_matrix.columns)}
li = []
brand_li = []
for brand_col in tqdm(cos_sim_matrix.columns):
    brand_li.append(brand_col)
    for brand_idx in cos_sim_matrix.index:
        if brand_idx not in brand_li and cos_sim_matrix.loc[brand_idx, brand_col]>=threshold:
            li.append({
                'source': hash_table[brand_col],
                'target': hash_table[brand_idx],
                'value': cos_sim_matrix.loc[brand_idx, brand_col]                
            })
        else:
            pass
# create knowledge graph data
data = {}
data['node'] = [{'name': name, 'group': group} for name, group in zip(df['brand_name'], df['group'])]
data['links'] = li
len(li)

# Get the number of nodes
N = len(data['nodes'])
# Define the list of edges and the graph object from edges
L = len(data['links'])
Edges = [(data['links'][k]['source'], data['links'][k]['target']) for k in range(L)]
G = ig.Graph(Edges, directed=False)

# Extract the node attributes, 'group', and 'name'
data['node'][0]

labels = []
group = []
for node in data['nodes']:
labels.append(node['name'])
group.append(node['group'])


# Get the positions, set by the Kamada-Kawai layout for 3D graphs
layt = G.layout('kk', dim=3)  # It is a list of three elements list(the coordinates of nodes)
len(layt)

# Set data for the Plotly plot of graph
Xn = [layt[k][0] for k in range(N)]  # x-coordinates of nodes
Yn = [layt[k][1] for k in range(N)]  # y-coordinates of nodes
Zn = [layt[k][2] for k in range(N)]  # z-coordinates of nodes

Xe = []
Ye = []
Ze = []
for e in Edges:
    Xe += [ layt[e[0]][0], layt[e[1]][0], None ]  # x-coordinates of edge ends
    Ye += [ layt[e[0]][1], layt[e[1]][1], None ]
    Ze += [ layt[e[0]][2], layt[e[1]][2], None ]

trace1 = go.Scatter3d(
    x = Xe,
    y = Ye,
    z = Ze,
    mode = 'lines',
    line=dic(color='rgb(125,125,125)', width=1),
    hoverinfo='none'
)

trace2 = go.Scatter3d(
    x=Xn,
    y=Yn,
    z=Zn,
    mode='markers+text',
    name='actors',
    marker=dic(
        symbol='circle',
        size=6,
        color=group,
        colorscale='Viridis',
        line=dic(color='rgb(50,50,50)', width=0.5)
    ),
    text=labels,
    hoverinfo='text'
)

axis = dic(
    showbackground=False,
    showline=False,
    zeroline=False,
    showgrid=False,
    showticklabels=False,
    title=''
)

layout = go.Layout(
    title='Network of attribute of brand',
    width=1000,
    height=1000,
    showlegend=False,
    scene=dic(
        xaxis=dic(axis),
        yaxis=dic(axis),
        zaxis=dic(axis)
    ),
    margin=dic(t=100),
    hovermode='closest',
    annotations=[
        dic(
            showarrow=False,
            text='',
            xref='paper',
            yref='paper',
            x=0,
            y=0.1,
            xanchor='left',
            yanchor='bottom',
            font=dic(size=14)
        )
    ]
)


data = [trace1, trace2]
fig = goFigure(data=data, layout=layout)
fig.update_traces(textposition='top center')
plotly.offline.plot(fig, auto_open=True), validation=False, filename='test.html', config={'displayModeBar': False}





















```
