!pip install pandas==1.2.3









import pandas as pd
from tqdm import tqdm
import numpy as np
import copy




def import_data_from_drive(id):
  !wget --save-cookies cookies.txt 'https://docs.google.com/uc?export=download&id='{id} -O- \
  | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1/p' > confirm.txt && wget --content-disposition --load-cookies cookies.txt 'https://docs.google.com/uc?export=download&id='{id}'&confirm='$(<confirm.txt)

import_data_from_drive('1sWQmlq9-CPfos1TOqiBxigN0BmPVr7Hs')
data = pd.read_json('/content/contacts.json')
data.to_csv('contacts.csv',index=False)

















df = pd.read_csv('contacts.csv', 
                 dtype = {'Id': int,'Email': str,'Phone': str, 'Contacts': int, 'OrderId': str},
                 engine = 'c')

Email_id = df.groupby('Email').agg({'Id': set}).rename(columns={'Id': 'same_email_id'}).reset_index()
Phone_id = df.groupby('Phone').agg({'Id': set}).rename(columns={'Id': 'same_phone_id'}).reset_index()
OrderId_id = df.groupby('OrderId').agg({'Id': set}).rename(columns={'Id': 'same_orderid_id'}).reset_index()

df = df.merge(Email_id, how='left', on='Email')
df = df.merge(Phone_id, how='left', on='Phone')
df = df.merge(OrderId_id, how='left', on='OrderId')

df[['same_email_id', 'same_phone_id', 'same_orderid_id']] = df[['same_email_id', 'same_phone_id', 'same_orderid_id']].replace({np.nan: set()})

def merge_column_value(x):
    s = set()
    s.update(x['same_email_id'])
    s.update(x['same_phone_id'])
    s.update(x['same_orderid_id'])
    return s

df['group_id'] = df[['same_email_id', 'same_phone_id', 'same_orderid_id']].apply(merge_column_value, axis=1)
df['group_id_count'] = df['group_id'].apply(len)

df = df.drop(['same_email_id', 'same_phone_id', 'same_orderid_id'], axis=1)

original_dic = copy.deepcopy(dict(zip(df['Id'], df['group_id'])))





dic = {}
for _, group_id in tqdm(original_dic.items()):
    li=[]
    for _id in group_id:
        if _id in dic:
            li.append(_id)
        else:
            pass
    if len(li)>0:
        s = set()
        [s.update(dic[i]) for i in li]
        s.update(group_id)
    else:
        s = group_id.copy()
    dic.update(
            dict( zip( s, [s]*len(s) ) )
        )        

    
    
    
data = [[dic[i]] for i in range(len(dic))]
result = pd.DataFrame(data)

result.columns = ['result']

df = pd.concat([df, result], axis=1)

del original_dic
del dic
del data

df['result'] = df['result'].apply(list)

def sort_list_value_and_convert_value_to_string(x):
    x = sorted(x)
    return '-'.join([str(i) for i in x])

df['result'] = df['result'].apply(sort_list_value_and_convert_value_to_string)

by_result = df.groupby('result').agg({'Contacts': 'sum'}).reset_index()
by_result['Contacts'] = by_result['Contacts'].apply(str)

by_result['final_result'] = by_result['result'] + ', ' + by_result['Contacts']

df = df.merge(by_result, on='result', how='left')