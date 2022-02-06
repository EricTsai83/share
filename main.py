import pandas as pd
import tkinter as tk
import tkinter.messagebox as msg # messagebox要另行匯入，否則會出錯。
from collections import defaultdict

from TkinterDnD2 import TkinterDnD, DND_FILES
import json

import os.path
import re


def auto_create_frame_with_button():
    global fm_middle
    global btn_li
    global btn_idx_li
    global tag_name_li
    radiostr.set(3)
    for rb in rb_li:
        rb['variable']=radiostr
        
        
    def button_event(tag_name, btn_idx):
        global drop_result_dic
        
        if btn_li[btn_idx]['bg'] == '#2874A6':
            btn_li[btn_idx]['bg'] = '#6d9bc9'
            drop_result_dic[fm_top_text['text']].update({tag_name})
        else:
            btn_li[btn_idx]['bg'] = '#2874A6'
            drop_result_dic[fm_top_text['text']].remove(tag_name)
        
    try:
        fm_middle.pack_forget()
        fm_middle = tk.Frame(win)
        fm_middle.pack(side='top', fill='both')
    except:    
        fm_middle = tk.Frame(win)
        fm_middle.pack(side='top', fill='both')
        
    

    
    try:
        key_li = list(dic.keys())
        ind = key_li.index(fm_top_text['text'])+1
        # asign top frame text
        fm_top_text['text'] = key_li[ind]  # id task is already done will get IndexError
        
        # asign empty list to brand key value in dictionary
        good_result_dic.update({fm_top_text['text']:set()})
        undecide_result_dic.update({fm_top_text['text']:set()})
        drop_result_dic.update({fm_top_text['text']:set()})

        tag_li = dic[key_li[ind]]


        row_pos = -1
        col_pos = -1
        i = -1
        btn_li = []
        btn_idx_li = []
        tag_name_li = []
        for btn_idx, tag_name in enumerate(tag_li):
            col_pos += 1
            i += 1
            if i % 5==0:
                row_pos += 1
                col_pos = 0

            # Specify Grid  
            tk.Grid.rowconfigure(fm_middle, row_pos, weight=1)
            tk.Grid.columnconfigure(fm_middle, col_pos, weight=1)

            btnstr = tk.StringVar()
            btnstr.set(f'{tag_name}')

            # pad是指兩個元件之間空出多少距離
            # Tkinter 使用文字單位而不是英寸或畫素來測量寬度和高度的原因是，該文字單位可確保 Tkinter 在不同平臺上的一致行為。
            btn_li.append(tk.Button(fm_middle,
                                    bg='#2874A6',  # '#F95E62'
                                    fg='white',  # '#2874A6'
                                    textvariable=btnstr,
                                    borderwidth = 1,
                                    font=('微軟正黑體', 16),
                                    command = lambda x=tag_name, y=btn_idx: button_event(x,y),
                                    padx=20, height=1, width=15))

            btn_li[btn_idx].grid(row=row_pos, column=col_pos) 
            btn_idx_li.append(btn_idx)
            tag_name_li.append(tag_name)
    except IndexError:
        msg.showinfo('Notice', 'Congratulations. You done this.\nbye~bye~')
        win.destroy()
    except NameError:  # 發生在沒有 load 資料就按送出鍵
        msg.showinfo('Notice', 'You need to drag a file to the textarea first.')
        
        
def button_event(tag_name, btn_idx):
    global drop_result_dic
    
    if btn_li[btn_idx]['bg'] != '#6d9bc9':
        btn_li[btn_idx]['bg'] = '#6d9bc9'
        drop_result_dic[fm_top_text['text']].update({tag_name})
    else:
        btn_li[btn_idx]['bg'] = '#2874A6'
        drop_result_dic[fm_top_text['text']].remove(tag_name)
        


def select():
    global good_result_dic
    global undecide_result_dic
    global drop_result_dic
    
    if radiostr.get()==1:
        def button_event(tag_name, btn_idx):
            if btn_li[btn_idx]['bg'] != '#FF9955':
                btn_li[btn_idx]['bg'] = '#FF9955'  # 橘
                good_result_dic[fm_top_text['text']].update({tag_name})
                undecide_result_dic[fm_top_text['text']].discard(tag_name)
                drop_result_dic[fm_top_text['text']].discard(tag_name)
            else:
                btn_li[btn_idx]['bg'] = '#2874A6'  # 藍
                good_result_dic[fm_top_text['text']].remove(tag_name)
    elif radiostr.get()==2:
        def button_event(tag_name, btn_idx):
            if btn_li[btn_idx]['bg'] != '#c96d6d':
                btn_li[btn_idx]['bg'] = '#c96d6d'
                undecide_result_dic[fm_top_text['text']].update({tag_name})
                good_result_dic[fm_top_text['text']].discard(tag_name)
                drop_result_dic[fm_top_text['text']].discard(tag_name)
            else:
                btn_li[btn_idx]['bg'] = '#2874A6'
                undecide_result_dic[fm_top_text['text']].remove(tag_name)
        
    elif radiostr.get()==3:
        def button_event(tag_name, btn_idx):
            if btn_li[btn_idx]['bg'] != '#6d9bc9':
                btn_li[btn_idx]['bg'] = '#6d9bc9'
                drop_result_dic[fm_top_text['text']].update({tag_name})
                good_result_dic[fm_top_text['text']].discard(tag_name)
                undecide_result_dic[fm_top_text['text']].discard(tag_name) 
            else:
                btn_li[btn_idx]['bg'] = '#2874A6'
                drop_result_dic[fm_top_text['text']].remove(tag_name)
    else: pass
        
    for btn, btn_idx, tag_name in zip(btn_li, btn_idx_li, tag_name_li):
        btn['command'] = lambda x=tag_name, y=btn_idx: button_event(x,y)
        
        
def initial_create_button():
    global fm_middle
    global btn_li
    global btn_idx_li
    global tag_name_li
    # asign empty list to brand key value in dictionary
    good_result_dic.update({fm_top_text['text']:set()})
    undecide_result_dic.update({fm_top_text['text']:set()})
    drop_result_dic.update({fm_top_text['text']:set()})
    
    
    
    try:
        fm_middle.pack_forget()
        fm_middle = tk.Frame(win)
        fm_middle.pack(side='top', fill='both')
    except:    
        fm_middle = tk.Frame(win)
        fm_middle.pack(side='top', fill='both')


    row_pos = -1
    col_pos = -1
    i = -1
    btn_li = []
    btn_idx_li = []
    tag_name_li = []
    for btn_idx, tag_name in enumerate(tag_li):
        col_pos += 1
        i += 1
        if i % 5==0:
            row_pos += 1
            col_pos = 0

        # Specify Grid  
        tk.Grid.rowconfigure(fm_middle, row_pos, weight=1)
        tk.Grid.columnconfigure(fm_middle, col_pos, weight=1)

        btnstr = tk.StringVar()
        btnstr.set(f'{tag_name}')

        # pad是指兩個元件之間空出多少距離
        # Tkinter 使用文字單位而不是英寸或畫素來測量寬度和高度的原因是，該文字單位可確保 Tkinter 在不同平臺上的一致行為。
        btn_li.append(tk.Button(fm_middle,
                                bg='#2874A6',  # '#F95E62'
                                fg='white',  # '#2874A6'
                                textvariable=btnstr,
                                borderwidth = 1,
                                font=('微軟正黑體', 16),
                                command = lambda x=tag_name, y=btn_idx: button_event(x,y),
                                padx=20, height=1, width=15))

        btn_li[btn_idx].grid(row=row_pos, column=col_pos)

        btn_idx_li.append(btn_idx)
        tag_name_li.append(tag_name)
        
        
        
def load_json_from_UI(event):
    global brand_li
    global drop_result_dic
    global good_result_dic
    global undecide_result_dic
    global tag_li
    global dic
    global data_dir
    global df
    data_dir = event.data
    
    textarea.delete("1.0","end")

    if event.data.endswith(".json"):
        with open(event.data, "r") as file:
            dic = json.load(file) 

        brand_li = list(dic.keys())
        drop_result_dic = defaultdict(set)
        good_result_dic = defaultdict(set)
        undecide_result_dic = defaultdict(set)
        textarea.insert('end', event.data)
        
        # global fm_top_text
        fm_top_text['text'] = list(dic.keys())[0]
        
        tag_li = dic[list(dic.keys())[0]]
        initial_create_button()
    elif event.data.endswith(".csv"):
        df = pd.read_csv(event.data, encoding='big5', index_col=False)
        dic = {key:eval(val) for key, val in zip(df.iloc[:,0], df.iloc[:,1])}
        
        brand_li = list(dic.keys())
        drop_result_dic = defaultdict(set)
        good_result_dic = defaultdict(set)
        undecide_result_dic = defaultdict(set)
        textarea.insert('end', event.data)
        
        # global fm_top_text
        fm_top_text['text'] = list(dic.keys())[0]
        
        tag_li = dic[list(dic.keys())[0]]
        initial_create_button() 

    else:
        msg.showinfo('Notice', 'Only support JSON file.')





def quit_shortcut(event):
    win.destroy()
    
    
def submit_shortcut(event):
    auto_create_frame_with_button()
    
    
def main():
    # define global variable
    global win
    global fm_top
    global fm_top_text
    global radiostr
    global rb_li
    global fm_bottom
    global change_brand_btn
    global quit_btn
    global textarea
    global drop_result_dic
    global good_result_dic
    global undecide_result_dic
    
    try:
        win = TkinterDnD.Tk()
        win.title('標籤查核')
        win.resizable(True, True)
        # win.iconbitmap('../Img/gudetama.ico')
        win.geometry(f'1120x300')

        ########################################################################################
        ########################################################################################
        # brand frame
        fm_top = tk.Frame(win, bd=20)
        fm_top.pack(side='top', fill='both')

        fm_top_text = tk.Label(
                            fm_top,
                            anchor='center',
                            fg='black',
                            text=None,
                            font=('微軟正黑體', 20, 'bold'), 
                            padx=10, pady=15)
        fm_top_text.place(in_= fm_top, relx=0.5, rely=0.5, anchor='center')


        evaluate = [("超讚", 1), ("待定", 2), ("丟棄", 3)]
        radiostr = tk.IntVar()
        radiostr.set(3)
        rb_li = []
        for e, num in evaluate:
            rb = tk.Radiobutton(fm_top,
                                text=e,
                                variable=radiostr,
                                value=num,
                                command=select)
            rb.pack(side='right')
            rb_li.append(rb)
        ########################################################################################
        ########################################################################################

        # 中間的區塊等到拉資料到UI介面後，再形成

        ########################################################################################
        ########################################################################################
        fm_bottom = tk.Frame(win)
        fm_bottom.pack(side='bottom', fill='both')


        btnstr = tk.StringVar()
        # btnstr.set('送出 <Enter>')
        change_brand_btn = tk.Button(fm_bottom,
                                     bg='#2874A6',  # '#F95E62'
                                     fg='white',  # '#2874A6'
                                     # textvariable=btnstr,
                                     borderwidth = 1,
                                     padx=20, height=1, width=15,
                                     text='送出 <Enter>',
                                     font=('微軟正黑體', 16),
                                     command=auto_create_frame_with_button,
                                     )
        change_brand_btn.pack(side='right')


        quit_btn = tk.Button(fm_bottom,
                             bg="OrangeRed3",
                             fg= "white",
                             borderwidth = 1,
                             padx=20, height=1, width=15, 
                             text = 'Quit and Save <ESC>',
                             font=('微軟正黑體', 16),
                             command = win.destroy)
        quit_btn.pack(side='left')


        # 檔案拖入視窗
        textarea = tk.Text(fm_bottom, height=2, width=40)
        textarea.pack(side='bottom')

        textarea.tag_config("tag_1", foreground="black", font=('微軟正黑體', 10))
        textarea.insert("end", "Drag JSON file here.", "tag_1")

        textarea.drop_target_register(DND_FILES)
        textarea.dnd_bind('<<Drop>>', load_json_from_UI)

        # shortcut
        win.bind("<Escape>", quit_shortcut)
        win.bind("<Return>", submit_shortcut)

        # 開始整個主程式
        win.mainloop()
    except:
        pass
    finally:
        try:
            # 因為 json 格式的資料不支援 set，所以要把value轉成list
            good_result_dic = {key: list(val) for key, val in good_result_dic.items()}
            drop_result_dic = {key: list(val) for key, val in drop_result_dic.items()}
            undecide_result_dic = {key: list(val) for key, val in undecide_result_dic.items()}

            # create output folder
            # Parent Directory path
            data_folder_dir = re.sub('[a-zA-z0-9]+\.(csv|json)', '', data_dir)
            # Directory
            directory = "output"
            # Path
            path = os.path.join(data_folder_dir, directory)
            # Create the directory
            try:
                os.makedirs(path, exist_ok = True)
                print("Directory '%s' created successfully" % directory)
            except OSError as error:
                print("Directory '%s' can not be created" % directory)

            try:
                final_df = df.copy()
            except NameError:  # 拉入 textarea 的資料不是 csv
                pass
        

            
            for dic_name, result_dic in zip(
                ['good_result', 'drop_result', 'undecide_result'],
                [good_result_dic, drop_result_dic, undecide_result_dic]
                ):
                if re.search('\.json$', data_dir):  # json output
                    file_name = re.search(r'[^/]+\.json$',  data_dir).group(0)
                    if os.path.isfile(f'{data_folder_dir}{directory}/{dic_name}_{file_name}.json'):
                        with open(f"{data_folder_dir}{directory}/{dic_name}_{file_name}", 'r') as fp:
                            old_dic = json.load(fp)

                        for key, val in result_dic.items():
                            old_dic[key] = val

                        with open(f"{data_folder_dir}{directory}/{dic_name}_{file_name}", 'w') as fp:
                            json.dump(old_dic, fp)                  
                    else:
                        with open(f"{data_folder_dir}{directory}/{dic_name}_{file_name}", 'w') as fp:
                            json.dump(result_dic, fp)
                            
                elif re.search('\.csv$', data_dir):  # csv output
                    file_name = re.search(r'[^/]+\.csv$',  data_dir).group(0)
                    temp_df = pd.DataFrame([result_dic]).T.reset_index()
                    temp_df.columns = [df.columns[0], dic_name]
                    final_df = final_df.merge(temp_df, on=df.columns[0], how='left')
                    final_df.to_csv(f"{data_folder_dir}{directory}/{file_name}",
                                    encoding='big5', index=False)

                else:
                    print('save data processing go wrong')
        except NameError:  # 發生情況為沒有拉入資料就關閉
            pass
            
            
if __name__ == '__main__':
    main()