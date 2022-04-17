# -*- coding: utf-8 -*-
import atexit
from time import time, strftime, localtime
from datetime import timedelta
import re


import datetime

def _timestamp():
    now = datetime.datetime.now()
    now_str = now.strftime('%Y-%m-%d %H:%M')
    return now_str

def _timestamp_pretty():
    now = datetime.datetime.now()
    now_str = now.strftime('%Y%m%d%H%M')
    return now_str



def seconds_to_str(elapsed=None):
    """
    1. if elapsed is None, convert Timestamp to local time format
    2. else, return the duration expressing the difference
       between two specific points in time
    """
    if elapsed is None:
        return strftime("%Y-%m-%d %H:%M:%S", localtime())
    else:
        return str(timedelta(seconds=elapsed))


def timelog(s, elapsed=None):
    """
    1. if elapsed is None, return the time right now
    2. else, return the time right now and duration of run time
    """
    time_str = f"{'='*40}\n"\
               f"{seconds_to_str()} - {s}\n"\
               f"{'='*40}"
    
    if elapsed is None:
        res = time_str + '\n'
    else:
        res = time_str + '\n' + 'Elapsed time: ' + str(elapsed) + '\n'
    return res


def endlog():
    """
    when process is done, return process run time
    """
    end = time()
    elapsed = end-start
    print(timelog( 
              "End Program",
              seconds_to_str(elapsed))
          )


def create_time_interval():
    """
    return a list which contain two specific time
    """
    ## print('輸入格式(YYYYMMDD): ex.19940805')
    for i in range(20):
        if i ==19:
            raise Exception("Sorry, yor datetime format was wrong too many time")

        start_time = input('請輸入開始時間: ')
        # regular expression" 4個數字，再來一個0搭配一個1-9或一個1搭配一個0-2，最後為一個0-2加上一個0-9或3加上一個0-1
        if re.fullmatch(pattern=r'\d{4}(0[1-9]|1[0-2])([0-2][0-9]|3[0-1])' , string = start_time, flags=0) != None:
            break
        else:
            print('\33[101m 開始時間格式錯誤，請重新輸入')

    for i in range(20):
        if i ==19:
            raise Exception("Sorry, yor datetime format was wrong too many time")

        end_time = input('請輸入結束時間: ')
        if re.fullmatch(pattern=r'\d{4}(0[1-9]|1[0-2])([0-2][0-9]|3[0-1])' , string = end_time, flags=0) != None:
            break        
        else:
            print('\33[101m 結束時間格式錯誤，請重新輸入')  # will print red background
    return [start_time, end_time]


# process started time
start = time()
# when process is done, run enlog function first
atexit.register(endlog)