"""
@author: Eric Tsai
@brief: utils for time
"""
import datetime

def current_timestamp():
    # getting the currently date and time
    dt = datetime.datetime.now()
    # getting the timestamp
    ts = datetime.datetime.timestamp(dt)
    return int(ts)


def current_time_pretty():
    now = datetime.datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
    return now_str


def current_time():
    now = datetime.datetime.now()
    now_str = now.strftime("%Y%m%d%H%M%S")
    return now_str