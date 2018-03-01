import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
plt.rcParams['font.family']='SimHei' #顯示中文


#取0301當天的query log來查看
query_0306 = pd.read_csv('data/query_log/0306.csv', encoding = "utf-8", header=None)
query_0520 = pd.read_csv('data/query_log/0520.csv', encoding = "utf-8", header=None)

def dict_init(dict_input, key):
    dict_input[key] = {}
    dict_input[key]["count"] = {}
    dict_input[key]["total"] = 0
    dict_input[key]["max"] = 0
    dict_input[key]["min"] = 99999
    dict_input[key]["mean"] = 0

def dict_add(dict_input, key, time):
    num_total = dict_input[key]["total"]
    #print("total", dict_input[key]["total"])
    dict_input[key]["total"] += 1
    day = datetime.datetime.fromtimestamp(time)
    day = day.month*12 + day.day
    if str(day) in dict_input[key]["count"]:
        dict_input[key]["count"][str(day)] += 1
    else:
        dict_input[key]["count"][str(day)] = 1

def file_count(dict_input, file):
    print("[Debug] in file_count")
    for i,item in enumerate(file.iterrows()):
        #if i > 1000: break
        if item[1][0] in dict_input:
            dict_add(dict_input, item[1][0], item[1][2])
        else:
            dict_init(dict_input, item[1][0])
            dict_add(dict_input, item[1][0], item[1][2])

dict_input = {}
file_count(dict_input, query_0306)
file_count(dict_input, query_0520)

def wrapup(dict_input):
    for key in dict_input:
        total = 0
        num_days = 0
        for key_day in dict_input[key]["count"]:
            v_max = dict_input[key]["max"]
            v_min = dict_input[key]["min"]
            num_days += 1
            total += dict_input[key]["count"][key_day]
            if dict_input[key]["count"][key_day] > v_max:
                v_max = dict_input[key]["count"][key_day]
                dict_input[key]["max"] = v_max
            if dict_input[key]["count"][key_day] < v_min:
                v_min = dict_input[key]["count"][key_day]
                dict_input[key]["min"] = v_min
        mean = total / num_days
        dict_input[key]["mean"] = mean
        dict_input[key]["total"] = total
        del dict_input[key]["count"]

wrapup(dict_input)
print(pd.DataFrame(dict_input).T.reset_index())


