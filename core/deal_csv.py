#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/9/8 11:19 上午
# @Author : alan
# @Site : 
# @File : deal_csv.py
# @Software: PyCharm

import numpy as np
import pandas as pd
import csv

def read_csv(url):
    '''
    默认跳过首行
    :param url: 文件路径
    :return:
    '''
    with open(url, encoding='utf-8') as f:
        data = np.loadtxt(f, str, delimiter=",", skiprows=1)
        print(data[:5])
    return  data

def get_data(url):
    '''
    获取数据
    :param url:
    :return:
    '''
    data = pd.read_csv(url)

    # Drop date variable
    data = data.drop(['DATE'], 1)

    # Dimensions of dataset
    n = data.shape[0]
    p = data.shape[1]

    # plt.plot (data['SP500'])
    # plt.show ()

    # Make data a np.array
    data = data.values

    # Training and test data
    train_start = 0
    train_end = int(np.floor(0.8 * n))
    test_start = train_end + 1
    test_end = n
    data_train = data[np.arange(train_start, train_end), :]
    data_test = data[np.arange(test_start, test_end), :]
    return data,data_train,data_test