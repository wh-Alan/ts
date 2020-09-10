#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/9/8 3:08 下午
# @Author : alan
# @Site : 
# @File : common.py
# @Software: PyCharm
from conf import settings
import logging
import logging.config
import json

def get_logger(name):
    logging.config.dictConfig(settings.LOGGING_DIC)  # 导入上面定义的logging配置
    logger = logging.getLogger(name)  # 生成一个log实例
    return logger


def conn_db():
    db_path=settings.DB_PATH
    dic=json.load(open(db_path,'r',encoding='utf-8'))
    return dic