#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time : 2020/9/8 2:23 下午
# @Author : alan
# @Site : 
# @File : factor.py
# @Software: PyCharm

class Factor:
    '''
    因子
    '''
    def __init__(self,data,shape1,shape2,shape3,trade1,trade2,bs_a,bs_b,price1):
        self.data=data
        self.shape1=shape1
        self.shape2=shape2
        self.shape3=shape3
        self.trade1=trade1
        self.trade2=trade2
        self.bs_a=bs_a
        self.bs_b=bs_b
        self.price1=price1
    def shape(self,m_flag,zf0_flag,zf1_flag=0):
        '''
        形态
        :param m_flag: 几分钟标准
        :param zf0_flag: 振幅标准
        :param zf1_flag: 涨幅标准
        :return:
        '''
        if zf1_flag==0 :
            return self.data[self.shape1]>m_flag and self.data[self.shape2]<zf0_flag
        else :
            return self.data[self.shape1]>m_flag and self.data[self.shape2]<zf0_flag and self.data[self.shape3]>zf1_flag

    def trade_vol(self,v_flag):
        '''
        放量
        :param v_flag: 标准
        :return:
        '''
        return self.data[self.trade1]/self.data[self.trade1]>v_flag

    def bs_vol_a(self,bs_flag):
        '''
        b+s
        :param bs_flag:
        :return:
        '''
        return self.data[self.bs_a]>bs_flag
    def bs_vol_b(self,bs_flag):
        '''
        委bs
        :param bs_flag:
        :return:
        '''
        return self.data[self.bs_b]>bs_flag

    def price(self,p_flag):
        '''
        价格
        :param p_flag:
        :return:
        '''
        return self.data[self.price1] > p_flag