# -*- coding: utf-8 -*-
"""
Created on Tue Jun 30 10:22:11 2020


@author: zhou kun CPSO-2D reconstruction CNN fuzzy penalty
"""

# coding: utf-8
import numpy as np
import random
import matplotlib.pyplot as plt
import scipy.io
import keras
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import RepeatedKFold
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, normalization, AveragePooling2D
from keras.layers import Dense, Dropout, Flatten
from keras.utils import np_utils
from keras import backend as K
import random
import pandas as pd
from keras.callbacks import ReduceLROnPlateau
from keras import regularizers
from keras.layers import LeakyReLU
import math
import gc 
from keras.regularizers import l2  #导入正则化l2（小写L）
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = ""
 
# ----------------------PSO参数设置---------------------------------
class PSO():
    def __init__(self, pN, dim, max_iter):
        self.w = 0.8
        self.c1 = 2
        self.c2 = 2
        self.r1 = random.random()
        self.r2 = random.random()
        self.pN = pN  # 粒子数量
        self.dim = dim  # 搜索维度
        self.max_iter = max_iter  # 迭代次数
        self.X = np.zeros((self.pN, self.dim))  # 所有粒子的位置和速度
        self.V = np.zeros((self.pN, self.dim))
        self.pbest = np.zeros((self.pN, self.dim))  # 个体经历的最佳位置和全局最佳位置
        self.gbest = np.zeros((1, self.dim))
        self.p_fit = np.zeros(self.pN)  # 每个个体的历史最佳适应值
        self.fit = 10  # 全局最佳适应值
        self.result = 0.1 # 最终结果
        self.down = 1
        self.up = 4
        self.dim_inp = [1,2,3,4]   #[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]
    # ---------------------目标函数Sphere函数-----------------------------
    def function(self, X, tt):
        Setup_data = 2  # 1. 
        Setup_reconstruction = 'PSO' 
        
        if Setup_data == 1:
            #data_path = "D:\데이터및프로그램\ucidata"
            df = pd.read_table('D:/데이터및프로그램/ucidata/Heart.dat',sep = '\t',header=None, engine='python')
            #load_matrix = np.fromfile('D:/데이터및프로그램/ucidata/wine.dat', dtype=int)
            df = df.dropna(axis=1)
            dataName = 'Heart'                # 'vehicle'  'sonar'  'wdbc'  'wine' 'pima' 'vowel' 
            load_matrix = df.values           # balance123 Heart Liver transfusion
        elif Setup_data == 2:  # R7
            data_path = r"C:\Users\Administrator\Desktop\desk 19.6.7\1Dto2Duci19.9.8\ucidata\iris.mat"  
            #  breastTissue winequality_red  banknote_authentication  iris ecoli  vertebral_column yeast seeds climate_model_simulation_crashes glass
            #  leafnew plrx   wireless_indoor_localization   hill_valley  discharge2Dmean   R7_9  birdhorse1t
            data = scipy.io.loadmat(data_path)
            load_matrix = data['iris']   #leaf
            dataName = 'iris'                  # discharge
            
        (N, D) = load_matrix.shape
        data_label = load_matrix[:, D - 1]
        data_order = X.reshape(D - 1, D - 1) - 1
        data_order = data_order.astype('int64')
        
        if dataName == 'vehicle' :             #or 'sonar' or 'wdbc' or 'wine'
           data_label = data_label - 1
        elif dataName == 'sonar' :
            data_label = data_label - 1
        elif dataName == 'wdbc' :    
            data_label = data_label - 1
        elif dataName == 'wine' :   
            data_label = data_label - 1
        elif dataName == 'discharge' :   
            data_label = data_label - 1   
        elif dataName == 'balance123' :   
            data_label = data_label - 1 
        elif dataName == 'seeds' :   
            data_label = data_label - 1     
        elif dataName == 'glass' :   
            data_label = data_label - 1    
        elif dataName == 'R1' :      # R7_9
            data_label = data_label - 1       
        
        data_all = []
        acc = []
        Acc = []
        Val_acc = []
        B = []
        C = []
        Y_pre = []
        Y_real = []
            
        for i in range(0, D - 1):
            data_load = load_matrix[:, i]
            data_all.append(data_load)
        
        data_all = np.transpose(data_all)
        #data_all = abs(data_all)   
        num_classes = 3
        batch_size = 30
        nb_epoch = 50
               
        learn_rate = 0.001
                
        
        X_train, X_test, y_train, y_test = train_test_split(data_all, data_label, test_size=0.2, random_state=2)

        (N_train, D_train) = X_train.shape
        (N_test, D_test) = X_test.shape
    
            
        if Setup_reconstruction == 'PSO':
            X_train_2D = np.ones((D_train,D_train), dtype=np.float64) 
            X_test_2D = np.ones((D_test,D_test), dtype=np.float64)
            sum_X_train_2D = np.ones((N_train, D_train,D_train), dtype=np.float64) 
            sum_X_test_2D = np.ones((N_test, D_test,D_test), dtype=np.float64) 
            
        
            
            for k in range(0, N_train):
                for ii in range(D_train):
                    a = X_train[k]
                    b = a[data_order[ii]]
                    X_train_2D[ii, :] = b
                    
                sum_X_train_2D[k, :, :] = X_train_2D
                
            for l in range(0, N_test):
                for ll in range(D_train):
                    c = X_test[l]
                    d = c[data_order[ll]]
                    X_test_2D[ll, :] = d
                    
                sum_X_test_2D[l, :, :] = X_test_2D  
        
        (W_1, H_1) = X_train_2D.shape
        #X_train = sum_X_train_2D
        X_train = sum_X_train_2D.reshape(sum_X_train_2D.shape[0], W_1, H_1, 1)
        #X_test = sum_X_test_2D
        X_test = sum_X_test_2D.reshape(sum_X_test_2D.shape[0], W_1, H_1, 1)
    
        y_train = np_utils.to_categorical(y_train, num_classes)
        y_real = y_test
        y_test = np_utils.to_categorical(y_test, num_classes)
        
        model = Sequential()
        # model.add(normalization.BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001, input_shape=(W_1, H_1, 1)))
        model.add(Conv2D(16, kernel_size=(2, 2), strides=(1, 1), padding='same',  input_shape=(W_1, H_1, 1)))  #kernel_regularizer=regularizers.l2(0.01),
        model.add(LeakyReLU(alpha=0.4))
        
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'))
       
        model.add(Conv2D(4, kernel_size=(2, 2), strides=1, padding='same'))
        model.add(LeakyReLU(alpha=0.4))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=1, padding='same'))
       
        model.add(Flatten())
       
        model.add(Dense(50, activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(num_classes, activation='softmax'))
        model.compile(loss=keras.losses.categorical_crossentropy,
                                 
                                  optimizer=keras.optimizers.Adam(lr=learn_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
                                  metrics=['accuracy'])

    
        h = model.fit(X_train, y_train,
                      batch_size=batch_size,
                      epochs=nb_epoch,
                      verbose=0,
                      shuffle=None,
                      validation_data=(X_test, y_test))
    
        
        acc, val_acc = h.history['accuracy'], h.history['val_accuracy']
    
        # B = np.array(acc)
        t_acc = acc[-1]
        t_val_acc = val_acc[-1]
        Acc = np.append(Acc, t_acc)
        Val_acc = np.append(Val_acc, t_val_acc)
            
        B = np.array(Acc)
        C = np.array(Val_acc)
        
        
        meanacc = np.mean(B, axis=0)
        meanval_acc = np.mean(C, axis=0)
        stdacc = np.std(B, axis=0)
        stdval_acc = np.std(C, axis=0)    
        del load_matrix, data_all, X_train, X_train_2D, X_test_2D, sum_X_train_2D, sum_X_test_2D, X_test, acc, val_acc, t_acc, t_val_acc, Acc,
        Val_acc, h
        gc.collect()
        return meanacc, stdacc, meanval_acc, stdval_acc
 
    # ---------------------初始化种群----------------------------------
    def init_Population(self):
        t = 1
        for i in range(self.pN):
            for j in range(0, self.dim, self.up):
                self.X[i][j:j + self.up] = random.sample(self.dim_inp, self.up)
                #self.V[i][j:j + self.up] = random.sample(self.dim_inp, self.up)
            for k in range(self.dim): 
                self.V[i][k] = random.randint(-1, 2)
            self.pbest[i] = self.X[i]
            tmp = self.function(self.X[i], t)
            self.p_fit[i] = 1 - tmp[0]
            #a = np.floor(self.X[i])
            if (1-tmp[0]) < self.fit:
                self.fit = 1-tmp[0]
                self.gbest = self.X[i].copy()
                self.result = tmp
 
                # ----------------------更新粒子位置----------------------------------
    def fisfunction(self, f_max, f_min, g_max, g_min, h_max, h_min, f1, g1, h1):      #Madani FIS
       
        #r_max = f_max - f_min
        f_norm = (f1-f_min)/(f_max-f_min)
        h_norm = (h1-h_min)/(h_max-h_min)
        g_norm = (g1-g_min)/(g_max-g_min)
        
        x_f_fitness_range = np.arange(0, 1.01, 0.01, np.float32)
        x_stain_range=np.arange(0, 1.01, 0.01, np.float32)
        x_oil_range=np.arange(0, 1.01, 0.01, np.float32)
        y_powder_range=np.arange(0, 1.01, 0.01, np.float32)
        
        # 创建模糊控制变量 
        x_f_fitness=ctrl.Antecedent(x_f_fitness_range, 'f_fitness or g')
        x_stain=ctrl.Antecedent(x_stain_range, 'stain')
        x_oil=ctrl.Antecedent(x_oil_range, 'oil')
        y_powder=ctrl.Consequent(y_powder_range, 'powder')
        
         
        # 定义模糊集和其隶属度函数
        
        x_f_fitness['LOW']=fuzz.gaussmf(x_f_fitness_range, 0, 0.27)
        
        x_f_fitness['HIGH']=fuzz.gaussmf(x_f_fitness_range, 1, 0.27)
        
        x_stain['LOW']=fuzz.gaussmf(x_stain_range,0, 0.27)
        
        x_stain['HIGH']=fuzz.gaussmf(x_stain_range, 1, 0.27)
        
        
        
        x_oil['LOW']=fuzz.gaussmf(x_oil_range, 0, 0.27)
        
        x_oil['HIGH']=fuzz.gaussmf(x_oil_range, 1, 0.27)
        
        
        y_powder['LOW']=fuzz.gaussmf(y_powder_range, 0, 0.18)
        y_powder['MEDIUM']=fuzz.gaussmf(y_powder_range, 0.5, 0.18)
        y_powder['HIGH']=fuzz.gaussmf(y_powder_range, 1, 0.18)
        
        # 设定输出powder的解模糊方法——质心解模糊方式
        y_powder.defuzzify_method='centroid'
        # 输出为N的规则
        rule0 = ctrl.Rule(antecedent=((x_f_fitness['HIGH'] & x_stain['LOW'] & x_oil['LOW']) |
        
                                      (x_f_fitness['HIGH'] & x_stain['HIGH'] & x_oil['LOW']) ),
        
                          consequent=y_powder['LOW'], label='rule 0')
        
        # 输出为M的规则
        rule1 = ctrl.Rule(antecedent=((x_f_fitness['LOW'] & x_oil['LOW']) |
        
                                      (x_f_fitness['HIGH'] & x_oil['HIGH'])),                             
        
                          consequent=y_powder['MEDIUM'], label='rule 1')
        
        # 输出为P的规则
        rule2 = ctrl.Rule(antecedent=((x_f_fitness['LOW'] & x_oil['HIGH'])),
                                     
        
                          consequent=y_powder['HIGH'], label='rule 2')
        
        # 系统和运行环境初始化
        system = ctrl.ControlSystem(rules=[rule0, rule1, rule2])
        sim = ctrl.ControlSystemSimulation(system)
        
        sim.input['f_fitness or g'] = f_norm
        sim.input['stain'] = h_norm
        sim.input['oil'] = g_norm
        
        sim.compute()   # 运行系统
        output_powder = sim.output['powder']
        
        # 打印输出结果
        #print(output_powder)
        output_p = output_powder*(h_norm+g_norm)
        del x_f_fitness_range, x_stain, x_oil_range, y_powder_range
        gc.collect()
        return output_p # , h_norm, g_norm

    def iterator(self):
        fitness = []
        f_function = []
        g_function = []
        h_function = []
        f_function_out = []
        g_function_out = []
        h_function_out = []
        for t in range(self.max_iter):
            for i in range(self.pN):  # 更新gbest\pbest
                penalty1 = 0
                penalty2 = 0
                temp = self.function(self.X[i], t)
                f_temp = 1-temp[0] 
                f_function = np.append(f_function, f_temp)
                f_max = max(f_function)
                f_min = min(f_function)
                
                
                for l in range(0, self.dim, self.up):
                    if sum(self.X[i][l:l + self.up]) != sum(self.dim_inp):
                        p1_temp = abs(sum(self.X[i][l:l + self.up]) - sum(self.dim_inp))
                        penalty1 = penalty1 + p1_temp
                                          
                    fea = self.X[i][l:l + self.up]
                    if len(set(fea)) == len(fea):
                        a12 = []
                    else: 
                        p2_temp = len(fea) - len(set(fea))                    
                        penalty2 = penalty2 + p2_temp
                       
                g_function = np.append(g_function, penalty1)  
                g_max = max(g_function)
                g_min = min(g_function)
                h_function = np.append(h_function, penalty2)
                h_max = max(h_function)
                h_min = min(h_function)
                
            for i in range(self.pN):
                if g_function[i] == 0 and h_function[i] == 0:
                    fit_temp =  f_function[i]
                else:
                    # start = time.clock()   # FIS
                    r = self.fisfunction(f_max, f_min, g_max, g_min, h_max, h_min, f_function[i], g_function[i], h_function[i])
                    fit_temp =  (f_temp*100 + r)/100
                    # print(time.clock() - start)    # FIS
            #for i in range(self.pN):
                
                if fit_temp < self.p_fit[i]:  # 更新个体最优
                    self.p_fit[i] = fit_temp
                    self.pbest[i] = self.X[i]
                    
                    if self.p_fit[i] < self.fit:  # 更新全局最优
                        self.gbest = self.X[i].copy()
                        self.fit = self.p_fit[i]
                        self.result = temp
            f_function_out.append(f_function)
            g_function_out.append(g_function)
            h_function_out.append(h_function)
            f_function = []
            g_function = []
            h_function = []   
            
            for i in range(self.pN):
                self.V[i] = self.w * self.V[i] + self.c1 * self.r1 * (self.pbest[i] - self.X[i]) + \
                            self.c2 * self.r2 * (self.gbest - self.X[i])
                self.X[i] = np.floor(self.X[i] + self.V[i])
                for j in range(self.dim):
                    if self.X[i][j] > self.up:
                        self.X[i][j] = self.up
                    elif self.X[i][j] < self.down:
                        self.X[i][j] = self.down
            fitness.append(self.fit)
            
            print('iter =', t)
            #print(self.gbest, end=" ")
            print(self.fit)  # 输出最优值
            print(self.result)
            #np.save('1.1_fg', self.gbest)
            #np.save('1.1_fg_f', f_function_out)
            #np.save('1.1_fg_g', g_function_out)
            #np.save('1.1_fg_h', h_function_out)
        return fitness
 
        # ----------------------程序执行-----------------------
 
# start = time.clock()   # FCPSO 
my_pso = PSO(pN=5, dim=16, max_iter=6)
my_pso.init_Population()
fitness = my_pso.iterator()
# print(time.clock() - start)    # FCPSO
# -------------------画图--------------------
# plt.figure(1)
# plt.title("Figure1")
# plt.xlabel("Generations", size=14)
# plt.ylabel("Fitness value", size=14)
# t = np.array([t for t in range(0, 42)])
# fitness = np.array(fitness)
# plt.plot(t, fitness, color='b', linewidth=3)
# plt.show()    