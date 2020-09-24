import pandas as pd
import numpy as np
import seaborn as sns

from pylab import *
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
from utils import myfct
import matplotlib.pyplot as plt

"""
Quelques unes de ces fonctions ont plusieurs version et parfois les premières versions ne sont plus utilisables. 
Cependant, les fonctions intéressantes sont observables dans jupyter notebooks "plan_exp_traitement" & "Online Scene by Scene y_exp".
"""
def error_rates_display_once(obj_error_rates,non_obj_error_rates,total_error_rates,X_axis,titre,y_lim,x_label):

    obj_meanst = np.zeros(len(obj_error_rates))
    obj_sdt = np.zeros(len(obj_error_rates))
    for i in range(len(obj_meanst)):
        for j in range(len(obj_error_rates[0])):
            obj_meanst[i] += obj_error_rates[i][j]
        obj_meanst[i] /= len(obj_error_rates[0])
    
    for i in range(len(obj_meanst)):
        for j in range(len(obj_error_rates[0])):
            obj_sdt[i] += (obj_error_rates[i][j]- obj_meanst[i])**2
        
        obj_sdt[i] /= len(obj_error_rates[0])
        obj_sdt[i] = obj_sdt[i]**(1/2)
        
    plt.figure(figsize=(18,10))
    fig, ax = plt.subplots(figsize=(18,10))
    clrs = sns.color_palette("husl", 5)
    with sns.axes_style("darkgrid"):
        epochs = X_axis#list(int((i)*nb_samples/len(obj_error_rates)) for i in range(1,len(obj_error_rates)+1))
        ax.plot(epochs, obj_meanst, c=clrs[3], marker='o',markerfacecolor=clrs[3], markersize=10, linewidth = 5, linestyle='dashed',label="Object Error Rate")
        ax.fill_between(epochs, obj_meanst-obj_sdt, obj_meanst+obj_sdt ,alpha=0.2, facecolor=clrs[3])
        #ax.fill_between(epochs, obj_meanst-3*obj_sdt, obj_meanst+3*obj_sdt ,alpha=0.3, facecolor=clrs[3])
        ax.legend()

        
    non_obj_meanst = np.zeros(len(non_obj_error_rates))
    non_obj_sdt = np.zeros(len(non_obj_error_rates))
    for i in range(len(non_obj_meanst)):
        for j in range(len(non_obj_error_rates[0])):
            non_obj_meanst[i] += non_obj_error_rates[i][j]
        non_obj_meanst[i] /= len(non_obj_error_rates[0])
    
    for i in range(len(non_obj_meanst)):
        for j in range(len(non_obj_error_rates[0])):
            non_obj_sdt[i] += (non_obj_error_rates[i][j]- non_obj_meanst[i])**2
        
        non_obj_sdt[i] /= len(non_obj_error_rates[0])
        non_obj_sdt[i] = non_obj_sdt[i]**(1/2)
        
    clrs = sns.color_palette("husl", 4)
    with sns.axes_style("darkgrid"):
        epochs = X_axis#list(int((i)*nb_samples/len(non_obj_error_rates)) for i in range(1,len(non_obj_error_rates)+1))
        ax.plot(epochs, non_obj_meanst, c=clrs[3], marker='o',markerfacecolor=clrs[3], markersize=10, linewidth = 5, linestyle='dashed',label="Non Object Error Rate")
        ax.fill_between(epochs, non_obj_meanst-non_obj_sdt, non_obj_meanst+non_obj_sdt ,alpha=0.2, facecolor=clrs[3])
        #ax.fill_between(epochs, non_obj_meanst-3*non_obj_sdt, non_obj_meanst+3*non_obj_sdt ,alpha=0.3, facecolor=clrs[3])


    total_meanst = np.zeros(len(total_error_rates))
    total_sdt = np.zeros(len(total_error_rates))
    for i in range(len(total_meanst)):
        for j in range(len(total_error_rates[0])):
            total_meanst[i] += total_error_rates[i][j]
        total_meanst[i] /= len(total_error_rates[0])
    
    for i in range(len(total_meanst)):
        for j in range(len(total_error_rates[0])):
            total_sdt[i] += (total_error_rates[i][j]- total_meanst[i])**2
        
        total_sdt[i] /= len(total_error_rates[0])
        total_sdt[i] = total_sdt[i]**(1/2)
        
    clrs = sns.color_palette("husl", 6)
    with sns.axes_style("darkgrid"):
        epochs = X_axis#list(int((i)*nb_samples/len(total_error_rates)) for i in range(1,len(total_error_rates)+1))
        ax.plot(epochs, total_meanst, c=clrs[3], marker='o',markerfacecolor=clrs[3], markersize=10, linewidth = 5, linestyle='dashed',label="Total Error Rate")
        ax.fill_between(epochs, total_meanst-total_sdt, total_meanst+total_sdt ,alpha=0.2, facecolor=clrs[2])
        #ax.fill_between(epochs, total_meanst-3*total_sdt, total_meanst+3*total_sdt ,alpha=0.3, facecolor=clrs[2])
        ax.legend()
        plt.title(titre,fontsize = 30)
        plt.xlabel(x_label,fontsize = 20)
        plt.ylabel('Error Rate',fontsize = 20)
        ax.yaxis.set_tick_params(which = 'major', length = 5, width = 3,
                           color = 'black', labelsize = 20, labelcolor = 'black')
        ax.xaxis.set_tick_params(which = 'major', length = 5, width = 3,
                           color = 'black', labelsize = 20, labelcolor = 'black')
        #ax.yaxis.grid(True, color = 'grey', linewidth = 0.5, linestyle = 'dashed')
        plt.gca().legend(fontsize = 20, loc = 'best')
        plt.ylim(y_lim)


def compare_error_rates_display_once2(error_rates,X_axis,titre,y_lim,x_label,curves_Names):
    colors = ["black","red","green","orange","purple","yellow","blue","brown"]
    if(len(error_rates)!=len(curves_Names)):
        raise Exception("len(error_rates)!=len(curves_Names)")
        
    plt.figure(figsize=(15,8))
    fig, ax = plt.subplots(figsize=(15,8))    
    for k in range(len(curves_Names)):
        meanst = np.zeros(len(error_rates[k]))
        sdt = np.zeros(len(error_rates[k]))
        for i in range(len(meanst)):
            for j in range(len(error_rates[k][0])):
                meanst[i] += error_rates[k][i][j]
            meanst[i] /= len(error_rates[k][0])
    
        for i in range(len(meanst)):
            for j in range(len(error_rates[k][0])):
                sdt[i] += (error_rates[k][i][j]- meanst[i])**2
        
            sdt[i] /= len(error_rates[k][0])
            sdt[i] = sdt[i]**(1/2)
        
        
        #clrs = sns.color_palette("husl", 5)
        with sns.axes_style("darkgrid"):
            epochs = X_axis
            ax.plot(epochs, meanst, marker=None, markersize=1, linewidth = 3,c=colors[k], linestyle='-',label=curves_Names[k])#"K Nearest Neighboors")
            ax.fill_between(epochs, meanst-sdt, meanst+sdt ,alpha=0.2, facecolor=colors[k])
            #ax.fill_between(epochs, meanst-3*sdt, meanst+3*sdt ,alpha=0.3, facecolor=clrs[3])
            ax.legend()


        plt.title(titre,fontsize = 30)
        plt.xlabel(x_label,fontsize = 20)
        plt.ylabel('Performance Score',fontsize = 20)
        plt.ylim(y_lim)
        ax.yaxis.set_tick_params(which = 'major', length = 5, width = 3,
                           color = 'black', labelsize = 20, labelcolor = 'black')
        ax.xaxis.set_tick_params(which = 'major', length = 5, width = 3,
                           color = 'black', labelsize = 20, labelcolor = 'black')
        #ax.yaxis.grid(True, color = 'grey', linewidth = 0.5, linestyle = 'dashed')
        plt.gca().legend(fontsize = 20, loc = 'best')
        
def compare_error_rates_display_once3(X_axis,titre,y_lim,x_label,error_rates_1,curves_Names_1,error_rates_2,curves_Names_2):
    colors = ["black","red","green","orange","purple","yellow","blue","brown"]
    if(len(error_rates_1)!=len(curves_Names_1)):
        raise Exception("len(error_rates)!=len(curves_Names)")
    if(len(error_rates_2)!=len(curves_Names_2)):
        raise Exception("len(error_rates)!=len(curves_Names)")
        
    plt.figure(figsize=(15,8))
    fig, ax = plt.subplots(figsize=(15,8))    
    
    for k in range(len(curves_Names_1)):
        meanst_1 = np.zeros(len(error_rates_1[k]))
        sdt_1 = np.zeros(len(error_rates_1[k]))
        for i in range(len(meanst_1)):
            for j in range(len(error_rates_1[k][0])):
                meanst_1[i] += error_rates_1[k][i][j]
            meanst_1[i] /= len(error_rates_1[k][0])
    
        for i in range(len(meanst_1)):
            for j in range(len(error_rates_1[k][0])):
                sdt_1[i] += (error_rates_1[k][i][j]- meanst_1[i])**2
        
            sdt_1[i] /= len(error_rates_1[k][0])
            sdt_1[i] = sdt_1[i]**(1/2)
        
        
        #clrs = sns.color_palette("husl", 5)
        with sns.axes_style("darkgrid"):
            epochs = X_axis
            ax.plot(epochs, meanst_1, marker=None, markersize=1, linewidth = 3,c=colors[k], linestyle='--')#"K Nearest Neighboors")
            #ax.fill_between(epochs, meanst_1-sdt_1, meanst_1+sdt_1 ,alpha=0.2, facecolor=colors[k])
            #ax.fill_between(epochs, meanst-3*sdt, meanst+3*sdt ,alpha=0.3, facecolor=clrs[3])
            #ax.legend()
            
    for k in range(len(curves_Names_2)):
        meanst_2 = np.zeros(len(error_rates_2[k]))
        sdt_2 = np.zeros(len(error_rates_2[k]))
        for i in range(len(meanst_2)):
            for j in range(len(error_rates_2[k][0])):
                meanst_2[i] += error_rates_2[k][i][j]
            meanst_2[i] /= len(error_rates_2[k][0])
    
        for i in range(len(meanst_2)):
            for j in range(len(error_rates_2[k][0])):
                sdt_2[i] += (error_rates_2[k][i][j]- meanst_2[i])**2
        
            sdt_2[i] /= len(error_rates_2[k][0])
            sdt_2[i] = sdt_2[i]**(1/2)
        
        
        #clrs = sns.color_palette("husl", 5)
        with sns.axes_style("darkgrid"):
            epochs = X_axis
            ax.plot(epochs, meanst_2, marker=None, markersize=1, linewidth = 3,c=colors[k], linestyle='-',label=curves_Names_2[k])#"K Nearest Neighboors")
            ax.fill_between(epochs, meanst_2-sdt_2, meanst_2+sdt_2 ,alpha=0.2, facecolor=colors[k])
            #ax.fill_between(epochs, meanst-3*sdt, meanst+3*sdt ,alpha=0.3, facecolor=clrs[3])
            ax.legend()


        plt.title(titre,fontsize = 30)
        plt.xlabel(x_label,fontsize = 20)
        plt.ylabel('Performance Score',fontsize = 20)
        plt.ylim(y_lim)
        ax.yaxis.set_tick_params(which = 'major', length = 5, width = 3,
                           color = 'black', labelsize = 20, labelcolor = 'black')
        ax.xaxis.set_tick_params(which = 'major', length = 5, width = 3,
                           color = 'black', labelsize = 20, labelcolor = 'black')
        #ax.yaxis.grid(True, color = 'grey', linewidth = 0.5, linestyle = 'dashed')
        plt.gca().legend(fontsize = 20, loc = 'best')
        
        
def displayNormalizedResults(title,batch_y_train_acc_exp,batch_y_train_acc_true,batch_X_train_acc_rough):
    min_len = len(batch_y_train_acc_exp[0])
    for i in range(len(batch_y_train_acc_exp)):
        if(len(batch_y_train_acc_exp[i])<min_len):
            min_len = len(batch_y_train_acc_exp[i])

    true_positive = np.ones(len(batch_y_train_acc_exp))
    true_negative = np.ones(len(batch_y_train_acc_exp))
    false_positive = np.ones(len(batch_y_train_acc_exp))
    false_negative = np.ones(len(batch_y_train_acc_exp))
    precisions = np.zeros((min_len,len(batch_y_train_acc_exp)))
    accuracies = np.zeros((min_len,len(batch_y_train_acc_exp)))
    recalls = np.zeros((min_len,len(batch_y_train_acc_exp)))

    for j in range(min_len):
        for i in range(len(batch_y_train_acc_exp)):
            if(batch_y_train_acc_exp[i][j] == batch_y_train_acc_true[i][j]):
                if(batch_y_train_acc_exp[i][j] ==1):
                    true_positive[i] += 1
                else:
                    true_negative[i] += 1
            else:
                if(batch_y_train_acc_exp[i][j] ==1):
                    false_positive[i] += 1
                else:
                    false_negative[i] += 1
            precisions[j][i] = true_positive[i]/(true_positive[i]+false_positive[i])
            accuracies[j][i] = (true_positive[i]+true_negative[i])/(true_positive[i]+true_negative[i]+false_positive[i]+false_negative[i])
            recalls[j][i] = true_positive[i]/(true_positive[i]+false_negative[i])
    
    
    X_axis = np.arange(min_len)   
    title = title
    y_lim = (0,1)
    x_label = "Number of interactions"

    prec_rec_accu = []
    curves_Names = []
    prec_rec_accu.append(precisions)
    curves_Names.append("precision")
    prec_rec_accu.append(recalls)
    curves_Names.append("recall")
    prec_rec_accu.append(accuracies)
    curves_Names.append("accuracy")
    compare_error_rates_display_once2(prec_rec_accu,X_axis,title,y_lim,x_label,curves_Names)
    
def displayNormalizedResults2(model,FPFHL_folder,nb_scenes,title,batch_y_train_acc_exp,batch_y_train_acc_true,batch_X_train_acc_rough):
    min_len = len(batch_y_train_acc_exp[0])
    for i in range(len(batch_y_train_acc_exp)):
        if(len(batch_y_train_acc_exp[i])<min_len):
            min_len = len(batch_y_train_acc_exp[i])

    true_positive = np.ones((min_len,len(batch_y_train_acc_exp)))
    true_negative = np.ones((min_len,len(batch_y_train_acc_exp)))
    false_positive = np.ones((min_len,len(batch_y_train_acc_exp)))
    false_negative = np.ones((min_len,len(batch_y_train_acc_exp)))
    precisions = np.zeros((min_len,len(batch_y_train_acc_exp)))
    accuracies = np.zeros((min_len,len(batch_y_train_acc_exp)))
    recalls = np.zeros((min_len,len(batch_y_train_acc_exp)))

    changed_first_sample = np.zeros(len(batch_y_train_acc_exp))

    X_test_, y_test_ = myfct.createTestData2(FPFHL_folder,nb_scenes,1)
    X_test_complet, y_test_complet = myfct.balanced_sample_maker(X_test_, y_test_)

    #num_batch = 0 
    #y_train_acc_exp = batch_y_train_acc_exp[num_batch]
    #X_train_acc_rough = batch_X_train_acc_rough[num_batch]

    
    for i in range(len(batch_y_train_acc_exp)):
        y_train_acc_exp = batch_y_train_acc_exp[i]
        X_train_acc_rough = batch_X_train_acc_rough[i]
        for j in range(2,min_len):
            if(j ==2 and y_train_acc_exp[0]==y_train_acc_exp[1]):
                y_train_acc_exp[0]= 1-y_train_acc_exp[0]
                changed_first_sample[i] = 1
            if(changed_first_sample[i] and len(y_train_acc_exp[:j])-sum(y_train_acc_exp[:j])>=2 and sum(y_train_acc_exp[:j])>=2):
                y_train_acc_exp[0]= 1-y_train_acc_exp[0]
                changed_first_sample[i] = 0 
            
            X_test_rough,y_true = X_test_complet, y_test_complet#myfct.getRandomDataTest(X_test_complet, y_test_complet,10000)
            X_all = myfct.normalize(np.concatenate((X_test_rough,X_train_acc_rough)))
            X_test,X_train = X_all[:len(X_test_rough)],X_all[len(X_test_rough):]
            #X_test = myfct.normalize(X_test_rough)
            #X_train = myfct.normalize(X_train_acc_rough)
            model.fit(X_train[:j],y_train_acc_exp[:j])
            y_pred = model.predict(X_test)###Ici on met un predict proba à 95% et bam on à seuillé; 
            #On va même faire un banc d'essai pour le seuillage!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            for k in range(len(y_true)):
                if(y_true[k] == 1 and y_pred[k] == 1):
                    true_positive[j][i] += 1
                elif(y_true[k] == 0 and y_pred[k] == 0):
                    true_negative[j][i] += 1
                elif(y_true[k] == 1 and y_pred[k] == 0):
                    false_negative[j][i] += 1
                elif(y_true[k] == 0 and y_pred[k] == 1):
                    false_positive[j][i] += 1
            precisions[j][i] = true_positive[j][i]/(true_positive[j][i]+false_positive[j][i])
            accuracies[j][i] = (true_positive[j][i]+true_negative[j][i])/(true_positive[j][i]+true_negative[j][i]+false_positive[j][i]+false_negative[j][i])
            recalls[j][i] = true_positive[j][i]/(true_positive[j][i]+false_negative[j][i])

    X_axis = np.arange(min_len)   
    title = title
    y_lim = (0.7,1)
    x_label = "Number of samples"

    prec_rec_accu = []
    curves_Names = []
    prec_rec_accu.append(precisions)
    curves_Names.append("precision")
    prec_rec_accu.append(recalls)
    curves_Names.append("recall")
    prec_rec_accu.append(accuracies)
    curves_Names.append("accuracy")
    compare_error_rates_display_once2(prec_rec_accu,X_axis,title,y_lim,x_label,curves_Names)
    
    
    
    
def displayNormalizedResults3(model,nb_scenes,title,FPFHL_folder_1,batch_y_train_acc_exp_1,batch_y_train_acc_true_1,batch_X_train_acc_rough_1,FPFHL_folder_2,batch_y_train_acc_exp_2,batch_y_train_acc_true_2,batch_X_train_acc_rough_2):
    min_len = len(batch_y_train_acc_exp_1[0]) + len(batch_y_train_acc_exp_2[0])
    for i in range(len(batch_y_train_acc_exp_1)):
        if(len(batch_y_train_acc_exp_1[i])<min_len):
            min_len = len(batch_y_train_acc_exp_1[i])
    for i in range(len(batch_y_train_acc_exp_2)):
        if(len(batch_y_train_acc_exp_2[i])<min_len):
            min_len = len(batch_y_train_acc_exp_2[i])

    true_positive_1 = np.ones((min_len,len(batch_y_train_acc_exp_1)))
    true_negative_1 = np.ones((min_len,len(batch_y_train_acc_exp_1)))
    false_positive_1 = np.ones((min_len,len(batch_y_train_acc_exp_1)))
    false_negative_1 = np.ones((min_len,len(batch_y_train_acc_exp_1)))
    precisions_1 = np.zeros((min_len,len(batch_y_train_acc_exp_1)))
    accuracies_1 = np.zeros((min_len,len(batch_y_train_acc_exp_1)))
    recalls_1 = np.zeros((min_len,len(batch_y_train_acc_exp_1)))
    
    true_positive_2 = np.ones((min_len,len(batch_y_train_acc_exp_2)))
    true_negative_2 = np.ones((min_len,len(batch_y_train_acc_exp_2)))
    false_positive_2 = np.ones((min_len,len(batch_y_train_acc_exp_2)))
    false_negative_2 = np.ones((min_len,len(batch_y_train_acc_exp_2)))
    precisions_2 = np.zeros((min_len,len(batch_y_train_acc_exp_2)))
    accuracies_2 = np.zeros((min_len,len(batch_y_train_acc_exp_2)))
    recalls_2 = np.zeros((min_len,len(batch_y_train_acc_exp_2)))
    

    changed_first_sample_1 = np.zeros(len(batch_y_train_acc_exp_1))
    changed_first_sample_2 = np.zeros(len(batch_y_train_acc_exp_2))

    X_test__1, y_test__1 = myfct.createTestData2(FPFHL_folder_1,nb_scenes,1)
    X_test_complet_1, y_test_complet_1 = myfct.balanced_sample_maker(X_test__1, y_test__1)
    
    X_test__2, y_test__2 = myfct.createTestData2(FPFHL_folder_2,nb_scenes,1)
    X_test_complet_2, y_test_complet_2 = myfct.balanced_sample_maker(X_test__2, y_test__2)

    #num_batch = 0 
    #y_train_acc_exp = batch_y_train_acc_exp[num_batch]
    #X_train_acc_rough = batch_X_train_acc_rough[num_batch]

    
    for i in range(len(batch_y_train_acc_exp_1)):
        y_train_acc_exp_1 = batch_y_train_acc_exp_1[i]
        X_train_acc_rough_1 = batch_X_train_acc_rough_1[i]
        for j in range(2,min_len):
            if(j ==2 and y_train_acc_exp_1[0]==y_train_acc_exp_1[1]):
                y_train_acc_exp_1[0]= 1-y_train_acc_exp_1[0]
                changed_first_sample_1[i] = 1
            if(changed_first_sample_1[i] and len(y_train_acc_exp_1[:j])-sum(y_train_acc_exp_1[:j])>=2 and sum(y_train_acc_exp_1[:j])>=2):
                y_train_acc_exp_1[0]= 1-y_train_acc_exp_1[0]
                changed_first_sample_1[i] = 0 
            
            X_test_rough_1,y_true_1 = X_test_complet_1, y_test_complet_1#myfct.getRandomDataTest(X_test_complet, y_test_complet,10000)
            X_all_1 = myfct.normalize(np.concatenate((X_test_rough_1,X_train_acc_rough_1)))
            X_test_1,X_train_1 = X_all_1[:len(X_test_rough_1)],X_all_1[len(X_test_rough_1):]
            #X_test = myfct.normalize(X_test_rough)
            #X_train = myfct.normalize(X_train_acc_rough)
            model.fit(X_train_1[:j],y_train_acc_exp_1[:j])
            y_pred_1 = model.predict(X_test_1)###Ici on met un predict proba à 95% et bam on à seuillé; 
            #On va même faire un banc d'essai pour le seuillage!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            for k in range(len(y_true_1)):
                if(y_true_1[k] == 1 and y_pred_1[k] == 1):
                    true_positive_1[j][i] += 1
                elif(y_true_1[k] == 0 and y_pred_1[k] == 0):
                    true_negative_1[j][i] += 1
                elif(y_true_1[k] == 1 and y_pred_1[k] == 0):
                    false_negative_1[j][i] += 1
                elif(y_true_1[k] == 0 and y_pred_1[k] == 1):
                    false_positive_1[j][i] += 1
            precisions_1[j][i] = true_positive_1[j][i]/(true_positive_1[j][i]+false_positive_1[j][i])
            accuracies_1[j][i] = (true_positive_1[j][i]+true_negative_1[j][i])/(true_positive_1[j][i]+true_negative_1[j][i]+false_positive_1[j][i]+false_negative_1[j][i])
            recalls_1[j][i] = true_positive_1[j][i]/(true_positive_1[j][i]+false_negative_1[j][i])
            
    for i in range(len(batch_y_train_acc_exp_2)):
        y_train_acc_exp_2 = batch_y_train_acc_exp_2[i]
        X_train_acc_rough_2 = batch_X_train_acc_rough_2[i]
        for j in range(2,min_len):
            if(j ==2 and y_train_acc_exp_2[0]==y_train_acc_exp_2[1]):
                y_train_acc_exp_2[0]= 1-y_train_acc_exp_2[0]
                changed_first_sample_2[i] = 1
            if(changed_first_sample_2[i] and len(y_train_acc_exp_2[:j])-sum(y_train_acc_exp_2[:j])>=2 and sum(y_train_acc_exp_2[:j])>=2):
                y_train_acc_exp_2[0]= 1-y_train_acc_exp_2[0]
                changed_first_sample_2[i] = 0 
            
            X_test_rough_2,y_true_2 = X_test_complet_2, y_test_complet_2#myfct.getRandomDataTest(X_test_complet, y_test_complet,10000)
            X_all_2 = myfct.normalize(np.concatenate((X_test_rough_2,X_train_acc_rough_2)))
            X_test_2,X_train_2 = X_all_2[:len(X_test_rough_2)],X_all_2[len(X_test_rough_2):]
            #X_test = myfct.normalize(X_test_rough)
            #X_train = myfct.normalize(X_train_acc_rough)
            model.fit(X_train_2[:j],y_train_acc_exp_2[:j])
            y_pred_2 = model.predict(X_test_2)###Ici on met un predict proba à 95% et bam on à seuillé; 
            #On va même faire un banc d'essai pour le seuillage!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            for k in range(len(y_true_2)):
                if(y_true_2[k] == 1 and y_pred_2[k] == 1):
                    true_positive_2[j][i] += 1
                elif(y_true_2[k] == 0 and y_pred_2[k] == 0):
                    true_negative_2[j][i] += 1
                elif(y_true_2[k] == 1 and y_pred_2[k] == 0):
                    false_negative_2[j][i] += 1
                elif(y_true_2[k] == 0 and y_pred_2[k] == 1):
                    false_positive_2[j][i] += 1
            precisions_2[j][i] = true_positive_2[j][i]/(true_positive_2[j][i]+false_positive_2[j][i])
            accuracies_2[j][i] = (true_positive_2[j][i]+true_negative_2[j][i])/(true_positive_2[j][i]+true_negative_2[j][i]+false_positive_2[j][i]+false_negative_2[j][i])
            recalls_2[j][i] = true_positive_2[j][i]/(true_positive_2[j][i]+false_negative_2[j][i])

    X_axis = np.arange(min_len)   
    title = title
    y_lim = (0.7,1)
    x_label = "Number of samples"

    prec_rec_accu_1 = []
    curves_Names_1 = []
    prec_rec_accu_1.append(precisions_1)
    curves_Names_1.append("precision")
    prec_rec_accu_1.append(recalls_1)
    curves_Names_1.append("recall")
    prec_rec_accu_1.append(accuracies_1)
    curves_Names_1.append("accuracy")
    
    prec_rec_accu_2 = []
    curves_Names_2 = []
    prec_rec_accu_2.append(precisions_2)
    curves_Names_2.append("precision")
    prec_rec_accu_2.append(recalls_2)
    curves_Names_2.append("recall")
    prec_rec_accu_2.append(accuracies_2)
    curves_Names_2.append("accuracy")
    
    compare_error_rates_display_once3(X_axis,title,y_lim,x_label,
                                                prec_rec_accu_1,curves_Names_1,prec_rec_accu_2,curves_Names_2)
    
    
    
    
def displayNormalizedResultsWithThreshold(model,nb_scenes,title,FPFHL_folder_1,batch_y_train_acc_exp_1,batch_y_train_acc_true_1,batch_X_train_acc_rough_1,FPFHL_folder_2,batch_y_train_acc_exp_2,batch_y_train_acc_true_2,batch_X_train_acc_rough_2,threshold):
    min_len = len(batch_y_train_acc_exp_1[0]) + len(batch_y_train_acc_exp_2[0])
    for i in range(len(batch_y_train_acc_exp_1)):
        if(len(batch_y_train_acc_exp_1[i])<min_len):
            min_len = len(batch_y_train_acc_exp_1[i])
    for i in range(len(batch_y_train_acc_exp_2)):
        if(len(batch_y_train_acc_exp_2[i])<min_len):
            min_len = len(batch_y_train_acc_exp_2[i])

    true_positive_1 = np.ones((min_len,len(batch_y_train_acc_exp_1)))
    true_negative_1 = np.ones((min_len,len(batch_y_train_acc_exp_1)))
    false_positive_1 = np.ones((min_len,len(batch_y_train_acc_exp_1)))
    false_negative_1 = np.ones((min_len,len(batch_y_train_acc_exp_1)))
    precisions_1 = np.zeros((min_len,len(batch_y_train_acc_exp_1)))
    accuracies_1 = np.zeros((min_len,len(batch_y_train_acc_exp_1)))
    recalls_1 = np.zeros((min_len,len(batch_y_train_acc_exp_1)))
    
    true_positive_2 = np.ones((min_len,len(batch_y_train_acc_exp_2)))
    true_negative_2 = np.ones((min_len,len(batch_y_train_acc_exp_2)))
    false_positive_2 = np.ones((min_len,len(batch_y_train_acc_exp_2)))
    false_negative_2 = np.ones((min_len,len(batch_y_train_acc_exp_2)))
    precisions_2 = np.zeros((min_len,len(batch_y_train_acc_exp_2)))
    accuracies_2 = np.zeros((min_len,len(batch_y_train_acc_exp_2)))
    recalls_2 = np.zeros((min_len,len(batch_y_train_acc_exp_2)))
    

    changed_first_sample_1 = np.zeros(len(batch_y_train_acc_exp_1))
    changed_first_sample_2 = np.zeros(len(batch_y_train_acc_exp_2))

    X_test__1, y_test__1 = myfct.createTestData2(FPFHL_folder_1,nb_scenes,1)
    X_test_complet_1, y_test_complet_1 = myfct.balanced_sample_maker(X_test__1, y_test__1)
    
    X_test__2, y_test__2 = myfct.createTestData2(FPFHL_folder_2,nb_scenes,1)
    X_test_complet_2, y_test_complet_2 = myfct.balanced_sample_maker(X_test__2, y_test__2)

    #num_batch = 0 
    #y_train_acc_exp = batch_y_train_acc_exp[num_batch]
    #X_train_acc_rough = batch_X_train_acc_rough[num_batch]

    
    for i in range(len(batch_y_train_acc_exp_1)):
        y_train_acc_exp_1 = batch_y_train_acc_exp_1[i]
        X_train_acc_rough_1 = batch_X_train_acc_rough_1[i]
        for j in range(2,min_len):
            if(j ==2 and y_train_acc_exp_1[0]==y_train_acc_exp_1[1]):
                y_train_acc_exp_1[0]= 1-y_train_acc_exp_1[0]
                changed_first_sample_1[i] = 1
            if(changed_first_sample_1[i] and len(y_train_acc_exp_1[:j])-sum(y_train_acc_exp_1[:j])>=2 and sum(y_train_acc_exp_1[:j])>=2):
                y_train_acc_exp_1[0]= 1-y_train_acc_exp_1[0]
                changed_first_sample_1[i] = 0 
            
            X_test_rough_1,y_true_1 = X_test_complet_1, y_test_complet_1#myfct.getRandomDataTest(X_test_complet, y_test_complet,10000)
            X_all_1 = myfct.normalize(np.concatenate((X_test_rough_1,X_train_acc_rough_1)))
            X_test_1,X_train_1 = X_all_1[:len(X_test_rough_1)],X_all_1[len(X_test_rough_1):]
            #X_test = myfct.normalize(X_test_rough)
            #X_train = myfct.normalize(X_train_acc_rough)
            model.fit(X_train_1[:j],y_train_acc_exp_1[:j])
            #y_pred_1 = model.predict(X_test_1)###Ici on met un predict proba à 95% et bam on à seuillé; 
            #On va même faire un banc d'essai pour le seuillage!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            y_pred_1_probs = model.predict_proba(X_test_1)
            y_pred_1 = np.zeros(len(y_true_1))
            for k in range(len(y_true_1)):
                if(y_pred_1_probs[k][1]>threshold):
                    y_pred_1[k] =1
                                            
            for k in range(len(y_true_1)):
                if(y_true_1[k] == 1 and y_pred_1[k] == 1):
                    true_positive_1[j][i] += 1
                elif(y_true_1[k] == 0 and y_pred_1[k] == 0):
                    true_negative_1[j][i] += 1
                elif(y_true_1[k] == 1 and y_pred_1[k] == 0):
                    false_negative_1[j][i] += 1
                elif(y_true_1[k] == 0 and y_pred_1[k] == 1):
                    false_positive_1[j][i] += 1
            precisions_1[j][i] = true_positive_1[j][i]/(true_positive_1[j][i]+false_positive_1[j][i])
            accuracies_1[j][i] = (true_positive_1[j][i]+true_negative_1[j][i])/(true_positive_1[j][i]+true_negative_1[j][i]+false_positive_1[j][i]+false_negative_1[j][i])
            recalls_1[j][i] = true_positive_1[j][i]/(true_positive_1[j][i]+false_negative_1[j][i])
            
    for i in range(len(batch_y_train_acc_exp_2)):
        y_train_acc_exp_2 = batch_y_train_acc_exp_2[i]
        X_train_acc_rough_2 = batch_X_train_acc_rough_2[i]
        for j in range(2,min_len):
            if(j ==2 and y_train_acc_exp_2[0]==y_train_acc_exp_2[1]):
                y_train_acc_exp_2[0]= 1-y_train_acc_exp_2[0]
                changed_first_sample_2[i] = 1
            if(changed_first_sample_2[i] and len(y_train_acc_exp_2[:j])-sum(y_train_acc_exp_2[:j])>=2 and sum(y_train_acc_exp_2[:j])>=2):
                y_train_acc_exp_2[0]= 1-y_train_acc_exp_2[0]
                changed_first_sample_2[i] = 0 
            
            X_test_rough_2,y_true_2 = X_test_complet_2, y_test_complet_2#myfct.getRandomDataTest(X_test_complet, y_test_complet,10000)
            X_all_2 = myfct.normalize(np.concatenate((X_test_rough_2,X_train_acc_rough_2)))
            X_test_2,X_train_2 = X_all_2[:len(X_test_rough_2)],X_all_2[len(X_test_rough_2):]
            #X_test = myfct.normalize(X_test_rough)
            #X_train = myfct.normalize(X_train_acc_rough)
            model.fit(X_train_2[:j],y_train_acc_exp_2[:j])
            #y_pred_2 = model.predict(X_test_2)###Ici on met un predict proba à 95% et bam on à seuillé; 
            #On va même faire un banc d'essai pour le seuillage!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            y_pred_2_probs = model.predict_proba(X_test_2)
            y_pred_2 = np.zeros(len(y_true_2))
            for k in range(len(y_true_2)):
                if(y_pred_2_probs[k][1]>threshold):
                    y_pred_2[k] =1

            for k in range(len(y_true_2)):
                if(y_true_2[k] == 1 and y_pred_2[k] == 1):
                    true_positive_2[j][i] += 1
                elif(y_true_2[k] == 0 and y_pred_2[k] == 0):
                    true_negative_2[j][i] += 1
                elif(y_true_2[k] == 1 and y_pred_2[k] == 0):
                    false_negative_2[j][i] += 1
                elif(y_true_2[k] == 0 and y_pred_2[k] == 1):
                    false_positive_2[j][i] += 1
            precisions_2[j][i] = true_positive_2[j][i]/(true_positive_2[j][i]+false_positive_2[j][i])
            accuracies_2[j][i] = (true_positive_2[j][i]+true_negative_2[j][i])/(true_positive_2[j][i]+true_negative_2[j][i]+false_positive_2[j][i]+false_negative_2[j][i])
            recalls_2[j][i] = true_positive_2[j][i]/(true_positive_2[j][i]+false_negative_2[j][i])

    X_axis = np.arange(min_len)   
    title = title
    y_lim = (0.7,1)
    x_label = "Number of samples"

    prec_rec_accu_1 = []
    curves_Names_1 = []
    prec_rec_accu_1.append(precisions_1)
    curves_Names_1.append("precision")
    prec_rec_accu_1.append(recalls_1)
    curves_Names_1.append("recall")
    prec_rec_accu_1.append(accuracies_1)
    curves_Names_1.append("accuracy")
    
    prec_rec_accu_2 = []
    curves_Names_2 = []
    prec_rec_accu_2.append(precisions_2)
    curves_Names_2.append("precision")
    prec_rec_accu_2.append(recalls_2)
    curves_Names_2.append("recall")
    prec_rec_accu_2.append(accuracies_2)
    curves_Names_2.append("accuracy")
    
    compare_error_rates_display_once3(X_axis,title,y_lim,x_label,
                                                prec_rec_accu_1,curves_Names_1,prec_rec_accu_2,curves_Names_2)
    
    
    
    
def error_rates_on_Dataset(num_batch,model,y_lim,FPFHL_folder,nb_scenes,title,batch_y_train_acc_pred,batch_y_train_acc_true,batch_X_train_acc_rough):
    #on créer un set de donnée équilibré aléatoire
    X_test_, y_test_ = myfct.createTestData2(FPFHL_folder,nb_scenes,1)
    X_test_complet, y_test_complet = myfct.balanced_sample_maker(X_test_, y_test_)
 
    y_train_acc_exp = batch_y_train_acc_pred[num_batch]
    X_train_acc_rough = batch_X_train_acc_rough[num_batch]


    error_rates = []
    min_i = 10
    for i in range(len(y_train_acc_exp)):
        if(len(y_train_acc_exp[:i])!=sum(y_train_acc_exp[:i])and len(y_train_acc_exp[:i])-sum(y_train_acc_exp[:i])!=len(y_train_acc_exp[:i])):
            min_i = i
            break
    for i in range(min_i,len(X_train_acc_rough)+1):
        X_train = myfct.normalize(X_train_acc_rough)
        X_test_rough,y_test = X_test_complet, y_test_complet#myfct.getRandomDataTest(X_test_complet, y_test_complet,10000)
        X_test = myfct.normalize(X_test_rough)

        model.fit(X_train[:i],y_train_acc_exp[:i])
        #for i in range(len(y_test)-1):
        #    y_test[len(y_test)-i-1] = y_test[len(y_test)-i-2]
        error_rates.append(myfct.error_pred(model.predict(X_test),y_test))

    obj_error_rates,non_obj_error_rates,total_error_rates = [],[],[]
    for i in range(len(error_rates)):
        non_obj_error_rates.append(error_rates[i][1])
        obj_error_rates.append(error_rates[i][0])
        total_error_rates.append(error_rates[i][2])

    X_axis = np.arange(len(error_rates))    

    y_lim = y_lim
    x_label = "nombres de samples"


    plt.figure(figsize=(18,10))
    plt.plot(X_axis,total_error_rates,label = "total error rates",markersize=10, linewidth = 5)
    plt.plot(X_axis,obj_error_rates,label = "obj error rates",markersize=10, linewidth = 5)
    plt.plot(X_axis,non_obj_error_rates,label = "non obj error rates",markersize=10, linewidth = 5)
    plt.gca().legend(fontsize = 20, loc = 'best')
    plt.title(title,fontsize = 30)
    plt.xlabel(x_label,fontsize = 20)
    plt.ylabel('Error Rate',fontsize = 20)
    plt.ylim(y_lim)
    plt.gca().yaxis.set_tick_params(which = 'major', length = 5, width = 3,
                               color = 'black', labelsize = 20, labelcolor = 'black')
    plt.gca().xaxis.set_tick_params(which = 'major', length = 5, width = 3,
                               color = 'black', labelsize = 20, labelcolor = 'black')
    



def ArraytoZ(array,a,b):
    DZ = np.zeros((512*424,1)).astype(float)#np.zeros((84*84,len(RGBA[0])-3))
    for i in range(len(array)):
        DZ[i][0] = 1-array[i][2]
    DZ = np.reshape(DZ,(424,512))
    return DZ

def read_xyzrgb(filename):

    coordinates = []
    xyz = open(filename)
    for line in xyz:
        x,y,z,r,g,b = line.split()
        coordinates.append([float(x), float(y), float(z), int(float(r)), int(float(g)), int(float(b))])
    xyz.close()

    return coordinates

def read_xyzrgbl(filename):

    coordinates = []
    xyz = open(filename)
    for line in xyz:
        x,y,z,r,g,b,l = line.split()
        coordinates.append([float(x), float(y), float(z), int(float(r)), int(float(g)), int(float(b)), int(float(l))])
    xyz.close()

    return coordinates


def read_xyzrgbls(filename):

    coordinates = []
    xyz = open(filename)
    for line in xyz:
        x,y,z,r,g,b,l,s = line.split()
        coordinates.append([float(x), float(y), float(z), int(float(r)), int(float(g)), int(float(b)), int(float(l)), int(float(s))])
    xyz.close()

    return coordinates

def display_Preds(paths_to_output,model_labels,true_labels,nomDossier,numeroDeScene,numIter,stride):
    XYZRGBL = read_xyzrgbl(paths_to_output+str(nomDossier)+"/nuage/nuage_scene"+str(numeroDeScene)+"iter"+numIter+".xyzrgbl")
    RGB = np.zeros((512*424,3)).astype(float)

    for i in range(len(XYZRGBL)):
        RGB[i][0] = XYZRGBL[i][3]/255
        RGB[i][1] = XYZRGBL[i][4]/255
        RGB[i][2] = XYZRGBL[i][5]/255
        
    segmented = read_xyzrgbls(paths_to_output+str(nomDossier)+"/vccs/vccs_scene"+str(numeroDeScene)+"iter"+numIter+".xyzrgbls")
    segRGB = np.zeros((512*424,3)).astype(float)

    for i in range(len(segmented)):
        if(model_labels[segmented[i][7]]==0):
            segRGB[i][0] = RGB[i][0]
            segRGB[i][1] = RGB[i][1]
            segRGB[i][2] = RGB[i][2]
        else:
            segRGB[i][0] = 1
            segRGB[i][1] = 1
            segRGB[i][2] = 0

    fig = plt.figure(figsize = (18,15))
    ax = fig.gca(projection='3d')
    ax.view_init(90, 180)
    plt.gca().invert_zaxis()
    segImg = np.reshape(segRGB,(424,512,3))
    X,Y = ogrid[0:segImg.shape[0], 0:segImg.shape[1]]
    Z = ArraytoZ(segmented,X,Y)
    surf = ax.plot_surface(X, -Y, -Z, rstride=stride, cstride=stride, facecolors=segImg,linewidth=0, antialiased=False)
    plt.savefig(paths_to_output+str(nomDossier)+"/predictionDisplay/preds_scene"+str(numeroDeScene)+"iter"+numIter)
    plt.show()

def display_certainty(paths_to_output,relevancy_map,model_labels,true_labels,nomDossier,numeroDeScene,numIter,stride):

    XYZRGBL = read_xyzrgbl(paths_to_output+str(nomDossier)+"/nuage/nuage_scene"+str(numeroDeScene)+"iter"+numIter+".xyzrgbl")
    RGB = np.zeros((512*424,3)).astype(float)

    for i in range(len(XYZRGBL)):
        RGB[i][0] = XYZRGBL[i][3]/255
        RGB[i][1] = XYZRGBL[i][4]/255
        RGB[i][2] = XYZRGBL[i][5]/255
        
    segmented = read_xyzrgbls(paths_to_output+str(nomDossier)+"/vccs/vccs_scene"+str(numeroDeScene)+"iter"+numIter+".xyzrgbls")
    segRGB = np.zeros((512*424,3)).astype(float)

    for i in range(len(segmented)):
        if(model_labels[segmented[i][7]]==0):
            segRGB[i][0] = RGB[i][0]*relevancy_map[segmented[i][7]]
            segRGB[i][1] = RGB[i][1]*relevancy_map[segmented[i][7]]
            segRGB[i][2] = RGB[i][2]*relevancy_map[segmented[i][7]]
        else:
            segRGB[i][0] = 1*relevancy_map[segmented[i][7]]
            segRGB[i][1] = 1*relevancy_map[segmented[i][7]]
            segRGB[i][2] = 0*relevancy_map[segmented[i][7]]

    fig = plt.figure(figsize = (18,15))
    ax = fig.gca(projection='3d')
    ax.view_init(90, 180)
    plt.gca().invert_zaxis()
    segImg = np.reshape(segRGB,(424,512,3))
    X,Y = ogrid[0:segImg.shape[0], 0:segImg.shape[1]]
    Z = ArraytoZ(segmented,X,Y)
    surf = ax.plot_surface(X, -Y, -Z, rstride=stride, cstride=stride, facecolors=segImg,linewidth=0, antialiased=False)
    plt.savefig(paths_to_output+str(nomDossier)+"/predictionDisplay/certainty_scene"+str(numeroDeScene)+"iter"+numIter)
    plt.show()
    
    
def display_RGBD(paths_to_output,nomDossier,numeroDeScene,numIter,stride):
    XYZRGBL = read_xyzrgbl(paths_to_output+str(nomDossier)+"/nuage/nuage_scene"+str(numeroDeScene)+"iter"+numIter+".xyzrgbl")
    RGB = np.zeros((512*424,3)).astype(float)

    for i in range(len(XYZRGBL)):
        RGB[i][0] = XYZRGBL[i][3]/255
        RGB[i][1] = XYZRGBL[i][4]/255
        RGB[i][2] = XYZRGBL[i][5]/255
        
    segmented = read_xyzrgbls(paths_to_output+str(nomDossier)+"/vccs/vccs_scene"+str(numeroDeScene)+"iter"+numIter+".xyzrgbls")
    segRGB = np.zeros((512*424,3)).astype(float)

    for i in range(len(segmented)):
        segRGB[i][0] = RGB[i][0]
        segRGB[i][1] = RGB[i][1]
        segRGB[i][2] = RGB[i][2]

    fig = plt.figure(figsize = (18,15))
    ax = fig.gca(projection='3d')
    ax.view_init(90, 180)
    plt.gca().invert_zaxis()
    segImg = np.reshape(segRGB,(424,512,3))
    X,Y = ogrid[0:segImg.shape[0], 0:segImg.shape[1]]
    Z = ArraytoZ(segmented,X,Y)
    surf = ax.plot_surface(X, -Y, -Z, rstride=stride, cstride=stride, facecolors=segImg,linewidth=0, antialiased=False)
    plt.savefig(paths_to_output+str(nomDossier)+"/predictionDisplay/RGBD_scene"+str(numeroDeScene)+"iter"+numIter)
    plt.show()
    
def display_certainty_treshold(treshold,paths_to_output,relevancy_map,nomDossier,numeroDeScene,numIter,stride):
    XYZRGBL = read_xyzrgbl(paths_to_output+str(nomDossier)+"/nuage/nuage_scene"+str(numeroDeScene)+"iter"+numIter+".xyzrgbl")
    RGB = np.zeros((512*424,3)).astype(float)

    for i in range(len(XYZRGBL)):
        RGB[i][0] = XYZRGBL[i][3]/255
        RGB[i][1] = XYZRGBL[i][4]/255
        RGB[i][2] = XYZRGBL[i][5]/255
        
    segmented = read_xyzrgbls(paths_to_output+str(nomDossier)+"/vccs/vccs_scene"+str(numeroDeScene)+"iter"+numIter+".xyzrgbls")
    segRGB = np.zeros((512*424,3)).astype(float)

    for i in range(len(segmented)):
    
        if(relevancy_map[segmented[i][7]]<treshold):
            segRGB[i][0] = RGB[i][0]
            segRGB[i][1] = RGB[i][1]
            segRGB[i][2] = RGB[i][2]
        else:
            
            segRGB[i][0] = 1
            segRGB[i][1] = 1
            segRGB[i][2] = 0

    fig = plt.figure(figsize = (18,15))
    ax = fig.gca(projection='3d')
    ax.view_init(90, 180)
    plt.gca().invert_zaxis()
    segImg = np.reshape(segRGB,(424,512,3))
    X,Y = ogrid[0:segImg.shape[0], 0:segImg.shape[1]]
    Z = ArraytoZ(segmented,X,Y)
    surf = ax.plot_surface(X, -Y, -Z, rstride=stride, cstride=stride, facecolors=segImg,linewidth=0, antialiased=False)
    plt.savefig(paths_to_output+str(nomDossier)+"/predictionDisplay/certainty_treshold_scene"+str(numeroDeScene)+"iter"+numIter)
    plt.show()
