import numpy as np
import random 

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torchvision
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision.transforms import ToTensor
from torchvision import datasets, models, transforms
from torchvision.io import read_image
from torch.optim.lr_scheduler import StepLR
from collections import Counter

import matplotlib.pyplot as plt
import time
import os
import copy

import pandas as pd
import numpy as np

from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import matthews_corrcoef, balanced_accuracy_score, accuracy_score,r2_score, f1_score, roc_auc_score, confusion_matrix, classification_report, mean_squared_error
import seaborn as sns
import math

device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

def roc_auc_score_multiclass(actual_class, pred_class, average = "micro"):
    
    #creating a set of all the unique classes using the actual class list
    unique_class = set(actual_class)
    roc_auc_dict = {}
    fpr = {}
    tpr = {}
    
    for per_class in unique_class:
        per_class = float(per_class)
        #creating a list of all the classes except the current class 
        other_class = [x for x in unique_class if x != per_class]

        #marking the current class as 1 and all other classes as 0
        new_actual_class = [0 if x in other_class else 1 for x in actual_class]
        new_pred_class = [0 if x in other_class else 1 for x in pred_class]

        #using the sklearn metrics method to calculate the roc_auc_score
        roc_auc = roc_auc_score(new_actual_class, new_pred_class, average = average)
        roc_auc_dict[per_class] = roc_auc

    return roc_auc_dict

def multi_acc(y_test, y_pred):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    
    return acc

def accuracy(y_true, y_pred):
    
    """
    Function to calculate accuracy
    -> param y_true: list of true values
    -> param y_pred: list of predicted values
    -> return: accuracy score
    
    """
    
    # Intitializing variable to store count of correctly predicted classes
    correct_predictions = 0
    for yt, yp in zip(y_true, y_pred):
        
        if yt == yp:
            
            correct_predictions += 1
    
    #returns accuracy
    return correct_predictions / len(y_true)

def multi_mcc(y_test, y_pred):
    mcc = matthews_corrcoef(y_test, y_pred)
    return mcc

def root_mean_squared_error(y_test, y_pred):
    MSE = mean_squared_error(y_test, y_pred)
    RMSE = math.sqrt(MSE)
    return RMSE

def save_confusion_matrix(y_true, y_pred, epoch=0):
    plt.figure(figsize = (12,8))
    sns.heatmap(confusion_matrix(y_true, y_pred), annot = True, xticklabels = list(set(y_true)), yticklabels = list(set(y_true)), cmap = 'summer')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.savefig(f"./logs/conf-matrix-epoch-{epoch}")


def save_csv_files(files):
    return

def save_best_dict(test_metrics, best_dict):
    acc, f1, roc, e = test_metrics

    if(best_dict["accuracy"] > acc):
        best_dict["accuracy"] = acc
    elif(best_dict["f1_score"] > f1):
        best_dict["f1_score"] = f1
        best_dict["roc_auc"] = roc
    
    return best_dict

def evaluate_test_set(model, test_loader, e, print_values=False, regression=False, class_weights = None):
    '''
    Params:
        model : The model used to evaluate
        test_loader : Data Loader for the test set
        print_values : Print values to console after evaluation
        regression : Evaluate as regression task
    '''
    y_pred_list = []
    y_true_list = []

    torch.cuda.empty_cache() # Clear CUDA memory
    with torch.no_grad():
        model.eval()    
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)

             # START Regression specific
            y_batch = y_batch.float()
            # END

     

            y_test_pred = model(X_batch)
            _, y_pred_hat = torch.max(y_test_pred.data, 1)

            if(regression == True):
                # START Regression specific
                y_pred_hat = torch.round(y_test_pred)
                y_pred_hat = y_pred_hat.detach()
                y_pred_hat = torch.squeeze(y_pred_hat)
                # END

    
            y_pred_list.extend(y_pred_hat.cpu().numpy())
            y_true_list.extend(y_batch.cpu().numpy())

    

    test_acc = accuracy_score(y_true_list, y_pred_list)
    #Do balanced accuracy if not none
    if(class_weights != None):
        # Weighted Sampling
        # class_count = [i for i in Counter(np.array(y_true_list)).values()]
        # class_weights = 1./torch.tensor(class_count, dtype=torch.float) 
        # class_weights_all = class_weights[np.squeeze(np.array(y_true_list))]

        class_weights_all = class_weights[np.squeeze(np.array(y_true_list))]
        test_acc = balanced_accuracy_score(y_true_list, y_pred_list, sample_weight=class_weights_all)
        
    
    test_f1 = f1_score(y_true_list, y_pred_list, average="weighted", labels=np.unique(y_pred_list))
    #test_roc = roc_auc_score_multiclass(y_true_list, y_pred_list)
    test_r2 = r2_score(y_true_list, y_pred_list)
    test_rmse = root_mean_squared_error(y_true_list, y_pred_list)
    test_roc = None
    try:
        test_roc = roc_auc_score(y_true_list, y_pred_list)
    except:
        test_roc = 0 # Placeholder for exception

    if(print_values == True):
        #print(f"Test Metrics | Accuracy {test_acc} | F1 {test_f1} | ROC {test_roc} | RMSE: {test_rmse}")
        print(f"Test Metrics | Accuracy:{test_acc} RMSE: {test_rmse} R2: {test_r2} F1: {test_f1} AUC: {test_roc}")
        
        #print(classification_report(y_true_list, y_pred_list))
    # return [test_acc, test_f1, test_rmse, test_roc, e]
    return {'accuracy':test_acc, 'f1':test_f1, 'rmse':test_rmse, 'r2':test_r2, 'roc_auc':test_roc}

def evaluate_test_set_multi_task(model, test_loader, e, print_values=False, regression=False, class_weights = None):
    '''
    Evaluate test set for multi-task specific model (Erosion and JSN score combined detection)
    Params:
        model : The model used to evaluate
        test_loader : Data Loader for the test set
        print_values : Print values to console after evaluation
        regression : Evaluate as regression task
    '''
    y_pred_e_list = []
    y_true_e_list = []

    y_pred_j_list = []
    y_true_j_list = []

   
    with torch.no_grad():
        model.eval()
        for X_batch, y_batch_e, y_batch_j in test_loader:
            
            X_batch = X_batch.to(device)
            y_batch_e, y_batch_j= y_batch_e.float(), y_batch_j.float()
     
            y_test_pred_e, y_test_pred_j  = model(X_batch)
        
         
            # START Regression specific
            y_pred_e_hat,y_pred_j_hat = torch.round(y_test_pred_e), torch.round(y_test_pred_j) # Round values to nearest digit
            y_pred_e_hat,y_pred_j_hat = y_pred_e_hat.detach(), y_pred_j_hat.detach()
            y_pred_e_hat,y_pred_j_hat = torch.squeeze(y_pred_e_hat), torch.squeeze(y_pred_j_hat)
            # END

    
            y_pred_e_list.extend(y_pred_e_hat.cpu().numpy())
            y_pred_j_list.extend(y_pred_j_hat.cpu().numpy())
            y_true_e_list.extend(y_batch_e.cpu().numpy())
            y_true_j_list.extend(y_batch_j.cpu().numpy())

    

    test_acc_e = accuracy_score(y_true_e_list, y_pred_e_list)
    test_acc_j = accuracy_score(y_true_j_list, y_pred_j_list)
    
    test_r2_e = r2_score(y_true_e_list, y_pred_e_list)
    test_r2_j = r2_score(y_true_j_list, y_pred_j_list)
    
    test_rmse_e = root_mean_squared_error(y_true_e_list, y_pred_e_list)
    test_rmse_j = root_mean_squared_error(y_true_j_list, y_pred_j_list)
    
    test_f1_e = f1_score(y_true_e_list, y_pred_e_list, average="weighted", labels=np.unique(y_pred_e_list))
    test_f1_j = f1_score(y_true_j_list, y_pred_j_list, average="weighted", labels=np.unique(y_pred_j_list))
    
    #test_roc = roc_auc_score_multiclass(y_true_list, y_pred_list)
    test_roc_e, test_roc_j = None, None
    try:
        test_roc_e = roc_auc_score(y_true_e_list, y_pred_e_list)
        test_roc_j = roc_auc_score(y_true_j_list, y_pred_j_list)
    except:
        test_roc_e, test_roc_j = 0, 0 # Placeholder for exception

    if(print_values == True):
        #print(f"Test Metrics | Accuracy {test_acc} | F1 {test_f1} | ROC {test_roc} | RMSE: {test_rmse}")
        print(f"Test Metrics | \nAccuracy (E|J):{test_acc_e} | {test_acc_j} \nRMSE: {test_rmse_e} | {test_rmse_j} \nF1: {test_f1_e} | {test_f1_j}  \nAUC: {test_roc_e} | {test_roc_j}")
        
        #print(classification_report(y_true_list, y_pred_list))
    # return [test_acc, test_f1, test_rmse, test_roc, e]
    return {'accuracy':(test_acc_e, test_acc_j), 'f1':(test_f1_e, test_f1_j), 'rmse':(test_rmse_e, test_rmse_j), 'r2':(test_r2_e, test_r2_j), 'roc_auc':(test_roc_e, test_roc_j)}

def evaluate_test_set_multi_wrist(model, test_loader, print_values=False, regression=False, class_weights = None):
    '''
    Evaluate test set for multi-task specific model (Wrist joints)
    Params:
        model : The model used to evaluate
        test_loader : Data Loader for the test set
        print_values : Print values to console after evaluation
        regression : Evaluate as regression task
    '''
    METRIC_NAMES = ['accuracy', 'loss', 'rmse', 'f1', 'r2','roc_auc']
    CAT_KEYS = ['lun', 'mc1', 'mul', 'nav']
    SPLIT_KEYS = ['train', 'val', 'test']

    pred_lists = {}
    true_lists = {}
    outputs = {}
    for metric in METRIC_NAMES:
        outputs[metric] = {}
        for jn in CAT_KEYS:
            outputs[metric][jn] = []
    for jn in CAT_KEYS:
        pred_lists[jn] = []
        true_lists[jn] = []


   
    with torch.no_grad():
        model.eval()
        for X_batch, y_batch_lun, y_batch_mc1, y_batch_mul, y_batch_nav in test_loader:
            
            X_batch = X_batch.to(device)
            y_batch_lun, y_batch_mc1, y_batch_mul, y_batch_nav = y_batch_lun.float(), y_batch_mc1.float(), y_batch_mul.float(), y_batch_nav.float()
     
            y_pred_lun, y_pred_mc1, y_pred_mul, y_pred_nav  = model(X_batch)
        
         
            # START Regression specific
            y_pred_lun, y_pred_mc1, y_pred_mul, y_pred_nav = torch.round(y_pred_lun), torch.round(y_pred_mc1), torch.round(y_pred_mul), torch.round(y_pred_nav) # Round values to nearest digit
            y_hat_lun, y_hat_mc1, y_hat_mul, y_hat_nav = y_pred_lun.detach(), y_pred_mc1.detach(), y_pred_mul.detach(), y_pred_nav.detach()
            y_hat_lun, y_hat_mc1, y_hat_mul, y_hat_nav = torch.squeeze(y_hat_lun), torch.squeeze(y_hat_mc1), torch.squeeze(y_hat_mul), torch.squeeze(y_hat_nav)
            # END


            pred_lists['lun'].extend(y_hat_lun.cpu().numpy())
            pred_lists['mc1'].extend(y_hat_mc1.cpu().numpy())
            pred_lists['mul'].extend(y_hat_mul.cpu().numpy())
            pred_lists['nav'].extend(y_hat_nav.cpu().numpy())
            
            true_lists['lun'].extend(y_batch_lun.cpu().numpy())
            true_lists['mc1'].extend(y_batch_mc1.cpu().numpy())
            true_lists['mul'].extend(y_batch_mul.cpu().numpy())
            true_lists['nav'].extend(y_batch_nav.cpu().numpy())
    


    for jn in CAT_KEYS:
        outputs['accuracy'][jn] =  accuracy_score(true_lists[jn], pred_lists[jn])
        outputs['rmse'][jn] =  root_mean_squared_error(true_lists[jn], pred_lists[jn])
        outputs['f1'][jn] =  f1_score(true_lists[jn], pred_lists[jn], average="weighted",labels=np.unique(pred_lists[jn]))
        outputs['r2'][jn] =  r2_score(true_lists[jn], pred_lists[jn])
        try:
            outputs['roc_auc'][jn] =  roc_auc_score(true_lists[jn], pred_lists[jn])
        except:
            outputs['roc_auc'][jn] = -1

    # test_acc_e = accuracy_score(y_true_e_list, y_pred_e_list)
    # test_acc_j = accuracy_score(y_true_j_list, y_pred_j_list)
    
    # test_r2_e = r2_score(y_true_e_list, y_pred_e_list)
    # test_r2_j = r2_score(y_true_j_list, y_pred_j_list)
    
    # test_rmse_e = root_mean_squared_error(y_true_e_list, y_pred_e_list)
    # test_rmse_j = root_mean_squared_error(y_true_j_list, y_pred_j_list)
    
    # test_f1_e = f1_score(y_true_e_list, y_pred_e_list, average="weighted", labels=np.unique(y_pred_e_list))
    # test_f1_j = f1_score(y_true_j_list, y_pred_j_list, average="weighted", labels=np.unique(y_pred_j_list))
    
 
    if(print_values == True):
        #print(f"Test Metrics | Accuracy {test_acc} | F1 {test_f1} | ROC {test_roc} | RMSE: {test_rmse}")
        print(f"Test Metrics | {outputs}")
        
        #print(classification_report(y_true_list, y_pred_list))
    # return [test_acc, test_f1, test_rmse, test_roc, e]
    return outputs



def calculate_roc_plot(y_score, y_test, n_classes):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[i], y_score[i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot of a ROC curve for a specific class
    for i in range(n_classes):
        plt.figure()
        plt.plot(fpr[i], tpr[i], label='ROC curve (area = %0.2f)' % roc_auc[i])
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()
    

def calculate_tpr_fpr(y_real, y_pred):
    '''
    Calculates the True Positive Rate (tpr) and the True Negative Rate (fpr) based on real and predicted observations
    
    Args:
        y_real: The list or series with the real classes
        y_pred: The list or series with the predicted classes
        
    Returns:
        tpr: The True Positive Rate of the classifier
        fpr: The False Positive Rate of the classifier
    '''
    
    # Calculates the confusion matrix and recover each element
    cm = confusion_matrix(y_real, y_pred)
    TN = cm[0, 0]
    FP = cm[0, 1]
    FN = cm[1, 0]
    TP = cm[1, 1]
    
    # Calculates tpr and fpr
    tpr =  TP/(TP + FN) # sensitivity - true positive rate
    fpr = 1 - TN/(TN+FP) # 1-specificity - false positive rate
    
    return tpr, fpr
    
## Calculate ROC curves
def get_all_roc_coordinates(y_real, y_proba):
    '''
    Calculates all the ROC Curve coordinates (tpr and fpr) by considering each point as a treshold for the predicion of the class.
    
    Args:
        y_real: The list or series with the real classes.
        y_proba: The array with the probabilities for each class, obtained by using the `.predict_proba()` method.
        
    Returns:
        tpr_list: The list of TPRs representing each threshold.
        fpr_list: The list of FPRs representing each threshold.
    '''
    tpr_list = [0]
    fpr_list = [0]
    for i in range(len(y_proba)):
        threshold = y_proba[i]
        y_pred = y_proba >= threshold
        tpr, fpr = calculate_tpr_fpr(y_real, y_pred)
        tpr_list.append(tpr)
        fpr_list.append(fpr)
    return tpr_list, fpr_list


def plot_roc_curve(tpr, fpr, scatter = True, ax = None):
    '''
    Plots the ROC Curve by using the list of coordinates (tpr and fpr).
    
    Args:
        tpr: The list of TPRs representing each coordinate.
        fpr: The list of FPRs representing each coordinate.
        scatter: When True, the points used on the calculation will be plotted with the line (default = True).
    '''
    if ax == None:
        plt.figure(figsize = (10, 10))
        ax = plt.axes()
    
    if scatter:
        sns.scatterplot(x = fpr, y = tpr, ax = ax)
    ax.set_title("ROC Curve OvR")
    sns.lineplot(x = fpr, y = tpr, ax = ax)
    sns.lineplot(x = [0, 1], y = [0, 1], color = 'green', ax = ax)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")



def draw_roc_curve(X_test, y_test, y_proba, classes, key):
    # Plots the Probability Distributions and the ROC Curves One vs Rest
    plt.figure(figsize = (12, 8))
    bins = [i/20 for i in range(20)] + [1]
    classes = classes
    roc_auc_ovr = {}
    for i in range(len(classes)):
        # Gets the class
        c = classes[i]
        
        # Prepares an auxiliar dataframe to help with the plots
        df_aux = pd.DataFrame()
        df_aux['class'] = [1 if y == c else 0 for y in y_test]
        df_aux['prob'] = y_proba[:, i]
        df_aux = df_aux.reset_index(drop = True)
        
        # Plots the probability distribution for the class and the rest
        ax = None
        # sns.histplot(x = "prob", data = df_aux, hue = 'class', color = 'b', ax = ax, bins = bins)
        # ax.set_title(c)
        # ax.legend([f"Class: {c}", "Rest"])
        # ax.set_xlabel(f"P(x = {c})")
        
        # # Calculates the ROC Coordinates and plots the ROC Curves
        # ax_bottom = plt.subplot(2, 6, i+7)
        tpr, fpr, thresholds = roc_curve(df_aux['class'].values, df_aux['prob'].values)
        plot_roc_curve(tpr, fpr, scatter = False, ax = ax)
        
        plt.savefig(f"./logs/roc_curve/{key}")
        # Calculates the ROC AUC OvR
        try:
            roc_auc_ovr[c] = roc_auc_score(df_aux['class'], df_aux['prob'], average = 'micro')
        except ValueError:
            pass
        
    plt.tight_layout()