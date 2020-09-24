import numpy as np
import pandas as pd
from numpy import unique
from numpy import random 
from sklearn.preprocessing import StandardScaler 
from random import normalvariate
import pickle

#Permet d'ecrir dans un fichier
def writePickler(path,array):
    with open(path,"wb") as file_befor_forgetting:
        my_pickler = pickle.Pickler(file_befor_forgetting)
        my_pickler.dump(array)
        
#Permet de lire dans un fichier
def readPickler(path):
    with open(path,"rb") as file_befor_forgetting:
        my_depickler = pickle.Unpickler(file_befor_forgetting)
        data = my_depickler.load()
    return data
        
#Fonction d'oublie des Outliers en simulation car il nous faut les étiquettages réels : y_train_acc_true
def forget_worst_sample(model,nb_model,X_train_acc_rough,y_train_acc_exp,y_train_acc_true):
    #model = SVC(kernel = 'rbf',probability=True,class_weight = 'balanced')
    nb_model = nb_model
    ratio_training = 0.9 

    sum_predictions = np.zeros(len(y_train_acc_exp))
    nb_predictions = np.zeros(len(y_train_acc_exp))

    for m in range(nb_model):
        X_model = X_train_acc_rough
        y_model = y_train_acc_exp
        label_model = np.arange(len(y_train_acc_exp))
        zipped = list(zip(X_model,y_model,label_model))
        random.shuffle(zipped)
        X_model,y_model,label_model = zip(*zipped)
        X_model = normalize(X_model)
        ratio = int(ratio_training*len(y_train_acc_exp))
        X_train_model,X_test_model = X_model[:ratio],X_model[ratio+1:]
        y_train_model,y_test_model = y_model[:ratio],y_model[ratio+1:]
        label_train_model,label_test_model = label_model[:ratio],label_model[ratio+1:]

        model.fit(X_train_model,y_train_model)
        y_test_model = model.predict(X_test_model)

        for i in range(len(y_test_model)):
            sum_predictions[label_test_model[i]] += y_test_model[i]
            nb_predictions[label_test_model[i]] +=1

    predictions = []

    for i in range(len(nb_predictions)):
        predictions.append(sum_predictions[i] / nb_predictions[i])

    predictions_int = np.zeros(len(predictions))
    for i in range(len(predictions)):
        if(predictions[i]>0.5):
            predictions_int[i] = 1
        else:
            predictions_int[i] = 0

    new_X = []
    new_y = []
    new_y_true = []
    for i in range(len(predictions_int)):
        #if(predictions_int[i]==y_train_acc_exp[i]):
        if(abs(predictions[i] - y_train_acc_exp[i]) <= 0.50):
            new_X.append(X_train_acc_rough[i])
            new_y.append(y_train_acc_exp[i])
            new_y_true.append(y_train_acc_true[i])  
        else:
            print("prediction removed :",predictions[i])
            print("is diferent :",int(abs(predictions_int[i]-y_train_acc_exp[i])))
   
    return new_X, new_y, new_y_true

#Fonction d'oublie des Outliers dans le cas ou nous n'avons pas accès aux véritables labels (i.e. dans le monde réel)
def forget_worst_sample2(model,nb_model,X_train_acc_rough,y_train_acc_exp):
    nb_model = nb_model#Nb de modeles utilisé pour le moyennage des prédictions (=1000 en géneral)
    ratio_training = 0.9#Proportion de données servant à l'entrainement de chacun des classifieurs

    sum_predictions = np.zeros(len(y_train_acc_exp))
    nb_predictions = np.zeros(len(y_train_acc_exp))

    for m in range(nb_model):
        X_model = X_train_acc_rough
        y_model = y_train_acc_exp
        label_model = np.arange(len(y_train_acc_exp))
        zipped = list(zip(X_model,y_model,label_model))
        random.shuffle(zipped)
        X_model,y_model,label_model = zip(*zipped)
        X_model = normalize(X_model)
        ratio = int(ratio_training*len(y_train_acc_exp))
        X_train_model,X_test_model = X_model[:ratio],X_model[ratio+1:]
        y_train_model,y_test_model = y_model[:ratio],y_model[ratio+1:]
        label_train_model,label_test_model = label_model[:ratio],label_model[ratio+1:]

        model.fit(X_train_model,y_train_model)
        y_test_model = model.predict(X_test_model)

        for i in range(len(y_test_model)):
            sum_predictions[label_test_model[i]] += y_test_model[i]
            nb_predictions[label_test_model[i]] +=1

    predictions = []

    for i in range(len(nb_predictions)):
        predictions.append(sum_predictions[i] / nb_predictions[i])

    predictions_int = np.zeros(len(predictions))
    for i in range(len(predictions)):
        if(predictions[i]>0.5):
            predictions_int[i] = 1
        else:
            predictions_int[i] = 0

    new_X = []
    new_y = []
    new_y_true = []
    for i in range(len(predictions_int)):
        #if(predictions_int[i]==y_train_acc_exp[i]):
        if(abs(predictions[i] - y_train_acc_exp[i]) <= 0.50):#0.50 peut etre changé en fonction du seuil de différence voulu
            new_X.append(X_train_acc_rough[i])
            new_y.append(y_train_acc_exp[i])
        else:
            print("prediction removed :",predictions[i])
            print("is diferent :",int(abs(predictions_int[i]-y_train_acc_exp[i])))
   
    return new_X, new_y

#Permet d'équilibrer une base de donnée pour qu'il y ait autant d'échantillon pour chacune des classes
def balanced_sample_maker(X, y, random_seed=None):
    """ return a balanced data set by oversampling minority class 
        current version is developed on assumption that the positive
        class is the minority.

    Parameters:
    ===========
    X: {numpy.ndarrray}
    y: {numpy.ndarray}
    """
    uniq_levels = unique(y)
    uniq_counts = {level: sum(y == level) for level in uniq_levels}

    if not random_seed is None:
        random.seed(random_seed)

    # find observation index of each class levels
    groupby_levels = {}
    for ii, level in enumerate(uniq_levels):
        obs_idx = [idx for idx, val in enumerate(y) if val == level]
        groupby_levels[level] = obs_idx

    # oversampling on observations of positive label
    sample_size = uniq_counts[0]
    over_sample_idx = random.choice(groupby_levels[1], size=sample_size, replace=True).tolist()
    balanced_copy_idx = groupby_levels[0] + over_sample_idx
    random.shuffle(balanced_copy_idx)

    return X[balanced_copy_idx, :], y[balanced_copy_idx]

#Simule des erreurs de prédiction (utile pour faire des projection sur l'impacte du taux d'erreur de prédiction sur notre classifieur)
def compromising_data(y,certitude_obj,certitude_bckg):
    compromised_y = np.zeros(len(y))
    for i in range(len(y)):
        if((random.uniform(0,1)>certitude_obj)and(y[i]==1)):
            compromised_y[i] = 0
        elif((random.uniform(0,1)>certitude_bckg)and(y[i]==0)):
            compromised_y[i] = 1
        else: 
            compromised_y[i] = y[i]
    return compromised_y

#Calcul les taux d'erreur entre les étiquettes prédites et les étiquettes réels
def error_pred(sup_label, real_label):
    if(len(sup_label)!=len(real_label)):
        raise Exception("len(sup_label)!=len(real_label)")
        
    error_rates = np.zeros(3)
    count_objet = 0
    for i in range (len(sup_label)):
        if(real_label[i]==1):
            error_rates[0] += 1-sup_label[i]
            error_rates[2] += 1-sup_label[i]
            count_objet +=1
        else:
            error_rates[1] += sup_label[i]
            error_rates[2] += sup_label[i]
    error_rates[0] /= count_objet
    error_rates[1] /= (len(sup_label) - count_objet)
    error_rates[2] /= len(sup_label)
    return error_rates#0 = objet ERate; 1 = Non Objet ERate; 2 = Total ERate    

#Créé une base de donnée pour tester notre(nos) classifieur(s)
def createTestData(nomDossier,nombreDeScene,numeroDeScene):
    df = pd.read_csv("../outputs/"+str(nomDossier)+"/fpfh/fpfh_scene"+str(numeroDeScene)+".txt")
    count = len(df)
    for i in range(2,nombreDeScene+1):
        nf = pd.read_csv("../outputs/"+str(nomDossier)+"/fpfh/fpfh_scene"+str(numeroDeScene+i)+".txt")
        count += len(nf)
        df = pd.concat([df,nf],ignore_index = True)
        
    X_prev = np.array(df.drop('label',axis=1))
    y_prev = np.array(df['label'])
    X_test,y_test = balanced_sample_maker(X_prev,y_prev)
    return X_test, y_test

#Idem que précedement avec un chemin d'accès différents vers les fichiers de données sauvegardés
def createTestData2(nomDossier,nombreDeScene,numeroDeScene):
    df = pd.read_csv("../outputs/"+str(nomDossier)+"/fpfh/fpfh_scene"+str(numeroDeScene)+"iter2.txt")
    count = len(df)
    for i in range(2,nombreDeScene+1):
        nf = pd.read_csv("../outputs/"+str(nomDossier)+"/fpfh/fpfh_scene"+str(numeroDeScene+i)+"iter2.txt")
        count += len(nf)
        df = pd.concat([df,nf],ignore_index = True)
        

    X_prev = np.array(df.drop('label',axis=1))
    y_prev = np.array(df['label'])
    X_test,y_test = balanced_sample_maker(X_prev,y_prev)
    return X_test, y_test

#Mélange les données de tests pour éviter de potentiel biais
def getRandomDataTest(X_test_complet, y_test_complet,nbTestSamples):
    X_test = np.zeros((nbTestSamples,48))
    y_test = np.zeros(nbTestSamples)
    for i in range(nbTestSamples):
        randy = int(random.uniform(0,len(X_test_complet)))
        X_test[i] = (X_test_complet[randy])
        y_test[i] = (y_test_complet[randy])
    return X_test, y_test
                    
#Fournis les vecteurs discriminants et les labels des SVP d'une scène (i.e. nuage de points)
def getDataScene(nomDossier,numeroScene):
    df =  pd.read_csv("../outputs/"+str(nomDossier)+"/fpfh/fpfh_scene"+str(numeroScene)+".txt")               
    #scaler=StandardScaler() 
    #scaler.fit(df.drop('label',axis=1))
    #scaled_features = scaler.transform(df.drop('label',axis=1))
    #df_feat = pd.DataFrame(df.drop('label',axis=1),columns=df.columns[:-1])
    #df_feat.head()
    #print("ow")
    X_train = np.array(df.drop('label',axis=1))
    y_train = np.array(df['label'])                
    return X_train, y_train
                    
#Fonction d'acquisition 1 (abandonnée depuis)
def getUncertainData(X_train_rough,y_train,preds_proba,y_acc,nbPointsParScene):
    relevancy_map = np.zeros(len(preds_proba), dtype=int)

    for i in range (len(relevancy_map)):
        #0 is the most uncertain
        relevancy_map[i] = int(1000*(abs(max(preds_proba[i][0],preds_proba[i][1]))-0.5)*2)
    idx = np.argpartition(relevancy_map, len(relevancy_map)-1)
    threshold_y = sum(y_acc)/len(y_acc)
    
    acc_i = 1

    if(threshold_y<random.uniform(0,0.8)):#0.8est choisi de manière empirique ici
        while (y_train[idx[acc_i]]!=1):
            acc_i +=1
            if(acc_i>=len(relevancy_map)-1):
                acc_i = 1
                break
    num_FPFH = idx[acc_i]

    return num_FPFH#Retourne le numéro du FPFH selectionné


def normal_choice(lst, mean=None, stddev=None):
    if mean is None:
        # if mean is not specified, use center of list
        mean = (len(lst) - 1) / 2

    if stddev is None:
        # if stddev is not specified, let list be -3 .. +3 standard deviations
        stddev = len(lst) / 6

    while True:
        index = int(normalvariate(mean, stddev) + 0.5)
        if 0 <= index < len(lst):
            return lst[index]
        
#Fonction d'acquisition actuelle        
def getRelevantFPFH(preds_proba,y_train_acc_exp,threshold_of_object):
    """
    inputs:
    preds_proba :      proba pour chaque FPFH d'apartenir à l'une ou à l'autre des deux classes
    y_train_acc_exp :  Label des samples accumulées au cours de l'experience    
    outputs:
    num_FPFH :         Numero du FPFH selectionné
    """
    certainty_map = np.zeros(len(preds_proba), dtype=int)
    stddev_idx = len(preds_proba)/20*pow(len(y_train_acc_exp),1/3)
    mean_idx = 0
    min_distance = 1
    
    for i in range (len(certainty_map)):
        certainty_map[i] = int(1000*(preds_proba[i][1]))#[|0,1000|]: 0 is the most uncertain
    ordered_FPFH = np.argpartition(certainty_map, len(certainty_map)-1)
    threshold_y_true = sum(y_train_acc_exp)/len(y_train_acc_exp)#proportion de FPFH d'objet dans la BDD accumulée
    if(threshold_of_object <= threshold_y_true):#i.e. on ne prends plus d'objet
        threshold_y = random.uniform(0.65,0.95)#i.e. on favorise le tirage d'un SVP du background
    else:
        threshold_y = threshold_y_true
    
    
    for i in range (len(ordered_FPFH)):
        if(abs(1-threshold_y-certainty_map[ordered_FPFH[i]]/1000)<min_distance):
            min_distance = abs(1-threshold_y-certainty_map[ordered_FPFH[i]]/1000)
            mean_idx = i
            
    num_FPFH = normal_choice(ordered_FPFH,mean=mean_idx, stddev=stddev_idx)
    print("threshold_y_true : ",int(100*threshold_y_true)/100,"Mean : ",int(0.1*certainty_map[ordered_FPFH[mean_idx]])/100,"  certainty_FPFH : ",int(0.1*certainty_map[num_FPFH])/100)
    return num_FPFH#Retourne le numéro du SVP que nous avons selectionner pour apliquer la primitive de poussée



def normalize(X):
    scaler=StandardScaler() 
    scaler.fit(X)
    scaled_features = scaler.transform(X)
    return scaled_features

def normalize_2(X1,X2):
    X = np.concatenate((X1,X2))
    scaler=StandardScaler() 
    scaler.fit(X)
    scaled_features = scaler.transform(X)
    X1_scaled  = np.array(scaled_features[:len(X1)])
    X2_scaled  = np.array(scaled_features[len(X1):])
    return X1_scaled, X2_scaled

def rgb2lab ( inputColor ) :

   num = 0
   RGB = [0, 0, 0]

   for value in inputColor :
       value = float(value) / 255

       if value > 0.04045 :
           value = ( ( value + 0.055 ) / 1.055 ) ** 2.4
       else :
           value = value / 12.92

       RGB[num] = value * 100
       num = num + 1

   XYZ = [0, 0, 0,]

   X = RGB [0] * 0.4124 + RGB [1] * 0.3576 + RGB [2] * 0.1805
   Y = RGB [0] * 0.2126 + RGB [1] * 0.7152 + RGB [2] * 0.0722
   Z = RGB [0] * 0.0193 + RGB [1] * 0.1192 + RGB [2] * 0.9505
   XYZ[ 0 ] = round( X, 4 )
   XYZ[ 1 ] = round( Y, 4 )
   XYZ[ 2 ] = round( Z, 4 )

   XYZ[ 0 ] = float( XYZ[ 0 ] ) / 95.047         # ref_X =  95.047   Observer= 2°, Illuminant= D65
   XYZ[ 1 ] = float( XYZ[ 1 ] ) / 100.0          # ref_Y = 100.000
   XYZ[ 2 ] = float( XYZ[ 2 ] ) / 108.883        # ref_Z = 108.883

   num = 0
   for value in XYZ :

       if value > 0.008856 :
           value = value ** ( 0.3333333333333333 )
       else :
           value = ( 7.787 * value ) + ( 16 / 116 )

       XYZ[num] = value
       num = num + 1

   Lab = [0, 0, 0]

   L = ( 116 * XYZ[ 1 ] ) - 16
   a = 500 * ( XYZ[ 0 ] - XYZ[ 1 ] )
   b = 200 * ( XYZ[ 1 ] - XYZ[ 2 ] )

   Lab [ 0 ] = round( L, 4 )
   Lab [ 1 ] = round( a, 4 )
   Lab [ 2 ] = round( b, 4 )

   return Lab
