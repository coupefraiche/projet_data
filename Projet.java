# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 10:46:23 2019

@author: PC
"""
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.metrics import *
from sklearn import *
from sklearn.metrics import classification_report
from sklearn.svm import SVC
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier


#Importer les données
def importer():
    data=pd.read_csv ('red_wines.csv', delimiter=',')
    
    #Citric acid
    value = data['citric acid'].median()
    data['citric acid'].fillna(value, inplace = True)  
        
        #Residual_sugar
    value = data['residual sugar'].median()
    data['residual sugar'].fillna(value, inplace = True) 

        #pH
    value =data.loc[data['pH'] <= 14,'pH' ].mean()
    data.loc[data['pH'] > 14,'pH' ] = value
    
    return data
  
#Supprimer une colonne
def supprimer_colonnes(data, nom):
    del data[nom]
  
    
#Creer une base de données TEST , Renvoie 2 tuple base de données avec 90% et 10%
def creer_nouvelle_base_donnees(data):
    new1 = pd.DataFrame({'fixed acidity':[], 'volatile acidity':[],'citric acid':[], 'residual sugar':[],'chlorides':[],'free sulfur dioxide':[], 'total sulfur dioxide':[],'density':[],'pH':[],'sulphates':[],'alcohol':[],'quality':[]} ) 
    new2 = pd.DataFrame({'fixed acidity':[], 'volatile acidity':[],'citric acid':[], 'residual sugar':[],'chlorides':[],'free sulfur dioxide':[], 'total sulfur dioxide':[],'density':[],'pH':[],'sulphates':[],'alcohol':[],'quality':[]} ) 
    compteur1 = 0
    compteur2 = 0
    compteur3 = 0
    while compteur1 < len(data):
        if compteur1%10 == 9:
            new1.loc[compteur2] = data.iloc[compteur1]
            compteur2 +=1
        else:
            new2.loc[compteur3] = data.iloc[compteur1]
            compteur3 +=1           
        compteur1+=1
    return new1, new2 

#Creer une base donnée avec 90%
def creer_nouvelle_base_donnees90(data):
    new1 = pd.DataFrame({'fixed acidity':[], 'volatile acidity':[],'citric acid':[], 'residual sugar':[],'chlorides':[],'free sulfur dioxide':[], 'total sulfur dioxide':[],'density':[],'pH':[],'sulphates':[],'alcohol':[],'quality':[]} ) 
    compteur1 = 0
    compteur2 = 0
    while compteur1 < len(data):
        if compteur1%10 != 9:
            new1.loc[compteur2] = data.iloc[compteur1]
            compteur2 +=1
        compteur1+=1
    return new1

#Creer une base donnée avec 10%
def creer_nouvelle_base_donnees10(data):
    new1 = pd.DataFrame({'fixed acidity':[], 'volatile acidity':[],'citric acid':[], 'residual sugar':[],'chlorides':[],'free sulfur dioxide':[], 'total sulfur dioxide':[],'density':[],'pH':[],'sulphates':[],'alcohol':[],'quality':[]} ) 
    compteur1 = 0
    compteur2 = 0
    while compteur1 < len(data):
        if compteur1%10 == 9:
            new1.loc[compteur2] = data.iloc[compteur1]
            compteur2 +=1
        compteur1+=1
    return new1


#Normaliser les données
def normaliser (data) :
    normalisation = StandardScaler()
    nrm = normalisation.fit_transform(data.iloc[:,:11].values)
    donnee_normalisee = pd.DataFrame(nrm, columns=data.iloc[:,:11].columns)
    donnee_normalisee.insert(11, "quality", data["quality"].values)
    return donnee_normalisee


#Regression logistique
def regression_logistique (data):
    data_10, data_90 = creer_nouvelle_base_donnees(data)  
    data_10_whithout_quality = data_10.iloc[:,:11]
    data_10_quality = data_10.iloc[:,11:]
    RegLog=LogisticRegression()
    RegLog=RegLog.fit(data_90.iloc[:,:11],data_90["quality"])
    prediction=RegLog.predict(data_10_whithout_quality)
    return data_10_quality, prediction

#SVM , SVC, Machines `a vecteurs supports
def machine_a_vecteur_support (data):
    data_10, data_90 = creer_nouvelle_base_donnees(data)  
    data_10_whithout_quality = data_10.iloc[:,:11]
    data_10_quality = data_10.iloc[:,11:]
    svc = SVC()
    svc=svc.fit(data_90.iloc[:,:11],data_90["quality"])
    prediction=svc.predict(data_10_whithout_quality)
    return data_10_quality, prediction

#Analyse discriminante linéaire
def analyse_discriminante_lineaire (data):
    data_10, data_90 = creer_nouvelle_base_donnees(data)  
    data_10_whithout_quality = data_10.iloc[:,:11]
    data_10_quality = data_10.iloc[:,11:]
    discriminante = LinearDiscriminantAnalysis()
    discriminante = discriminante.fit(data_90.iloc[:,:11],data_90["quality"])
    prediction=discriminante.predict(data_10_whithout_quality)
    return data_10_quality, prediction  

#Analyse discriminante quadratique
def analyse_discriminante_quadratique (data):
    data_10, data_90 = creer_nouvelle_base_donnees(data)  
    data_10_whithout_quality = data_10.iloc[:,:11]
    data_10_quality = data_10.iloc[:,11:]
    discriminante = QuadraticDiscriminantAnalysis()
    discriminante=discriminante.fit(data_90.iloc[:,:11],data_90["quality"])
    prediction=discriminante.predict(data_10_whithout_quality)
    return data_10_quality, prediction 

#K-plus proches voisins
def k_plus_proches_voisins (data):
    data_10, data_90 = creer_nouvelle_base_donnees(data)  
    data_10_whithout_quality = data_10.iloc[:,:11]
    data_10_quality = data_10.iloc[:,11:]
    k = KNeighborsClassifier()
    k = k.fit(data_90.iloc[:,:11],data_90["quality"])
    prediction=k.predict(data_10_whithout_quality)
    return data_10_quality, prediction  

#Arbres de decision
def arbre_de_decision(data):
    data_10, data_90 = creer_nouvelle_base_donnees(data)  
    data_10_whithout_quality = data_10.iloc[:,:11]
    data_10_quality = data_10.iloc[:,11:]
    arbre = DecisionTreeClassifier()
    arbre = arbre.fit(data_90.iloc[:,:11],data_90["quality"])
    prediction=arbre.predict(data_10_whithout_quality)
    return data_10_quality, prediction 


def afficher_resultat(data):
   
        #Regression logistique
    data_1,prediction_1 = regression_logistique (data)
    regression=metrics.classification_report(data_1,prediction_1)
    print("Regression logistique")
    print(regression)
    
        #SVM , SVC, Machines a vecteurs supports
    data_2,prediction_2 = machine_a_vecteur_support (data)
    machine =metrics.classification_report(data_2,prediction_2)
    print("Machines a vecteurs supports")
    print(machine)
    
        #Analyse discriminante linéaire
    data_3,prediction_3 = analyse_discriminante_lineaire (data)
    lineaire =metrics.classification_report(data_3,prediction_3)
    print("Analyse discriminante linéaire")
    print(lineaire)
    
        #Analyse discriminante quadratique
    data_4,prediction_4 = analyse_discriminante_quadratique (data)
    quadratique =metrics.classification_report(data_4,prediction_4)
    print("Analyse discriminante quadratique")
    print(quadratique)
    
        #K-plus proches voisins
    data_5,prediction_5 = k_plus_proches_voisins (data)
    voisins =metrics.classification_report(data_5,prediction_5)
    print("K-plus proches voisins")
    print(voisins)       
    
        #Arbres de decision
    data_6,prediction_6 = arbre_de_decision (data)
    decision = metrics.classification_report(data_6,prediction_6)
    print("Arbres de decision")
    print(decision) 
    
   
    
    
    
    
    
    
    
    
    
    
