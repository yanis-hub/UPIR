#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 15 13:22:38 2024

@author: malatchoumymarine
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2 as cv
import os
from cherche_vitesse import vitesse #vitesse en fonction du nom de fichier
import matplotlib.ticker as ticker

N=2 #nombre de branche de l'analyse

if N==0:
    Ur_min=2.5
    Ur_max=25
    f_max=8
    amp_max=0.5
    ddl=4
    mod=2 #nombre de mode que l'on veut afficher 
    freq_propre=3.15
    dossier_traite=['/Volumes/lm2/Public Folders/Ferraut Samuel/Video_originale/N=0/0802mainpump/output_csv'
                    ,'/Volumes/lm2/Public Folders/Ferraut Samuel/Video_originale/N=0/0103secondpump/output_csv'
                    ,'/Volumes/lm2/Public Folders/Ferraut Samuel/Video_originale/N=0/0902mainpump/output_csv']
if N==1:
    Ur_min=4
    Ur_max=20
    f_max=5
    amp_max=1.2
    ddl=10
    mod=4 #nombre de mode que l'on veut afficher 
    freq_propre=1.88  
    dossier_traite=['/Volumes/lm2/Public Folders/Ferraut Samuel/Video_originale/N=1/0103secondpump/output_csv'
                    ,'/Volumes/lm2/Public Folders/Ferraut Samuel/Video_originale/N=1/3001secondpump/output_csv'
                    ,'/Volumes/lm2/Public Folders/Ferraut Samuel/Video_originale/N=1/0802mainpump/output_csv']
if N==2:
    Ur_min=4
    Ur_max=25
    f_max=5
    amp_max=1.4
    ddl=16
    mod=6 #nombre de mode que l'on veut afficher 
    freq_propre=1.41
    dossier_traite=['/Volumes/lm2/Public Folders/Ferraut Samuel/Video_originale/N=2/0103secondpump/output_csv'
                    ,'/Volumes/lm2/Public Folders/Ferraut Samuel/Video_originale/N=2/0902mainpump/output_csv']

#les lignes suivante servent a recuper le chemin d'accés de la premiere video pour extraire ses propriété

def rms_amplitude(data):
    return np.sqrt(np.mean(np.square(data)))

nb_fichier=0
element=[]
for i in dossier_traite:
    nb_fichier+=len(os.listdir(i))
    dossier_parent = os.path.dirname(i)
    a=os.listdir(dossier_parent)
    a.pop()
    element.append(a)


matrix_list = [] #matrice qui va contenir toutes les datas
c=np.zeros((ddl,nb_fichier)) #pour la récupération de la moyenne
nb_snap=[0] #pour stocker longeur de chaque fichier
fps_liste=[] #stocker les fps de chaque video
vit=[] #vitesse des videos à récuperer

compteur_tot=0
compteur_dos=0
for dos in dossier_traite:
    compteur_fichier=0
    for nom_fichier in os.listdir(dos):
        chemin_csv_i = os.path.join(dos, nom_fichier) #on recupére le nom de chaque fichier csv
        dossier_parent = os.path.dirname(os.path.dirname(chemin_csv_i)) #chemin parent de chaque fichier pour accéder aux videos
        chemin_vid=os.path.join(dossier_parent, element[compteur_dos][compteur_fichier]) #chemin de chaque video
        #video = cv.VideoCapture(chemin_vid)
        #width = video.get(cv.CAP_PROP_FRAME_WIDTH)
        #if width==2160.0:
            #print(chemin_vid)
        #print(width)
    
        u=vitesse(chemin_csv_i) #récupére la vitesse en fonction de nom du fichier 
        vit.append(u) #ajout a la liste globale
        
        video = cv.VideoCapture(chemin_vid)
        fps = video.get(cv.CAP_PROP_FPS)
        height=int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps_liste.append(fps)
        
        data = pd.read_csv(chemin_csv_i)
        n, m = data.shape[1], data.shape[0]
        A=np.zeros((n-1,m))
        
        for k in range(n-1):  #transfert des coloness du fichier csv dans une matrice
            A[k,:]=data.iloc[:,k+1]
            
        cols_with_neg_one = np.any(A == -1, axis=0) # Crée un masque booléen pour les colonnes contenant -1
        A = A[:, ~cols_with_neg_one] # Sélectionne les colonnes qui ne contiennent pas -1
        
        for k in range(1,len(A[:,0]),2): #remettre les données dans le bon ordre pour avoir un bon tracé
            A[k,:]=height-A[k,:]
            
        for i in range(len(A[:,0])):  #centrage des données sur 0
             c[i,compteur_fichier]=np.mean(A[i,:]) #recupération de la moyenne sur chaque point tracké
             A[i,:]*=5/42.9 #convertir les pixels en mm
             A[i,:]=(A[i,:]-np.mean(A[i,:]))  #centrage des données sur 0
             
        l=len(A[0,:]) #nombre de snapshot aprés filtre des mauvais tracking
        

        A[n-2,:]=np.zeros(l)
        A[n-3,:]=np.zeros(l)
             
        l=len(A[0,:]) #nombre de snapshot aprés filtre des mauvais tracking
        nb_snap.append(l)
        matrix_list.append(A)
        compteur_fichier+=1
        compteur_tot+=1
    compteur_dos+=1
    
D = np.hstack(matrix_list)

h=len(D[:,0]) 
ns=len(D[0,:]) #sanpshot non buggué

U, S, Vt = np.linalg.svd(D, full_matrices=False) #SVD pour séparer les modes de vibrations

plt.plot(np.arange(1,len(S)+1,1),np.cumsum(S) / np.sum(S), '-o', color='k') #energie par mode en %
plt.xlabel('Modes')
plt.ylabel(r'$\lambda $')
plt.title(f'Singular Values vs Modes, N={N}')
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.show()

frequence_glob=[]
vitesse_glob=[]
amp_mode=[]

for k in range(mod):
    freq_mode=[] #pour récuper la fréquence de chaque mode
    Amplitude=[]
    for i in range(nb_fichier):
        signal = Vt[k, :] #récupere seulement le signal pour 1 seul mode
        signal=signal[sum(nb_snap[:i+1]):sum(nb_snap[:i+2])] #slicing pour récupérerer la partir intéréssante
        ni=nb_snap[i+1]
        time=np.linspace(0,ni/fps_liste[i],ni) #vecteur temporel

        Amplitude.append(rms_amplitude(signal))
        # Appliquer la FFT
        signal_fft = np.fft.fft(signal)
        freq = np.fft.fftfreq(len(signal), d=1/fps_liste[i])

        # Magnitude de la FFT
        magnitude = np.abs(signal_fft)

        # Filtrer les fréquences sup à 1 Hz
        mask = freq >= 1
        freq = freq[mask]
        magnitude = magnitude[mask]
        
        # Trouver l'indice de la plus grandes magnitudes
        top_indice = np.argsort(magnitude)[-1:]

        # Extraire les fréquences correspondantes
        top_freq = freq[top_indice]
        top_magnitude = magnitude[top_indice]

        # Stocker la fréquence ou les fréquences
        freq_mode.append(top_freq)
        """
        plt.figure(figsize=(12, 6))  # Graph FFT
        plt.subplot(2, 1, 1)
        plt.plot(time, signal)
        plt.title(f'Signal temporel du mode {k+1}, u={vit[i]} m/s')
        plt.xlabel('Temps (s)')
        plt.ylabel('Amplitude')
        
        plt.subplot(2, 1, 2)
        plt.plot(freq, magnitude)
        plt.title(f'Transformée de Fourier du mode {k+1}, u={vit[i]} m/s')
        plt.xlabel('Fréquence (Hz)')
        plt.ylabel('Magnitude')
        plt.xlim(1, 20)
        plt.grid()

        # Annoter les plus grandes fréquences sur le graph
        for j in range(len(top_freqs)):
            plt.annotate(f'{top_freqs[j]:.2f} Hz', 
                         xy=(top_freqs[j], top_magnitudes[j]), 
                         xytext=(top_freqs[j], top_magnitudes[j] + 0.1 * np.max(magnitude)),
                         arrowprops=dict(facecolor='red', shrink=0.05, headwidth=5),
                         horizontalalignment='center',
                         fontsize=12,  # Augmenter légèrement la taille de la police
                         color='blue')
        plt.tight_layout()
        plt.show()"""
    vitesse_dupliquee = []
    freq_finale=[]
    # Dupliquer les valeurs côte à côte
    for i in range(len(vit)):
        if len(freq_mode[i])==2:
            vitesse_dupliquee.append(vit[i])
            vitesse_dupliquee.append(vit[i])
            freq_finale.append(float(freq_mode[i][0]))
            freq_finale.append(float(freq_mode[i][1]))
        else:
            vitesse_dupliquee.append(vit[i])
            freq_finale.append(float(freq_mode[i][0]))
    frequence_glob.append(freq_finale)
    vitesse_glob.append(vitesse_dupliquee)
    amp_mode.append(Amplitude)
    
Dia_branche=0.005 #en m

sym=['o','^','+','*','o','^','+']
plt.figure(figsize=(14, 8))

# Premier graphique
plt.subplot(2, 1, 1)
plt.title(f'RMS Amplitude as a function of Ur, N={N}')
plt.ylabel('A/D')
plt.xlim(Ur_min, Ur_max)
plt.ylim(0,amp_max)


# Deuxième graphique
plt.subplot(2, 1, 2)
plt.title(f'Frequencies as a function of Ur, N={N}')
plt.ylabel('Frequency/natural frequency')
plt.xlabel('Reduce velocity (Ur)')
plt.xlim(Ur_min, Ur_max)
plt.ylim(0,f_max)


# Boucle pour ajouter les données aux graphiques
for k in range(mod):
    plt.subplot(2, 1, 1)
    plt.plot(np.array(vit)/(Dia_branche*freq_propre),np.array(amp_mode[k])/Dia_branche, sym[k], label=f'mode {k+1}')
    
    plt.subplot(2, 1, 2)
    plt.plot(np.array(vitesse_glob[k])/(Dia_branche*freq_propre), np.array(frequence_glob[k])/freq_propre, sym[k], label=f'mode {k+1}')

plt.subplot(2, 1, 1)
plt.legend()
plt.subplot(2, 1, 2)
plt.legend()
plt.show()

Amp=200 #amplitude des modes propres 
for j in range(mod):
    for p in np.arange(0, 1.25, 0.1):
        linewidth = 1 - p  # diminuer l'épaisseur à chaque itération
        alpha = 0.8 - p * 0.15  # Diminuer l'opacité à chaque itération
        x, y = np.zeros(h//2), np.zeros(h//2)
        for i in range(0,h,2):
            x[i//2]=np.mean(c[i,:])+np.sin(p*np.pi)*U[i,j]*Amp
                
            y[i//2]=np.mean(c[i+1,:])+np.sin(p*np.pi)*U[i+1,j]*Amp
        if h==4:
            pairs = [(1, 0)]
        if h==10:
            pairs = [(0, 3), (1, 3), (2, 3), (4, 3)] #paterne de la structure pour 1 paire de branches
        if h==16:
            pairs = [(0, 6), (1, 5), (2, 5), (3, 5), (4, 6), (5, 6), (6, 7)] #paterne de la structure pour 2 paires de branches
        for f in pairs:
            plt.plot(np.array([x[f[0]],x[f[1]]]),np.array([y[f[0]],y[f[1]]]),color='black',linewidth=linewidth, alpha=alpha)
    plt.title(f'Mode {j+1}, Energy : {S[j]/np.sum(S)*100:.1f} %'.format(S[j]))
    plt.xlabel('X-axis (pixels)')
    plt.ylabel('Y-axis (pixels)')
    plt.axis('equal')
    plt.axis('off')
    plt.show()

     
"""
#animation
mode=2
U_i = U[:, mode-1].reshape(-1, 1)  # i-ème vecteur singulier de gauche
Sigma_i = S[mode-1]  # i-ème valeur singulière
Vt_i = Vt[mode-1, :].reshape(1, -1)  # i-ème vecteur singulier de droite

# Reconstruire le signal en utilisant seulement le i-ème mode
A_i = U_i @ np.diag([Sigma_i]) @ Vt_i

fs=20 #agrandissement de l'amplitude de vibration
A_i=fs*A_i

for j in range(h):
   A_i[j,:]+=np.mean(c[j,:])  

x, y =np.zeros((h//2,ns)), np.zeros((h//2,ns))
for i in range(h):  #association des x et y de chaque de point
    if i%2==0:
        x[i//2,:]=A_i[i,:]
    else:
        y[i//2,:]=A_i[i,:]

if h==10:
    pairs = [(0, 3), (1, 3), (2, 3), (4, 3)]
elif h==16:
    pairs = [(0, 6), (1, 5), (2, 5), (3, 5), (4, 6), (5, 6), (6, 7)]

for i in range(ns):
    for f in pairs:
        plt.plot(np.array([x[f[0],i],x[f[1],i]]),np.array([y[f[0],i],y[f[1],i]]),'-')
        #plt.ylim(np.min(y)-20,np.max(y)+20)
        #plt.xlim(np.min(x)-20,np.max(x)+20)
        plt.ylim(100,800)
        plt.xlim(800,1600)
        plt.title(f'mode {mode}')
    if i==80:
        break
    plt.pause(0.05)
    plt.axis('equal') 
    plt.axis('off') 
plt.show()
"""
