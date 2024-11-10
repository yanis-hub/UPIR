#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 11:35:17 2024
@author: malatchoumymarine"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import cv2 as cv
import matplotlib.ticker as ticker
from cherche_vitesse import vitesse #vitesse en fonction du nom de fichier

csv_file='/Volumes/lm2/Public Folders/Ferraut Samuel/Video_originale/N=2/0103secondpump/output_csv/N=2 (36).csv'
video_path='/Volumes/lm2/Public Folders/Ferraut Samuel/Video_originale/N=2/0103secondpump/N=2 (36).MP4'

u=vitesse(csv_file) #récupération vitesse 
UR=u/(0.005*1.41)
video = cv.VideoCapture(video_path)
data = pd.read_csv(csv_file)

#inofrmations video
fps = video.get(cv.CAP_PROP_FPS)
height=int(video.get(cv.CAP_PROP_FRAME_HEIGHT))

n, m = data.shape[1], data.shape[0] #nombre de point et nombre de snapshots

A=np.zeros((n-1,m))

for k in range(n-1):  #transfert des coloness du fichier csv dans une matrice
    A[k,:]=data.iloc[:,k+1]

cols_with_neg_one = np.any(A == -1, axis=0) # Crée un masque booléen pour les colonnes contenant -1

A = A[:, ~cols_with_neg_one] # Sélectionne les colonnes qui ne contiennent pas -1

for k in range(1,len(A[:,0]),2): #remettre les données dans le bon ordre pour avoir un bon tracé
    A[k,:]=height-A[k,:]

B=np.array(A) #sauvegarde des données pour plot

for i in range(len(A[:,0])):  #centrage des données sur 0
    A[i,:]=(A[i,:]-np.mean(A[i,:])) 
    
#analyse SVD    
U, S, Vt = np.linalg.svd(A, full_matrices=True) #SVD pour séparer les modes de vibrations

h=n-1
ns=len(A[0,:]) #sanpshot non buggué


#singular value 
plt.plot(np.arange(1,len(S)+1,1),np.cumsum(S) / np.sum(S), '-o', color='k') #energie par mode en %
plt.xlabel('Modes')
plt.ylabel(r'$\lambda $')
plt.title('Singular Values vs Modes')
plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
plt.show()

#FFT 
mod=4 #nombre de mode que l'on veut afficher
freq_mode=np.zeros(h) #pour récuper la fréquence de chaque mode
time=np.linspace(0,ns/fps,ns) #vecteur temporel
Amplitude=np.zeros(mod)

def rms_amplitude(data):
    return np.sqrt(np.mean(np.square(data)))

for i in range(mod):
    signal = Vt[i, :]
    Amplitude[i]=rms_amplitude(signal)

    # Appliquer la FFT
    signal_fft = np.fft.fft(signal)
    freq = np.fft.fftfreq(len(signal), d=1/fps)

    # Magnitude de la FFT
    magnitude = np.abs(signal_fft)

    # Filtrer les fréquences sup à 1 Hz
    mask = freq >= 1
    freq = freq[mask]
    magnitude = magnitude[mask]

    # Trouver les indices des 5 plus grandes magnitudes
    top_indices = np.argsort(magnitude)[-5:]

    # Extraire les fréquences correspondantes
    top_freqs = freq[top_indices]
    top_magnitudes = magnitude[top_indices]

    freq_mode[i] = freq[np.argmax(magnitude)]
    plt.figure(figsize=(12, 6))  # Graph FFT

    plt.subplot(2, 1, 1)
    plt.plot(time, signal)
    plt.title(f'Signal temporel du mode {i+1}, u={u} m/s')
    plt.xlabel('Temps (s)')
    plt.ylabel('Amplitude')

    plt.subplot(2, 1, 2)
    plt.plot(freq, magnitude)
    plt.title(f'Transformée de Fourier du mode {i+1}, u={u} m/s')
    plt.xlabel('Fréquence (Hz)')
    plt.ylabel('Magnitude')
    plt.xlim(1, 15)
    plt.grid()

    # Annoter les 5 plus grandes fréquences avec un décalage vertical
    for j in range(len(top_freqs)):
        plt.annotate(f'{top_freqs[j]:.2f} Hz', 
                     xy=(top_freqs[j], top_magnitudes[j]), 
                     xytext=(top_freqs[j], top_magnitudes[j] + 0.1 * np.max(magnitude)),
                     arrowprops=dict(facecolor='red', shrink=0.05, headwidth=5),
                     horizontalalignment='center',
                     fontsize=12,  # Augmenter légèrement la taille de la police
                     color='blue')
    plt.tight_layout()
    plt.show()
 
#plot modes   
Amp=100 #amplitude des modes propres 

for j in range(mod):
    for p in np.arange(0, 1.25, 0.1):
        linewidth = 1 - p  # diminuer l'épaisseur à chaque itération
        alpha = 0.8 - p * 0.15  # Diminuer l'opacité à chaque itération
        x, y = np.zeros(h//2), np.zeros(h//2)
        for i in range(0,h,2):
            
            x[i//2]=np.mean(B[i,:])+np.sin(p*np.pi)*U[i,j]*Amp
                
            y[i//2]=np.mean(B[i+1,:])+np.sin(p*np.pi)*U[i+1,j]*Amp
        if h==4:
            pairs = [(0, 1)] 
        if h==10:
            pairs = [(0, 3), (1, 3), (2, 3), (4, 3)] #paterne de la structure pour 1 paire de branches
        if h==16:
            pairs = [(0, 6), (1, 5), (2, 5), (3, 5), (4, 6), (5, 6), (6, 7)] #paterne de la structure pour 2 paires de branches
        for f in pairs:
            plt.plot(np.array([x[f[0]],x[f[1]]]),np.array([y[f[0]],y[f[1]]]),color='black',linewidth=linewidth, alpha=alpha)
    plt.title(f'Mode {j+1}, Vitesse={u} m/s, amp={Amplitude[j]}, Energie : {S[j]/np.sum(S)*100:.1f} % '.format(S[j]))
    plt.xlabel('X-axis (pixels)')
    plt.ylabel('Y-axis (pixels)')
    plt.axis('equal')
    plt.show()
    

"""for j in range(mod): #avec spline
      for p in np.arange(0, 1.25, 0.1):
          linewidth = 1 - p  # Augmenter l'épaisseur à chaque itération
          alpha = 1.0 - p * 0.15  # Diminuer l'opacité à chaque itération
          x, y = np.zeros(h//2), np.zeros(h//2)
          for i in range(0, h, 2):
            
              x[i//2]=np.mean(c[i,:])+np.sin(p*np.pi)*U[i,j]*Amp
                 
              y[i//2]=np.mean(c[i+1,:])+np.sin(p*np.pi)*U[i+1,j]*Amp
          if h==4:
             pairs = [(1, 0)]
          elif h == 10:
             pairs = [(0, 3), (1, 3), (2, 3), (4, 3)]
          elif h == 16:
             pairs = [(0, 6), (1, 5), (2, 5), (3, 5), (4, 6), (5, 6), (6, 7)]
        
          for f in pairs:
              # Créer des points intermédiaires pour une courbe lisse
              mid_x = (x[f[0]] + x[f[1]]) / 2 + np.random.randn() * 10  # Ajouter un peu de variabilité
              mid_y = (y[f[0]] + y[f[1]]) / 2 + np.random.randn() * 10
                
              points_x = np.array([x[f[0]], mid_x, x[f[1]]])
              points_y = np.array([y[f[0]], mid_y, y[f[1]]])
                
              cs = CubicSpline(np.arange(len(points_x)), points_x)
              cs_y = CubicSpline(np.arange(len(points_y)), points_y)
                
              t = np.linspace(0, len(points_x) - 1, 100)
              spline_x = cs(t)
              spline_y = cs_y(t)
              plt.plot(spline_x,spline_y,color='black',linewidth=linewidth, alpha=alpha)
    
      plt.title(f'Mode {j+1} , fréquence : {freq_mode[j]:.3f}'.format(S[j]))
      plt.xlabel('X-axis (pixels)')
      plt.ylabel('Y-axis (pixels)')
      plt.show() """
    