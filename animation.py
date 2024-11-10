#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 13:07:47 2024

@author: malatchoumymarine
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2 as cv
from matplotlib.animation import FuncAnimation

csv_file = '/Users/yanis/Desktop/output_csv/output.csv'
video_path = '/Users/yanis/Desktop/output.MP4'

data = pd.read_csv(csv_file)
video = cv.VideoCapture(video_path)
height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))

frames = data['frame'].values

# Nombre de points et nombre de snapshots
n, m = data.shape[1], data.shape[0]

A = np.zeros((n - 1, m))
for k in range(n - 1):  # Transfert des colonnes du fichier CSV dans une matrice
    A[k, :] = data.iloc[:, k + 1]

A = A[~np.any(A == -1, axis=1)]  # Enlever les points où le suivi a échoué

h = len(A[:, 0])  # Nombre de degrés de liberté
ns = len(A[0, :])  # Nombre de snapshots

# Ajustement des coordonnées y
for k in range(1, h, 2):
    A[k, :] = height - A[k, :]  # Inverser les coordonnées y pour un bon tracé

# Centrage des données
c = np.zeros(h)
for i in range(h):
    c[i] = np.mean(A[i, :])  # Moyenne sur chaque point tracké
    A[i, :] -= c[i]  # Centrage des données sur 0

# Décomposition en valeurs singulières (SVD) pour extraire les modes de vibration
U, S, Vt = np.linalg.svd(A, full_matrices=True)

mode = 4  # Sélection du i-ème mode

U_i = U[:, mode - 1].reshape(-1, 1)  # i-ème vecteur singulier de gauche
Sigma_i = S[mode - 1]  # i-ème valeur singulière
Vt_i = Vt[mode - 1, :].reshape(1, -1)  # i-ème vecteur singulier de droite

# Reconstruction du signal en utilisant uniquement le i-ème mode
A_i = U_i @ np.diag([Sigma_i]) @ Vt_i

fs = 10  # Facteur d'agrandissement de l'amplitude des vibrations
A_i *= fs

# Recentrage des données
for j in range(h):
    A_i[j, :] += c[j]

# Séparation des coordonnées x et y
x = np.zeros((h // 2, m))
y = np.zeros((h // 2, m))
for i in range(h):
    if i % 2 == 0:
        x[i // 2, :] = A_i[i, :]
    else:
        y[i // 2, :] = A_i[i, :]

# Fonction pour générer les paires de points en fonction de h
def generate_pairs(h):
    pairs = []

    if h % 2 != 0:
        raise ValueError("Le nombre de degrés de liberté 'h' doit être pair pour correspondre aux coordonnées x et y.")

    num_points = h // 2  # Nombre de points suivis

    if num_points == 2:
        # Pour h = 4 (2 points), connecter les deux points
        pairs = [(0, 1)]
    elif num_points == 3:
        # Pour h = 6 (3 points), former un triangle
        pairs = [(0, 1), (1, 2), (2, 0)]
    elif num_points == 5:
        # Pour h = 10 (5 points), connecter les points à un point central (par exemple, le point 2)
        central_point = 2
        pairs = [(i, central_point) for i in range(num_points) if i != central_point]
    elif num_points == 8:
        # Pour h = 16 (8 points), définir un motif spécifique
        pairs = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 0)]
    else:
        # Cas général : connecter les points de manière consécutive
        pairs = [(i, i + 1) for i in range(num_points - 1)]
        # Optionnel : fermer la boucle
        # pairs.append((num_points - 1, 0))

    return pairs

# Générer les paires de points
try:
    pairs = generate_pairs(h)
except ValueError as e:
    print(e)
    pairs = []



# Tracé des points et des connexions
if pairs:
    plt.figure()
    for i in range(ns):
        plt.clf()  # Effacer la figure pour le prochain frame
        for f in pairs:
            plt.plot([x[f[0], i], x[f[1], i]], [y[f[0], i], y[f[1], i]], '-o')
        plt.ylim(np.min(y) - 50, np.max(y) + 50)
        plt.xlim(np.min(x) - 50, np.max(x) + 50)
        plt.title(f'Mode {mode}')
        plt.axis('equal')
        plt.pause(0.05)
    plt.show()
else:
    print("Aucune paire de points à tracer.")
