import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_data(csv_path):
    """Charge les données CSV et retourne les positions des noeuds."""
    data = pd.read_csv(csv_path)
    node_0_x = data['0_u'].values
    node_0_y = data['0_v'].values
    node_1_x = data['1_u'].values
    node_1_y = data['1_v'].values
    return node_0_x, node_0_y, node_1_x, node_1_y

def calculate_error(reference_x, reference_y, algo_x, algo_y, norm_type='Linf'):
    """
    Calcule l'erreur entre la référence et l'algorithme en fonction de la norme choisie.
    
    norm_type : str
        'L1' pour la norme de Manhattan,
        'L2' pour la norme Euclidienne (par défaut),
        'Linf' pour la norme de Chebyshev.
    """
    if norm_type == 'L1':
        # Norme de Manhattan
        error = np.abs(reference_x - algo_x) + np.abs(reference_y - algo_y)
    elif norm_type == 'L2':
        # Norme Euclidienne
        error = np.sqrt((reference_x - algo_x) ** 2 + (reference_y - algo_y) ** 2)
    elif norm_type == 'Linf':
        # Norme de Chebyshev
        error = np.maximum(np.abs(reference_x - algo_x), np.abs(reference_y - algo_y))
    else:
        raise ValueError("Type de norme non reconnu. Utilisez 'L1', 'L2', ou 'Linf'.")
    
    return error


def compare_algorithms(mosse_csv, kcf_csv, csrt_csv):
    # Charger les résultats de chaque algorithme
    mosse_0_x, mosse_0_y, mosse_1_x, mosse_1_y = load_data(mosse_csv)
    kcf_0_x, kcf_0_y, kcf_1_x, kcf_1_y = load_data(kcf_csv)
    csrt_0_x, csrt_0_y, csrt_1_x, csrt_1_y = load_data(csrt_csv)
    
    # Calculer la trajectoire de référence (moyenne des trois algorithmes)
    ref_0_x = (mosse_0_x + kcf_0_x + csrt_0_x) / 3
    ref_0_y = (mosse_0_y + kcf_0_y + csrt_0_y) / 3
    ref_1_x = (mosse_1_x + kcf_1_x + csrt_1_x) / 3
    ref_1_y = (mosse_1_y + kcf_1_y + csrt_1_y) / 3
    
    # Calculer l'erreur pour chaque noeud et chaque algorithme
    mosse_error_0 = calculate_error(ref_0_x, ref_0_y, mosse_0_x, mosse_0_y)
    mosse_error_1 = calculate_error(ref_1_x, ref_1_y, mosse_1_x, mosse_1_y)
    kcf_error_0 = calculate_error(ref_0_x, ref_0_y, kcf_0_x, kcf_0_y)
    kcf_error_1 = calculate_error(ref_1_x, ref_1_y, kcf_1_x, kcf_1_y)
    csrt_error_0 = calculate_error(ref_0_x, ref_0_y, csrt_0_x, csrt_0_y)
    csrt_error_1 = calculate_error(ref_1_x, ref_1_y, csrt_1_x, csrt_1_y)
    
    # Calculer l'erreur moyenne pour chaque algorithme
    mosse_avg_error = (mosse_error_0 + mosse_error_1) / 2
    kcf_avg_error = (kcf_error_0 + kcf_error_1) / 2
    csrt_avg_error = (csrt_error_0 + csrt_error_1) / 2
    
    # Tracer les erreurs moyennes au cours du temps
    frames = np.arange(len(mosse_avg_error))
    plt.figure(figsize=(10, 6))
    plt.plot(frames, mosse_avg_error, label='MOSSE', color='blue')
    plt.plot(frames, kcf_avg_error, label='KCF', color='green')
    plt.plot(frames, csrt_avg_error, label='CSRT', color='red')
    
    plt.xlabel('Frame')
    plt.ylabel('Erreur de suivi (pixels)')
    plt.title('Comparaison de l\'erreur de suivi pour chaque algorithme')
    plt.legend()
    plt.grid()
    # plt.savefig("Comparaison-erreur.png")
    plt.show()

# Chemins vers les fichiers CSV
mosse_csv = "/Users/yanis/Desktop/output_csv/output-mosse.csv"
kcf_csv = "/Users/yanis/Desktop/output_csv/output-kcf.csv"
csrt_csv = "/Users/yanis/Desktop/output_csv/output-csrt.csv"

compare_algorithms(mosse_csv, kcf_csv, csrt_csv)
