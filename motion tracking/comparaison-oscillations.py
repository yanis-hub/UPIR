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

def calculate_second_derivative(position_x, position_y):
    """Calcule la dérivée seconde pour les coordonnées x et y."""
    dx = np.diff(position_x, n=1)
    dy = np.diff(position_y, n=1)
    ddx = np.diff(dx, n=1)
    ddy = np.diff(dy, n=1)
    return np.sqrt(ddx**2 + ddy**2)

def moving_average(data, window_size=10):
    """Calcule la moyenne mobile pour lisser les données."""
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

def plot_regression_line(frames, oscillations, label, color):
    """Ajuste et trace une droite de régression linéaire."""
    slope, intercept = np.polyfit(frames, oscillations, 1)
    regression_line = slope * frames + intercept
    plt.plot(frames, regression_line, label=f'{label} (régression)', color=color, linestyle='--')

def compare_oscillations(mosse_csv, kcf_csv, csrt_csv, window_size=10):
    # Charger les résultats de chaque algorithme
    mosse_0_x, mosse_0_y, mosse_1_x, mosse_1_y = load_data(mosse_csv)
    kcf_0_x, kcf_0_y, kcf_1_x, kcf_1_y = load_data(kcf_csv)
    csrt_0_x, csrt_0_y, csrt_1_x, csrt_1_y = load_data(csrt_csv)
    
    # Calculer les oscillations (dérivée seconde) pour chaque noeud et chaque algorithme
    mosse_osc_0 = calculate_second_derivative(mosse_0_x, mosse_0_y)
    mosse_osc_1 = calculate_second_derivative(mosse_1_x, mosse_1_y)
    kcf_osc_0 = calculate_second_derivative(kcf_0_x, kcf_0_y)
    kcf_osc_1 = calculate_second_derivative(kcf_1_x, kcf_1_y)
    csrt_osc_0 = calculate_second_derivative(csrt_0_x, csrt_0_y)
    csrt_osc_1 = calculate_second_derivative(csrt_1_x, csrt_1_y)
    
    # Calculer l'oscillation moyenne pour chaque algorithme (pour les deux noeuds)
    mosse_avg_osc = (mosse_osc_0 + mosse_osc_1) / 2
    kcf_avg_osc = (kcf_osc_0 + kcf_osc_1) / 2
    csrt_avg_osc = (csrt_osc_0 + csrt_osc_1) / 2
    
    # Lissage des oscillations avec une moyenne mobile
    mosse_avg_osc_smooth = moving_average(mosse_avg_osc, window_size)
    kcf_avg_osc_smooth = moving_average(kcf_avg_osc, window_size)
    csrt_avg_osc_smooth = moving_average(csrt_avg_osc, window_size)
    
    # Tracer uniquement les droites de régression linéaire
    frames = np.arange(len(mosse_avg_osc_smooth))
    plt.figure(figsize=(10, 6))
    plot_regression_line(frames, mosse_avg_osc_smooth, 'MOSSE', 'blue')
    plot_regression_line(frames, kcf_avg_osc_smooth, 'KCF', 'green')
    plot_regression_line(frames, csrt_avg_osc_smooth, 'CSRT', 'red')
    
    # Personnalisation du graphique
    plt.xlabel('Frame')
    plt.ylabel('Oscillation (variation de l\'accélération)')
    plt.title('Comparaison des oscillations pour chaque algorithme (régression linéaire uniquement)')
    plt.legend()
    plt.grid()
    # plt.savefig("Comparaison-oscillations-regression.png")
    plt.show()

# Chemins vers les fichiers CSV
mosse_csv = "/Users/yanis/Desktop/output_csv/N=0/output-mosse.csv"
kcf_csv = "/Users/yanis/Desktop/output_csv/N=0/output-kcf.csv"
csrt_csv = "/Users/yanis/Desktop/output_csv/N=0/output-csrt.csv"

compare_oscillations(mosse_csv, kcf_csv, csrt_csv)
