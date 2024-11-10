import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color
from skimage.transform import resize

# Charger une image en niveaux de gris et redimensionner pour simplifier l'exemple
image = color.rgb2gray(data.camera())
image = resize(image, (100, 100))  # Redimensionner pour une taille plus petite

# Appliquer la SVD
U, S, Vt = np.linalg.svd(image, full_matrices=False)

# Fonction pour reconstruire l'image avec un certain nombre de valeurs singulières
def reconstruct_image(U, S, Vt, k):
    # Utiliser les k premières valeurs singulières pour la reconstruction
    S_k = np.zeros((k, k))
    np.fill_diagonal(S_k, S[:k])
    U_k = U[:, :k]
    Vt_k = Vt[:k, :]
    return U_k @ S_k @ Vt_k

# Visualiser les reconstructions avec différents nombres de valeurs singulières
k_values = [5, 10, 20, 50, 100]  # Différents niveaux de compression
plt.figure(figsize=(10, 6))
for i, k in enumerate(k_values, 1):
    reconstructed_image = reconstruct_image(U, S, Vt, k)
    plt.subplot(2, 3, i)
    plt.imshow(reconstructed_image, cmap='gray')
    plt.title(f'k = {k} valeurs singulières')
    plt.axis('off')

# Afficher l'image d'origine pour comparaison
plt.subplot(2, 3, 6)
plt.imshow(image, cmap='gray')
plt.title('Image originale')
plt.axis('off')
plt.show()
