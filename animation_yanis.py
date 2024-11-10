import pandas as pd
import cv2 as cv

def main(csv_file: str, video_path: str, output_path: str):
    # Lire le fichier CSV contenant les coordonnées des noeuds
    data = pd.read_csv(csv_file)

    # Déterminer automatiquement le nombre de noeuds à partir des colonnes du fichier CSV
    num_nodes = len([col for col in data.columns if '_u' in col])
    print(f"Nombre de noeuds détectés : {num_nodes}")

    # Ouvrir la vidéo originale
    video = cv.VideoCapture(video_path)

    if not video.isOpened():
        print("Erreur : Impossible d'ouvrir la vidéo.")
        return

    # Obtenir les propriétés de la vidéo
    total_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    fps = int(video.get(cv.CAP_PROP_FPS))
    width = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))

    # Configurer l'enregistrement de la vidéo de sortie
    fourcc = cv.VideoWriter_fourcc(*'mp4v')  # Codec pour .mp4
    out = cv.VideoWriter(output_path, fourcc, fps, (width, height))

    # Obtenir les coordonnées des noeuds à partir du CSV
    node_coords = [(data[f'{i}_u'].values, data[f'{i}_v'].values) for i in range(num_nodes)]

    frame_id = 0

    while frame_id < len(node_coords[0][0]):
        ret, frame = video.read()
        if not ret:
            print("Fin de la vidéo ou erreur de lecture.")
            break

        # Dessiner chaque noeud et stocker les positions
        points = []
        for i in range(num_nodes):
            point = (int(node_coords[i][0][frame_id]), int(node_coords[i][1][frame_id]))
            points.append(point)
            # Dessiner le noeud avec une couleur spécifique
            color = (0, 0, 255) if i == 2 else (255, 0, 0)  # Rouge pour le noeud central (2), bleu pour les autres
            cv.circle(frame, point, 10, color, -1)

        # Relier les noeuds en fonction de la structure du corail
        if num_nodes == 8:
            # Configuration en corail pour 8 noeuds (structure en arbre)
            cv.line(frame, points[2], points[0], (0, 0, 0), 5)  # Centre (2) à noeud 0 (haut gauche)
            cv.line(frame, points[2], points[1], (0, 0, 0), 5)  # Centre (2) à noeud 1 (haut droit)
            cv.line(frame, points[2], points[3], (0, 0, 0), 5)  # Centre (2) à noeud 3 (gauche)
            cv.line(frame, points[4], points[6], (0, 0, 0), 5)  # Noeud 4 à noeud 6 (bas droit)
            cv.line(frame, points[5], points[6], (0, 0, 0), 5)  # Noeud 5 à noeud 6 (liaison diagonale)
            cv.line(frame, points[2], points[6], (0, 0, 0), 5)  # Noeud 2 à noeud 6 (liaison diagonale)
            cv.line(frame, points[7], points[6], (0, 0, 0), 5)  # Noeud 7 à noeud 6 (nouvelle liaison)

        elif num_nodes == 5:
            # Configuration en "Y" pour 5 noeuds
            cv.line(frame, points[2], points[0], (0, 0, 0), 5)  # Centre à gauche
            cv.line(frame, points[2], points[1], (0, 0, 0), 5)  # Centre à droite
            cv.line(frame, points[2], points[3], (0, 0, 0), 5)  # Centre en bas gauche
            cv.line(frame, points[2], points[4], (0, 0, 0), 5)  # Centre en bas droite
        elif num_nodes == 2:
            # Pour 2 noeuds, relier les deux points simplement
            cv.line(frame, points[0], points[1], (0, 0, 0), 5)

        # Écrire la frame dans la vidéo de sortie
        out.write(frame)

        # Afficher la frame dans une fenêtre OpenCV
        cv.imshow('Video avec Noeuds et Lignes', frame)

        # Attendre 25 ms entre chaque frame (peut être ajusté en fonction du FPS)
        if cv.waitKey(25) & 0xFF == ord('q'):
            break

        frame_id += 1

    # Libérer les ressources
    video.release()
    out.release()
    cv.destroyAllWindows()
    
if __name__ == "__main__":
    # Chemin vers le fichier CSV, la vidéo originale et la vidéo de sortie
    csv_file = "/Users/yanis/Desktop/output_csv/N=1 (croped).csv"  # Remplace par ton chemin
    video_path = "/Users/yanis/Desktop/N=1 (croped).MP4"          # Remplace par ton chemin
    output_path = "/Users/yanis/Desktop/animation/N=1.mp4" # Remplace par ton chemin

    main(csv_file, video_path, output_path)
