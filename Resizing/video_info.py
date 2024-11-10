import cv2

def get_video_info(video_path: str):
    # Ouvrir la vidéo
    cap = cv2.VideoCapture(video_path)

    # Vérifier si la vidéo s'ouvre correctement
    if not cap.isOpened():
        print("Erreur : Impossible d'ouvrir la vidéo.")
        return

    # Extraire les propriétés de la vidéo
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # Largeur de la vidéo
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # Hauteur de la vidéo
    fps = cap.get(cv2.CAP_PROP_FPS)  # Fréquence d'images (fps)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # Nombre total de frames

    # Calculer la durée de la vidéo en secondes
    duration = total_frames / fps

    # Afficher les informations
    print(f"Informations sur la vidéo :")
    print(f"  - Résolution : {width} x {height} pixels")
    print(f"  - Fréquence d'images : {fps} fps")
    print(f"  - Nombre total de frames : {total_frames}")
    print(f"  - Durée : {duration:.2f} secondes")

    # Libérer l'objet VideoCapture
    cap.release()

if __name__ == "__main__":
    video_path = "/Users/yanis/Desktop/N=1 (1).MP4"  # Remplace par le chemin de ta vidéo
    get_video_info(video_path)

