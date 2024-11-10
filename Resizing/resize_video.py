import cv2
import argparse

def resize_video(input_path: str, output_path: str, width: int, height: int):
    # Ouvrir la vidéo d'entrée
    cap = cv2.VideoCapture(input_path)

    # Vérifier si la vidéo s'ouvre correctement
    if not cap.isOpened():
        print("Erreur : Impossible d'ouvrir la vidéo.")
        return

    # Déterminer le codec et créer l'objet VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec pour .mp4
    out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Fin de la vidéo

        # Redimensionner le cadre
        resized_frame = cv2.resize(frame, (width, height))

        # Écrire le cadre redimensionné dans la vidéo de sortie
        out.write(resized_frame)

    # Libérer les objets VideoCapture et VideoWriter
    cap.release()
    out.release()
    print("Vidéo redimensionnée enregistrée sous :", output_path)

if __name__ == "__main__":
    # Argument parser pour la ligne de commande
    parser = argparse.ArgumentParser(description="Redimensionner une vidéo.")
    parser.add_argument("input_path", type=str, help="Chemin d'accès à la vidéo d'entrée.")
    parser.add_argument("output_path", type=str, help="Chemin d'accès à la vidéo de sortie.")
    parser.add_argument("width", type=int, help="Largeur de la vidéo redimensionnée.")
    parser.add_argument("height", type=int, help="Hauteur de la vidéo redimensionnée.")

    args = parser.parse_args()
    
    # Appeler la fonction pour redimensionner la vidéo
    resize_video(args.input_path, args.output_path, args.width, args.height)
