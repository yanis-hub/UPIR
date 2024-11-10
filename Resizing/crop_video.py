import cv2
import argparse

def crop_video(input_path: str, output_path: str, x: int, y: int, width: int, height: int):
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

        # Extraire la région d'intérêt (ROI)
        roi = frame[y:y + height, x:x + width]

        # Écrire le cadre découpé dans la vidéo de sortie
        out.write(roi)

    # Libérer les objets VideoCapture et VideoWriter
    cap.release()
    out.release()
    print("Vidéo recadrée enregistrée sous :", output_path)

if __name__ == "__main__":
    # Argument parser pour la ligne de commande
    parser = argparse.ArgumentParser(description="Recadrer une vidéo.")
    parser.add_argument("input_path", type=str, help="Chemin d'accès à la vidéo d'entrée.")
    parser.add_argument("output_path", type=str, help="Chemin d'accès à la vidéo de sortie.")
    parser.add_argument("x", type=int, help="Coordonnée x du coin supérieur gauche de la zone rectangulaire.")
    parser.add_argument("y", type=int, help="Coordonnée y du coin supérieur gauche de la zone rectangulaire.")
    parser.add_argument("width", type=int, help="Largeur de la zone rectangulaire.")
    parser.add_argument("height", type=int, help="Hauteur de la zone rectangulaire.")

    args = parser.parse_args()

    # Appeler la fonction pour recadrer la vidéo
    crop_video(args.input_path, args.output_path, args.x, args.y, args.width, args.height)
