# Object tracking
L'algorithme KCF, intégré dans OpenCV, offre une méthode efficace pour suivre la position d'un objet à travers différentes frames. Cette technique est basée sur l'algorithme décrit dans le papier de J.F. Henriques et al., 'Exploiting the circulant structure of tracking-by-detection with kernels', présenté à l'ECCV 2012. Le KCF utilise des filtres de corrélation pour prédire la position d'un objet, en exploitant les structures circulantes dans les calculs de détection, ce qui le rend particulièrement adapté pour des applications de suivi en temps réel."

Pour plus de détails techniques, vous pouvez consulter la documentation d'OpenCV sur le TrackerKCF.

## Installation

```commandline

# for virtualenv user 
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```

### Utilisation:
Spécifiez les paramètres suivants :

video_path: str - Chemin d'accès à la vidéo à suivre.   
write_csv: bool - Si True, les résultats sont écrits dans un fichier .csv.  
visualize: bool - Si True, affiche la vidéo avec le suivi en temps réel.    


Au début de l'algorithme, une fenêtre s'ouvrira pour permettre à l'utilisateur de dessiner une boîte de sélection (bbox) autour de chaque objet à suivre. Utilisez la souris pour dessiner une boîte autour de l'objet à suivre, puis appuyez deux fois sur la touche Entrée pour valider. Appuyez sur Échap une fois tous les objets sélectionnés.


#### Pour lancer une prédiction avec écriture d'un fichier csv AVEC visualisation
```commandline
python main.py /home/hugo/Project/MotionTracking/data/N=2_1.MP4 --write-csv --visualize
```

#### Pour lancer une prédiction avec écriture d'un fichier csv SANS visualisation
```commandline
python main.py /home/hugo/Project/MotionTracking/data/N=2_1.MP4 --write-csv
```