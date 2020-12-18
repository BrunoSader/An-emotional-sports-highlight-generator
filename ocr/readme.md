## Fonctionnement de l'OCR

L'OCR fonctionne grâce à deux fichiers :
-main.py, qui lit la vidéo et va analyser le scoreboard en temps réel (utilise tesseract)
-highlights.py, qui va sortir les highlights

### Pour faire fonctionner l'OCR :

Dans le fichier <u>main.py</u> , remplir les variables suivantes :
**left_x**
**upper_y**
**right_x**
**lower_y**

Qui sont les positions de la scoreboard dans la video, en pixels.

**time_divide** 

Correspond à la position de séparation entre la partie "temps" de la scoreboard et la partie "score"

**time_width**

Correspond à la largeur de la partie "temps" de la scoreboard

**time_position = 'left' ou 'right'**

Selon si le temps est positionné à gauche ou à droite du score

**video_length**

La longueur de la video en secondes

Renseigner également le nom de la vidéo à analyser dans la classe ImageHandler:   **self.video_source_path** = 'ocr/tmp/test.mkv' (par exemple)

Puis, lancer depuis le fichier source : python ocr/main.py

(pour l'instant : arrêter avec ctrl+C)

### Générer les highlights

Les temps seront renseignés dans le fichier times.txt

Il faut rentrer, dans le fichier **highlights.py**, les infos suivantes :

**filename='times.txt'** (nom du fichier avec les temps)
**highlight_length = 10** (longueur d'un highlight que l'on va considérer comme pertinent, en secondes)
**video_path = 'ocr/highlights_videos'** (où stocker les vidéos que l'on va générer)
**video_name = 'but.mkv'** (nom de la vidéo à couper, il faut qu'elle soit dans le dossier video_path)

Executer avec python ocr/highlights.py (pour l'instant : va juste sortir des segments de vidéo et les mettre dans video_path)