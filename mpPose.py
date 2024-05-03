import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import cv2
import mediapipe as mp
import os

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

# Directory contenente le immagini di input
input_dir = "samples/img"

# Trova tutti i file immagine nella directory
image_files = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.endswith(('.jpg', '.jpeg', '.png'))]

with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2) as pose:
    for idx, file in enumerate(image_files):
        image = cv2.imread(file)
        image_height, image_width, _ = image.shape
        # Converti l'immagine BGR in RGB prima dell'elaborazione.
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        annotated_image = image.copy()

        # Disegna i landmark della posa e delle mani sull'immagine.
        mp_drawing.draw_landmarks(
            annotated_image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        # Salva l'immagine annotata con un nome modificato nella stessa cartella di input
        cv2.imwrite(file.replace("samples/img/", "samples/imgPose/pose_"), annotated_image)
