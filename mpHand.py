import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import cv2
import mediapipe as mp
import os

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Directory contenente le immagini di input
input_dir = "samples/img"

# Trova tutti i file immagine nella directory
image_files = [os.path.join(input_dir, file) for file in os.listdir(input_dir) if file.endswith(('.jpg', '.jpeg', '.png'))]

with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2) as hands:
    for idx, file in enumerate(image_files):
        image = cv2.imread(file)
        image_height, image_width, _ = image.shape
        # Converti l'immagine BGR in RGB prima dell'elaborazione.
        results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        annotated_image = image.copy()

        # Disegna i landmark della posa e delle mani sull'immagine.
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())


        # Salva l'immagine annotata con un nome modificato nella stessa cartella di input
        cv2.imwrite(file.replace("samples/img/", "samples/imgHand/hand_"), annotated_image)
