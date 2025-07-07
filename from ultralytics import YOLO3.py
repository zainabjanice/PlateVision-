from ultralytics import YOLO
import cv2
import easyocr
import re
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime
import os

# Charger YOLOv8 pour détecter les voitures
car_model = YOLO(r"C:\path\to\yolov8n.pt")

# Initialiser EasyOCR
reader = easyocr.Reader(['en'])

# Définir le regex pour les plaques espagnoles
spain_plate_regex = r"^[0-9]{4}\s?[A-Z]{3}$"

# Charger la vidéo
video_path = r"C:\Users\USER\Desktop\prjt computer vision\tc.mp4"
cap = cv2.VideoCapture(video_path)

# Initialiser les listes et compteurs
confidences = []  # Liste pour stocker les confiances
all_plate_texts = []  # Liste pour stocker les plaques détectées
correct_plates = 0  # Compteur des bonnes détections
total_plates = 0  # Compteur total des détections

# Créer un dossier pour enregistrer la vidéo
output_video_dir = r"C:\Users\USER\Desktop\prjt computer vision"
os.makedirs(output_video_dir, exist_ok=True)

# Préparer l'enregistrement de la vidéo
output_video_path = os.path.join(output_video_dir, f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec pour MP4
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

# DataFrame pour stocker les plaques détectées
plates_df = pd.DataFrame(columns=["Plaque", "Confiance", "Timestamp"])

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Étape 1 : Détection des voitures avec YOLOv8
    results = car_model.predict(frame, conf=0.5)  # Ajuster conf pour réduire les faux positifs
    car_boxes = []
    for box in results[0].boxes:
        class_id = int(box.cls[0])
        if class_id == 2:  # Classe voiture
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            car_boxes.append((x1, y1, x2, y2))

    # Étape 2 : OCR pour chaque voiture
    for (x1, y1, x2, y2) in car_boxes:
        car_roi = frame[y1:y2, x1:x2]
        gray = cv2.cvtColor(car_roi, cv2.COLOR_BGR2GRAY)

        # Appliquer EasyOCR sur la voiture entière
        result = reader.readtext(gray)
        for (bbox, text, confidence) in result:
            if confidence > 0.5:
                plate_text = text.strip()
                plate_confidence = confidence
                confidences.append(plate_confidence)
                all_plate_texts.append(plate_text)

                if re.match(spain_plate_regex, plate_text):  # Vérifier si la plaque correspond
                    correct_plates += 1
                total_plates += 1

                # Ajouter la plaque détectée au DataFrame
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                plates_df = pd.concat([plates_df, pd.DataFrame([{
                    "Plaque": plate_text,
                    "Confiance": plate_confidence,
                    "Timestamp": timestamp
                }])], ignore_index=True)

                print(f"Plaque détectée : {plate_text}, Confiance : {plate_confidence}")

                # Annoter l'image
                cv2.putText(
                    frame, plate_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
                )
                break

    # Afficher les voitures détectées
    for (x1, y1, x2, y2) in car_boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Enregistrer la frame dans la vidéo
    out.write(frame)

    # Afficher la vidéo (facultatif)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()  # Fermer l'enregistrement vidéo
cv2.destroyAllWindows()

# Afficher la répartition des confiances
print(f"Nombre total de plaques détectées : {total_plates}")
print(f"Nombre de plaques correctes : {correct_plates}")
print(f"Confiances des détections : {confidences}")

# Tracer un histogramme des confiances
plt.hist(confidences, bins=20, color='blue', edgecolor='black')
plt.title('Répartition des confiances des détections de plaques')
plt.xlabel('Confiance')
plt.ylabel('Nombre de détections')
plt.show()

# Enregistrer les plaques détectées dans un fichier Excel
output_excel_path = os.path.join(output_video_dir, f"plates_detected_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
plates_df.to_excel(output_excel_path, index=False)
print(f"Les plaques détectées ont été enregistrées dans : {output_excel_path}")
