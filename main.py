import face_recognition
import cv2
import numpy as np
from datetime import datetime

def initialize_camera():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Impossible d'accéder à la caméra")
    return camera

def process_frame(frame, scale=0.25):
    # Redimensionner le frame pour de meilleures performances
    small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    rgb_small_frame = small_frame[:, :, ::-1]
    
    # Détecter les visages
    face_locations = face_recognition.face_locations(rgb_small_frame)
    
    # Redimensionner les coordonnées au format original
    face_locations_original = [(int(top/scale), int(right/scale), 
                              int(bottom/scale), int(left/scale)) 
                             for top, right, bottom, left in face_locations]
    
    return face_locations_original

def main():
    try:
        video_capture = initialize_camera()
        frame_count = 0
        fps_start_time = datetime.now()
        
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Erreur lors de la capture de l'image")
                break
                
            frame_count += 1
            # Calculer et afficher FPS toutes les 30 frames
            if frame_count % 30 == 0:
                fps = frame_count / (datetime.now() - fps_start_time).total_seconds()
                print(f"FPS: {fps:.2f}")
                
            face_locations = process_frame(frame)
            
            # Dessiner les rectangles et ajouter du texte
            for top, right, bottom, left in face_locations:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, 'Visage detecte', (left, top - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Afficher le nombre de visages détectés
            cv2.putText(frame, f'Visages: {len(face_locations)}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("Detection Faciale", frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"Une erreur est survenue: {str(e)}")
    finally:
        if 'video_capture' in locals():
            video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
