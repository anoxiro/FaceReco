import face_recognition
import cv2
import numpy as np
from datetime import datetime, timedelta
import os
import pickle

def initialize_camera():
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Impossible d'accéder à la caméra")
    return camera

def create_storage_directory():
    base_dir = "faces"
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    return base_dir

class FaceTracker:
    def __init__(self, base_dir, min_delay_seconds=5, tolerance=0.6):
        self.base_dir = base_dir
        self.encodings_file = os.path.join(base_dir, "known_faces.pkl")
        self.saved_faces = {}  # encoding -> (dernière_capture, person_id)
        self.face_encodings = []
        self.person_count = self._get_last_person_id()
        self.min_delay = timedelta(seconds=min_delay_seconds)
        self.tolerance = tolerance
        self._load_known_faces()

    def _get_last_person_id(self):
        person_dirs = [d for d in os.listdir(self.base_dir) 
                      if os.path.isdir(os.path.join(self.base_dir, d)) 
                      and d.startswith("personne_")]
        if not person_dirs:
            return 0
        person_ids = [int(d.split("_")[1]) for d in person_dirs]
        return max(person_ids)

    def _load_known_faces(self):
        if os.path.exists(self.encodings_file):
            try:
                with open(self.encodings_file, 'rb') as f:
                    data = pickle.load(f)
                    self.face_encodings = data['encodings']
                    self.saved_faces = data['saved_faces']
                    print(f"Chargement de {len(self.face_encodings)} visages connus")
            except Exception as e:
                print(f"Erreur lors du chargement des visages connus: {e}")
                self.face_encodings = []
                self.saved_faces = {}

    def _save_known_faces(self):
        try:
            with open(self.encodings_file, 'wb') as f:
                data = {
                    'encodings': self.face_encodings,
                    'saved_faces': self.saved_faces
                }
                pickle.dump(data, f)
        except Exception as e:
            print(f"Erreur lors de la sauvegarde des visages connus: {e}")

    def get_person_directory(self, person_id):
        person_dir = os.path.join(self.base_dir, f"personne_{person_id}")
        if not os.path.exists(person_dir):
            os.makedirs(person_dir)
        return person_dir

    def find_matching_face(self, face_encoding):
        if not self.face_encodings:
            return None
        
        face_distances = face_recognition.face_distance(self.face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        
        if face_distances[best_match_index] <= self.tolerance:
            return self.face_encodings[best_match_index]
        return None

    def can_save_face(self, face_encoding):
        matching_face = self.find_matching_face(face_encoding)
        if matching_face is None:
            self.person_count += 1
            return True, None, self.person_count
        
        current_time = datetime.now()
        last_capture, person_id = self.saved_faces[tuple(matching_face)]
        can_save = (current_time - last_capture) >= self.min_delay
        return can_save, matching_face, person_id

    def update_face(self, face_encoding, person_id):
        self.face_encodings.append(face_encoding)
        self.saved_faces[tuple(face_encoding)] = (datetime.now(), person_id)
        self._save_known_faces()  # Sauvegarder après chaque nouveau visage

    def get_saved_count(self):
        return len(set(person_id for _, person_id in self.saved_faces.values()))

    def get_time_until_next_capture(self, face_encoding):
        matching_face = self.find_matching_face(face_encoding)
        if matching_face is None:
            return 0
        
        last_capture, _ = self.saved_faces[tuple(matching_face)]
        remaining = (last_capture + self.min_delay - datetime.now()).seconds
        return max(0, remaining)

def process_frame(frame, scale=0.25):
    small_frame = cv2.resize(frame, (0, 0), fx=scale, fy=scale)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
    
    face_locations = face_recognition.face_locations(rgb_small_frame, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
    
    face_locations_original = [(int(top/scale), int(right/scale), 
                              int(bottom/scale), int(left/scale)) 
                             for top, right, bottom, left in face_locations]
    
    return face_locations_original, face_encodings

def save_detected_face(frame, face_location, face_encoding, face_tracker):
    top, right, bottom, left = face_location
    face_img = frame[top:bottom, left:right]
    
    can_save, matching_face, person_id = face_tracker.can_save_face(face_encoding)
    
    if can_save:
        person_dir = face_tracker.get_person_directory(person_id)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{timestamp}.jpg"
        
        if matching_face is None:
            status = f"Nouvelle personne #{person_id}"
            face_tracker.update_face(face_encoding, person_id)
        else:
            status = f"Personne #{person_id} reconnue"
            face_tracker.saved_faces[tuple(matching_face)] = (datetime.now(), person_id)
            face_tracker._save_known_faces()
        
        filepath = os.path.join(person_dir, filename)
        cv2.imwrite(filepath, face_img)
        return True, status
    else:
        seconds = face_tracker.get_time_until_next_capture(face_encoding)
        return False, f"Attente: {seconds}s"

def main():
    try:
        video_capture = initialize_camera()
        frame_count = 0
        fps_start_time = datetime.now()
        base_dir = create_storage_directory()
        face_tracker = FaceTracker(base_dir, min_delay_seconds=5, tolerance=0.6)
        
        print(f"Dossier de sauvegarde: {base_dir}")
        print(f"Organisation: Un dossier par personne (personne_X)")
        print(f"Format des fichiers: YYYYMMDD_HHMMSS_microseconds.jpg")
        print("Délai entre captures: 5 secondes")
        print("Appuyez sur 'q' pour quitter")
        
        while True:
            ret, frame = video_capture.read()
            if not ret:
                print("Erreur lors de la capture de l'image")
                break
                
            frame_count += 1
            if frame_count % 30 == 0:
                fps = frame_count / (datetime.now() - fps_start_time).total_seconds()
                print(f"FPS: {fps:.2f}")
                
            face_locations, face_encodings = process_frame(frame)
            
            for face_location, face_encoding in zip(face_locations, face_encodings):
                top, right, bottom, left = face_location
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                
                saved, status = save_detected_face(frame, face_location, face_encoding, face_tracker)
                status_color = (0, 255, 0) if saved else (0, 165, 255)
                
                cv2.putText(frame, status, (left, top - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
            
            cv2.putText(frame, f'Personnes uniques: {face_tracker.get_saved_count()}', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
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
