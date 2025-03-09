import face_recognition
import cv2
import numpy as np
from datetime import datetime, timedelta
import os
import pickle
import sqlite3

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

def initialize_database():
    """Initialise la base de données SQLite"""
    conn = sqlite3.connect('faces.db')
    c = conn.cursor()
    
    # Créer la table des personnes si elle n'existe pas
    c.execute('''
        CREATE TABLE IF NOT EXISTS personnes
        (id INTEGER PRIMARY KEY,
         prenom TEXT,
         nom TEXT,
         date_ajout TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
    ''')
    
    conn.commit()
    return conn

class FaceTracker:
    def __init__(self, base_dir, min_delay_seconds=5, tolerance=0.6):
        self.base_dir = base_dir
        self.encodings_file = os.path.join(base_dir, "known_faces.pkl")
        self.saved_faces = {}  # encoding -> (dernière_capture, person_id)
        self.face_encodings = []
        self.person_count = self._get_last_person_id()
        self.min_delay = timedelta(seconds=min_delay_seconds)
        self.tolerance = tolerance
        self.db_conn = initialize_database()
        self.save_photos = False  # Mode de sauvegarde des photos désactivé par défaut
        self._load_known_faces()

    def toggle_photo_saving(self):
        """Active/désactive la sauvegarde des photos"""
        self.save_photos = not self.save_photos
        return self.save_photos

    def __del__(self):
        """Ferme la connexion à la base de données"""
        if hasattr(self, 'db_conn'):
            self.db_conn.close()

    def add_person_info(self, person_id, prenom, nom):
        """Ajoute ou met à jour les informations d'une personne"""
        c = self.db_conn.cursor()
        c.execute('''
            INSERT OR REPLACE INTO personnes (id, prenom, nom)
            VALUES (?, ?, ?)
        ''', (person_id, prenom, nom))
        self.db_conn.commit()

    def add_new_face(self, face_encoding):
        """Ajoute un nouveau visage sans photo"""
        self.person_count += 1
        self.face_encodings.append(face_encoding)
        self.saved_faces[tuple(face_encoding)] = (datetime.now(), self.person_count)
        self._save_known_faces()
        return self.person_count

    def get_person_info(self, person_id):
        """Récupère les informations d'une personne"""
        c = self.db_conn.cursor()
        c.execute('SELECT prenom, nom FROM personnes WHERE id = ?', (person_id,))
        result = c.fetchone()
        if result:
            return f"{result[0]} {result[1]}"
        return f"ID #{person_id}"

    def _get_last_person_id(self):
        # Vérifier dans la base de données
        if hasattr(self, 'db_conn'):
            c = self.db_conn.cursor()
            c.execute('SELECT MAX(id) FROM personnes')
            result = c.fetchone()[0]
            if result is not None:
                return result

        # Vérifier dans les dossiers si pas de résultat dans la BD
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

    def process_face(self, frame, face_location, face_encoding):
        """Traite un visage détecté (identification et sauvegarde si activée)"""
        top, right, bottom, left = face_location
        person_id, person_info = self.identify_face(face_encoding)
        status = None

        if person_id is None:
            # Nouveau visage détecté
            person_id = self.add_new_face(face_encoding)
            status = "Nouvelle personne détectée"
        elif self.save_photos:
            # Visage connu, sauvegarder la photo si le mode est activé
            current_time = datetime.now()
            last_capture, _ = self.saved_faces[tuple(self.find_matching_face(face_encoding))]
            if (current_time - last_capture) >= self.min_delay:
                if self.save_photos:
                    person_dir = self.get_person_directory(person_id)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = f"{timestamp}.jpg"
                    filepath = os.path.join(person_dir, filename)
                    face_img = frame[top:bottom, left:right]
                    cv2.imwrite(filepath, face_img)
                    status = "Photo sauvegardée"
                self.saved_faces[tuple(self.find_matching_face(face_encoding))] = (current_time, person_id)

        return person_id, person_info, status

    def identify_face(self, face_encoding):
        """Identifie une personne et retourne son ID et son statut"""
        matching_face = self.find_matching_face(face_encoding)
        if matching_face is None:
            return None, "Inconnu"
        _, person_id = self.saved_faces[tuple(matching_face)]
        person_info = self.get_person_info(person_id)
        return person_id, person_info

    def find_matching_face(self, face_encoding):
        if not self.face_encodings:
            return None
        
        face_distances = face_recognition.face_distance(self.face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        
        if face_distances[best_match_index] <= self.tolerance:
            return self.face_encodings[best_match_index]
        return None

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

def draw_face_info(frame, face_location, person_id, person_info, status=None):
    """Dessine les informations du visage sur l'image"""
    top, right, bottom, left = face_location
    
    # Couleur du rectangle selon le statut
    if person_id is not None:
        color = (0, 255, 0)  # Vert pour les visages connus
    else:
        color = (0, 165, 255)  # Orange pour les inconnus
    
    # Dessiner le rectangle et l'info
    cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
    
    # Afficher le nom/ID au-dessus du rectangle
    font = cv2.FONT_HERSHEY_DUPLEX
    cv2.putText(frame, person_info, (left + 6, top - 6),
                font, 0.6, color, 1)
    
    # Afficher le status si présent
    if status:
        cv2.putText(frame, status, (left, bottom + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
    
    return frame

def handle_keyboard_input(face_tracker):
    """Gère les entrées clavier pour l'ajout/modification des noms"""
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q'):
        return False
    elif key == ord('a'):  # 'a' pour ajouter/modifier un nom
        try:
            person_id = int(input("Entrez l'ID de la personne: "))
            prenom = input("Entrez le prénom: ")
            nom = input("Entrez le nom: ")
            face_tracker.add_person_info(person_id, prenom, nom)
            print(f"Informations mises à jour pour la personne {person_id}")
        except ValueError:
            print("ID invalide")
        except Exception as e:
            print(f"Erreur: {e}")
    elif key == ord('s'):  # 's' pour activer/désactiver la sauvegarde des photos
        is_saving = face_tracker.toggle_photo_saving()
        print(f"Sauvegarde des photos {'activée' if is_saving else 'désactivée'}")
    
    return True

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
        print("Commandes:")
        print("  'q' pour quitter")
        print("  'a' pour ajouter/modifier le nom d'une personne")
        print("  's' pour activer/désactiver la sauvegarde des photos")
        print(f"Mode actuel: {'Sauvegarde photos activée' if face_tracker.save_photos else 'Identification seule'}")
        
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
                person_id, person_info, status = face_tracker.process_face(frame, face_location, face_encoding)
                frame = draw_face_info(frame, face_location, person_id, person_info, status)
            
            mode_text = "Mode: Sauvegarde photos" if face_tracker.save_photos else "Mode: Identification seule"
            cv2.putText(frame, mode_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f'Personnes connues: {face_tracker.get_saved_count()}', 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow("Detection Faciale", frame)
            
            if not handle_keyboard_input(face_tracker):
                break
                
    except Exception as e:
        print(f"Une erreur est survenue: {str(e)}")
    finally:
        if 'video_capture' in locals():
            video_capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
