from flask import Flask, Response, jsonify, request, render_template
import face_recognition
import cv2
import numpy as np
from datetime import datetime, timedelta
import os
import pickle
import sqlite3
import json
from flask_cors import CORS
import atexit
import signal
import multiprocessing

app = Flask(__name__)
CORS(app)

class FaceTracker:
    def __init__(self, base_dir, min_delay_seconds=5, tolerance=0.6):
        self.base_dir = base_dir
        self.encodings_file = os.path.join(base_dir, "known_faces.pkl")
        self.saved_faces = {}
        self.face_encodings = []
        self.min_delay = timedelta(seconds=min_delay_seconds)
        self.tolerance = tolerance
        self.db_conn = self._initialize_database()
        self.save_photos = False
        self.person_count = 0  # Initialiser à 0 avant de charger
        self.detection_model = "hog"  # Mode par défaut
        self._load_known_faces()
        self.video_capture = cv2.VideoCapture(0)
        self.person_count = self._get_last_person_id()  # Mettre à jour après le chargement
        self._cnn_pool = None

    def _initialize_cnn_pool(self):
        if self._cnn_pool is None and self.detection_model == "cnn":
            self._cnn_pool = multiprocessing.Pool(processes=1)
        elif self.detection_model == "hog" and self._cnn_pool is not None:
            self._cnn_pool.close()
            self._cnn_pool.join()
            self._cnn_pool = None

    def _initialize_database(self):
        conn = sqlite3.connect('faces.db', check_same_thread=False)
        c = conn.cursor()
        c.execute('''
            CREATE TABLE IF NOT EXISTS personnes
            (id INTEGER PRIMARY KEY,
             prenom TEXT,
             nom TEXT,
             date_ajout TIMESTAMP DEFAULT CURRENT_TIMESTAMP)
        ''')
        conn.commit()
        return conn

    def _get_last_person_id(self):
        # Vérifier dans la base de données
        c = self.db_conn.cursor()
        c.execute('SELECT MAX(id) FROM personnes')
        db_max = c.fetchone()[0]
        
        # Vérifier dans les encodages sauvegardés
        saved_max = max([pid for _, pid in self.saved_faces.values()]) if self.saved_faces else 0
        
        # Vérifier dans les dossiers
        person_dirs = [d for d in os.listdir(self.base_dir) 
                      if os.path.isdir(os.path.join(self.base_dir, d)) 
                      and d.startswith("personne_")]
        dir_max = max([int(d.split("_")[1]) for d in person_dirs]) if person_dirs else 0
        
        # Retourner le maximum des trois sources
        return max(filter(None, [db_max, saved_max, dir_max, 0]))

    def __del__(self):
        self.cleanup()

    def cleanup(self):
        if hasattr(self, 'db_conn'):
            self.db_conn.close()
        if hasattr(self, 'video_capture'):
            self.video_capture.release()
        if hasattr(self, '_cnn_pool') and self._cnn_pool is not None:
            self._cnn_pool.close()
            self._cnn_pool.join()

    def _load_known_faces(self):
        if os.path.exists(self.encodings_file):
            try:
                with open(self.encodings_file, 'rb') as f:
                    data = pickle.load(f)
                    self.face_encodings = data.get('encodings', [])
                    self.saved_faces = data.get('saved_faces', {})
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

    def add_person_info(self, person_id, prenom, nom):
        c = self.db_conn.cursor()
        c.execute('''
            INSERT OR REPLACE INTO personnes (id, prenom, nom)
            VALUES (?, ?, ?)
        ''', (person_id, prenom, nom))
        self.db_conn.commit()

    def get_person_info(self, person_id):
        c = self.db_conn.cursor()
        c.execute('SELECT prenom, nom FROM personnes WHERE id = ?', (person_id,))
        result = c.fetchone()
        if result:
            return f"{result[0]} {result[1]}"
        return f"ID #{person_id}"

    def identify_face(self, face_encoding):
        matching_face = self.find_matching_face(face_encoding)
        if matching_face is None:
            return None, "Inconnu"
        _, person_id = self.saved_faces[tuple(matching_face)]
        person_info = self.get_person_info(person_id)
        return person_id, person_info

    def find_matching_face(self, face_encoding):
        if not self.face_encodings:
            return None
        
        # Calculer toutes les distances
        face_distances = face_recognition.face_distance(self.face_encodings, face_encoding)
        
        # Trouver les meilleurs matches potentiels
        potential_matches = []
        for idx, distance in enumerate(face_distances):
            if distance <= self.tolerance:
                potential_matches.append((distance, idx))
        
        if not potential_matches:
            return None
        
        # Trier par distance et prendre le meilleur match
        best_match = min(potential_matches, key=lambda x: x[0])
        return self.face_encodings[best_match[1]]

    def add_new_face(self, face_encoding):
        self.person_count += 1
        self.face_encodings.append(face_encoding)
        self.saved_faces[tuple(face_encoding)] = (datetime.now(), self.person_count)
        self._save_known_faces()
        return self.person_count

    def get_frame(self):
        success, frame = self.video_capture.read()
        if not success:
            return None, []

        face_locations, face_encodings = self._process_frame(frame)
        detections = []

        for face_location, face_encoding in zip(face_locations, face_encodings):
            person_id, person_info, status = self._process_face(frame, face_location, face_encoding)
            detections.append({
                'person_id': person_id,
                'name': person_info,
                'location': face_location,
                'status': status
            })

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes(), detections

    def _process_frame(self, frame, scale=0.25):
        # Convertir en RGB pour face_recognition
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Créer plusieurs versions de l'image avec différentes rotations légères
        frames_to_process = [
            rgb_frame,  # Image originale
        ]
        
        # Ajouter les rotations uniquement en mode CNN pour plus de précision
        if self.detection_model == "cnn":
            frames_to_process.extend([
                cv2.rotate(rgb_frame, cv2.ROTATE_90_CLOCKWISE),  # Rotation 90°
                cv2.rotate(rgb_frame, cv2.ROTATE_90_COUNTERCLOCKWISE),  # Rotation -90°
            ])
        
        all_face_locations = []
        all_face_encodings = []

        try:
            self._initialize_cnn_pool()
            
            for idx, process_frame in enumerate(frames_to_process):
                # Redimensionner pour la performance
                small_frame = cv2.resize(process_frame, (0, 0), fx=scale, fy=scale)
                
                # Utiliser le modèle sélectionné avec gestion du pool pour CNN
                if self.detection_model == "cnn" and self._cnn_pool is not None:
                    face_locations = self._cnn_pool.apply(
                        face_recognition.face_locations,
                        args=(small_frame,),
                        kwds={'model': self.detection_model, 'number_of_times_to_upsample': 1}
                    )
                else:
                    face_locations = face_recognition.face_locations(
                        small_frame,
                        model=self.detection_model,
                        number_of_times_to_upsample=1
                    )
                
                # Ajuster les paramètres selon le mode
                num_jitters = 2 if self.detection_model == "cnn" else 1
                face_encodings = face_recognition.face_encodings(
                    small_frame, 
                    face_locations,
                    num_jitters=num_jitters
                )
                
                # Reconvertir les coordonnées à l'échelle originale
                face_locations_original = []
                for top, right, bottom, left in face_locations:
                    # Ajuster les coordonnées selon la rotation
                    if self.detection_model == "cnn" and idx == 1:  # Rotation 90°
                        new_top = left
                        new_right = top
                        new_bottom = right
                        new_left = bottom
                    elif self.detection_model == "cnn" and idx == 2:  # Rotation -90°
                        new_top = right
                        new_right = bottom
                        new_bottom = left
                        new_left = top
                    else:  # Pas de rotation
                        new_top, new_right, new_bottom, new_left = top, right, bottom, left
                    
                    # Mettre à l'échelle originale
                    face_locations_original.append((
                        int(new_top/scale),
                        int(new_right/scale),
                        int(new_bottom/scale),
                        int(new_left/scale)
                    ))
                
                all_face_locations.extend(face_locations_original)
                all_face_encodings.extend(face_encodings)

        except Exception as e:
            print(f"Erreur lors du traitement de l'image: {e}")
            return [], []
        
        # Supprimer les doublons potentiels
        unique_faces = {}
        for loc, enc in zip(all_face_locations, all_face_encodings):
            # Utiliser le centre du visage comme clé pour éviter les doublons
            center = ((loc[0] + loc[2])//2, (loc[1] + loc[3])//2)
            if center not in unique_faces:
                unique_faces[center] = (loc, enc)
        
        return list(face[0] for face in unique_faces.values()), list(face[1] for face in unique_faces.values())

    def _process_face(self, frame, face_location, face_encoding):
        top, right, bottom, left = face_location
        person_id, person_info = self.identify_face(face_encoding)
        status = None

        if person_id is None:
            person_id = self.add_new_face(face_encoding)
            status = "Nouvelle personne détectée"
        elif self.save_photos:
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

    def get_saved_count(self):
        return len(set(person_id for _, person_id in self.saved_faces.values()))

def gen_frames(face_tracker):
    while True:
        frame_bytes, detections = face_tracker.get_frame()
        if frame_bytes is None:
            break
            
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

# Créer le dossier de stockage et initialiser le tracker
storage_dir = "faces"
if not os.path.exists(storage_dir):
    os.makedirs(storage_dir)

face_tracker = None

def create_face_tracker():
    global face_tracker
    face_tracker = FaceTracker(storage_dir)

def cleanup_resources(signum=None, frame=None):
    global face_tracker
    if face_tracker is not None:
        face_tracker.cleanup()

# Enregistrer les gestionnaires de nettoyage
atexit.register(cleanup_resources)
signal.signal(signal.SIGINT, cleanup_resources)
signal.signal(signal.SIGTERM, cleanup_resources)

# Initialiser le FaceTracker
create_face_tracker()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(face_tracker),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detections')
def get_detections():
    _, detections = face_tracker.get_frame()
    return jsonify(detections)

@app.route('/toggle_save_photos', methods=['POST'])
def toggle_save_photos():
    face_tracker.save_photos = not face_tracker.save_photos
    return jsonify({'save_photos': face_tracker.save_photos})

@app.route('/toggle_detection_model', methods=['POST'])
def toggle_detection_model():
    face_tracker.detection_model = "cnn" if face_tracker.detection_model == "hog" else "hog"
    return jsonify({
        'detection_model': face_tracker.detection_model
    })

@app.route('/add_person', methods=['POST'])
def add_person():
    data = request.json
    try:
        face_tracker.add_person_info(
            int(data['person_id']),
            data['prenom'],
            data['nom']
        )
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/get_stats')
def get_stats():
    return jsonify({
        'total_persons': face_tracker.get_saved_count(),
        'save_photos': face_tracker.save_photos,
        'detection_model': face_tracker.detection_model
    })

@app.route('/shutdown', methods=['POST'])
def shutdown():
    cleanup_resources()
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Pas dans un environnement Werkzeug')
    func()
    return jsonify({'success': True, 'message': 'Serveur arrêté'})

if __name__ == '__main__':
    app.run(debug=True) 