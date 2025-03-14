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
import multiprocessing.util
from multiprocessing import freeze_support
import time

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
        self.person_count = 0
        self.detection_model = "hog"
        self._last_cnn_time = 0
        self._cnn_interval = 1.0  # Augmenter l'intervalle à 1 seconde
        self._cnn_enabled = False  # Flag pour désactiver complètement le CNN si nécessaire
        self._load_known_faces()
        self.video_capture = cv2.VideoCapture(0)
        self.person_count = self._get_last_person_id()

    def _initialize_cnn_pool(self):
        """Initialise le contexte et le pool CNN de manière sécurisée"""
        try:
            # Nettoyer les anciennes ressources si elles existent
            self._cleanup_cnn_resources()
            
            # Créer un nouveau pool uniquement si en mode CNN
            if self.detection_model == "cnn":
                try:
                    self._cnn_context = multiprocessing.get_context('spawn')
                    self._cnn_pool = self._cnn_context.Pool(processes=1)
                    print("Mode CNN initialisé avec succès")
                except Exception as e:
                    print(f"Erreur lors de l'initialisation du pool CNN: {e}")
                    self._cleanup_cnn_resources()
                    self.detection_model = "hog"
                    print("Retour au mode HOG suite à une erreur")
        except Exception as e:
            print(f"Erreur lors de l'initialisation du mode CNN: {e}")
            self._cleanup_cnn_resources()
            self.detection_model = "hog"
            print("Retour au mode HOG suite à une erreur")

    def _cleanup_cnn_resources(self):
        """Nettoie proprement les ressources CNN"""
        if self._cnn_pool is not None:
            try:
                self._cnn_pool.terminate()
                self._cnn_pool.join(timeout=0.5)
            except:
                pass
            finally:
                self._cnn_pool = None
                
        self._cnn_context = None

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
        try:
            # Vérifier dans la base de données
            c = self.db_conn.cursor()
            c.execute('SELECT MAX(id) FROM personnes')
            db_max = c.fetchone()[0]
            if db_max is None:
                db_max = 0
            
            # Vérifier dans les encodages sauvegardés
            saved_max = 0
            if self.saved_faces:
                saved_ids = [pid for _, pid in self.saved_faces.values()]
                if saved_ids:
                    saved_max = max(saved_ids)
            
            # Vérifier dans les dossiers
            dir_max = 0
            try:
                person_dirs = [d for d in os.listdir(self.base_dir) 
                             if os.path.isdir(os.path.join(self.base_dir, d)) 
                             and d.startswith("personne_")]
                if person_dirs:
                    dir_ids = [int(d.split("_")[1]) for d in person_dirs]
                    if dir_ids:
                        dir_max = max(dir_ids)
            except (ValueError, IndexError):
                pass
            
            # Retourner le maximum des trois sources
            return max(db_max, saved_max, dir_max, 0)
        except Exception as e:
            print(f"Erreur lors de la récupération du dernier ID: {e}")
            return 0

    def __del__(self):
        self.cleanup()

    def cleanup(self):
        """Nettoyage propre des ressources"""
        # Nettoyer les ressources CNN
        self._cleanup_cnn_resources()

        # Fermer la base de données
        if hasattr(self, 'db_conn'):
            try:
                self.db_conn.close()
            except:
                pass

        # Libérer la caméra
        if hasattr(self, 'video_capture'):
            try:
                self.video_capture.release()
            except:
                pass

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
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Ajuster l'échelle selon le mode
            if self.detection_model == "cnn":
                scale = 0.1  # Réduire encore plus pour CNN
            
            # Redimensionner l'image
            small_frame = cv2.resize(rgb_frame, (0, 0), fx=scale, fy=scale)
            
            current_time = time.time()
            
            try:
                if self.detection_model == "cnn" and not self._cnn_enabled:
                    # Premier essai avec CNN
                    try:
                        face_recognition.face_locations(
                            small_frame,
                            model="cnn",
                            number_of_times_to_upsample=1
                        )
                        self._cnn_enabled = True
                        print("Mode CNN activé avec succès")
                    except Exception as e:
                        print(f"CNN non supporté sur cette machine: {e}")
                        self.detection_model = "hog"
                        self._cnn_enabled = False
                
                if self.detection_model == "cnn" and self._cnn_enabled:
                    if current_time - self._last_cnn_time >= self._cnn_interval:
                        try:
                            face_locations = face_recognition.face_locations(
                                small_frame,
                                model="cnn",
                                number_of_times_to_upsample=1
                            )
                            self._last_cnn_time = current_time
                        except Exception as e:
                            print(f"Erreur CNN, retour à HOG: {e}")
                            face_locations = face_recognition.face_locations(
                                small_frame,
                                model="hog",
                                number_of_times_to_upsample=1
                            )
                    else:
                        face_locations = face_recognition.face_locations(
                            small_frame,
                            model="hog",
                            number_of_times_to_upsample=1
                        )
                else:
                    face_locations = face_recognition.face_locations(
                        small_frame,
                        model="hog",
                        number_of_times_to_upsample=1
                    )
                
            except Exception as e:
                print(f"Erreur de détection: {str(e)}")
                self.detection_model = "hog"
                self._cnn_enabled = False
                face_locations = face_recognition.face_locations(
                    small_frame,
                    model="hog",
                    number_of_times_to_upsample=1
                )
            
            # Encodage des visages avec paramètres réduits
            face_encodings = face_recognition.face_encodings(
                small_frame, 
                face_locations,
                num_jitters=1
            )
            
            # Reconvertir les coordonnées
            face_locations_original = [
                (int(top/scale), int(right/scale), 
                 int(bottom/scale), int(left/scale))
                for top, right, bottom, left in face_locations
            ]
            
            return face_locations_original, face_encodings
            
        except Exception as e:
            print(f"Erreur générale lors du traitement de l'image: {str(e)}")
            if self.detection_model == "cnn":
                print("Désactivation du mode CNN suite à une erreur")
                self.detection_model = "hog"
                self._cnn_enabled = False
            return [], []

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

    def toggle_detection_model(self):
        """Change le mode de détection de manière sécurisée"""
        try:
            new_model = "cnn" if self.detection_model == "hog" else "hog"
            print(f"Changement de mode: {self.detection_model} -> {new_model}")
            
            if new_model == "cnn" and not self._cnn_enabled:
                print("Test de compatibilité CNN...")
                try:
                    # Test avec une petite image
                    test_frame = np.zeros((100, 100, 3), dtype=np.uint8)
                    face_recognition.face_locations(test_frame, model="cnn")
                    self._cnn_enabled = True
                    print("Mode CNN compatible")
                except Exception as e:
                    print(f"Mode CNN non supporté: {e}")
                    return False
            
            self.detection_model = new_model
            self._last_cnn_time = 0
            
            print(f"Mode changé avec succès: {self.detection_model}")
            return True
            
        except Exception as e:
            print(f"Erreur lors du changement de mode: {str(e)}")
            self.detection_model = "hog"
            self._cnn_enabled = False
            print("Retour forcé au mode HOG")
            return False

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
    """Nettoyage propre des ressources au shutdown"""
    global face_tracker
    if face_tracker is not None:
        try:
            face_tracker.cleanup()
        except:
            pass
        finally:
            face_tracker = None

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
    success = face_tracker.toggle_detection_model()
    return jsonify({
        'success': success,
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
    try:
        # Configuration du multiprocessing
        multiprocessing.freeze_support()
        multiprocessing.set_start_method('spawn', force=True)
        
        # Créer le dossier de stockage
        storage_dir = "faces"
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)
        
        # Initialiser le FaceTracker
        create_face_tracker()
        
        # Enregistrer les gestionnaires de nettoyage
        atexit.register(cleanup_resources)
        signal.signal(signal.SIGINT, cleanup_resources)
        signal.signal(signal.SIGTERM, cleanup_resources)
        
        # Démarrer l'application Flask
        app.run(debug=False, use_reloader=False, threaded=True)
    except Exception as e:
        print(f"Erreur lors du démarrage: {e}")
        cleanup_resources()
    finally:
        cleanup_resources() 