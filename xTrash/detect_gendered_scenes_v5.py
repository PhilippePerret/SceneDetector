#!/usr/bin/env python3

import cv2
import torch
from PIL import Image
import numpy as np
import sys
import os
from datetime import timedelta
from scenedetect import open_video, SceneManager, ContentDetector, AdaptiveDetector
# BLIP supprimé - inutile pour analyse de films
from deep_translator import GoogleTranslator
import json
import re
from collections import defaultdict
from faster_whisper import WhisperModel
import argparse
import csv
from datetime import datetime
import ffmpeg
from ultralytics import YOLO

# Utilitaires v2 pour chronologie fiable (plans + INT/EXT + Jour/Nuit)
import math

def _hms(seconds: int) -> str:
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h}:{m:02d}:{s:02d}"

def _ms_to_hms(ms: int) -> str:
    return _hms(int(round(ms / 1000)))

def _grab_frame_at_ms(video_path: str, time_ms: int):
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, time_ms)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return frame

def detect_scenes_pyscenedetect(video_path: str, start_time_s: float = 0, end_time_s: float = None):
    """Détection professionnelle de scènes avec PySceneDetect"""
    from scenedetect import detect, ContentDetector
    
    # Utiliser la fonction detect simplifiée
    scene_list = detect(video_path, ContentDetector(threshold=30.0, min_scene_len=15))  # 0.5 seconde minimum
    
    # Filtrer les scènes selon start/end
    scenes = []
    for scene in scene_list:
        start_s = scene[0].get_seconds()
        end_s = scene[1].get_seconds()
        
        # Vérifier si la scène est dans la plage demandée
        if end_time_s and start_s >= end_time_s:
            continue
        if start_time_s and end_s <= start_time_s:
            continue
            
        # Ajuster aux limites
        if start_time_s and start_s < start_time_s:
            start_s = start_time_s
        if end_time_s and end_s > end_time_s:
            end_s = end_time_s
            
        start_ms = int(start_s * 1000)
        end_ms = int(end_s * 1000)
        mid_ms = (start_ms + end_ms) // 2
        scenes.append((start_ms, end_ms, mid_ms))
    
    # Si aucune scène détectée, créer une scène unique
    if not scenes:
        start_ms = int(start_time_s * 1000) if start_time_s else 0
        end_ms = int(end_time_s * 1000) if end_time_s else 60000
        scenes = [(start_ms, end_ms, (start_ms + end_ms) // 2)]
    
    return scenes

def _detect_day_night(frame_bgr) -> str:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    avg = float(np.mean(gray))
    return "Jour" if avg > 90 else "Nuit"

def _detect_int_ext(frame_bgr) -> str:
    h, w, _ = frame_bgr.shape
    top = frame_bgr[: max(h // 3, 1), :, :]
    top_hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
    mask_blue = cv2.inRange(top_hsv, (90, 20, 120), (130, 80, 255))
    ratio_blue = float(np.count_nonzero(mask_blue)) / float(mask_blue.size)
    return "Ext" if ratio_blue > 0.03 else "Int"

def extract_ambient_light(frame_bgr, objects_boxes=None):
    """Extrait la luminosité et teinte ambiante (hors objets détectés)"""
    h, w, _ = frame_bgr.shape
    mask = np.ones((h, w), dtype=np.uint8) * 255
    
    # Masquer les zones des objets détectés
    if objects_boxes:
        for box in objects_boxes:
            x1, y1, x2, y2 = box
            mask[y1:y2, x1:x2] = 0
    
    # Analyser uniquement les zones hors objets
    frame_hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    ambient_pixels = frame_hsv[mask > 0]
    
    if len(ambient_pixels) == 0:
        # Si tout est masqué, analyser tout
        ambient_pixels = frame_hsv.reshape(-1, 3)
    
    # Calculer luminosité moyenne et teinte dominante
    avg_brightness = np.mean(ambient_pixels[:, 2])  # V channel
    avg_hue = np.mean(ambient_pixels[:, 0])  # H channel
    avg_saturation = np.mean(ambient_pixels[:, 1])  # S channel
    
    return {
        'brightness': float(avg_brightness),
        'hue': float(avg_hue),
        'saturation': float(avg_saturation)
    }

def compare_ambient_light(light1, light2, threshold=20):
    """Compare deux ambiances lumineuses"""
    if not light1 or not light2:
        return False
    
    brightness_diff = abs(light1['brightness'] - light2['brightness'])
    hue_diff = abs(light1['hue'] - light2['hue'])
    
    # Les hues bouclent à 180, gérer la circularité
    if hue_diff > 90:
        hue_diff = 180 - hue_diff
    
    return brightness_diff < threshold and hue_diff < 30

# Supprimer le warning tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class ScenePatternDetector:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Utilisation du device: {self.device}")
        
        # Charger YOLOv8x pour détection d'objets PRÉCISE
        print("Chargement du modèle YOLOv8x (le plus précis)...")
        self.yolo_model = YOLO('yolov8x.pt')  # Version X = maximum de précision
        print("YOLOv8x chargé!")
        
        # PAS DE BLIP - on le supprime complètement
        
        # Traducteur (on garde pour les dialogues)
        self.translator = GoogleTranslator(source='en', target='fr')
        
        # Charger Whisper pour transcription
        print("Chargement du modèle Whisper...")
        try:
            self.whisper_model = WhisperModel("small", device="cpu", compute_type="float32")
            print("Whisper chargé!")
        except Exception as e:
            print(f"Erreur Whisper: {e}")
            self.whisper_model = None
        
        # Définir les critères de détection pour chaque type de scène
        self.scene_detectors = self._initialize_scene_detectors()
        
        # Historique des ambiances lumineuses pour continuité
        self.light_history = []
    
    def _initialize_scene_detectors(self):
        """Initialise les critères de détection pour chaque type de scène sans catégorisation préjugée"""
        return {
            'shopping': {
                'visual_keywords': ['magasin', 'boutique', 'shop', 'store', 'vêtement', 'clothing', 'dress', 'robe', 'mirror', 'miroir'],
                'audio_keywords': ['essayer', 'try on', 'acheter', 'buy', 'prix', 'price', 'taille', 'size'],
                'min_duration': 30  # 30 secondes minimum
            },
            'long_dialogue': {
                'min_participants': 3,
                'min_duration': 120,  # 2 minutes minimum
                'dialogue_density_threshold': 0.7  # 70% de la scène doit contenir du dialogue
            },
            'soins_toilettage': {
                'visual_keywords': ['coiffure', 'hair', 'maquillage', 'makeup', 'bathroom', 'salle de bain', 'miroir', 'mirror'],
                'audio_keywords': ['coiffer', 'maquiller', 'brush', 'brosser'],
                'min_duration': 20
            },
            'cuisine': {
                'visual_keywords': ['cuisine', 'kitchen', 'four', 'oven', 'cuisiner', 'cooking', 'chef', 'recipe'],
                'audio_keywords': ['cuisiner', 'cook', 'préparer', 'prepare', 'recette', 'recipe'],
                'min_duration': 30
            },
            'decoration': {
                'visual_keywords': ['décorer', 'decorating', 'furniture', 'meuble', 'interior', 'intérieur'],
                'audio_keywords': ['décorer', 'arrange', 'ranger'],
                'min_duration': 20
            },
            'emotion_expression': {
                'visual_keywords': ['pleur', 'crying', 'tears', 'larmes', 'console', 'comfort'],
                'audio_keywords': ['pleure', 'cry', 'triste', 'sad', 'console', 'comfort'],
                'min_duration': 15
            },
            'voiture': {
                'visual_keywords': ['voiture', 'car', 'auto', 'vehicle', 'conduire', 'driving', 'road', 'route', 'volant', 'steering', 'truck', 'pickup'],
                'audio_keywords': ['rouler', 'drive', 'conduire', 'car', 'voiture', 'moteur', 'engine', 'démarrer'],
                'min_duration': 20
            },
            'sport_affrontement': {
                'visual_keywords': ['sport', 'fight', 'combat', 'battle', 'competition', 'match', 'game'],
                'audio_keywords': ['fight', 'combat', 'bataille', 'match', 'jouer', 'play'],
                'min_duration': 30
            },
            'bricolage': {
                'visual_keywords': ['tool', 'outil', 'hammer', 'marteau', 'repair', 'réparer', 'workshop', 'atelier'],
                'audio_keywords': ['réparer', 'repair', 'fix', 'bricoler'],
                'min_duration': 20
            },
            'bar_alcool': {
                'visual_keywords': ['bar', 'pub', 'alcool', 'alcohol', 'beer', 'bière', 'whisky', 'drink'],
                'audio_keywords': ['boire', 'drink', 'beer', 'bière', 'alcohol'],
                'min_duration': 30
            },
            'jeux': {
                'visual_keywords': ['cartes', 'cards', 'poker', 'billard', 'pool', 'game', 'jeu'],
                'audio_keywords': ['jouer', 'play', 'game', 'bet', 'pari'],
                'min_duration': 20
            },
            'hierarchie': {
                'visual_keywords': ['bureau', 'office', 'uniform', 'uniforme', 'military', 'militaire'],
                'audio_keywords': ['ordre', 'order', 'command', 'chef', 'boss', 'patron'],
                'min_duration': 15
            },
            'solitude': {
                'max_people_visible': 1,
                'min_duration': 30,
                'silence_ratio_threshold': 0.6  # 60% de silence
            },
            'commandement': {
                'visual_keywords': ['pointing', 'pointer', 'directive', 'commander', 'instruct'],
                'audio_keywords': ['order', 'ordre', 'command', 'must', 'immediately', 'immédiatement', 'maintenant', 'now'],
                'min_duration': 15
            }
        }
    
    def extract_audio(self, video_path, start_time_s=0, end_time_s=None):
        """Extrait l'audio de la vidéo"""
        import ffmpeg
        import tempfile
        
        try:
            temp_audio = tempfile.mktemp(suffix=".wav")
            
            input_params = {'ss': start_time_s}
            if end_time_s:
                input_params['t'] = end_time_s - start_time_s
            
            input_stream = ffmpeg.input(video_path, **input_params)
            out = ffmpeg.output(
                input_stream['a'],
                temp_audio,
                acodec='pcm_s16le',
                ac=1,
                ar=16000,
                loglevel='quiet'
            )
            
            # Première tentative avec paramètres standards
            try:
                ffmpeg.run(out, overwrite_output=True, quiet=True)
                return temp_audio
            except Exception as first_error:
                print(f"Première tentative d'extraction audio échouée: {first_error}")
                
                # Deuxième tentative avec options plus permissives
                try:
                    print("Nouvelle tentative avec paramètres alternatifs...")
                    input_stream = ffmpeg.input(video_path, probesize='100M', analyzeduration='100M', **input_params)
                    out = ffmpeg.output(
                        input_stream['a'],
                        temp_audio,
                        acodec='pcm_s16le',
                        ac=1,
                        ar=16000,
                        loglevel='error'  # Afficher les erreurs critiques seulement
                    )
                    ffmpeg.run(out, overwrite_output=True, quiet=False)
                    return temp_audio
                    
                except Exception as second_error:
                    print(f"Deuxième tentative d'extraction audio échouée: {second_error}")
                    
                    # Troisième tentative avec format audio plus simple
                    try:
                        print("Tentative finale avec format audio basique...")
                        input_stream = ffmpeg.input(video_path, **input_params)
                        out = ffmpeg.output(
                            input_stream['a'],
                            temp_audio,
                            f='wav',
                            acodec='pcm_s16le',
                            loglevel='error'
                        )
                        ffmpeg.run(out, overwrite_output=True)
                        return temp_audio
                        
                    except Exception as final_error:
                        print(f"Impossible d'extraire l'audio après 3 tentatives: {final_error}")
                        # Nettoyer le fichier temporaire si créé
                        if os.path.exists(temp_audio):
                            os.remove(temp_audio)
                        return None
            
        except Exception as e:
            print(f"Erreur extraction audio définitive pour {os.path.basename(video_path)}: {e}")
            print("Analyse continuée sans audio pour ce fichier.")
            return None
    
    def transcribe_audio(self, audio_path):
        """Transcrit l'audio avec Whisper"""
        if not self.whisper_model:
            return None
            
        try:
            segments, info = self.whisper_model.transcribe(
                audio_path,
                beam_size=5,
                word_timestamps=True,
                vad_filter=True,
                temperature=0.0,
                compression_ratio_threshold=2.4,
                no_speech_threshold=0.6
            )
            
            segments_list = []
            for segment in segments:
                segments_list.append({
                    'start': segment.start,
                    'end': segment.end,
                    'text': segment.text
                })
            
            # Nettoyer le fichier temporaire
            if os.path.exists(audio_path):
                os.remove(audio_path)
            
            return {
                'language': info.language,
                'segments': segments_list
            }
            
        except Exception as e:
            print(f"Erreur transcription: {e}")
            if os.path.exists(audio_path):
                os.remove(audio_path)
            return None
    
    def extract_frames_with_timecodes(self, video_path, interval_seconds=2, start_time_s=0, end_time_s=None):
        """Extrait des frames pour analyse"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_ms = (total_frames / fps) * 1000
        
        start_ms = start_time_s * 1000
        end_ms = end_time_s * 1000 if end_time_s else duration_ms
        
        frames_data = []
        interval_ms = interval_seconds * 1000
        current_time = start_ms
        
        print(f"Extraction frames toutes les {interval_seconds}s de {self.format_timecode(start_ms)} à {self.format_timecode(end_ms)}")
        
        while current_time < end_ms:
            cap.set(cv2.CAP_PROP_POS_MSEC, current_time)
            ret, frame = cap.read()
            
            if ret:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
                
                frames_data.append({
                    'image': pil_image,
                    'timecode': self.format_timecode(current_time),
                    'time_ms': current_time
                })
            
            current_time += interval_ms
        
        cap.release()
        return frames_data
    
    def format_timecode(self, milliseconds):
        """Convertit les millisecondes en format HH:MM:SS"""
        td = timedelta(milliseconds=milliseconds)
        total_seconds = int(td.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    
    def detect_objects(self, image):
        """Détecte les objets dans l'image avec YOLOv8"""
        try:
            results = self.yolo_model(image, conf=0.25, verbose=False)
            detections = []
            boxes = []
            
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Récupérer classe et confidence
                        cls_id = int(box.cls[0])
                        conf = float(box.conf[0])
                        class_name = self.yolo_model.names[cls_id]
                        
                        # Récupérer coordonnées
                        x1, y1, x2, y2 = box.xyxy[0].tolist()
                        boxes.append([int(x1), int(y1), int(x2), int(y2)])
                        
                        detections.append({
                            'class': class_name,
                            'confidence': conf,
                            'box': [int(x1), int(y1), int(x2), int(y2)]
                        })
            
            return detections, boxes
        except Exception as e:
            print(f"Erreur détection YOLO: {e}")
            return [], []
    
    def analyze_frame_content(self, frame_bgr):
        """Analyse le contenu d'une frame avec YOLO uniquement"""
        # Détection YOLO
        objects, boxes = self.detect_objects(Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)))
        
        # Compter les personnes
        person_count = sum(1 for obj in objects if obj['class'] == 'person')
        
        # Détecter les véhicules
        vehicles = [obj['class'] for obj in objects if obj['class'] in ['car', 'truck', 'bus', 'motorcycle', 'bicycle']]
        
        return {
            'person_count': person_count,
            'vehicles': vehicles,
            'all_objects': objects
        }
    
    # BLIP supprimé - on utilise uniquement YOLO pour l'analyse
    
    def analyze_film(self, video_path, interval_seconds=2, start_time_s=0, end_time_s=None, output_prefix=None):
        """Analyse complète d'un film pour détecter les scènes genrées"""
        if not os.path.exists(video_path):
            print(f"Erreur: Le fichier {video_path} n'existe pas")
            return None
        
        print(f"Analyse des scènes genrées: {video_path}")
        
        # Extraction audio et transcription (désactivé temporairement)
        transcription = None
        print("Analyse sans audio (ffmpeg désactivé)")
        
        # Extraction des frames
        frames_data = self.extract_frames_with_timecodes(video_path, interval_seconds, start_time_s, end_time_s)
        print(f"{len(frames_data)} frames extraites")
        
        # Analyse de chaque frame
        frames_analysis = []
        prev_light = None
        
        # Préparer la barre de progression
        total_frames = len(frames_data)
        print(f"\nAnalyse des frames: {'*' * total_frames}")
        print("Progression:         ", end='', flush=True)
        
        for i, frame_data in enumerate(frames_data):
            # Afficher progression avec astérisque vert
            print("\033[92m*\033[0m", end='', flush=True)  # Vert
            
            # Détection d'objets avec YOLO
            objects, boxes = self.detect_objects(frame_data['image'])
            
            # Extraction de l'ambiance lumineuse
            frame_bgr = cv2.cvtColor(np.array(frame_data['image']), cv2.COLOR_RGB2BGR)
            ambient_light = extract_ambient_light(frame_bgr, boxes)
            
            # Vérifier continuité lumineuse
            same_light_as_prev = compare_ambient_light(ambient_light, prev_light) if prev_light else False
            
            # Analyse du contenu avec YOLO seulement
            frame_analysis = self.analyze_frame_content(frame_bgr)
            description_fr = f"{frame_analysis['person_count']} personne(s)"
            
            # Dialogue correspondant
            dialogue = self.get_dialogue_for_frame(transcription, frame_data['time_ms'], start_time_s)
            
            frames_analysis.append({
                'timecode': frame_data['timecode'],
                'time_ms': frame_data['time_ms'],
                'description_en': '',  # Plus de BLIP
                'description_fr': description_fr,
                'dialogue': dialogue.get('text', ''),
                'image': frame_data['image'],
                'objects': objects,
                'ambient_light': ambient_light,
                'same_light_as_prev': same_light_as_prev,
                'person_count': frame_analysis['person_count'],
                'vehicles': frame_analysis['vehicles']
            })
            
            prev_light = ambient_light
        
        print()  # Nouvelle ligne après la progression
        
        # Détection des scènes spécialisées
        detected_scenes = self.detect_gendered_scenes(frames_analysis)
        
        # Calcul des métriques globales
        metrics = self.calculate_film_metrics(detected_scenes, frames_analysis)
        
        # Détecter TOUTES les scènes du film (pas seulement nos types prédéfinis)
        all_film_scenes = self.detect_all_film_scenes(frames_analysis)
        
        # Construire une chronologie fiable avec PySceneDetect
        print("Détection professionnelle des scènes avec PySceneDetect...")
        shots = detect_scenes_pyscenedetect(video_path, start_time_s, end_time_s)
        timeline = []
        for (s_ms, e_ms, m_ms) in shots:
            frame = _grab_frame_at_ms(video_path, m_ms)
            if frame is None:
                continue
            ie = _detect_int_ext(frame)
            dn = _detect_day_night(frame)
            duration_ms = e_ms - s_ms
            timeline.append({
                'start_timecode': _ms_to_hms(s_ms),
                'end_timecode': _ms_to_hms(e_ms),
                'start_seconds': int(round(s_ms/1000)),
                'duration': self.format_duration(duration_ms),
                'duration_seconds': duration_ms/1000,
                'int_ext': ie,
                'day_night': dn
            })
        
        # Préparer les résultats
        total_duration = timeline[-1]['end_timecode'] if timeline else "0:00:00"
        
        results = {
            'film': os.path.basename(video_path),
            'analysis_date': datetime.now().isoformat(),
            'total_duration': total_duration,
            'detected_scenes': detected_scenes,
            'metrics': metrics,
            'all_film_scenes': all_film_scenes,
            'timeline': timeline
        }
        
        # Sauvegarder dans les 4 formats
        if output_prefix:
            self.save_results_quadruple_format(results, output_prefix, video_path)
        else:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            self.save_results_quadruple_format(results, f"{base_name}_gendered_analysis", video_path)
        
        return results
    
    def translate_text(self, text):
        """Traduit un texte en français"""
        if not text.strip():
            return ""
        try:
            return self.translator.translate(text)
        except:
            return text
    
    def get_dialogue_for_frame(self, transcription, frame_time_ms, start_offset_s=0):
        """Récupère le dialogue correspondant à une frame"""
        if not transcription or not transcription.get('segments'):
            return {'text': ''}
        
        frame_time_s = (frame_time_ms / 1000.0) - start_offset_s
        dialogue_window = 2.0  # Fenêtre de 2 secondes
        
        dialogue_parts = []
        for segment in transcription['segments']:
            if (segment['start'] <= frame_time_s <= segment['end'] + dialogue_window or
                abs(segment['start'] - frame_time_s) <= dialogue_window):
                dialogue_parts.append(segment['text'])
        
        return {'text': ' '.join(dialogue_parts)}
    
    def detect_gendered_scenes(self, frames_analysis):
        """Détecte toutes les scènes genrées dans le film"""
        detected_scenes = {scene_type: [] for scene_type in self.scene_detectors.keys()}
        
        for scene_type, criteria in self.scene_detectors.items():
            print(f"Détection: {scene_type}")
            scenes = getattr(self, f'detect_{scene_type}')(frames_analysis, criteria)
            detected_scenes[scene_type] = scenes
        
        return detected_scenes
    
    def detect_shopping(self, frames_analysis, criteria):
        """Détecte les scènes de shopping"""
        scenes = []
        current_scene = None
        
        for frame in frames_analysis:
            is_shopping = self.check_keywords_match(
                frame['description_fr'] + ' ' + frame['dialogue'],
                criteria['visual_keywords'] + criteria['audio_keywords']
            )
            
            if is_shopping:
                if current_scene is None:
                    current_scene = {
                        'start_timecode': frame['timecode'],
                        'start_time_ms': frame['time_ms'],
                        'confidence_frames': 1
                    }
                else:
                    current_scene['confidence_frames'] += 1
                
                current_scene['end_timecode'] = frame['timecode']
                current_scene['end_time_ms'] = frame['time_ms']
            else:
                if current_scene and self.is_scene_valid(current_scene, criteria):
                    scenes.append(self.finalize_scene(current_scene))
                current_scene = None
        
        # Dernière scène
        if current_scene and self.is_scene_valid(current_scene, criteria):
            scenes.append(self.finalize_scene(current_scene))
        
        return scenes
    
    def detect_voiture(self, frames_analysis, criteria):
        """Détecte les scènes de voiture avec YOLO et continuité temporelle"""
        scenes = []
        current_scene = None
        
        # Objets YOLO liés aux voitures
        car_objects = ['car', 'truck', 'bus', 'motorcycle', 'steering wheel']
        
        for i, frame in enumerate(frames_analysis):
            is_car_scene = False
            
            # 1. Vérifier objets YOLO
            for obj in frame.get('objects', []):
                if any(car_obj in obj['class'].lower() for car_obj in car_objects):
                    is_car_scene = True
                    break
            
            # 2. Vérifier mots-clés si pas détecté par YOLO
            if not is_car_scene:
                is_car_scene = self.check_keywords_match(
                    frame['description_fr'] + ' ' + frame['dialogue'],
                    criteria['visual_keywords'] + criteria['audio_keywords']
                )
            
            # 3. Propagation temporelle : vérifier frames adjacentes
            if not is_car_scene and i > 0 and i < len(frames_analysis) - 1:
                # Si frame précédente et suivante sont des scènes de voiture avec même ambiance
                prev_frame = frames_analysis[i-1]
                next_frame = frames_analysis[i+1] if i+1 < len(frames_analysis) else None
                
                if prev_frame and next_frame:
                    prev_is_car = any(any(car_obj in obj['class'].lower() for car_obj in car_objects) 
                                    for obj in prev_frame.get('objects', []))
                    next_is_car = any(any(car_obj in obj['class'].lower() for car_obj in car_objects) 
                                    for obj in next_frame.get('objects', []))
                    
                    # Si les deux sont voiture et même ambiance lumineuse
                    if prev_is_car and next_is_car and frame.get('same_light_as_prev'):
                        is_car_scene = True
            
            if is_car_scene:
                if current_scene is None:
                    current_scene = {
                        'start_timecode': frame['timecode'],
                        'start_time_ms': frame['time_ms'],
                        'confidence_frames': 1,
                        'detected_objects': [obj['class'] for obj in frame.get('objects', [])]
                    }
                else:
                    current_scene['confidence_frames'] += 1
                    current_scene['detected_objects'].extend([obj['class'] for obj in frame.get('objects', [])])
                
                current_scene['end_timecode'] = frame['timecode']
                current_scene['end_time_ms'] = frame['time_ms']
            else:
                if current_scene and self.is_scene_valid(current_scene, criteria):
                    # Calculer confiance basée sur objets détectés
                    unique_objects = set(current_scene['detected_objects'])
                    car_object_count = sum(1 for obj in unique_objects if any(car in obj.lower() for car in car_objects))
                    current_scene['confidence'] = min(1.0, current_scene['confidence_frames'] * 0.1 + car_object_count * 0.2)
                    scenes.append(self.finalize_scene(current_scene))
                current_scene = None
        
        # Dernière scène
        if current_scene and self.is_scene_valid(current_scene, criteria):
            scenes.append(self.finalize_scene(current_scene))
        
        return scenes
    
    def detect_sport_affrontement(self, frames_analysis, criteria):
        """Détecte les scènes de sport/affrontement"""
        return self.detect_keyword_based_scene(frames_analysis, criteria, 'sport_affrontement')
    
    def detect_long_dialogue(self, frames_analysis, criteria):
        """Détecte les longs dialogues multi-personnes"""
        scenes = []
        current_scene = None
        
        for frame in frames_analysis:
            has_dialogue = len(frame['dialogue'].strip()) > 10
            multiple_people = any(word in frame['description_fr'].lower() for word in ['gens', 'people', 'group', 'personnes'])
            
            if has_dialogue and multiple_people:
                if current_scene is None:
                    current_scene = {
                        'start_timecode': frame['timecode'],
                        'start_time_ms': frame['time_ms'],
                        'dialogue_frames': 1
                    }
                else:
                    current_scene['dialogue_frames'] += 1
                
                current_scene['end_timecode'] = frame['timecode']
                current_scene['end_time_ms'] = frame['time_ms']
            else:
                if current_scene:
                    end_time = current_scene.get('end_time_ms', 0)
                    start_time = current_scene.get('start_time_ms', 0)
                    duration_s = (end_time - start_time) / 1000
                    if duration_s >= criteria['min_duration']:
                        scene = self.finalize_scene(current_scene)
                        scene['participants_estimated'] = 'multiple'
                        scenes.append(scene)
                current_scene = None
        
        return scenes
    
    def detect_keyword_based_scene(self, frames_analysis, criteria, scene_name):
        """Détecteur générique basé sur les mots-clés"""
        scenes = []
        current_scene = None
        
        for frame in frames_analysis:
            is_match = self.check_keywords_match(
                frame['description_fr'] + ' ' + frame['dialogue'],
                criteria['visual_keywords'] + criteria['audio_keywords']
            )
            
            if is_match:
                if current_scene is None:
                    current_scene = {
                        'start_timecode': frame['timecode'],
                        'start_time_ms': frame['time_ms'],
                        'confidence_frames': 1
                    }
                else:
                    current_scene['confidence_frames'] += 1
                
                current_scene['end_timecode'] = frame['timecode']
                current_scene['end_time_ms'] = frame['time_ms']
            else:
                if current_scene and self.is_scene_valid(current_scene, criteria):
                    scenes.append(self.finalize_scene(current_scene))
                current_scene = None
        
        # Dernière scène
        if current_scene and self.is_scene_valid(current_scene, criteria):
            scenes.append(self.finalize_scene(current_scene))
        
        return scenes
    
    # Ajouter les autres détecteurs (soins_toilettage, cuisine, etc.)
    def detect_soins_toilettage(self, frames_analysis, criteria):
        return self.detect_keyword_based_scene(frames_analysis, criteria, 'soins_toilettage')
    
    def detect_cuisine(self, frames_analysis, criteria):
        return self.detect_keyword_based_scene(frames_analysis, criteria, 'cuisine')
    
    def detect_decoration(self, frames_analysis, criteria):
        return self.detect_keyword_based_scene(frames_analysis, criteria, 'decoration')
    
    def detect_emotion_expression(self, frames_analysis, criteria):
        return self.detect_keyword_based_scene(frames_analysis, criteria, 'emotion_expression')
    
    def detect_bricolage(self, frames_analysis, criteria):
        return self.detect_keyword_based_scene(frames_analysis, criteria, 'bricolage')
    
    def detect_bar_alcool(self, frames_analysis, criteria):
        return self.detect_keyword_based_scene(frames_analysis, criteria, 'bar_alcool')
    
    def detect_jeux(self, frames_analysis, criteria):
        return self.detect_keyword_based_scene(frames_analysis, criteria, 'jeux')
    
    def detect_hierarchie(self, frames_analysis, criteria):
        return self.detect_keyword_based_scene(frames_analysis, criteria, 'hierarchie')
    
    def detect_solitude(self, frames_analysis, criteria):
        """Détecte les scènes de solitude avec VRAI comptage de personnes"""
        scenes = []
        current_scene = None
        
        for frame in frames_analysis:
            # Utiliser le VRAI comptage de personnes de YOLO
            person_count = frame.get('person_count', 0)
            single_person = (person_count == 1)  # EXACTEMENT 1 personne
            
            has_little_dialogue = len(frame['dialogue'].strip()) < 20
            
            if single_person and has_little_dialogue:
                if current_scene is None:
                    current_scene = {
                        'start_timecode': frame['timecode'],
                        'start_time_ms': frame['time_ms'],
                        'solitude_frames': 1
                    }
                else:
                    current_scene['solitude_frames'] += 1
                
                current_scene['end_timecode'] = frame['timecode']
                current_scene['end_time_ms'] = frame['time_ms']
            else:
                if current_scene and self.is_scene_valid(current_scene, criteria):
                    scenes.append(self.finalize_scene(current_scene))
                current_scene = None
        
        return scenes
    
    def detect_commandement(self, frames_analysis, criteria):
        """Détecte les scènes de commandement"""
        return self.detect_keyword_based_scene(frames_analysis, criteria, 'commandement')
    
    def check_keywords_match(self, text, keywords):
        """Vérifie si le texte contient des mots-clés"""
        text_lower = text.lower()
        return any(keyword.lower() in text_lower for keyword in keywords)
    
    def is_scene_valid(self, scene, criteria):
        """Vérifie si une scène respecte les critères de durée minimale"""
        if not scene.get('end_time_ms') or not scene.get('start_time_ms'):
            return False
        duration_s = (scene.get('end_time_ms', 0) - scene.get('start_time_ms', 0)) / 1000
        return duration_s >= criteria.get('min_duration', 15)
    
    def finalize_scene(self, scene):
        """Finalise une scène détectée"""
        end_time = scene.get('end_time_ms', 0)
        start_time = scene.get('start_time_ms', 0)
        duration_ms = end_time - start_time
        scene['duration'] = self.format_duration(duration_ms)
        scene['duration_seconds'] = duration_ms / 1000
        scene['confidence'] = scene.get('confidence_frames', 1) * 0.1  # Score basique
        return scene
    
    def format_duration(self, duration_ms):
        """Formate une durée en format lisible"""
        seconds = int(duration_ms / 1000)
        if seconds < 60:
            return f"{seconds}s"
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes}m{seconds:02d}s"
    
    def calculate_film_metrics(self, detected_scenes, frames_analysis):
        """Calcule les métriques globales du film"""
        metrics = {
            'total_scenes_detected': sum(len(scenes) for scenes in detected_scenes.values()),
            'scene_type_summary': {}
        }
        
        for scene_type, scenes in detected_scenes.items():
            total_duration = sum(scene['duration_seconds'] for scene in scenes)
            
            metrics['scene_type_summary'][scene_type] = {
                'count': len(scenes),
                'total_duration_seconds': total_duration
            }
        
        return metrics
    
    def save_results_quadruple_format(self, results, output_prefix, video_path):
        """Sauvegarde dans les 4 formats : CSV, JSON, TXT, HTML"""
        
        # 1. FORMAT CSV (pour analyse statistique)
        self.save_as_csv(results, f"{output_prefix}.csv")
        
        # 2. FORMAT JSON (pour détails complets)
        self.save_as_json(results, f"{output_prefix}.json")
        
        # 3. FORMAT TXT (pour vérification humaine)
        self.save_as_text(results, f"{output_prefix}.txt")
        
        # 4. FORMAT HTML (pour vérification interactive)
        self.save_as_html(results, f"{output_prefix}.html", video_path)
        
        print(f"\nRésultats sauvegardés:")
        print(f"  - Statistiques: {output_prefix}.csv")
        print(f"  - Détails: {output_prefix}.json") 
        print(f"  - Vérification: {output_prefix}.txt")
        print(f"  - Interactive: {output_prefix}.html")
    
    def save_as_csv(self, results, filename):
        """Sauvegarde au format CSV pour analyse statistique"""
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # En-tête
            headers = ['Film', 'Total_Duration', 'Total_Scenes_Detected']
            
            # Ajouter une colonne pour chaque type de scène
            for scene_type in self.scene_detectors.keys():
                headers.extend([f"{scene_type}_count", f"{scene_type}_duration"])
            
            writer.writerow(headers)
            
            # Données
            row = [
                results['film'],
                results['total_duration'],
                results['metrics']['total_scenes_detected']
            ]
            
            # Ajouter les données par type de scène
            for scene_type in self.scene_detectors.keys():
                scene_data = results['metrics']['scene_type_summary'].get(scene_type, {'count': 0, 'total_duration_seconds': 0})
                row.extend([scene_data['count'], f"{scene_data['total_duration_seconds']:.1f}s"])
            
            writer.writerow(row)
    
    def save_as_json(self, results, filename):
        """Sauvegarde au format JSON avec tous les détails"""
        # Nettoyer les objets Image pour la sérialisation JSON
        clean_results = results.copy()
        
        # Supprimer les objets Image qui ne peuvent pas être sérialisés
        if 'all_film_scenes' in clean_results:
            for scene in clean_results['all_film_scenes']:
                if 'descriptions' in scene:
                    # Garder seulement un échantillon des descriptions
                    scene['descriptions'] = scene['descriptions'][:3] if len(scene['descriptions']) > 3 else scene['descriptions']
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(clean_results, f, indent=2, ensure_ascii=False, default=str)
    
    def save_as_text(self, results, filename):
        """Sauvegarde au format TXT pour vérification humaine"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"ANALYSE DES PATTERNS DE SCÈNES : {results['film']}\n")
            f.write("=" * 80 + "\n")
            f.write(f"Date d'analyse : {results['analysis_date']}\n")
            f.write(f"Durée totale : {results['total_duration']}\n\n")
            
            f.write("RÉSUMÉ STATISTIQUE :\n")
            f.write("-" * 40 + "\n")
            metrics = results['metrics']
            f.write(f"Total scènes détectées : {metrics['total_scenes_detected']}\n\n")
            
            f.write("DÉTAIL PAR TYPE DE SCÈNE :\n")
            f.write("-" * 40 + "\n")
            
            for scene_type, scenes in results['detected_scenes'].items():
                if scenes:
                    f.write(f"\n{scene_type.upper()} - {len(scenes)} scène(s) :\n")
                    
                    for i, scene in enumerate(scenes, 1):
                        f.write(f"  {i}. {scene['start_timecode']} → {scene['end_timecode']} ({scene['duration']})\n")
                        f.write(f"     Confiance : {scene['confidence']:.2f}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("Fin de l'analyse\n")
    
    def save_as_html(self, results, filename, video_path):
        """Sauvegarde au format HTML interactif avec vidéo embarquée"""
        import os
        import urllib.parse
        
        # Chemin vers le dossier _films-done avec encodage URL correct
        video_filename = os.path.basename(video_path)
        # Encoder les espaces et caractères spéciaux pour l'URL
        encoded_filename = urllib.parse.quote(video_filename)
        video_rel_path = f"_films-done/{encoded_filename}"
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(self.generate_html_content(results, video_rel_path))
    
    def generate_html_content(self, results, video_path):
        """Génère le contenu HTML complet"""
        html = f'''
<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Analyse : {results['film']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; background: #f5f5f5; }}
        .header {{ background: white; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); margin-bottom: 0; }}
        h1 {{ color: #333; border-bottom: 3px solid #007acc; padding-bottom: 10px; margin: 0; }}
        .main-container {{ display: flex; height: calc(100vh - 120px); }}
        
        .video-panel {{
            position: fixed;
            left: 0;
            top: 120px;
            width: 45%;
            height: calc(100vh - 120px);
            background: white;
            padding: 20px;
            box-sizing: border-box;
            border-right: 1px solid #ddd;
        }}
        
        .video-container {{ text-align: center; margin-bottom: 20px; position: relative; }}
        
        #timecode-display {{
            background: #000;
            color: #EEEEEE;
            font-family: 'Courier New', Courier, monospace;
            font-size: 24px;
            padding: 10px 20px;
            display: inline-block;
            margin-bottom: 10px;
            border-radius: 4px;
            min-width: 150px;
            text-align: center;
        }}
        
        video {{ 
            width: 100%; 
            max-width: 100%; 
            height: auto;
            max-height: 60vh;
            border: 2px solid #ddd; 
            border-radius: 4px; 
        }}
        
        
        .stats {{ background: #f9f9f9; padding: 15px; border-radius: 4px; }}
        .stats h2 {{ color: #007acc; margin-top: 0; }}
        
        .content-panel {{
            margin-left: 45%;
            width: 55%;
            height: calc(100vh - 120px);
            overflow-y: auto;
            padding: 20px;
            box-sizing: border-box;
            background: white;
        }}
        
        h2 {{ color: #007acc; margin-top: 0; }}
        .scene-type {{ margin: 20px 0; border: 1px solid #ddd; border-radius: 4px; }}
        .scene-type h3 {{ background: #007acc; color: white; margin: 0; padding: 10px; cursor: pointer; user-select: none; }}
        .scene-type h3:hover {{ background: #005a99; }}
        .scene-type-empty {{ border-color: #ccc; }}
        .scene-type-empty h3 {{ background: #999; color: #666; cursor: default; }}
        .scene-type-empty h3:hover {{ background: #999; }}
        .scene-type-all {{ border-color: #333; }}
        .scene-type-all h3 {{ background: #333; color: #ccc; }}
        .scene-type-all h3:hover {{ background: #444; }}
        .scene-list {{ display: none; padding: 0; }}
        .scene-list.expanded {{ display: block; }}
        .scene-item {{ padding: 10px; border-bottom: 1px solid #eee; }}
        .scene-item:last-child {{ border-bottom: none; }}
        
        .timecode-link {{ 
            color: #007acc; font-weight: bold; cursor: pointer; 
            text-decoration: underline; margin-right: 10px;
            transition: all 0.2s;
        }}
        .timecode-link:hover {{ color: #005a99; }}
        
        .scene-duration {{ color: #666; font-style: italic; }}
        .confidence {{ color: #888; font-size: 0.9em; }}
        .no-scenes {{ color: #666; font-style: italic; padding: 20px; text-align: center; }}
        
        .gendered-scene {{ background-color: #fff3e0; border-left: 4px solid #ff9800; }}
        .natural-scene {{ background-color: #f3f3f3; border-left: 4px solid #9e9e9e; }}
        
        @media (max-width: 1200px) {{
            .video-panel {{ width: 40%; }}
            .content-panel {{ margin-left: 40%; width: 60%; }}
        }}
        
        @media (max-width: 900px) {{
            .main-container {{ flex-direction: column; height: auto; }}
            .video-panel {{ position: relative; width: 100%; height: auto; top: auto; }}
            .content-panel {{ margin-left: 0; width: 100%; height: auto; }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Analyse des patterns de scènes : {results['film']}</h1>
    </div>
    
    <div class="main-container">
        <div class="video-panel">
            <div class="video-container">
                <div id="timecode-display">0:00:00</div>
                <video id="mainVideo" controls preload="metadata">
                    <source src="{video_path}" type="video/mp4">
                    Votre navigateur ne supporte pas la balise vidéo.
                </video>
            </div>
            
            <div class="stats">
                <h2>Statistiques globales</h2>
                <p><strong>Date d'analyse :</strong> {results['analysis_date']}</p>
                <p><strong>Durée totale :</strong> {results['total_duration']}</p>
                <p><strong>Total scènes détectées :</strong> {results['metrics']['total_scenes_detected']}</p>
            </div>
        </div>
        
        <div class="content-panel">
            <h2>Détail par type de scène</h2>
'''
        
        # Séparer scènes détectées et non détectées
        scenes_detected = []
        scenes_empty = []
        
        for scene_type, scenes in results['detected_scenes'].items():
            if len(scenes) > 0:
                scenes_detected.append((scene_type, scenes))
            else:
                scenes_empty.append(scene_type)
        
        # ORDRE CONDITIONNEL SELON LES RÈGLES :
        # Si des scènes genrées existent: Scènes détectées → Chronologie → Groupes vides
        # Si AUCUNE scène genrée: Chronologie directement → Groupes vides
        
        has_gendered_scenes = len(scenes_detected) > 0
        
        if has_gendered_scenes:
            # == SECTION 1: SCÈNES GENRÉES DÉTECTÉES PAR THÈME ==
            for scene_type, scenes in scenes_detected:
                scene_count = len(scenes)
                total_duration = sum(scene.get('duration_seconds', 0) for scene in scenes)
                
                html += f'''
        <div class="scene-type">
            <h3 onclick="toggleSceneList('{scene_type}')" id="header-{scene_type}">
                {scene_type.upper().replace('_', ' ')} ({scene_count} scène{'s' if scene_count > 1 else ''} - {total_duration:.1f}s total) ▼
            </h3>
            <div class="scene-list expanded" id="list-{scene_type}">
'''
                
                for i, scene in enumerate(scenes, 1):
                    start_time = self.timecode_to_seconds(scene['start_timecode'])
                    end_time = self.timecode_to_seconds(scene['end_timecode'])
                    confidence = scene.get('confidence', 0)
                    
                    # Formater les timecodes en H:MM:SS pour les scènes genrées
                    start_formatted = self.format_timecode_complete(scene['start_timecode'])
                    end_formatted = self.format_timecode_complete(scene['end_timecode'])
                    
                    html += f'''
                <div class="scene-item">
                    <span class="timecode-link" onclick="seekToTime({start_time})">{start_formatted}</span>
                    <span class="timecode-link" onclick="seekToTime({end_time})">→ {end_formatted}</span>
                    <span class="scene-duration">({scene.get('duration', 'N/A')})</span>
                    <span class="confidence">Conf: {confidence:.2f}</span>
                </div>
'''
                
                html += '''
            </div>
        </div>
'''
        
        # == SECTION 2: CHRONOLOGIE (toujours affichée, après scènes genrées si elles existent) ==
        
        # Créer la liste unifiée de toutes les scènes
        all_scenes_unified = []
        
        # Ajouter les scènes genrées
        for scene_type, scenes in scenes_detected:
            for scene in scenes:
                all_scenes_unified.append({
                    'start_timecode': scene['start_timecode'],
                    'end_timecode': scene.get('end_timecode', scene['start_timecode']),
                    'duration': scene.get('duration', 'N/A'),
                    'type': f"{scene_type.upper().replace('_', ' ')}",
                    'start_seconds': self.timecode_to_seconds(scene['start_timecode']),
                    'is_gendered': True
                })
        
        # Ajouter les scènes non-genrées depuis la timeline
        for scene in results.get('timeline', results.get('all_film_scenes', [])):
            all_scenes_unified.append({
                'start_timecode': scene['start_timecode'],
                'end_timecode': scene.get('end_timecode', scene['start_timecode']),
                'duration': scene.get('duration', 'N/A'),
                'type': 'SCÈNE',
                'start_seconds': scene.get('start_seconds', self.timecode_to_seconds(scene['start_timecode'])),
                'is_gendered': False,
                'int_ext': scene.get('int_ext', 'Int'),
                'day_night': scene.get('day_night', 'Jour')
            })
        
        # Trier chronologiquement et éliminer doublons
        all_scenes_sorted = sorted(all_scenes_unified, key=lambda x: x['start_seconds'])
        
        # Déduplication par start_seconds (garder genrée en priorité)
        scenes_deduplicated = []
        seen_seconds = set()
        
        for scene in all_scenes_sorted:
            ssec = scene['start_seconds']
            if ssec not in seen_seconds:
                scenes_deduplicated.append(scene)
                seen_seconds.add(ssec)
            elif scene['is_gendered']:
                for i, existing in enumerate(scenes_deduplicated):
                    if existing['start_seconds'] == ssec:
                        scenes_deduplicated[i] = scene
                        break
        
        all_scenes_sorted = scenes_deduplicated
        
        html += '''
        <div class="scene-type scene-type-all">
            <h3 onclick="toggleSceneList('chronologie')" id="header-chronologie">
                CHRONOLOGIE ▲
            </h3>
            <div class="scene-list expanded" id="list-chronologie">'''
        
        # Afficher toutes les scènes dans l'ordre chronologique
        for scene in all_scenes_sorted:
            timecode = self.format_timecode_complete(scene['start_timecode'])
            duration_fr = self.format_duration_french(scene['duration'])
            start_seconds = scene['start_seconds']
            
            # Format d'affichage selon le type
            if scene['is_gendered']:
                # Pour scènes genrées : <timecode> <thème> (<durée>)
                theme = scene['type']
                display_text = f"{theme} ({duration_fr})"
            else:
                # Pour scènes non-genrées : <timecode> Int|Ext - Jour|Nuit (<durée>)
                ie = scene.get('int_ext', 'Int')
                dn = scene.get('day_night', 'Jour')
                display_text = f"{ie} - {dn} ({duration_fr})"
            
            # Style différent pour genrées vs non-genrées
            style_class = 'gendered-scene' if scene['is_gendered'] else 'natural-scene'
            
            html += f'''
                <div class="scene-item {style_class}">
                    <span class="timecode-link" onclick="seekToTime({start_seconds})">{timecode}</span>
                    <span class="scene-duration"> {display_text}</span>
                </div>
'''
        
        html += '''
            </div>
        </div>
'''
        
        # == SECTION 3: THÈMES VIDES (seulement si il y a des scènes genrées détectées) ==
        if has_gendered_scenes:
            for scene_type in scenes_empty:
                html += f'''
        <div class="scene-type scene-type-empty">
            <h3>{scene_type.upper().replace('_', ' ')}</h3>
            <div class="no-scenes">Aucune scène détectée</div>
        </div>
'''
        else:
            # Si aucune scène genrée détectée, montrer tous les groupes comme vides
            for scene_type in self.scene_detectors.keys():
                html += f'''
        <div class="scene-type scene-type-empty">
            <h3>{scene_type.upper().replace('_', ' ')}</h3>
            <div class="no-scenes">Aucune scène détectée</div>
        </div>
'''
        
        # Fermer le HTML
        html += '''
        </div>
    </div>
    
    <script>
        const video = document.getElementById('mainVideo');
        const timecodeDisplay = document.getElementById('timecode-display');
        
        // Formater le temps en H:MM:SS
        function formatTime(seconds) {
            const h = Math.floor(seconds / 3600);
            const m = Math.floor((seconds % 3600) / 60);
            const s = Math.floor(seconds % 60);
            return h + ':' + String(m).padStart(2, '0') + ':' + String(s).padStart(2, '0');
        }
        
        // Mettre à jour le timecode
        function updateTimecode() {
            if (video && timecodeDisplay) {
                timecodeDisplay.textContent = formatTime(video.currentTime);
            }
        }
        
        // Écouter tous les événements de changement de temps
        video.addEventListener('timeupdate', updateTimecode);
        video.addEventListener('play', updateTimecode);
        video.addEventListener('pause', updateTimecode);
        video.addEventListener('seeked', updateTimecode);
        video.addEventListener('loadedmetadata', updateTimecode);
        
        function seekToTime(seconds) {
            try {
                video.currentTime = seconds;
                updateTimecode();
                if (video.paused) {
                    video.play().catch(function(error) {
                        console.log('Lecture automatique bloquée par le navigateur:', error);
                    });
                }
            } catch (error) {
                console.error('Erreur lors du positionnement de la vidéo:', error);
            }
        }
        
        function toggleSceneList(sceneType) {
            const list = document.getElementById('list-' + sceneType);
            const header = document.getElementById('header-' + sceneType);
            
            if (!list || !header) {
                console.error('Element non trouvé pour:', sceneType);
                return;
            }
            
            if (list.classList.contains('expanded')) {
                list.classList.remove('expanded');
                header.innerHTML = header.innerHTML.replace('▲', '▼');
            } else {
                list.classList.add('expanded');
                header.innerHTML = header.innerHTML.replace('▼', '▲');
            }
        }
        
    </script>
</body>
</html>
'''
        
        return html
    
    def timecode_to_seconds(self, timecode):
        """Convertit un timecode HH:MM:SS en secondes"""
        parts = timecode.split(':')
        hours, minutes, seconds = map(int, parts)
        return hours * 3600 + minutes * 60 + seconds
    
    def format_timecode_complete(self, timecode):
        """Formate un timecode en format complet H:MM:SS"""
        parts = timecode.split(':')
        if len(parts) == 3:
            h, m, s = parts
            return f"{int(h)}:{m}:{s}"
        return timecode
    
    def format_duration_french(self, duration_str):
        """Convertit une durée comme '2m15s' en format français '2"15'"""
        if not duration_str or duration_str == 'N/A':
            return '0"00'
        
        # Extraire minutes et secondes
        import re
        match = re.match(r'(\d+)m(\d+)s', duration_str)
        if match:
            minutes, seconds = match.groups()
            return f'{minutes}"{seconds.zfill(2)}'
        
        # Si juste des secondes
        match = re.match(r'(\d+)s', duration_str)
        if match:
            seconds = int(match.group(1))
            if seconds >= 60:
                minutes = seconds // 60
                seconds = seconds % 60
                return f'{minutes}"{seconds:02d}'
            else:
                return f'0"{seconds:02d}'
        
        return duration_str
    
    def detect_all_film_scenes(self, frames_analysis):
        """Détecte toutes les scènes du film basé sur les changements de lieux, temps, etc."""
        print("Détection des scènes non-genrées du film...")
        
        if not frames_analysis:
            return []
        
        scenes = []
        current_scene = None
        prev_description = None
        scene_threshold = 0.6  # Seuil de similarité pour détecter un changement de scène
        
        # Commencer toujours par une première scène dès la première frame
        first_frame = frames_analysis[0]
        current_scene = {
            'type': 'scene_naturelle',
            'start_timecode': first_frame['timecode'],
            'start_time_ms': first_frame['time_ms'],
            'descriptions': [first_frame.get('description_fr', first_frame.get('description_en', ''))],
            'locations': self.extract_location_info(first_frame.get('description_fr', '')),
            'lighting': self.extract_lighting_info(first_frame.get('description_fr', '')),
            'setting': self.extract_setting_info(first_frame.get('description_fr', ''))
        }
        
        # Traiter les frames suivantes
        for i, frame in enumerate(frames_analysis[1:], 1):
            description = frame.get('description_fr', frame.get('description_en', ''))
            
            # Analyser les indicateurs de changement de scène
            scene_change_detected = self.is_scene_change(
                description, prev_description, frame
            )
            
            if scene_change_detected and i > 5:  # Pas de changement dans les 5 premières frames
                # Finaliser la scène précédente
                if current_scene:
                    current_scene['end_timecode'] = prev_frame['timecode'] if 'prev_frame' in locals() else frame['timecode']
                    current_scene['end_time_ms'] = prev_frame['time_ms'] if 'prev_frame' in locals() else frame['time_ms']
                    if self.is_natural_scene_valid(current_scene):
                        scenes.append(self.finalize_natural_scene(current_scene))
                
                # Commencer une nouvelle scène
                current_scene = {
                    'type': 'scene_naturelle',
                    'start_timecode': frame['timecode'],
                    'start_time_ms': frame['time_ms'],
                    'descriptions': [description],
                    'locations': self.extract_location_info(description),
                    'lighting': self.extract_lighting_info(description),
                    'setting': self.extract_setting_info(description)
                }
            else:
                # Continuer la scène actuelle
                if current_scene:
                    current_scene['descriptions'].append(description)
                    current_scene['end_timecode'] = frame['timecode']
                    current_scene['end_time_ms'] = frame['time_ms']
                    
                    # Mettre à jour les informations de contexte
                    current_scene['locations'].update(self.extract_location_info(description))
                    current_scene['lighting'].update(self.extract_lighting_info(description))
                    current_scene['setting'].update(self.extract_setting_info(description))
            
            prev_description = description
            prev_frame = frame
        
        # Finaliser la dernière scène
        if current_scene:
            if not current_scene.get('end_timecode'):
                current_scene['end_timecode'] = frames_analysis[-1]['timecode']
                current_scene['end_time_ms'] = frames_analysis[-1]['time_ms']
            if self.is_natural_scene_valid(current_scene):
                scenes.append(self.finalize_natural_scene(current_scene))
        
        print(f"Détectées {len(scenes)} scènes non-genrées (de {frames_analysis[0]['timecode']} à {frames_analysis[-1]['timecode']})")
        return scenes
    
    def is_scene_change(self, current_desc, prev_desc, frame):
        """Détermine si il y a un changement de scène naturelle"""
        if prev_desc is None:
            return True
        
        # Mots-clés indiquant un changement de lieu
        location_changes = [
            'outside', 'inside', 'interior', 'exterior', 'dehors', 'dedans',
            'kitchen', 'bedroom', 'bathroom', 'living room', 'office', 'car',
            'cuisine', 'chambre', 'salon', 'bureau', 'voiture', 'restaurant',
            'street', 'road', 'park', 'house', 'building', 'shop', 'store'
        ]
        
        # Mots-clés indiquant un changement de temps/éclairage
        time_changes = [
            'night', 'day', 'evening', 'morning', 'dark', 'bright', 'light',
            'nuit', 'jour', 'soir', 'matin', 'sombre', 'clair', 'éclairé'
        ]
        
        # Vérifier les changements significatifs
        current_lower = current_desc.lower()
        prev_lower = prev_desc.lower()
        
        # Changement de lieu détecté
        current_locations = [word for word in location_changes if word in current_lower]
        prev_locations = [word for word in location_changes if word in prev_lower]
        
        if current_locations and prev_locations:
            if not any(loc in prev_locations for loc in current_locations):
                return True
        
        # Changement de temps/éclairage détecté
        current_time = [word for word in time_changes if word in current_lower]
        prev_time = [word for word in time_changes if word in prev_lower]
        
        if current_time and prev_time:
            if not any(time in prev_time for time in current_time):
                return True
        
        # Changement radical de contexte (similarity check basique)
        similarity = self.calculate_description_similarity(current_desc, prev_desc)
        if similarity < 0.3:  # Très différent
            return True
        
        return False
    
    def calculate_description_similarity(self, desc1, desc2):
        """Calcule la similarité basique entre deux descriptions"""
        if not desc1 or not desc2:
            return 0.0
        
        words1 = set(desc1.lower().split())
        words2 = set(desc2.lower().split())
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def extract_location_info(self, description):
        """Extrait les informations de lieu de la description"""
        locations = set()
        location_keywords = {
            'interior': ['inside', 'interior', 'room', 'kitchen', 'bedroom', 'bathroom', 'office'],
            'exterior': ['outside', 'exterior', 'street', 'park', 'garden', 'road'],
            'vehicle': ['car', 'bus', 'train', 'vehicle', 'voiture'],
            'commercial': ['shop', 'store', 'restaurant', 'bar', 'café', 'magasin']
        }
        
        desc_lower = description.lower()
        for category, keywords in location_keywords.items():
            if any(keyword in desc_lower for keyword in keywords):
                locations.add(category)
        
        return locations
    
    def extract_lighting_info(self, description):
        """Extrait les informations d'éclairage de la description"""
        lighting = set()
        lighting_keywords = {
            'day': ['day', 'daylight', 'bright', 'sunny', 'jour', 'clair'],
            'night': ['night', 'dark', 'evening', 'nuit', 'sombre', 'soir'],
            'artificial': ['lamp', 'light', 'illuminated', 'éclairé', 'lampe']
        }
        
        desc_lower = description.lower()
        for category, keywords in lighting_keywords.items():
            if any(keyword in desc_lower for keyword in keywords):
                lighting.add(category)
        
        return lighting
    
    def extract_setting_info(self, description):
        """Extrait les informations de contexte de la description"""
        settings = set()
        setting_keywords = {
            'urban': ['city', 'urban', 'building', 'street', 'ville', 'urbain'],
            'domestic': ['home', 'house', 'kitchen', 'bedroom', 'maison', 'domestique'],
            'professional': ['office', 'work', 'business', 'bureau', 'travail'],
            'social': ['restaurant', 'bar', 'party', 'meeting', 'social']
        }
        
        desc_lower = description.lower()
        for category, keywords in setting_keywords.items():
            if any(keyword in desc_lower for keyword in keywords):
                settings.add(category)
        
        return settings
    
    def is_natural_scene_valid(self, scene):
        """Vérifie si une scène non-genrée est valide (durée minimale)"""
        end_time = scene.get('end_time_ms')
        start_time = scene.get('start_time_ms')
        if not end_time or not start_time:
            return False
        
        duration_s = (end_time - start_time) / 1000
        return duration_s >= 5  # Au moins 5 secondes
    
    def finalize_natural_scene(self, scene):
        """Finalise une scène non-genrée détectée"""
        end_time = scene.get('end_time_ms', 0)
        start_time = scene.get('start_time_ms', 0)
        duration_ms = end_time - start_time
        scene['duration'] = self.format_duration(duration_ms)
        scene['duration_seconds'] = duration_ms / 1000
        scene['confidence'] = 0.8  # Score fixe pour les scènes non-genrées
        
        # Résumé du contexte
        scene['context_summary'] = {
            'locations': list(scene.get('locations', [])),
            'lighting': list(scene.get('lighting', [])),
            'setting': list(scene.get('setting', []))
        }
        
        return scene


def parse_time(time_str):
    """Convertit un temps au format HH:MM:SS ou MM:SS en secondes"""
    parts = time_str.split(':')
    if len(parts) == 2:  # MM:SS
        minutes, seconds = map(int, parts)
        return minutes * 60 + seconds
    elif len(parts) == 3:  # HH:MM:SS
        hours, minutes, seconds = map(int, parts)
        return hours * 3600 + minutes * 60 + seconds
    else:
        raise ValueError("Format de temps invalide. Utilisez MM:SS ou HH:MM:SS")


def analyze_folder(folder_path, interval_seconds=4, output_dir=None, start_time_s=0, end_time_s=None):
    """Analyse tous les films d'un dossier"""
    from datetime import datetime
    
    start_time = datetime.now()
    print(f"Début de l'analyse: {start_time.strftime('%d/%m/%Y %H:%M:%S')}")
    
    if not os.path.isdir(folder_path):
        print(f"Erreur: {folder_path} n'est pas un dossier")
        return
    
    # Extensions vidéo supportées
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
    
    # Trouver tous les fichiers vidéo
    video_files = []
    for file in os.listdir(folder_path):
        if os.path.splitext(file.lower())[1] in video_extensions:
            video_files.append(os.path.join(folder_path, file))
    
    if not video_files:
        print(f"Aucun fichier vidéo trouvé dans {folder_path}")
        return
    
    print(f"Trouvé {len(video_files)} fichier(s) vidéo à analyser:")
    for video in video_files:
        print(f"  - {os.path.basename(video)}")
    
    # Créer dossier de sortie si nécessaire
    if output_dir is None:
        output_dir = os.path.join(folder_path, 'analyses')
    
    os.makedirs(output_dir, exist_ok=True)
    print(f"Résultats seront sauvegardés dans: {output_dir}")
    
    # Initialiser le fichier de liste des films analysés (remise à zéro)
    films_list_path = os.path.join(output_dir, 'FILMS_ANALYSES.txt')
    with open(films_list_path, 'w', encoding='utf-8') as f:
        f.write(f"LISTE DES FILMS ANALYSÉS\n")
        f.write(f"Démarré le: {start_time.strftime('%d/%m/%Y à %H:%M:%S')}\n")
        f.write(f"="*50 + "\n\n")
    
    # Analyser chaque film
    detector = ScenePatternDetector()
    all_results = []
    
    for i, video_path in enumerate(video_files, 1):
        print(f"\n{'='*60}")
        print(f"ANALYSE {i}/{len(video_files)}: {os.path.basename(video_path)}")
        print(f"{'='*60}")
        
        # Nom de fichier de sortie
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_prefix = os.path.join(output_dir, f"{base_name}_gendered_analysis")
        
        try:
            results = detector.analyze_film(
                video_path,
                interval_seconds=interval_seconds,
                start_time_s=start_time_s,
                end_time_s=end_time_s,
                output_prefix=output_prefix
            )
            
            if results:
                all_results.append(results)
                print(f"\n✓ Analyse terminée pour {results['film']}")
                print(f"  Scènes détectées: {results['metrics']['total_scenes_detected']}")
                
                # Ajouter à la liste des films analysés
                with open(films_list_path, 'a', encoding='utf-8') as f:
                    f.write(f"✓ {results['film']} - {results['metrics']['total_scenes_detected']} scènes détectées\n")
            else:
                print(f"\n✗ Échec de l'analyse pour {os.path.basename(video_path)}")
                
                # Ajouter l'échec à la liste
                with open(films_list_path, 'a', encoding='utf-8') as f:
                    f.write(f"✗ {os.path.basename(video_path)} - ÉCHEC DE L'ANALYSE\n")
                
        except Exception as e:
            print(f"\n✗ Erreur lors de l'analyse de {os.path.basename(video_path)}: {e}")
            
            # Ajouter l'erreur à la liste
            with open(films_list_path, 'a', encoding='utf-8') as f:
                f.write(f"✗ {os.path.basename(video_path)} - ERREUR: {str(e)}\n")
    
    # Générer un rapport de synthèse
    if all_results:
        generate_summary_report(all_results, output_dir, start_time)
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    # Finaliser le fichier de liste des films
    with open(films_list_path, 'a', encoding='utf-8') as f:
        f.write(f"\n" + "="*50 + "\n")
        f.write(f"ANALYSE TERMINÉE\n")
        f.write(f"Succès: {len(all_results)}/{len(video_files)} films\n")
        f.write(f"Fin: {end_time.strftime('%d/%m/%Y à %H:%M:%S')}\n")
        f.write(f"Durée totale: {str(duration).split('.')[0]}\n")
    
    print(f"\n{'='*60}")
    print(f"ANALYSE TERMINÉE - {len(all_results)}/{len(video_files)} films analysés avec succès")
    print(f"Début: {start_time.strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"Fin: {end_time.strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"Durée totale: {str(duration).split('.')[0]}")
    print(f"Liste des films : {films_list_path}")
    
    print(f"\n{'='*60}")
    print("FILMS ANALYSÉS AVEC SUCCÈS :")
    print(f"{'='*60}")
    
    for result in all_results:
        total_scenes = result['metrics']['total_scenes_detected']
        print(f"  ✓ {result['film']} - {total_scenes} scènes détectées")
    
    # Définir le chemin du dossier films-done (mais ne pas le créer automatiquement)
    films_done_dir = os.path.join(folder_path, 'analyses', 'films-done')
    
    print(f"\n{'='*60}")
    print("ACTION REQUISE : DÉPLACER LES FILMS")
    print(f"{'='*60}")
    print(f"Pour que les liens vidéo HTML fonctionnent, créez le dossier et déplacez les films analysés vers :")
    print(f"  mkdir -p '{films_done_dir}'")
    print("\nCommandes à exécuter :")
    
    for result in all_results:
        film_name = result['film']
        source_path = None
        # Trouver le chemin source du film
        for video_file in video_files:
            if os.path.basename(video_file) == film_name:
                source_path = video_file
                break
        
        if source_path:
            print(f"  mv '{source_path}' '{films_done_dir}/'")
    
    print(f"\n{'='*60}")

def generate_summary_report(all_results, output_dir, start_time):
    """Génère un rapport de synthèse pour tous les films"""
    from datetime import datetime
    
    summary_path = os.path.join(output_dir, 'SYNTHESE_ANALYSE_PATTERNS.csv')
    
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # En-tête avec informations temporelles
        writer.writerow(['# Analyse effectuée le', start_time.strftime('%d/%m/%Y à %H:%M:%S')])
        writer.writerow(['# Fin le', datetime.now().strftime('%d/%m/%Y à %H:%M:%S')])
        writer.writerow([''])  # Ligne vide
        
        # En-tête des données
        headers = ['Film', 'Duree_Totale', 'Total_Scenes_Detectees']
        
        # Ajouter colonnes détaillées pour chaque type de scène
        scene_types = list(all_results[0]['detected_scenes'].keys())
        for scene_type in scene_types:
            headers.extend([f"{scene_type}_count", f"{scene_type}_duree"])
        
        writer.writerow(headers)
        
        # Données pour chaque film
        for result in all_results:
            metrics = result['metrics']
            
            row = [
                result['film'],
                result['total_duration'],
                metrics['total_scenes_detected']
            ]
            
            # Ajouter détails par type de scène
            for scene_type in scene_types:
                scene_data = metrics['scene_type_summary'].get(scene_type, {'count': 0, 'total_duration_seconds': 0})
                row.extend([scene_data['count'], f"{scene_data['total_duration_seconds']:.1f}s"])
            
            writer.writerow(row)
    
    print(f"\n✓ Rapport de synthèse généré: {summary_path}")

def main():
    parser = argparse.ArgumentParser(description='Détection de scènes genrées dans les films')
    parser.add_argument('path', help='Fichier vidéo ou dossier à analyser')
    parser.add_argument('--interval', type=int, default=4, help='Intervalle entre frames (défaut: 4s)')
    parser.add_argument('--start', type=str, help='Temps de début (HH:MM:SS ou MM:SS)')
    parser.add_argument('--end', type=str, help='Temps de fin (HH:MM:SS ou MM:SS)')
    parser.add_argument('--output', type=str, help='Préfixe/dossier de sortie')
    
    args = parser.parse_args()
    
    # Conversion des temps pour dossier aussi
    start_time_s = 0
    end_time_s = None
    
    if args.start:
        start_time_s = parse_time(args.start)
    
    if args.end:
        end_time_s = parse_time(args.end)
    
    # Vérifier si c'est un dossier ou un fichier
    if os.path.isdir(args.path):
        # Analyse de dossier
        period_info = f" de {args.start or '00:00'} à {args.end or 'fin'}" if args.start or args.end else ""
        print(f"Analyse du dossier: {args.path}{period_info}")
        analyze_folder(args.path, args.interval, args.output, start_time_s, end_time_s)
    
    elif os.path.isfile(args.path):
        # Analyse d'un seul fichier
        period_info = f" de {args.start or '00:00'} à {args.end or 'fin'}" if args.start or args.end else ""
        print(f"Analyse du fichier: {args.path}{period_info}")
        
        # Analyse
        detector = ScenePatternDetector()
        results = detector.analyze_film(
            args.path, 
            interval_seconds=args.interval,
            start_time_s=start_time_s, 
            end_time_s=end_time_s,
            output_prefix=args.output
        )
        
        if results:
            print(f"\nAnalyse terminée pour {results['film']}")
        print(f"Total scènes détectées : {results['metrics']['total_scenes_detected']}")
    
    else:
        print(f"Erreur: {args.path} n'existe pas ou n'est ni un fichier ni un dossier")


if __name__ == "__main__":
    main()
