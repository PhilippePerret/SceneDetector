#!/usr/bin/env python3
"""
Détection exhaustive multi-modèles pour l'analyse de films v1.0
Utilise plusieurs modèles de vision pour une détection complète
"""

import cv2
import torch
import numpy as np
import json
import argparse
import os
import sys
from pathlib import Path
from datetime import timedelta
from collections import defaultdict
from typing import Dict, List, Any, Tuple
import hashlib
from PIL import Image

# Import des modèles disponibles
from ultralytics import YOLO
from transformers import (
    AutoProcessor, AutoModelForZeroShotImageClassification,  # CLIP
    AutoModelForImageClassification, AutoImageProcessor,     # Vision classifiers
    pipeline                                                  # Pipelines génériques
)
import torch.nn.functional as F


class MultiModelDetector:
    """Détecteur multi-modèles pour analyse exhaustive de films"""
    
    def __init__(self, video_path: str, device: str = None):
        self.video_path = Path(video_path)
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.processors = {}
        print(f"🎬 Initialisation pour: {self.video_path.name}")
        print(f"🖥️  Device: {self.device}")
        
    def load_models(self):
        """Charge tous les modèles disponibles"""
        print("\n📦 Chargement des modèles...")
        
        # 1. YOLOv8x - Détection d'objets de base (80 classes)
        try:
            self.models['yolo'] = YOLO('yolov8x.pt')
            print("✅ YOLOv8x chargé (80 catégories de base)")
        except Exception as e:
            print(f"❌ YOLOv8x: {e}")
        
        # 2. CLIP - Classification zero-shot (illimité)
        try:
            model_name = "openai/clip-vit-large-patch14"
            self.processors['clip'] = AutoProcessor.from_pretrained(model_name)
            self.models['clip'] = AutoModelForZeroShotImageClassification.from_pretrained(model_name)
            if self.device != 'cpu':
                self.models['clip'].to(self.device)
            print("✅ CLIP chargé (classification zero-shot)")
        except Exception as e:
            print(f"❌ CLIP: {e}")
        
        # 3. DINOv2 - Extraction de features visuelles
        try:
            self.processors['dino'] = AutoImageProcessor.from_pretrained('facebook/dinov2-large')
            self.models['dino'] = AutoModelForImageClassification.from_pretrained('facebook/dinov2-large')
            if self.device != 'cpu':
                self.models['dino'].to(self.device)
            print("✅ DINOv2 chargé (features visuelles)")
        except Exception as e:
            print(f"❌ DINOv2: {e}")
        
        # 4. Pipeline de détection d'objets générique
        try:
            device_id = 0 if torch.cuda.is_available() else -1
            self.models['object_detection'] = pipeline(
                "object-detection", 
                model="facebook/detr-resnet-50",
                device=device_id
            )
            print("✅ DETR chargé (détection d'objets avancée)")
        except Exception as e:
            print(f"❌ DETR: {e}")
        
        # 5. Pipeline de classification d'images
        try:
            device_id = 0 if torch.cuda.is_available() else -1
            self.models['image_classification'] = pipeline(
                "image-classification",
                model="google/vit-base-patch16-224",
                device=device_id
            )
            print("✅ ViT chargé (classification d'images)")
        except Exception as e:
            print(f"❌ ViT: {e}")
            
        print(f"\n✨ {len(self.models)} modèles chargés avec succès\n")
        
    def detect_with_yolo(self, frame: np.ndarray) -> List[Dict]:
        """Détection YOLO standard"""
        if 'yolo' not in self.models:
            return []
        
        try:
            results = self.models['yolo'](frame, verbose=False)
        except Exception:
            return []
        detections = []
        
        for r in results:
            if getattr(r, 'boxes', None) is not None and len(r.boxes) > 0:
                try:
                    classes = r.boxes.cls if hasattr(r.boxes, 'cls') else []
                    confs = r.boxes.conf if hasattr(r.boxes, 'conf') else []
                    bboxes = r.boxes.xyxy if hasattr(r.boxes, 'xyxy') else []
                    for cls_tensor, conf_tensor, bbox_tensor in zip(classes, confs, bboxes):
                        try:
                            cls_idx = int(cls_tensor.item()) if hasattr(cls_tensor, 'item') else int(cls_tensor)
                        except Exception:
                            cls_idx = 0
                        names = getattr(self.models.get('yolo'), 'names', None)
                        if isinstance(names, dict):
                            cls_name = names.get(cls_idx, 'unknown')
                        elif isinstance(names, list):
                            cls_name = names[cls_idx] if 0 <= cls_idx < len(names) else 'unknown'
                        else:
                            cls_name = str(cls_idx)
                        try:
                            conf = float(conf_tensor.item()) if hasattr(conf_tensor, 'item') else float(conf_tensor)
                        except Exception:
                            conf = 0.0
                        try:
                            bbox = [float(v) for v in bbox_tensor.tolist()[:4]]
                        except Exception:
                            bbox = []
                        detections.append({
                            'class': cls_name,
                            'confidence': conf,
                            'bbox': bbox,
                            'model': 'yolo'
                        })
                except Exception:
                    continue
        
        return detections
    
    def detect_with_clip(self, frame: np.ndarray, categories: List[str]) -> Dict[str, float]:
        """Classification CLIP avec catégories personnalisées"""
        if 'clip' not in self.models or 'clip' not in self.processors:
            return {}
        
        # Convertir frame en PIL Image
        try:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        except (cv2.error, AttributeError, TypeError):
            return {}
        
        # Préparer les inputs
        try:
            inputs = self.processors['clip'](
                images=image,
                text=categories,
                return_tensors="pt",
                padding=True
            )
            if self.device != 'cpu':
                inputs = inputs.to(self.device)
            
            # Prédiction
            with torch.no_grad():
                outputs = self.models['clip'](**inputs)
                probs = F.softmax(outputs.logits, dim=1)
                
            # Résultats
            results = {}
            if probs is not None and len(probs.shape) > 1 and probs.shape[0] > 0:
                for i, category in enumerate(categories):
                    if i < probs.shape[1]:
                        results[category] = float(probs[0][i])
            return results
        except Exception:
            return {}
    
    def detect_with_detr(self, frame: np.ndarray) -> List[Dict]:
        """Détection DETR (Facebook)"""
        if 'object_detection' not in self.models:
            return []
        
        # Convertir en PIL
        try:
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        except (cv2.error, AttributeError, TypeError):
            return []
        
        # Détection
        try:
            results = self.models['object_detection'](image)
        except Exception:
            return []
        
        detections = []
        for item in results:
            try:
                bbox = []
                if 'box' in item and isinstance(item['box'], dict):
                    box = item['box']
                    bbox = [
                        box.get('xmin', 0),
                        box.get('ymin', 0),
                        box.get('xmax', 0),
                        box.get('ymax', 0)
                    ]
                
                detections.append({
                    'class': item.get('label', 'unknown'),
                    'confidence': item.get('score', 0.0),
                    'bbox': bbox,
                    'model': 'detr'
                })
            except (KeyError, TypeError, AttributeError):
                continue
        
        return detections
    
    def classify_scene_context(self, frame: np.ndarray, yolo_detections: List[Dict] = None) -> Dict[str, Any]:
        """Analyse contextuelle de la scène"""
        context = {
            'lighting': self.analyze_lighting(frame),
            'location': self.detect_location_type(frame, yolo_detections or []),
            'time_of_day': self.detect_time_of_day(frame),
            'mood': self.detect_mood(frame),
            'activity': self.detect_activity(frame)
        }
        return context
    
    def analyze_lighting(self, frame: np.ndarray) -> Dict[str, Any]:
        """Analyse de l'éclairage"""
        try:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        except (cv2.error, AttributeError, TypeError):
            return {'type': 'inconnu', 'brightness': 0.0, 'saturation': 0.0}
        brightness = np.mean(hsv[:, :, 2])
        saturation = np.mean(hsv[:, :, 1])
        
        lighting_type = "sombre" if brightness < 80 else "clair"
        if brightness < 50:
            lighting_type = "très sombre"
        elif brightness > 180:
            lighting_type = "très lumineux"
        
        return {
            'type': lighting_type,
            'brightness': float(brightness),
            'saturation': float(saturation)
        }
    
    def detect_location_type(self, frame: np.ndarray, yolo_detections: List[Dict]) -> str:
        """Détecte le type de lieu. Utilise CLIP si dispo, sinon fallback YOLO."""
        # 1) CLIP si disponible
        if 'clip' in self.models and 'clip' in self.processors:
            locations = [
                "intérieur", "extérieur", "bureau", "bar", "restaurant",
                "voiture", "nature", "plage", "montagne", "ville", "campagne",
                "magasin", "hôpital", "école", "église", "aéroport", "gare"
            ]
            scores = self.detect_with_clip(frame, locations)
            if scores:
                # Normaliser les libellés complexes en intérieur/extérieur si nécessaire
                best = max(scores, key=scores.get)
                if best.startswith('intérieur'):
                    return 'intérieur'
                if best.startswith('extérieur'):
                    return 'extérieur'
                return best
        
        # 2) Fallback YOLO: heuristique basée sur les objets COCO
        interior_objs = {
            'couch','tv','bed','dining table','toilet','sink','refrigerator','microwave','oven','book','clock','vase',
            'scissors','teddy bear','hair drier','toothbrush','potted plant','chair','cup','knife','fork','spoon','bowl'
        }
        exterior_objs = {
            'car','truck','bus','motorcycle','bicycle','boat','train','traffic light','stop sign','fire hydrant',
            'parking meter','bench','kite','skateboard','surfboard','snowboard','sports ball'
        }
        int_score = sum(1 for d in yolo_detections if d.get('class') in interior_objs)
        ext_score = sum(1 for d in yolo_detections if d.get('class') in exterior_objs)
        
        # Analyse de luminosité/haut de l'image pour aider la décision
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape[:2]
            top_third = gray[:h//3, :]
            top_brightness = float(np.mean(top_third))
        except Exception:
            top_brightness = 0.0
        
        if int_score == 0 and ext_score == 0:
            # Pas d'indice d'objets, décider via luminosité/sky proxy
            return 'extérieur' if top_brightness > 120 else 'intérieur'
        if int_score >= ext_score:
            return 'intérieur'
        return 'extérieur'
    
    def detect_time_of_day(self, frame: np.ndarray) -> str:
        """Détecte le moment de la journée"""
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        except (cv2.error, AttributeError, TypeError):
            return "inconnu"
        avg_brightness = np.mean(gray)
        
        # Analyser le tiers supérieur pour le ciel
        if len(gray.shape) >= 2:
            h, w = gray.shape[:2]
            top_third = gray[:h//3, :]
            top_brightness = np.mean(top_third)
        else:
            top_brightness = np.mean(gray)
        
        if top_brightness < 50:
            return "nuit"
        elif top_brightness < 100:
            return "crépuscule"
        elif top_brightness < 150:
            return "matin/soir"
        else:
            return "jour"
    
    def detect_mood(self, frame: np.ndarray) -> str:
        """Détecte l'ambiance de la scène avec CLIP"""
        if 'clip' not in self.models:
            return "neutre"
        
        moods = [
            "joyeux", "triste", "tendu", "romantique", "dramatique",
            "comique", "effrayant", "paisible", "action", "mystérieux"
        ]
        
        scores = self.detect_with_clip(frame, moods)
        if scores:
            return max(scores, key=scores.get)
        return "neutre"
    
    def detect_activity(self, frame: np.ndarray) -> str:
        """Détecte l'activité principale avec CLIP"""
        if 'clip' not in self.models:
            return "statique"
        
        activities = [
            "conversation", "conduite", "marche", "course", "repas",
            "travail", "danse", "combat", "lecture", "attente",
            "shopping", "sport", "cuisine", "sommeil", "fête"
        ]
        
        scores = self.detect_with_clip(frame, activities)
        if scores:
            return max(scores, key=scores.get)
        return "statique"
    
    def detect_specific_objects(self, frame: np.ndarray, yolo_detections: List[Dict] = None) -> Dict[str, List]:
        """Détection d'objets spécifiques pour l'analyse de films"""
        specific_detections = defaultdict(list)
        yolo_detections = yolo_detections or []
        
        # 1) Fallback rapide via YOLO (si dispos)
        if yolo_detections:
            y2cat = {
                'vêtements': {
                    'handbag','backpack','umbrella','tie','suitcase','shoe'
                },
                'véhicules': {
                    'car','truck','bus','motorcycle','bicycle','train','boat'
                },
                'objets_quotidiens': {
                    'cell phone','laptop','keyboard','remote','book','scissors','bottle','wine glass','cup','fork','knife','spoon','bowl','umbrella','backpack','handbag','suitcase'
                },
                'armes': {
                    'knife'  # COCO ne contient pas pistolet/fusil
                },
                'meubles': {
                    'chair','couch','bed','dining table','toilet','tv','refrigerator'
                }
            }
            for d in yolo_detections:
                cls = d.get('class')
                conf = float(d.get('confidence', 0.0))
                for cat, names in y2cat.items():
                    if cls in names:
                        specific_detections[cat].append({'item': cls, 'confidence': conf, 'model': 'yolo'})
        
        # 2) CLIP (si dispo)
        if 'clip' in self.models and 'clip' in self.processors:
            categories = {
                'vêtements': [
                    "costume", "robe", "jeans", "t-shirt", "manteau",
                    "uniforme", "chapeau", "lunettes", "bijoux"
                ],
                'véhicules': [
                    "voiture", "moto", "vélo", "bus", "camion", "train", "avion", "bateau"
                ],
                'objets_quotidiens': [
                    "téléphone", "ordinateur", "livre", "journal", "cigarette",
                    "verre", "bouteille", "assiette", "tasse", "sac"
                ],
                'armes': ["couteau"],
                'meubles': [
                    "chaise", "table", "lit", "canapé", "bureau",
                    "armoire", "étagère", "lampe"
                ]
            }
            for category, items in categories.items():
                scores = self.detect_with_clip(frame, items)
                for item, score in scores.items():
                    if score > 0.3:
                        specific_detections[category].append({
                            'item': item,
                            'confidence': float(score),
                            'model': 'clip'
                        })
        
        return dict(specific_detections)
    
    def analyze_frame_comprehensive(self, frame: np.ndarray, frame_idx: int, timestamp: float) -> Dict:
        """Analyse complète d'une frame avec tous les modèles"""
        
        # Hash de la frame pour déduplication
        frame_hash = hashlib.md5(frame.tobytes()).hexdigest()[:8]
        
        analysis = {
            'frame_idx': frame_idx,
            'timestamp': timestamp,
            'timestamp_str': str(timedelta(seconds=timestamp)),
            'frame_hash': frame_hash,
            'detections': {}
        }
        
        # 1. Détection YOLO
        print(f"  🎯 YOLO...", end='')
        yolo_detections = self.detect_with_yolo(frame)
        analysis['detections']['yolo'] = yolo_detections
        print(f" {len(yolo_detections)} objets")
        
        # 2. Détection DETR
        print(f"  🎯 DETR...", end='')
        detr_detections = self.detect_with_detr(frame)
        analysis['detections']['detr'] = detr_detections
        print(f" {len(detr_detections)} objets")
        
        # 3. Contexte de scène
        print(f"  🎯 Contexte...", end='')
        analysis['context'] = self.classify_scene_context(frame, yolo_detections)
        location = analysis.get('context', {}).get('location', 'inconnu')
        print(f" {location}")
        
        # 4. Objets spécifiques
        print(f"  🎯 Objets spécifiques...", end='')
        analysis['specific_objects'] = self.detect_specific_objects(frame, yolo_detections)
        total_specific = 0
        if analysis.get('specific_objects'):
            total_specific = sum(len(v) for v in analysis['specific_objects'].values())
        print(f" {total_specific} détections")
        
        # 5. Comptage de personnes
        person_count = sum(1 for d in yolo_detections if d['class'] == 'person')
        analysis['person_count'] = person_count
        
        # 6. Présence de véhicules
        vehicle_classes = {'car', 'truck', 'bus', 'motorcycle', 'bicycle'}
        vehicle_count = sum(1 for d in yolo_detections if d['class'] in vehicle_classes)
        analysis['vehicle_count'] = vehicle_count
        
        return analysis
    
    def process_video(self, start_time: float = 0, end_time: float = None, 
                     interval: float = 2.0, output_dir: str = "analyses"):
        """Traite la vidéo avec tous les modèles"""
        
        # Créer le dossier de sortie
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Ouvrir la vidéo
        cap = cv2.VideoCapture(str(self.video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            print(f"❌ Impossible de lire le FPS de la vidéo")
            cap.release()
            return []
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        # Ajuster les temps
        if end_time is None or end_time > duration:
            end_time = duration
        if start_time < 0:
            start_time = 0
        if start_time >= end_time:
            print("❌ Intervalle invalide: start >= end")
            cap.release()
            return []
        
        print(f"\n📹 Vidéo: {self.video_path.name}")
        print(f"⏱️  Durée: {timedelta(seconds=duration)}")
        print(f"🎬 FPS: {fps:.2f}")
        print(f"🔍 Analyse: {timedelta(seconds=start_time)} → {timedelta(seconds=end_time)}")
        print(f"⏳ Intervalle: {interval}s\n")
        
        # Frames à analyser
        effective_interval = interval if interval and interval > 0 else 2.0
        frame_times = np.arange(start_time, end_time, effective_interval)
        all_analyses = []
        
        for i, time_sec in enumerate(frame_times):
            print(f"\n[{i+1}/{len(frame_times)}] Frame à {timedelta(seconds=time_sec)}:")
            
            # Extraire la frame
            frame_pos = int(time_sec * fps) if fps > 0 else 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_pos)
            ret, frame = cap.read()
            
            if not ret:
                print("  ❌ Impossible d'extraire la frame")
                continue
            
            # Analyser
            frame_idx = int(time_sec * fps) if fps > 0 else 0
            analysis = self.analyze_frame_comprehensive(
                frame, 
                frame_idx,
                time_sec
            )
            
            all_analyses.append(analysis)
        
        cap.release()
        
        # Sauvegarder les résultats
        output_file = output_path / f"{self.video_path.stem}_analysis.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_analyses, f, indent=2, ensure_ascii=False)
        
        print(f"\n✅ Analyse terminée!")
        print(f"📊 {len(all_analyses)} frames analysées")
        print(f"💾 Résultats: {output_file}")
        
        # Générer le rapport HTML
        self.generate_html_report(all_analyses, output_path)
        
        return all_analyses
    
    
    def generate_html_report(self, analyses: List[Dict], output_path: Path):
        """Génère un rapport HTML simple et fonctionnel"""
        
        # Préparer le chemin de la vidéo
        import urllib.parse
        video_filename = self.video_path.name
        encoded_filename = urllib.parse.quote(video_filename)
        video_rel_path = f"_films-done/{encoded_filename}"
        
        html_content = """<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Analyse - {video_name}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background: white;
            color: black;
        }}
        
        .header {{
            background: #f0f0f0;
            padding: 10px 20px;
            border-bottom: 1px solid #ccc;
        }}
        
        h1 {{
            margin: 0;
            font-size: 1.5em;
        }}
        
        .main-container {{
            display: flex;
            height: calc(100vh - 60px);
        }}
        
        .video-panel {{
            position: fixed;
            left: 0;
            top: 60px;
            width: 45%;
            height: calc(100vh - 60px);
            padding: 20px;
            box-sizing: border-box;
            border-right: 1px solid #ccc;
        }}
        
        #timecode-display {{
            background: #000;
            color: #0F0;
            font-family: 'Courier New', monospace;
            font-size: 28px;
            padding: 10px 20px;
            text-align: center;
            margin-bottom: 10px;
        }}
        
        video {{
            width: 100%;
            max-height: 70%;
            border: 1px solid #ccc;
        }}
        
        .detection-panel {{
            margin-left: 45%;
            width: 55%;
            height: calc(100vh - 60px);
            overflow-y: auto;
            padding: 20px;
            box-sizing: border-box;
        }}
        
        .detection-entry {{
            padding: 10px;
            margin: 5px 0;
            border: 1px solid #ddd;
            cursor: default;
        }}
        
        .detection-entry.selected {{
            background-color: #007bff;
            color: white;
        }}
        
        .timecode-link {{
            color: #0066cc;
            cursor: pointer;
            text-decoration: underline;
            font-family: 'Courier New', monospace;
            font-weight: bold;
            margin-right: 10px;
        }}
        
        .detection-entry.selected .timecode-link {{
            color: white;
        }}
        
        .detection-content {{
            margin-left: 100px;
            font-size: 0.9em;
        }}
        
        .object-list {{
            display: inline;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Analyse : {video_name}</h1>
    </div>
    
    <div class="main-container">
        <div class="video-panel">
            <div id="timecode-display">00:00:00</div>
            <video id="mainVideo" controls preload="metadata">
                <source src="{video_path}" type="video/mp4">
                Votre navigateur ne supporte pas la balise vidéo.
            </video>
        </div>
        
        <div class="detection-panel">
            <h2>Détections</h2>
            {detections_content}
        </div>
    </div>
    
    <script>
        const video = document.getElementById('mainVideo');
        const timecodeDisplay = document.getElementById('timecode-display');
        
        function formatTime(seconds) {{
            const h = Math.floor(seconds / 3600);
            const m = Math.floor((seconds % 3600) / 60);
            const s = Math.floor(seconds % 60);
            return String(h).padStart(2, '0') + ':' + String(m).padStart(2, '0') + ':' + String(s).padStart(2, '0');
        }}
        
        function updateTimecode() {{
            if (video && timecodeDisplay) {{
                timecodeDisplay.textContent = formatTime(video.currentTime);
            }}
        }}
        
        function updateSelection() {{
            const currentTime = video.currentTime;
            const allEntries = document.querySelectorAll('.detection-entry');
            
            allEntries.forEach(entry => {{
                const entryTime = parseFloat(entry.dataset.timestamp);
                const nextTime = parseFloat(entry.dataset.nexttimestamp) || entryTime + 10;
                
                if (currentTime >= entryTime && currentTime < nextTime) {{
                    entry.classList.add('selected');
                }} else {{
                    entry.classList.remove('selected');
                }}
            }});
        }}
        
        video.addEventListener('timeupdate', () => {{
            updateTimecode();
            updateSelection();
        }});
        
        video.addEventListener('seeked', () => {{
            updateTimecode();
            updateSelection();
        }});
        
        video.addEventListener('loadedmetadata', updateTimecode);
        
        function seekToTime(seconds) {{
            video.currentTime = seconds;
            updateTimecode();
            updateSelection();
        }}
    </script>
</body>
</html>"""
        
        # Générer le contenu des détections
        detections_html = []
        
        for i, analysis in enumerate(analyses):
            # Récupérer tous les objets détectés
            objects_yolo = [d.get('class', 'unknown') for d in analysis.get('detections', {}).get('yolo', [])]
            
            # Contexte
            context = analysis.get('context', {})
            
            # Temps suivant pour la sélection
            next_timestamp = analyses[i+1]['timestamp'] if i+1 < len(analyses) else analysis['timestamp'] + 10
            
            detection_html = f'''
            <div class="detection-entry" data-timestamp="{analysis['timestamp']}" data-nexttimestamp="{next_timestamp}">
                <span class="timecode-link" onclick="seekToTime({analysis['timestamp']})">{analysis.get('timestamp_str', 'N/A')}</span>
                <div class="detection-content">
                    <strong>Lieu:</strong> {context.get('location', 'inconnu')} | 
                    <strong>Moment:</strong> {context.get('time_of_day', 'inconnu')} | 
                    <strong>Personnes:</strong> {analysis.get('person_count', 0)} | 
                    <strong>Véhicules:</strong> {analysis.get('vehicle_count', 0)}<br>
                    <strong>Objets:</strong> {', '.join(set(objects_yolo)) if objects_yolo else 'Aucun'}
                </div>
            </div>
            '''
            detections_html.append(detection_html)
        
        # Remplir le template
        html_final = html_content.format(
            video_name=self.video_path.name,
            video_path=video_rel_path,
            detections_content=''.join(detections_html)
        )
        
        # Sauvegarder
        html_file = output_path / f"{self.video_path.stem}_report.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_final)
        
        print(f"📄 Rapport HTML: {html_file}")


def parse_time(time_str):
    """Convertit un temps au format HH:MM:SS ou MM:SS en secondes"""
    if not time_str:
        return None
    parts = time_str.split(':')
    if len(parts) == 2:  # MM:SS
        minutes, seconds = map(int, parts)
        return minutes * 60 + seconds
    elif len(parts) == 3:  # HH:MM:SS ou H:MM:SS
        hours, minutes, seconds = map(int, parts)
        return hours * 3600 + minutes * 60 + seconds
    else:
        raise ValueError(f"Format de temps invalide: {time_str}. Utilisez MM:SS ou H:MM:SS")


def main():
    # Charger la config si elle existe
    config = {}
    if os.path.exists('detect_config.json'):
        with open('detect_config.json', 'r') as f:
            config = json.load(f)
    
    parser = argparse.ArgumentParser(description='Détection exhaustive multi-modèles v1.0')
    parser.add_argument('path', nargs='?', help='Fichier vidéo ou dossier à analyser')
    parser.add_argument('--interval', type=int, default=config.get('interval', 4), help='Intervalle entre frames (secondes)')
    parser.add_argument('--start', type=str, default=config.get('start'), help='Temps de début (H:MM:SS ou MM:SS)')
    parser.add_argument('--end', type=str, default=config.get('end'), help='Temps de fin (H:MM:SS ou MM:SS)')
    parser.add_argument('--output', type=str, default=config.get('output', 'analyses'), help='Préfixe/dossier de sortie')
    
    args = parser.parse_args()
    
    # Utiliser path depuis config si non spécifié
    if not args.path:
        args.path = config.get('film_path') or config.get('films_folder')
    
    if not args.path:
        print("Erreur: Aucun fichier ou dossier spécifié (ni en argument ni dans detect_config.json)")
        sys.exit(1)
    
    # Conversion des temps
    start_time_s = parse_time(args.start) if args.start else 0
    end_time_s = parse_time(args.end) if args.end else None
    
    # Vérifier si c'est un dossier ou un fichier
    if os.path.isdir(args.path):
        # Analyse de dossier - traiter tous les fichiers vidéo
        print(f"Analyse du dossier: {args.path}")
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
        video_files = []
        for file in os.listdir(args.path):
            if os.path.splitext(file.lower())[1] in video_extensions:
                video_files.append(os.path.join(args.path, file))
        
        if not video_files:
            print(f"Aucun fichier vidéo trouvé dans {args.path}")
            sys.exit(1)
        
        for video_path in video_files:
            print(f"\n{'='*60}")
            print(f"Analyse de: {os.path.basename(video_path)}")
            print(f"{'='*60}")
            
            # Créer le détecteur
            detector = MultiModelDetector(video_path)
            detector.load_models()
            
            # Nom de sortie spécifique au fichier
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_dir = os.path.join(args.output, base_name)
            
            # Analyser
            detector.process_video(
                start_time=start_time_s,
                end_time=end_time_s,
                interval=float(args.interval),
                output_dir=output_dir
            )
    
    elif os.path.isfile(args.path):
        # Analyse d'un seul fichier
        print(f"Analyse du fichier: {args.path}")
        if args.start or args.end:
            print(f"Période: {args.start or '00:00'} → {args.end or 'fin'}")
        
        # Créer le détecteur
        detector = MultiModelDetector(args.path)
        detector.load_models()
        
        # Analyser
        detector.process_video(
            start_time=start_time_s,
            end_time=end_time_s,
            interval=float(args.interval),
            output_dir=args.output
        )
    
    else:
        print(f"Erreur: {args.path} n'existe pas ou n'est ni un fichier ni un dossier")
        sys.exit(1)


if __name__ == "__main__":
    main()
