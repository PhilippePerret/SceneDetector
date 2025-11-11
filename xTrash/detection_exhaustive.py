#!/usr/bin/env python3
"""
Extraction exhaustive avec tous les modèles de détection disponibles
"""

import json
import cv2
from pathlib import Path
from datetime import timedelta
from typing import Dict, List, Any
import torch

class DetectionExhaustive:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.detections = {}
        self.models_loaded = {}
        
    def load_all_models(self):
        """Charge tous les modèles de détection"""
        print("Chargement des modèles...")
        
        # 1. RAM++ (Recognize Anything Model Plus Plus)
        try:
            from ram import ram_plus
            self.models_loaded['ram++'] = ram_plus.load_model()
            print("✓ RAM++ chargé (14,000+ catégories)")
        except:
            print("✗ RAM++ non disponible")
            
        # 2. CLIP (OpenAI)
        try:
            import clip
            self.models_loaded['clip'] = clip.load("ViT-L/14@336px", device="cuda" if torch.cuda.is_available() else "cpu")
            print("✓ CLIP chargé")
        except:
            print("✗ CLIP non disponible")
            
        # 3. OWL-ViT (Google)
        try:
            from transformers import OwlViTProcessor, OwlViTForObjectDetection
            self.models_loaded['owl-vit'] = {
                'processor': OwlViTProcessor.from_pretrained("google/owlvit-large-patch14"),
                'model': OwlViTForObjectDetection.from_pretrained("google/owlvit-large-patch14")
            }
            print("✓ OWL-ViT chargé")
        except:
            print("✗ OWL-ViT non disponible")
            
        # 4. DETIC (Meta)
        try:
            from detectron2.config import get_cfg
            from detic import add_detic_config
            cfg = get_cfg()
            add_detic_config(cfg)
            self.models_loaded['detic'] = cfg
            print("✓ DETIC chargé (21,000 catégories)")
        except:
            print("✗ DETIC non disponible")
            
        # 5. Grounding DINO
        try:
            from groundingdino.util.inference import load_model
            self.models_loaded['grounding-dino'] = load_model("groundingdino_swint_ogc.pth")
            print("✓ Grounding DINO chargé")
        except:
            print("✗ Grounding DINO non disponible")
            
        # 6. X-CLIP
        try:
            from x_clip import XCLIP
            self.models_loaded['x-clip'] = XCLIP.load()
            print("✓ X-CLIP chargé (vidéo-aware)")
        except:
            print("✗ X-CLIP non disponible")
            
        # 7. YOLOv8 (notre base actuelle)
        try:
            from ultralytics import YOLO
            self.models_loaded['yolo'] = YOLO('yolov8x.pt')
            print("✓ YOLOv8x chargé (80 catégories)")
        except:
            print("✗ YOLO non disponible")
            
        # 8. BLIP-2 (description de scène)
        try:
            from transformers import Blip2Processor, Blip2ForConditionalGeneration
            self.models_loaded['blip2'] = {
                'processor': Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b"),
                'model': Blip2ForConditionalGeneration.from_pretrained("Salesforce/blip2-opt-2.7b")
            }
            print("✓ BLIP-2 chargé (descriptions)")
        except:
            print("✗ BLIP-2 non disponible")
            
        # 9. DeepFashion2 (vêtements détaillés)
        try:
            import mmdet
            self.models_loaded['deepfashion'] = mmdet.apis.init_detector('deepfashion2_config.py', 'deepfashion2.pth')
            print("✓ DeepFashion2 chargé (13 types vêtements)")
        except:
            print("✗ DeepFashion2 non disponible")
            
        # 10. Places365 (reconnaissance de lieux)
        try:
            import torchvision.models as models
            self.models_loaded['places365'] = models.places365()
            print("✓ Places365 chargé (365 lieux)")
        except:
            print("✗ Places365 non disponible")
            
    def detect_with_ram_plus(self, frame):
        """RAM++ - détection automatique de 14,000+ catégories"""
        if 'ram++' not in self.models_loaded:
            return []
        
        # RAM++ détecte automatiquement sans prompt
        tags = self.models_loaded['ram++'].predict(frame)
        return tags
        
    def detect_with_clip(self, frame, text_queries):
        """CLIP - recherche par requête textuelle"""
        if 'clip' not in self.models_loaded:
            return {}
            
        model, preprocess = self.models_loaded['clip']
        # Implémentation CLIP...
        return {}
        
    def detect_with_owl_vit(self, frame, text_queries):
        """OWL-ViT - détection par requête textuelle"""
        if 'owl-vit' not in self.models_loaded:
            return []
            
        processor = self.models_loaded['owl-vit']['processor']
        model = self.models_loaded['owl-vit']['model']
        # Implémentation OWL-ViT...
        return []
        
    def detect_all_in_frame(self, frame, timestamp):
        """Lance TOUS les modèles sur une frame"""
        results = {
            'timestamp': timestamp,
            'detections': {}
        }
        
        # 1. RAM++ (le plus complet, sans requête)
        if 'ram++' in self.models_loaded:
            results['detections']['ram++'] = self.detect_with_ram_plus(frame)
            
        # 2. YOLO (rapide, basique)
        if 'yolo' in self.models_loaded:
            yolo_results = self.models_loaded['yolo'](frame)
            results['detections']['yolo'] = self.parse_yolo_results(yolo_results)
            
        # 3. CLIP avec requêtes prédéfinies
        if 'clip' in self.models_loaded:
            queries = [
                "person in a bar", "car interior", "outdoor scene",
                "person wearing suit", "person wearing dress", 
                "vintage car", "bottles", "glasses on table"
            ]
            results['detections']['clip'] = self.detect_with_clip(frame, queries)
            
        # 4. OWL-ViT avec requêtes
        if 'owl-vit' in self.models_loaded:
            queries = ["person", "car", "bottle", "glass", "suit", "dress"]
            results['detections']['owl-vit'] = self.detect_with_owl_vit(frame, queries)
            
        # 5. BLIP-2 pour description générale
        if 'blip2' in self.models_loaded:
            results['detections']['blip2_description'] = self.get_blip_description(frame)
            
        # Et ainsi de suite pour tous les modèles...
        
        return results
        
    def process_video_segment(self, start_time, end_time):
        """Traite un segment vidéo avec tous les modèles"""
        cap = cv2.VideoCapture(self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Convertir temps en frames
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        all_detections = []
        frame_num = start_frame
        
        while frame_num <= end_frame:
            ret, frame = cap.read()
            if not ret:
                break
                
            timestamp = frame_num / fps
            
            # Détection toutes les N frames pour économiser
            if frame_num % 15 == 0:  # ~2 fois par seconde à 30fps
                detections = self.detect_all_in_frame(frame, timestamp)
                all_detections.append(detections)
                print(f"Frame {frame_num}/{end_frame}: {len(detections['detections'])} modèles utilisés")
                
            frame_num += 1
            
        cap.release()
        return all_detections
        
    def generate_html_report(self, detections, output_path):
        """Génère le rapport HTML détaillé"""
        # TODO: Créer le HTML avec timeline, filtres, etc.
        pass

if __name__ == "__main__":
    # Test sur Harold et Maude
    detector = DetectionExhaustive("./HaroldAndMaude.mp4")
    detector.load_all_models()
    
    # Segment de test (20:00 à 21:17)
    detections = detector.process_video_segment(20*60, 21*60+17)
    
    # Sauvegarder résultats
    with open("detections_exhaustives.json", "w") as f:
        json.dump(detections, f, indent=2)
        
    print(f"\n{len(detections)} frames analysées")
    print("Résultats sauvés dans detections_exhaustives.json")
