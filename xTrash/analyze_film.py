#!/usr/bin/env python3

import cv2
import torch
from PIL import Image
import numpy as np
import sys
import os
from datetime import timedelta
from transformers import BlipProcessor, BlipForConditionalGeneration
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from deep_translator import GoogleTranslator
import json
import re
from collections import defaultdict
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer
import spacy
import subprocess
import sys
import ffmpeg
import tempfile
import hashlib
from datetime import datetime

# Supprimer le warning tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class FilmAnalyzer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Utilisation du device: {self.device}")
        
        # Charger BLIP
        print("Chargement du modèle BLIP...")
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-base"
        ).to(self.device)
        print("Modèle chargé!")
        
        # Initialiser le traducteur
        self.translator = GoogleTranslator(source='en', target='fr')
        
        # Pour le suivi des personnages
        self.character_embeddings = []
        self.character_count = 0
        
        # Charger faster-whisper avec détection de performance
        print("Chargement du modèle Whisper (faster-whisper)...")
        try:
            # Profil de performance adaptatif
            performance_profile = self.detect_performance_profile()
            model_size = performance_profile['whisper_model']
            compute_type = performance_profile['compute_type']
            
            print(f"Profil détecté: {performance_profile['name']} - Modèle: {model_size}")
            self.whisper_model = WhisperModel(model_size, device="cpu", compute_type=compute_type)
            print(f"Whisper chargé ({model_size})!")
        except Exception as e:
            print(f"Erreur Whisper: {e}")
            print("Désactivation de la transcription audio...")
            self.whisper_model = None
        
        # Pour le suivi des lieux identiques
        self.location_embeddings = {}
        self.location_counters = defaultdict(int)
        
        # Pipeline hybride d'analyse sémantique avec modèle adaptatif
        print("Chargement du modèle Sentence-Transformers...")
        st_model = performance_profile['sentence_model']
        self.sentence_model = SentenceTransformer(st_model)
        print(f"Sentence-Transformers chargé ({st_model})!")
        
        # Cache des modèles spaCy
        self.spacy_models = {}
        
        # Contextes de référence pour classification
        self.reference_contexts = {
            'academic': self.sentence_model.encode("université cours professeur étudiant science mathématiques recherche enseignement MIT école classe tableau"),
            'medical': self.sentence_model.encode("hôpital médecin patient traitement maladie santé diagnostic chirurgie infirmier"),
            'business': self.sentence_model.encode("bureau entreprise travail patron employé réunion projet client affaires"),
            'domestic': self.sentence_model.encode("maison cuisine salon chambre famille repas conversation personnel intime"),
            'social': self.sentence_model.encode("bar restaurant café fête amis discussion rendez-vous sortie rencontre"),
            'action': self.sentence_model.encode("poursuite combat bagarre course urgence danger police crime"),
            'transport': self.sentence_model.encode("voiture train avion bus voyage route conduire transport déplacement")
        }

    def extract_audio(self, video_path, start_time_s=0, end_time_s=None):
        """Extrait l'audio de la vidéo pour la transcription"""
        try:
            # Créer un fichier temporaire pour l'audio
            temp_audio = tempfile.mktemp(suffix=".wav")
            
            # Paramètres d'extraction
            input_params = {'ss': start_time_s}  # Start time
            if end_time_s:
                input_params['t'] = end_time_s - start_time_s  # Duration
            
            input_stream = ffmpeg.input(video_path, **input_params)
            
            out = ffmpeg.output(
                input_stream['a'],  # Select audio stream explicitly
                temp_audio,
                acodec='pcm_s16le',
                ac=1,  # mono
                ar=16000,  # sample rate pour Whisper
                loglevel='quiet'
            )
            
            ffmpeg.run(out, overwrite_output=True)
            return temp_audio
            
        except Exception as e:
            print(f"Erreur lors de l'extraction audio: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def transcribe_audio(self, audio_path):
        """Transcrit l'audio avec Whisper"""
        if not self.whisper_model:
            print("Whisper non disponible, transcription désactivée")
            return None
            
        try:
            print("Transcription en cours...")
            segments, info = self.whisper_model.transcribe(
                audio_path, 
                beam_size=5,
                word_timestamps=True,
                vad_filter=True,  # Détection automatique de la voix
                temperature=0.0,  # Plus déterministe
                compression_ratio_threshold=2.4,  # Filtrer les répétitions
                no_speech_threshold=0.6  # Seuil pour détecter le silence
            )
            
            # Convertir les segments en format compatible
            segments_list = []
            full_text = ""
            
            for segment in segments:
                segment_dict = {
                    'start': segment.start,
                    'end': segment.end,
                    'text': segment.text
                }
                segments_list.append(segment_dict)
                full_text += segment.text + " "
            
            # Nettoyer le fichier temporaire
            if os.path.exists(audio_path):
                os.remove(audio_path)
            
            return {
                'language': info.language,
                'text': full_text.strip(),
                'segments': segments_list
            }
            
        except Exception as e:
            print(f"Erreur lors de la transcription: {e}")
            # Nettoyer quand même
            if os.path.exists(audio_path):
                os.remove(audio_path)
            return None
    
    def translate_text(self, text, source_lang, target_lang='fr'):
        """Traduit un texte si nécessaire"""
        if source_lang == target_lang or source_lang == 'fr':
            return text
        
        try:
            translator = GoogleTranslator(source=source_lang, target=target_lang)
            return translator.translate(text)
        except:
            return text  # Retourner original si traduction échoue
    
    def format_timecode(self, milliseconds):
        """Convertit les millisecondes en format HH:MM:SS"""
        td = timedelta(milliseconds=milliseconds)
        total_seconds = int(td.total_seconds())
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def extract_frames_with_timecodes(self, video_path, interval_seconds=30, start_time_s=0, end_time_s=None):
        """Extrait des frames à intervalles réguliers avec leurs timecodes"""
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_ms = (total_frames / fps) * 1000
        
        # Convertir les temps en millisecondes
        start_ms = start_time_s * 1000
        end_ms = end_time_s * 1000 if end_time_s else duration_ms
        
        frames_data = []
        interval_ms = interval_seconds * 1000
        current_time = start_ms
        
        print(f"Durée du film: {self.format_timecode(duration_ms)}")
        print(f"Analyse de {self.format_timecode(start_ms)} à {self.format_timecode(end_ms)}")
        print(f"Extraction d'une frame toutes les {interval_seconds} secondes")
        
        while current_time < end_ms:
            # Positionner à la frame correspondant au temps voulu
            cap.set(cv2.CAP_PROP_POS_MSEC, current_time)
            ret, frame = cap.read()
            
            if ret:
                # Convertir BGR vers RGB
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

    def detect_day_night(self, image):
        """Détecte si c'est JOUR ou NUIT basé sur la luminosité"""
        # Convertir en niveaux de gris et calculer luminosité moyenne
        gray = np.array(image.convert('L'))
        avg_brightness = np.mean(gray)
        return "JOUR" if avg_brightness > 100 else "NUIT"
    
    def load_locations_config(self):
        """Charge la configuration des lieux depuis le fichier JSON"""
        try:
            config_path = os.path.join(os.path.dirname(__file__), 'lieux_config.json')
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
                return config['lieux']
        except:
            # Fallback si le fichier n'existe pas
            return {
                'CUISINE': ['cuisine', 'frigo', 'évier', 'four', 'placard'],
                'SALON': ['salon', 'canapé', 'télé', 'fauteuil'],
                'CHAMBRE': ['chambre', 'lit', 'oreiller', 'armoire'],
                'BUREAU': ['bureau', 'ordinateur', 'chaise', 'table'],
                'RUE': ['rue', 'route', 'trottoir', 'voiture', 'extérieur']
            }
    
    def get_location_embedding(self, description_fr, image):
        """Crée un embedding pour comparer les lieux"""
        # Combinaison simple : hash de mots-clés + luminosité moyenne de l'image
        description_words = set(description_fr.lower().split())
        # Réduire à des mots-clés pertinents
        relevant_words = [w for w in description_words if len(w) > 3 and w not in ['avec', 'dans', 'pour', 'elle', 'homme', 'femme']]
        
        # Signature visuelle basique
        gray = np.array(image.convert('L'))
        brightness = np.mean(gray)
        contrast = np.std(gray)
        
        # Créer une signature combinée
        text_signature = str(sorted(relevant_words))
        visual_signature = f"{brightness:.0f}_{contrast:.0f}"
        
        return f"{text_signature}_{visual_signature}"
    
    def detect_location(self, description_fr, image):
        """Détecte le lieu avec numérotation intelligente"""
        if not hasattr(self, '_lieu_keywords'):
            self._lieu_keywords = self.load_locations_config()
        
        # Détecter le type de lieu de base
        description_lower = description_fr.lower()
        base_location = None
        confidence = "~"
        
        for lieu, mots_cles in self._lieu_keywords.items():
            matches = sum(1 for mot in mots_cles if mot in description_lower)
            if matches >= 1:
                base_location = lieu
                confidence = "" if matches >= 2 else "~"
                break
        
        if not base_location:
            return "DÉCOR NON IDENTIFIÉ"
        
        # Obtenir l'embedding pour ce lieu
        embedding = self.get_location_embedding(description_fr, image)
        
        # Vérifier si c'est un lieu déjà vu
        if base_location not in self.location_embeddings:
            self.location_embeddings[base_location] = {}
        
        # Chercher un lieu similaire déjà enregistré
        for existing_id, existing_embedding in self.location_embeddings[base_location].items():
            similarity = self.calculate_embedding_similarity(embedding, existing_embedding)
            if similarity > 0.7:  # Seuil de similarité
                return f"{confidence}{base_location}#{existing_id}"
        
        # Nouveau lieu, créer un nouvel ID
        self.location_counters[base_location] += 1
        new_id = self.location_counters[base_location]
        self.location_embeddings[base_location][new_id] = embedding
        
        return f"{confidence}{base_location}#{new_id}"
    
    def detect_performance_profile(self):
        """Détecte le profil de performance de la machine"""
        import psutil
        import platform
        
        # Détecter les specs de la machine
        ram_gb = psutil.virtual_memory().total / (1024**3)
        cpu_count = psutil.cpu_count()
        system = platform.system()
        
        print(f"Détection machine: {system}, {cpu_count} cores, {ram_gb:.1f}GB RAM")
        
        # Profils adaptatifs
        if ram_gb >= 16 and cpu_count >= 8:  # Machine puissante
            return {
                'name': 'Haute performance',
                'whisper_model': 'medium',  # Au lieu de tiny
                'sentence_model': 'all-mpnet-base-v2',  # Plus précis
                'compute_type': 'float32'  # Meilleure précision
            }
        elif ram_gb >= 8 and cpu_count >= 4:  # Machine moyenne
            return {
                'name': 'Performance standard',
                'whisper_model': 'small',
                'sentence_model': 'all-MiniLM-L12-v2',
                'compute_type': 'float32'
            }
        else:  # Machine limitée (macOS ARM souvent)
            return {
                'name': 'Performance conservatrice',
                'whisper_model': 'tiny',
                'sentence_model': 'all-MiniLM-L6-v2',
                'compute_type': 'int8'
            }
    
    def calculate_embedding_similarity(self, emb1, emb2):
        """Calcule la similarité entre deux embeddings de lieux"""
        # Méthode simple basée sur les mots communs et caractéristiques visuelles
        parts1 = emb1.split('_')
        parts2 = emb2.split('_')
        
        # Comparer les mots-clés (partie texte)
        if len(parts1) >= 2 and len(parts2) >= 2:
            words1 = set(parts1[0].replace('[', '').replace(']', '').replace("'", '').split(', '))
            words2 = set(parts2[0].replace('[', '').replace(']', '').replace("'", '').split(', '))
            
            if words1 and words2:
                word_similarity = len(words1.intersection(words2)) / len(words1.union(words2))
                
                # Comparer les caractéristiques visuelles
                try:
                    brightness1, contrast1 = map(float, parts1[-1].split('_')[-2:])
                    brightness2, contrast2 = map(float, parts2[-1].split('_')[-2:])
                    
                    brightness_diff = abs(brightness1 - brightness2) / 255.0
                    contrast_diff = abs(contrast1 - contrast2) / 100.0
                    
                    visual_similarity = 1.0 - (brightness_diff + contrast_diff) / 2.0
                    
                    return (word_similarity * 0.7 + visual_similarity * 0.3)
                except:
                    return word_similarity
                
        return 0.0
    
    def get_dialogue_for_frame(self, transcription, frame_time_ms, start_offset_s=0):
        """Récupère les dialogues correspondant à une frame donnée"""
        if not transcription or not transcription.get('segments'):
            return {'original': '', 'french': ''}
        
        frame_time_s = (frame_time_ms / 1000.0) - start_offset_s
        dialogue_window = 2.0  # Fenêtre de 2 secondes autour de la frame
        
        dialogue_parts = []
        
        for segment in transcription['segments']:
            segment_start = segment['start']
            segment_end = segment['end']
            
            # Vérifier si le segment overlap avec la fenêtre de la frame
            if (segment_start <= frame_time_s + dialogue_window and 
                segment_end >= frame_time_s - dialogue_window):
                dialogue_parts.append(segment['text'].strip())
        
        original_text = ' '.join(dialogue_parts).strip()
        
        if original_text:
            # Traduire en français si nécessaire
            french_text = self.translate_text(
                original_text, 
                transcription['language'], 
                'fr'
            )
            return {
                'original': original_text,
                'french': french_text
            }
        
        return {'original': '', 'french': ''}
    
    def process_characters(self, description_fr):
        """Remplace les références génériques par des identifiants de personnages"""
        # Patterns pour hommes et femmes
        patterns = {
            r'\b(un homme|homme|l\'homme|cet homme)\b': 'HOMME',
            r'\b(une femme|femme|la femme|cette femme)\b': 'FEMME',
            r'\b(une personne|personne|la personne)\b': 'PERSO',
            r'\b(un enfant|enfant|l\'enfant)\b': 'ENFANT'
        }
        
        result = description_fr
        for pattern, replacement in patterns.items():
            if re.search(pattern, result, re.IGNORECASE):
                # Simuler une reconnaissance basique (à améliorer avec vrais embeddings)
                char_id = hash(result) % 5 + 1  # Basique, juste pour la démo
                result = re.sub(pattern, f'{replacement}#{char_id}', result, flags=re.IGNORECASE)
        
        return result
    
    def describe_scene(self, image):
        """Génère une description libre de la scène en français avec analyse"""
        inputs = self.processor(image, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=50, num_beams=3)
            description_en = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
        
        # Traduire en français
        try:
            description_fr = self.translator.translate(description_en)
        except:
            description_fr = description_en  # Fallback si traduction échoue
        
        # Traiter les personnages
        description_fr = self.process_characters(description_fr)
        
        # Détecter jour/nuit et lieu
        day_night = self.detect_day_night(image)
        location = self.detect_location(description_fr, image)
        
        return {
            'description': description_fr,
            'day_night': day_night,
            'location': location,
            'raw_en': description_en
        }

    def cluster_descriptions(self, scenes_data, n_clusters=8):
        """Groupe les descriptions similaires"""
        descriptions = [scene['description'] for scene in scenes_data]
        
        # Vectoriser les descriptions
        vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(descriptions)
        
        # Clustering
        kmeans = KMeans(n_clusters=min(n_clusters, len(descriptions)), random_state=42)
        clusters = kmeans.fit_predict(tfidf_matrix)
        
        # Ajouter les clusters aux données
        for i, scene in enumerate(scenes_data):
            scene['cluster'] = int(clusters[i])
        
        return scenes_data, vectorizer, kmeans

    def find_representative_scenes(self, scenes_data, vectorizer, kmeans):
        """Trouve les scènes représentatives de chaque cluster"""
        descriptions = [scene['description'] for scene in scenes_data]
        tfidf_matrix = vectorizer.transform(descriptions)
        
        representatives = {}
        
        for cluster_id in range(kmeans.n_clusters):
            # Trouver les scènes de ce cluster
            cluster_scenes = [i for i, scene in enumerate(scenes_data) if scene['cluster'] == cluster_id]
            
            if not cluster_scenes:
                continue
            
            # Calculer la similarité avec le centroïde
            cluster_center = kmeans.cluster_centers_[cluster_id].reshape(1, -1)
            similarities = cosine_similarity(tfidf_matrix[cluster_scenes], cluster_center).flatten()
            
            # Prendre la scène la plus proche du centre
            most_representative_idx = cluster_scenes[similarities.argmax()]
            representatives[cluster_id] = most_representative_idx
        
        return representatives

    def analyze_film(self, video_path, interval_seconds=30, start_time_s=0, end_time_s=None, output_file=None):
        """Analyse complète d'un film avec audio et vidéo"""
        if not os.path.exists(video_path):
            print(f"Erreur: Le fichier {video_path} n'existe pas")
            return None
        
        print(f"Analyse audio-visuelle de: {video_path}")
        
        # Extraire et transcrire l'audio
        print("Extraction et transcription de l'audio...")
        audio_path = self.extract_audio(video_path, start_time_s, end_time_s)
        transcription = None
        if audio_path:
            transcription = self.transcribe_audio(audio_path)
            if transcription:
                print(f"Audio transcrit - Langue détectée: {transcription['language']}")
                print(f"Longueur du texte: {len(transcription['text'])} caractères")
        
        # Extraire les frames
        frames_data = self.extract_frames_with_timecodes(video_path, interval_seconds, start_time_s, end_time_s)
        print(f"{len(frames_data)} frames extraites")
        
        # Analyser chaque frame avec contexte audio et déduplication intelligente
        frames_analysis = []
        previous_analysis = None
        
        for i, frame_data in enumerate(frames_data):
            print(f"Analyse frame {i+1}/{len(frames_data)} ({frame_data['timecode']})")
            
            # Vérifier si cette frame est redondante avec la précédente
            if previous_analysis and self.is_frame_redundant(frame_data['image'], previous_analysis.get('image')):
                print(f"  Frame similaire à la précédente, réutilisation avec mise à jour dialogue")
                
                # Réutiliser l'analyse visuelle, mettre à jour le dialogue
                frame_dialogue = self.get_dialogue_for_frame(transcription, frame_data['time_ms'], start_time_s)
                
                current_analysis = {
                    'timecode': frame_data['timecode'],
                    'time_ms': frame_data['time_ms'],
                    'description': previous_analysis['description'],
                    'day_night': previous_analysis['day_night'],
                    'location': previous_analysis['location'],
                    'raw_en': previous_analysis.get('raw_en', ''),
                    'dialogue_original': frame_dialogue.get('original', ''),
                    'dialogue_fr': frame_dialogue.get('french', ''),
                    'has_speech': bool(frame_dialogue.get('original', '').strip()),
                    'image': frame_data['image']  # Pour comparaison suivante
                }
            else:
                print(f"  Nouvelle analyse complète avec fusion multimodale")
                
                # Analyse visuelle complète
                visual_analysis = self.describe_scene(frame_data['image'])
                frame_dialogue = self.get_dialogue_for_frame(transcription, frame_data['time_ms'], start_time_s)
                
                # FUSION MULTIMODALE pour améliorer la compréhension
                enhanced_context = self.analyze_context_multimodal(
                    visual_analysis['description'],
                    frame_dialogue.get('french', ''),
                    frame_data['image']
                )
                
                # Utiliser les informations enrichies si confidence suffisante
                final_description = enhanced_context.get('enhanced_description', visual_analysis['description'])
                final_location = enhanced_context.get('refined_location') or visual_analysis['location']
                
                if enhanced_context['confidence'] > 0.6:
                    print(f"    Contexte détecté: {enhanced_context['context_type']} (conf: {enhanced_context['confidence']:.2f})")
                
                current_analysis = {
                    'timecode': frame_data['timecode'],
                    'time_ms': frame_data['time_ms'],
                    'description': final_description,
                    'day_night': visual_analysis['day_night'],
                    'location': final_location,
                    'raw_en': visual_analysis['raw_en'],
                    'dialogue_original': frame_dialogue.get('original', ''),
                    'dialogue_fr': frame_dialogue.get('french', ''),
                    'has_speech': bool(frame_dialogue.get('original', '').strip()),
                    'context_confidence': enhanced_context['confidence'],
                    'image': frame_data['image']
                }
            
            frames_analysis.append(current_analysis)
            previous_analysis = current_analysis
        
        # Analyse avec révision rétroactive des lieux
        frames_analysis = self.apply_retroactive_refinement(frames_analysis)
        
        # Détecter les scènes selon tes critères
        scenes = self.detect_scenes(frames_analysis)
        
        # Appliquer la numérotation intelligente des lieux
        scenes = self.apply_smart_location_numbering(scenes)
        
        # Clustering des descriptions pour les types de scènes
        print("\nRegroupement des types de scènes similaires...")
        scenes, vectorizer, kmeans = self.cluster_scenes(scenes)
        representatives = self.find_representative_scenes(scenes, vectorizer, kmeans)
        
        # Préparer les résultats
        results = {
            'film': video_path,
            'total_scenes': len(scenes),
            'clusters': kmeans.n_clusters,
            'scenes': scenes,
            'scene_types': {},
            'audio_language': transcription['language'] if transcription else 'Non détectée'
        }
        
        # Identifier les types de scènes
        for cluster_id, rep_idx in representatives.items():
            cluster_scenes = [s for s in scenes if s['cluster'] == cluster_id]
            results['scene_types'][f"Type {cluster_id + 1}"] = {
                'representative_description': scenes[rep_idx]['description'],
                'representative_timecode': scenes[rep_idx]['start_timecode'],
                'occurrences': len(cluster_scenes),
                'timecodes': [f"{s['start_timecode']} - {s['end_timecode']}" for s in cluster_scenes]
            }
        
        # Sauvegarder selon le format demandé
        if output_file:
            self.save_results(results, output_file)
        else:
            # Génération automatique du nom de fichier texte
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            output_file = f"{base_name}_analyse.txt"
            self.save_results(results, output_file)
        
        return results
    
    def save_results(self, results, filename):
        """Sauvegarde dans le format déterminé par l'extension"""
        extension = os.path.splitext(filename)[1].lower()
        
        if extension == '.json':
            self.save_as_json(results, filename)
        elif extension == '.yaml' or extension == '.yml':
            self.save_as_yaml(results, filename)
        elif extension == '.csv':
            self.save_as_csv(results, filename)
        else:
            self.save_as_text(results, filename)
    
    def save_as_json(self, results, filename):
        """Sauvegarde au format JSON"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nRésultats JSON sauvegardés dans: {filename}")
    
    def save_as_yaml(self, results, filename):
        """Sauvegarde au format YAML"""
        try:
            import yaml
            with open(filename, 'w', encoding='utf-8') as f:
                yaml.dump(results, f, default_flow_style=False, allow_unicode=True)
            print(f"\nRésultats YAML sauvegardés dans: {filename}")
        except ImportError:
            print("Erreur: PyYAML n'est pas installé. Utilisez: pip install PyYAML")
            # Fallback vers JSON
            json_filename = filename.replace('.yaml', '.json').replace('.yml', '.json')
            self.save_as_json(results, json_filename)
    
    def save_as_csv(self, results, filename):
        """Sauvegarde au format CSV (scènes uniquement)"""
        import csv
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            # En-tête
            writer.writerow([
                'Scene_Number', 'Start_Timecode', 'End_Timecode', 'Duration', 
                'Day_Night', 'Location', 'Description', 'Enriched_Summary',
                'Has_Dialogue', 'Dialogue_Count', 'Classification'
            ])
            
            # Données des scènes
            for scene in results['scenes']:
                writer.writerow([
                    scene['scene_number'],
                    scene['start_timecode'],
                    scene['end_timecode'], 
                    scene['duration'],
                    scene['day_night'],
                    scene['location'],
                    scene['description'],
                    scene.get('enriched_summary', ''),
                    scene.get('has_dialogue', False),
                    len(scene.get('dialogues_original', [])),
                    f"Type {scene['cluster'] + 1}"
                ])
        
        print(f"\nRésultats CSV sauvegardés dans: {filename}")
    
    def detect_scenes(self, frames_analysis):
        """Détecte les scènes avec critères améliorés incluant continuité de dialogue"""
        scenes = []
        current_scene_start = 0
        
        for i in range(1, len(frames_analysis)):
            prev_frame = frames_analysis[i-1]
            curr_frame = frames_analysis[i]
            
            # NOUVELL REGLE PRIORITAIRE : Continuité de dialogue
            dialogue_continuity = self.check_dialogue_continuity(prev_frame, curr_frame)
            
            if dialogue_continuity['is_continuous']:
                # Si dialogue continu, on ne change PAS de scène même avec changements visuels
                print(f"Continuité dialogue détectée: {dialogue_continuity['reason']}")
                continue
            
            # Critères visuels de changement de scène
            location_changed = self.extract_base_location(prev_frame['location']) != self.extract_base_location(curr_frame['location'])
            day_night_changed = prev_frame['day_night'] != curr_frame['day_night']
            
            # EXCEPTION : Changements INT/EXT ou JOUR/NUIT forcent un changement même avec dialogue
            major_context_change = day_night_changed or self.detect_major_location_change(
                prev_frame['location'], curr_frame['location']
            )
            
            if major_context_change:
                print(f"Changement majeur détecté: JOUR/NUIT ou INT/EXT")
                # Terminer la scène précédente
                scene_frames = frames_analysis[current_scene_start:i]
                if scene_frames:
                    scenes.append(self.create_scene_summary(scene_frames, current_scene_start))
                current_scene_start = i
                continue
            
            # Changement de lieu standard
            if location_changed:
                # Terminer la scène précédente
                scene_frames = frames_analysis[current_scene_start:i]
                if scene_frames:
                    scenes.append(self.create_scene_summary(scene_frames, current_scene_start))
                current_scene_start = i
        
        # Dernière scène
        if current_scene_start < len(frames_analysis):
            scene_frames = frames_analysis[current_scene_start:]
            scenes.append(self.create_scene_summary(scene_frames, current_scene_start))
        
        return scenes
    
    def extract_base_location(self, location_str):
        """Extrait le type de lieu de base (MAISON#1 -> MAISON)"""
        if '#' in location_str:
            return location_str.split('#')[0].replace('~', '')
        return location_str.replace('~', '')
    
    def detect_dialogue_break(self, prev_frame, curr_frame, all_frames, current_index):
        """Détecte une rupture dans la continuité du dialogue"""
        # Vérifier s'il y a une pause significative dans le dialogue
        silence_threshold = 3.0  # 3 secondes de silence
        
        prev_has_speech = prev_frame.get('has_speech', False)
        curr_has_speech = curr_frame.get('has_speech', False)
        
        # Si les deux frames n'ont pas de dialogue, pas de rupture
        if not prev_has_speech and not curr_has_speech:
            return False
        
        # Changement de présence de dialogue
        speech_pattern_changed = prev_has_speech != curr_has_speech
        
        # Vérifier la continuité thématique du dialogue
        if prev_has_speech and curr_has_speech:
            dialogue_similarity = self.calculate_description_similarity(
                prev_frame.get('dialogue_fr', ''),
                curr_frame.get('dialogue_fr', '')
            )
            theme_break = dialogue_similarity < 0.2
            return theme_break
        
        return speech_pattern_changed
    
    def calculate_description_similarity(self, desc1, desc2):
        """Calcule la similarité entre deux descriptions"""
        # Méthode simple basée sur les mots communs
        words1 = set(desc1.lower().split())
        words2 = set(desc2.lower().split())
        
        if not words1 or not words2:
            return 0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0
    
    def create_scene_summary(self, scene_frames, scene_index):
        """Crée un résumé enrichi de scène avec dialogues"""
        if not scene_frames:
            return None
        
        # Prendre les infos de la première frame pour l'en-tête
        first_frame = scene_frames[0]
        last_frame = scene_frames[-1]
        
        # Durée de la scène
        duration_ms = last_frame['time_ms'] - first_frame['time_ms']
        duration_str = self.format_duration(duration_ms)
        
        # Collecter tous les dialogues de la scène (avec dédoublonnage intelligent)
        dialogues_original = []
        dialogues_french = []
        
        for frame in scene_frames:
            dialogue_orig = frame.get('dialogue_original', '').strip()
            dialogue_fr = frame.get('dialogue_fr', '').strip()
            
            if dialogue_orig and not self.is_dialogue_duplicate(dialogue_orig, dialogues_original):
                dialogues_original.append(dialogue_orig)
                dialogues_french.append(dialogue_fr)
        
        # Description consolidée (prendre la plus représentative)
        descriptions = [f['description'] for f in scene_frames]
        main_description = descriptions[len(descriptions)//2]  # Prendre celle du milieu
        
        # Générer résumés multiples selon tes spécifications
        hyper_summary = self.generate_hyper_summary(
            main_description, dialogues_french, first_frame
        )
        
        enriched_summary = self.generate_enriched_summary_v2(
            main_description, dialogues_french, first_frame
        )
        
        # Analyser et organiser les dialogues avec détection de genre
        organized_dialogues = self.organize_dialogues_with_speakers(
            dialogues_original, dialogues_french, first_frame
        )
        
        return {
            'scene_number': scene_index + 1,
            'start_timecode': first_frame['timecode'],
            'end_timecode': last_frame['timecode'],
            'duration': duration_str,
            'day_night': first_frame['day_night'],
            'location': self.refine_location_with_context(first_frame, dialogues_french),
            'description': main_description,
            'hyper_summary': hyper_summary,
            'enriched_summary': enriched_summary,
            'organized_dialogues': organized_dialogues,
            'has_dialogue': bool(dialogues_original),
            'frame_count': len(scene_frames)
        }
    
    def generate_enriched_summary(self, visual_desc, dialogues, frame_info):
        """Génère un résumé enrichi combinant image et dialogue"""
        summary_parts = []
        
        # Contexte visuel
        location = frame_info['location']
        day_night = frame_info['day_night']
        summary_parts.append(f"Scène en {location} ({day_night.lower()})")
        
        # Description visuelle simplifiée
        if visual_desc:
            summary_parts.append(visual_desc)
        
        # Résumé du dialogue si présent
        if dialogues:
            dialogue_text = ' '.join(dialogues[:3])  # Limiter à 3 premiers dialogues
            if len(dialogue_text) > 150:
                dialogue_text = dialogue_text[:147] + "..."
            summary_parts.append(f"Dialogue: {dialogue_text}")
        
        return " - ".join(summary_parts)
    
    def generate_hyper_summary(self, visual_desc, dialogues, frame_info):
        """Génère un hyper-résumé ultra-concis de l'action"""
        location = frame_info['location']
        
        # Analyser le contexte sémantique si dialogues disponibles
        if dialogues:
            combined_dialogue = ' '.join(dialogues[:2])  # Limiter pour rapidité
            context_analysis = self.analyze_context_hybrid(
                combined_dialogue, combined_dialogue, 'fr'
            )
            primary_context = context_analysis.get('primary_context', '')
        else:
            primary_context = 'unknown'
        
        # Templates d'hyper-résumés basés sur contexte + lieu
        action_templates = {
            ('academic', 'ECOLE'): 'Professeur faisant un cours devant ses élèves',
            ('academic', 'BUREAU'): 'Réunion de travail ou présentation',
            ('medical', 'HOPITAL'): 'Consultation médicale ou intervention',
            ('social', 'BAR'): 'Conversation dans un bar',
            ('social', 'RESTAURANT'): 'Repas ou discussion au restaurant',
            ('domestic', 'CUISINE'): 'Scène domestique en cuisine',
            ('domestic', 'SALON'): 'Conversation familiale ou intime',
            ('business', 'BUREAU'): 'Réunion daffaires',
            ('transport', 'VOITURE'): 'Trajet en voiture',
            ('action', 'RUE'): 'Scène daction dans la rue'
        }
        
        base_location = self.extract_base_location(location)
        template_key = (primary_context, base_location)
        
        hyper_summary = action_templates.get(template_key)
        
        if not hyper_summary:
            # Fallback basé sur analyse visuelle
            hyper_summary = self.generate_fallback_hyper_summary(visual_desc, location)
        
        return hyper_summary
    
    def generate_fallback_hyper_summary(self, visual_desc, location):
        """Génère un hyper-résumé de fallback"""
        # Analyse simple de la description visuelle
        desc_lower = visual_desc.lower()
        
        if 'debout' in desc_lower and 'tableau' in desc_lower:
            return 'Présentation devant un public'
        elif 'assis' in desc_lower and 'table' in desc_lower:
            return 'Discussion autour dune table'
        elif 'marche' in desc_lower or 'marchant' in desc_lower:
            return 'Personne se déplaçant'
        elif 'conduire' in desc_lower or 'volant' in desc_lower:
            return 'Conduite de véhicule'
        else:
            base_loc = self.extract_base_location(location)
            return f'Scène en {base_loc.lower()}'
    
    def generate_enriched_summary_v2(self, visual_desc, dialogues, frame_info):
        """Version améliorée du résumé enrichi sans dialogues explicites"""
        summary_parts = []
        
        # Contexte visuel
        location = frame_info['location']
        day_night = frame_info['day_night']
        summary_parts.append(f"Scène en {location} ({day_night.lower()})")
        
        # Description visuelle simplifiée
        if visual_desc:
            summary_parts.append(visual_desc)
        
        # Résumé thématique du dialogue (pas le dialogue lui-même)
        if dialogues:
            dialogue_theme = self.summarize_dialogue_theme(dialogues)
            if dialogue_theme:
                summary_parts.append(f"Sujet: {dialogue_theme}")
        
        return " - ".join(summary_parts)
    
    def summarize_dialogue_theme(self, dialogues):
        """Résume le thème du dialogue sans le citer"""
        combined_text = ' '.join(dialogues)
        
        # Analyse sémantique pour déterminer le thème
        context_analysis = self.analyze_context_hybrid(
            combined_text, combined_text, 'fr'
        )
        
        primary_context = context_analysis.get('primary_context', '')
        confidence = context_analysis.get('confidence', 0)
        
        if confidence > 0.4:
            theme_mapping = {
                'academic': 'enseignement et recherche',
                'medical': 'santé et soins médicaux',
                'business': 'affaires et travail',
                'domestic': 'vie quotidienne',
                'social': 'conversation sociale',
                'action': 'situation durgence',
                'transport': 'voyage et déplacement'
            }
            return theme_mapping.get(primary_context, 'conversation générale')
        
        return 'conversation générale'
    
    def organize_dialogues_with_speakers(self, dialogues_original, dialogues_french, frame_info):
        """Organise les dialogues avec détection de genre des locuteurs optimisée"""
        if not dialogues_original:
            return {'speakers': [], 'original_block': '', 'french_block': ''}
        
        # Détecter les locuteurs avec analyse basique
        speakers_info = self.detect_speakers_basic(dialogues_french)
        
        # Organiser le bloc original avec locuteurs (pas de répétitions)
        original_with_speakers = self.format_dialogue_block(dialogues_original, speakers_info)
        
        # Organiser le bloc traduit avec locuteurs (pas de répétitions)
        french_with_speakers = self.format_dialogue_block(dialogues_french, speakers_info)
        
        return {
            'speakers': speakers_info,
            'original_block': original_with_speakers,
            'french_block': french_with_speakers
        }
    
    def format_dialogue_block(self, dialogues, speakers):
        """Formate un bloc de dialogue sans répétitions de locuteurs"""
        if not dialogues:
            return ''
        
        formatted_lines = []
        last_speaker = None
        
        for i, dialogue in enumerate(dialogues):
            speaker = speakers[i] if i < len(speakers) else 'INCONNU'
            
            if speaker != last_speaker:
                # Nouveau locuteur, l'afficher
                formatted_lines.append(f"{speaker}: {dialogue}")
                last_speaker = speaker
            else:
                # Même locuteur, pas de répétition
                formatted_lines.append(f"  {dialogue}")
        
        return '\n'.join(formatted_lines)
    
    def detect_speakers_basic(self, dialogues_french):
        """Détection basique des locuteurs avec logique de doute améliorée"""
        speakers = []
        
        # Compteurs pour suivi des personnages
        male_count = 0
        female_count = 0
        
        for dialogue in dialogues_french:
            # Analyse sémantique avec niveau de confiance
            gender_analysis = self.detect_speaker_gender_with_confidence(dialogue)
            gender = gender_analysis['gender']
            confidence = gender_analysis['confidence']
            
            if gender == 'male':
                male_count += 1
                speaker_id = f"HOMME#{male_count}" if male_count > 1 or self.will_have_multiple_males(dialogues_french) else "HOMME"
                
                # Ajouter tilde si confiance faible
                if confidence < 0.7:
                    speaker_id = f"~{speaker_id}"
                    
                speakers.append(speaker_id)
                
            elif gender == 'female':
                female_count += 1
                speaker_id = f"FEMME#{female_count}" if female_count > 1 or self.will_have_multiple_females(dialogues_french) else "FEMME"
                
                # Ajouter tilde si confiance faible
                if confidence < 0.7:
                    speaker_id = f"~{speaker_id}"
                    
                speakers.append(speaker_id)
                
            else:
                # Personnage non visible à l'écran (voix-off)
                speakers.append("INCONNU")
        
        return speakers
    
    def will_have_multiple_males(self, dialogues):
        """Prédit s'il y aura plusieurs hommes dans la scène"""
        male_count = 0
        for dialogue in dialogues:
            if self.detect_speaker_gender_with_confidence(dialogue)['gender'] == 'male':
                male_count += 1
                if male_count > 1:
                    return True
        return False
    
    def will_have_multiple_females(self, dialogues):
        """Prédit s'il y aura plusieurs femmes dans la scène"""
        female_count = 0
        for dialogue in dialogues:
            if self.detect_speaker_gender_with_confidence(dialogue)['gender'] == 'female':
                female_count += 1
                if female_count > 1:
                    return True
        return False
    
    def detect_speaker_gender_with_confidence(self, dialogue_text):
        """Détection du genre avec niveau de confiance"""
        text_lower = dialogue_text.lower()
        
        # Indices linguistiques basiques avec poids
        male_indicators = {
            'monsieur': 2, 'homme': 2, 'père': 2, 'frère': 2, 'fils': 2, 'mari': 2,
            'il': 1, 'lui': 1, 'son': 1, 'mec': 2, 'gars': 2
        }
        
        female_indicators = {
            'madame': 2, 'femme': 2, 'mère': 2, 'sœur': 2, 'fille': 2, 'épouse': 2,
            'elle': 1, 'sa': 1, 'nana': 2, 'meuf': 2
        }
        
        male_score = sum(weight for word, weight in male_indicators.items() if word in text_lower)
        female_score = sum(weight for word, weight in female_indicators.items() if word in text_lower)
        
        total_score = male_score + female_score
        
        if male_score > female_score and male_score > 0:
            confidence = male_score / max(total_score, 1)
            return {'gender': 'male', 'confidence': confidence}
        elif female_score > male_score and female_score > 0:
            confidence = female_score / max(total_score, 1)
            return {'gender': 'female', 'confidence': confidence}
        else:
            return {'gender': 'unknown', 'confidence': 0.0}
    
    def detect_speaker_gender(self, dialogue_text):
        """Version simplifiée pour rétrocompatibilité"""
        return self.detect_speaker_gender_with_confidence(dialogue_text)['gender']
    
    def is_dialogue_duplicate(self, new_dialogue, existing_dialogues):
        """Détecte si un dialogue est un vrai doublon (pas une continuation)"""
        new_dialogue_lower = new_dialogue.lower().strip()
        
        for existing in existing_dialogues:
            existing_lower = existing.lower().strip()
            
            # Véritable doublon : textes identiques ou l'un contient entièrement l'autre
            if new_dialogue_lower == existing_lower:
                return True
                
            # Cas : "Please finish Percival by next time." vs "Please finish Percival by next time. I know many..."
            if new_dialogue_lower in existing_lower or existing_lower in new_dialogue_lower:
                # Vérifier si c'est vraiment un début identique (pas juste des mots en commun)
                shorter = new_dialogue_lower if len(new_dialogue_lower) < len(existing_lower) else existing_lower
                longer = existing_lower if len(new_dialogue_lower) < len(existing_lower) else new_dialogue_lower
                
                # Si le plus court est exactement le début du plus long, c'est probablement un doublon
                if longer.startswith(shorter) and len(shorter) > 10:  # Au moins 10 caractères
                    return True
        
        return False
    
    def is_frame_redundant(self, current_image, previous_image, threshold=0.95):
        """Détermine si deux frames sont trop similaires pour éviter l'analyse redondante"""
        if previous_image is None:
            return False
            
        try:
            # Conversion en niveaux de gris et redimensionnement pour comparaison rapide
            current_gray = np.array(current_image.convert('L').resize((64, 64)))
            previous_gray = np.array(previous_image.convert('L').resize((64, 64)))
            
            # Calcul de similarité basique (corrélation)
            correlation = np.corrcoef(current_gray.flatten(), previous_gray.flatten())[0, 1]
            
            return correlation > threshold
        except:
            return False
    
    def analyze_context_multimodal(self, visual_description, dialogue_text, image):
        """Analyse combinée vision + audio + linguistique pour inférer le vrai contexte"""
        enhanced_info = {
            'enhanced_description': visual_description,
            'refined_location': None,
            'confidence': 0.0,
            'context_type': 'unknown'
        }
        
        # 1. INDICES VISUELS AVANCÉS
        visual_indicators = self.extract_advanced_visual_cues(visual_description, image)
        
        # 2. INDICES LINGUISTIQUES STRUCTURELS 
        linguistic_patterns = self.analyze_linguistic_structure(dialogue_text)
        
        # 3. FUSION ET INFÉRENCE
        context_analysis = self.infer_real_context(visual_indicators, linguistic_patterns, dialogue_text)
        
        # 4. ENRICHISSEMENT DE LA DESCRIPTION
        if context_analysis['confidence'] > 0.6:
            enhanced_info['enhanced_description'] = self.enhance_description_with_context(
                visual_description, context_analysis
            )
            enhanced_info['refined_location'] = context_analysis.get('inferred_location')
            enhanced_info['confidence'] = context_analysis['confidence']
            enhanced_info['context_type'] = context_analysis['type']
        
        return enhanced_info
    
    def extract_advanced_visual_cues(self, description, image):
        """Extrait des indices visuels avancés pour inférence contextuelle"""
        cues = {
            'has_blackboard': any(word in description.lower() for word in ['tableau', 'blackboard', 'board']),
            'has_desks': any(word in description.lower() for word in ['bureau', 'desk', 'table']),
            'multiple_people': any(word in description.lower() for word in ['gens', 'people', 'personnes', 'group']),
            'formal_setting': any(word in description.lower() for word in ['salle', 'room', 'auditorium']),
            'academic_objects': any(word in description.lower() for word in ['livre', 'book', 'paper', 'document']),
        }
        
        # Analyse de la luminosité pour déduire le type d'environnement
        gray = np.array(image.convert('L'))
        brightness = np.mean(gray)
        cues['bright_indoor'] = brightness > 120  # Éclairage artificiel fort
        cues['indoor_lighting'] = 80 < brightness < 180  # Éclairage intérieur typique
        
        return cues
    
    def analyze_linguistic_structure(self, dialogue_text):
        """Analyse la structure linguistique pour détecter des patterns contextuels"""
        if not dialogue_text.strip():
            return {'patterns': [], 'confidence': 0.0}
        
        text_lower = dialogue_text.lower()
        patterns = []
        
        # Patterns académiques
        if any(pattern in text_lower for pattern in [
            'finish .+ by next time', 'homework', 'assignment', 'chapter', 'page \\d+',
            'quiz', 'exam', 'test', 'study', 'lecture', 'class'
        ]):
            patterns.append(('academic', 0.8))
        
        # Patterns d'instruction (impératifs + deadline)
        if any(word in text_lower for word in ['please', 'finish', 'complete', 'read']) and \
           any(time_ref in text_lower for time_ref in ['next time', 'tomorrow', 'deadline']):
            patterns.append(('instructional', 0.9))
        
        # Patterns de questions pédagogiques
        if text_lower.startswith(('can anyone', 'who can', 'what is', 'how do', 'why')):
            patterns.append(('pedagogical_question', 0.7))
        
        # Patterns formels vs informels
        formal_indicators = ['please', 'would you', 'could you', 'thank you']
        informal_indicators = ['yeah', 'ok', 'sure', 'hey']
        
        formal_count = sum(1 for word in formal_indicators if word in text_lower)
        informal_count = sum(1 for word in informal_indicators if word in text_lower)
        
        if formal_count > informal_count:
            patterns.append(('formal_context', min(0.8, formal_count * 0.2)))
        
        return {
            'patterns': patterns,
            'confidence': max([conf for _, conf in patterns], default=0.0)
        }
    
    def infer_real_context(self, visual_cues, linguistic_patterns, dialogue_text):
        """Infère le vrai contexte en combinant tous les indices"""
        context_scores = defaultdict(float)
        
        # SCORING VISUEL
        if visual_cues['has_blackboard'] and visual_cues['multiple_people']:
            context_scores['classroom'] += 0.6
            
        if visual_cues['has_desks'] and visual_cues['formal_setting']:
            context_scores['classroom'] += 0.4
            context_scores['office_meeting'] += 0.3
        
        if visual_cues['bright_indoor'] and visual_cues['academic_objects']:
            context_scores['classroom'] += 0.3
        
        # SCORING LINGUISTIQUE
        for pattern, confidence in linguistic_patterns['patterns']:
            if pattern == 'academic':
                context_scores['classroom'] += confidence * 0.5
                context_scores['university_lecture'] += confidence * 0.4
            elif pattern == 'instructional':
                context_scores['classroom'] += confidence * 0.7
            elif pattern == 'pedagogical_question':
                context_scores['classroom'] += confidence * 0.6
            elif pattern == 'formal_context':
                context_scores['office_meeting'] += confidence * 0.3
                context_scores['classroom'] += confidence * 0.2
        
        # CONTEXTE FINAL
        if context_scores:
            best_context = max(context_scores.items(), key=lambda x: x[1])
            context_type, confidence = best_context
            
            # Mapping vers lieux
            location_mapping = {
                'classroom': 'ECOLE',
                'university_lecture': 'UNIVERSITE', 
                'office_meeting': 'BUREAU'
            }
            
            return {
                'type': context_type,
                'confidence': min(confidence, 1.0),
                'inferred_location': location_mapping.get(context_type, 'DÉCOR NON IDENTIFIÉ')
            }
        
        return {'type': 'unknown', 'confidence': 0.0, 'inferred_location': None}
    
    def enhance_description_with_context(self, original_description, context_analysis):
        """Enrichit la description avec le contexte inféré"""
        context_type = context_analysis['type']
        confidence = context_analysis['confidence']
        
        enhancements = {
            'classroom': 'Scène de cours avec professeur et étudiants',
            'university_lecture': 'Cours magistral en amphithéâtre universitaire', 
            'office_meeting': 'Réunion ou présentation en milieu professionnel'
        }
        
        if context_type in enhancements and confidence > 0.7:
            return f"{enhancements[context_type]} - {original_description}"
        
        return original_description
    
    def refine_location_with_context(self, frame, dialogues):
        """Affine la localisation avec le contexte du dialogue"""
        base_location = frame['location']
        
        if not dialogues:
            return base_location
        
        # Analyser le contexte sémantique
        combined_dialogue = ' '.join(dialogues)
        context_analysis = self.analyze_context_hybrid(
            combined_dialogue, combined_dialogue, 'fr'
        )
        
        primary_context = context_analysis.get('primary_context', '')
        confidence = context_analysis.get('confidence', 0)
        
        # Correction contextuelle pour éviter BUREAU quand c'est ECOLE
        if confidence > 0.5:
            context_location_mapping = {
                'academic': 'ECOLE',
                'medical': 'HOPITAL',
                'social': 'BAR' if 'bar' in combined_dialogue.lower() else 'RESTAURANT'
            }
            
            suggested_location = context_location_mapping.get(primary_context)
            if suggested_location:
                # Garder la numérotation si elle existe
                if '#' in base_location:
                    number = base_location.split('#')[1]
                    return f"{suggested_location}#{number}"
                else:
                    return f"{suggested_location}#1"
        
        return base_location
    
    def apply_smart_location_numbering(self, scenes):
        """Applique une numérotation intelligente des lieux (pas de #1 si unique)"""
        # Compter les occurrences de chaque type de lieu
        location_counts = defaultdict(int)
        
        for scene in scenes:
            location = scene.get('location', '')
            base_location = self.extract_base_location(location)
            if base_location and base_location != 'INCONNU':
                location_counts[base_location] += 1
        
        # Mettre à jour les scènes avec numérotation intelligente
        location_counters = defaultdict(int)
        
        for scene in scenes:
            location = scene.get('location', '')
            base_location = self.extract_base_location(location)
            
            if base_location and base_location != 'INCONNU':
                total_count = location_counts[base_location]
                
                if total_count == 1:
                    # Lieu unique, pas de numérotation
                    confidence_prefix = '~' if location.startswith('~') else ''
                    scene['location'] = f"{confidence_prefix}{base_location}"
                else:
                    # Lieux multiples, garder la numérotation
                    location_counters[base_location] += 1
                    confidence_prefix = '~' if location.startswith('~') else ''
                    scene['location'] = f"{confidence_prefix}{base_location}#{location_counters[base_location]}"
        
        return scenes
    
    def format_duration(self, duration_ms):
        """Formate la durée en format lisible"""
        if duration_ms < 1000:
            return "<1s"
        
        seconds = int(duration_ms / 1000)
        if seconds < 60:
            return f"{seconds}s"
        
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes}m{seconds:02d}s"
    
    def cluster_scenes(self, scenes, n_clusters=6):
        """Groupe les scènes similaires"""
        descriptions = [scene['description'] for scene in scenes]
        
        # Liste des mots vides français
        french_stop_words = ['le', 'de', 'un', 'et', 'est', 'en', 'avoir', 'que', 'pour', 
                           'dans', 'ce', 'il', 'une', 'sur', 'avec', 'ne', 'se', 'pas', 
                           'tout', 'plus', 'par', 'grand', 'elle', 'du', 'la', 'des', 
                           'au', 'aux', 'les', 'sa', 'son', 'ses', 'mes', 'tes', 'nos', 'vos']
        
        # Vectoriser les descriptions
        vectorizer = TfidfVectorizer(stop_words=french_stop_words, max_features=1000)
        tfidf_matrix = vectorizer.fit_transform(descriptions)
        
        # Clustering
        kmeans = KMeans(n_clusters=min(n_clusters, len(descriptions)), random_state=42)
        clusters = kmeans.fit_predict(tfidf_matrix)
        
        # Ajouter les clusters aux données
        for i, scene in enumerate(scenes):
            scene['cluster'] = int(clusters[i])
        
        return scenes, vectorizer, kmeans
    
    def load_spacy_model(self, language_code):
        """Charge ou télécharge automatiquement le modèle spaCy pour la langue"""
        if language_code in self.spacy_models:
            return self.spacy_models[language_code]
        
        model_mapping = {
            'en': 'en_core_web_sm',
            'fr': 'fr_core_news_sm', 
            'es': 'es_core_news_sm',
            'de': 'de_core_news_sm',
            'it': 'it_core_news_sm',
            'pt': 'pt_core_news_sm',
            'nl': 'nl_core_news_sm',
            'ja': 'ja_core_news_sm',
            'zh': 'zh_core_web_sm'
        }
        
        model_name = model_mapping.get(language_code)
        if not model_name:
            print(f"Pas de modèle spaCy pour {language_code}, utilisation de Sentence-Transformers uniquement")
            self.spacy_models[language_code] = None
            return None
        
        try:
            model = spacy.load(model_name)
            self.spacy_models[language_code] = model
            return model
        except OSError:
            print(f"Téléchargement du modèle spaCy {model_name}...")
            try:
                subprocess.run([sys.executable, "-m", "spacy", "download", model_name], 
                             check=True, capture_output=True)
                model = spacy.load(model_name)
                self.spacy_models[language_code] = model
                print(f"Modèle {model_name} installé avec succès!")
                return model
            except (subprocess.CalledProcessError, OSError) as e:
                print(f"Impossible d'installer {model_name}: {e}")
                self.spacy_models[language_code] = None
                return None
    
    def analyze_context_hybrid(self, dialogue_original, dialogue_fr, detected_language):
        """Analyse hybride spaCy + Sentence-Transformers pour déterminer le contexte"""
        # Étape 1: Analyse spaCy (si disponible)
        spacy_features = self.extract_spacy_features(dialogue_original, detected_language)
        
        # Étape 2: Analyse sémantique avec Sentence-Transformers
        semantic_scores = self.analyze_semantic_context(dialogue_fr, spacy_features)
        
        return {
            'primary_context': max(semantic_scores.items(), key=lambda x: x[1])[0],
            'confidence': max(semantic_scores.values()),
            'all_scores': semantic_scores,
            'spacy_entities': spacy_features.get('entities', []),
            'spacy_keywords': spacy_features.get('keywords', [])
        }
    
    def extract_spacy_features(self, text, language_code):
        """Extrait les caractéristiques avec spaCy"""
        nlp = self.load_spacy_model(language_code)
        if not nlp:
            return {'entities': [], 'keywords': [], 'pos_tags': []}
        
        doc = nlp(text)
        
        # Entités nommées
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        # Mots-clés (noms et adjectifs lemmatizés)
        keywords = [token.lemma_.lower() for token in doc 
                   if token.pos_ in ['NOUN', 'ADJ'] and not token.is_stop and len(token.text) > 2]
        
        # Tags POS pour analyse grammaticale
        pos_tags = [(token.text, token.pos_) for token in doc]
        
        return {
            'entities': entities,
            'keywords': keywords,
            'pos_tags': pos_tags
        }
    
    def analyze_semantic_context(self, dialogue_fr, spacy_features):
        """Analyse sémantique avec Sentence-Transformers enrichi par spaCy"""
        # Enrichir le texte avec les découvertes spaCy
        enriched_text = dialogue_fr
        
        if spacy_features['keywords']:
            enriched_text += " " + " ".join(spacy_features['keywords'])
        
        if spacy_features['entities']:
            entity_texts = [ent[0] for ent in spacy_features['entities']]
            enriched_text += " " + " ".join(entity_texts)
        
        # Encoder le texte enrichi
        text_embedding = self.sentence_model.encode(enriched_text)
        
        # Calculer similarités avec contextes de référence
        scores = {}
        for context_name, context_embedding in self.reference_contexts.items():
            similarity = np.dot(text_embedding, context_embedding) / (
                np.linalg.norm(text_embedding) * np.linalg.norm(context_embedding)
            )
            scores[context_name] = float(similarity)
        
        # Bonus pour entités spécifiques
        entity_bonuses = {
            'ORG': {'academic': 0.1, 'business': 0.1},  # MIT, entreprises
            'PERSON': {'academic': 0.05, 'medical': 0.05},  # Noms de personnes
            'GPE': {'transport': 0.1}  # Lieux géographiques
        }
        
        for entity_text, entity_type in spacy_features['entities']:
            if entity_type in entity_bonuses:
                for context, bonus in entity_bonuses[entity_type].items():
                    scores[context] = scores.get(context, 0) + bonus
        
        return scores
    
    def apply_retroactive_refinement(self, frames_analysis):
        """Applique une révision rétroactive des lieux basée sur le contexte futur"""
        print("Application de la révision rétroactive des lieux...")
        
        # PASSE 1: Sauvegarder les localisations initiales
        for frame in frames_analysis:
            frame['location_initial'] = frame['location']
            frame['confidence_initial'] = self.get_location_confidence(frame['location'])
        
        # PASSE 2: Révision avec contexte futur
        revised_count = 0
        
        for i, frame in enumerate(frames_analysis):
            # Analyser le contexte futur (fenêtre de 3-5 frames)
            future_context = self.get_future_context(frames_analysis, i, window=5)
            
            # Réviser seulement les lieux incertains (~INCONNU ou faible confiance)
            if frame['confidence_initial'] < 0.7 or '~INCONNU' in frame['location']:
                revised_location = self.revise_location_with_context(
                    frame, future_context, frames_analysis
                )
                
                if revised_location != frame['location']:
                    print(f"Révision frame {i}: {frame['location']} -> {revised_location}")
                    
                    # Réviser rétroactivement toute la séquence en cours
                    self.apply_retroactive_revision(frames_analysis, i, revised_location)
                    revised_count += 1
        
        print(f"Révisions rétroactives appliquées: {revised_count}")
        return frames_analysis
    
    def get_location_confidence(self, location_str):
        """Calcule le niveau de confiance d'une localisation"""
        if '~INCONNU' in location_str:
            return 0.0
        elif location_str.startswith('~'):
            return 0.4  # Faible confiance
        else:
            return 0.8  # Confiance élevée
    
    def get_future_context(self, frames_analysis, current_index, window=5):
        """Extrait le contexte des frames futures"""
        end_index = min(current_index + window + 1, len(frames_analysis))
        future_frames = frames_analysis[current_index + 1:end_index]
        
        context = {
            'locations': [f.get('location', '') for f in future_frames],
            'descriptions': [f.get('description', '') for f in future_frames],
            'dialogues': [f.get('dialogue_fr', '') for f in future_frames if f.get('dialogue_fr', '').strip()],
            'day_night': [f.get('day_night', '') for f in future_frames]
        }
        
        return context
    
    def revise_location_with_context(self, current_frame, future_context, all_frames):
        """Révise la localisation avec le contexte futur"""
        current_location = current_frame['location']
        
        # 1. Chercher des indices forts dans les futures localisations
        future_locations = [loc for loc in future_context['locations'] if not loc.startswith('~INCONNU')]
        
        if future_locations:
            # Prendre la localisation la plus fréquente et fiable
            location_counts = defaultdict(int)
            for loc in future_locations:
                base_loc = self.extract_base_location(loc)
                if not loc.startswith('~'):
                    location_counts[base_loc] += 2  # Bonus pour haute confiance
                else:
                    location_counts[base_loc] += 1
            
            if location_counts:
                most_likely_location = max(location_counts.items(), key=lambda x: x[1])[0]
                
                # 2. Analyse sémantique des dialogues pour confirmation
                if future_context['dialogues']:
                    combined_dialogue = ' '.join(future_context['dialogues'])
                    context_analysis = self.analyze_context_hybrid(
                        combined_dialogue, combined_dialogue, 'fr'
                    )
                    
                    # Vérifier la cohérence sémantique
                    if self.is_location_context_coherent(most_likely_location, context_analysis):
                        return f"{most_likely_location}#1"  # Nouvelle numérotation
        
        # 3. Analyser les descriptions visuelles futures
        future_descriptions = ' '.join(future_context['descriptions'])
        if future_descriptions:
            potential_location = self.detect_location_from_description(future_descriptions)
            if potential_location and potential_location != '~INCONNU':
                return potential_location
        
        # Pas de révision possible
        return current_location
    
    def is_location_context_coherent(self, location, context_analysis):
        """Vérifie la cohérence entre lieu et contexte sémantique"""
        coherence_mapping = {
            'ECOLE': ['academic'],
            'HOPITAL': ['medical'],
            'BUREAU': ['business', 'academic'],
            'BAR': ['social'],
            'RESTAURANT': ['social'],
            'CUISINE': ['domestic'],
            'SALON': ['domestic'],
            'VOITURE': ['transport']
        }
        
        expected_contexts = coherence_mapping.get(location, [])
        if not expected_contexts:
            return True  # Pas de contrainte
        
        # Vérifier si le contexte détecté correspond
        primary_context = context_analysis.get('primary_context', '')
        return primary_context in expected_contexts or context_analysis.get('confidence', 0) < 0.3
    
    def detect_location_from_description(self, description):
        """Détecte le lieu à partir d'une description visuelle enrichie"""
        # Utiliser la méthode existante mais sans image (fallback)
        # Créer une image factice pour l'interface
        from PIL import Image
        dummy_image = Image.new('RGB', (100, 100), color='white')
        
        return self.detect_location(description, dummy_image)
    
    def apply_retroactive_revision(self, frames_analysis, start_index, new_location):
        """Applique une révision rétroactive à une séquence de frames"""
        # Remonter pour trouver le début de la séquence incertaine
        revision_start = start_index
        
        # Chercher en arrière jusqu'aux frames avec faible confiance ou même contexte
        while revision_start > 0:
            prev_frame = frames_analysis[revision_start - 1]
            
            # S'arrêter si on trouve une frame avec haute confiance ET lieu différent
            if (prev_frame['confidence_initial'] > 0.7 and 
                self.extract_base_location(prev_frame['location']) != self.extract_base_location(new_location)):
                break
            
            # Continuer si faible confiance ou contexte similaire
            if (prev_frame['confidence_initial'] < 0.5 or 
                self.has_similar_context(prev_frame, frames_analysis[start_index])):
                revision_start -= 1
            else:
                break
        
        # Appliquer la révision à toute la séquence
        for i in range(revision_start, start_index + 1):
            frames_analysis[i]['location'] = new_location
            frames_analysis[i]['revised'] = True
    
    def has_similar_context(self, frame1, frame2):
        """Vérifie si deux frames ont un contexte similaire"""
        # Comparaison basique de contexte visuel et temporel
        same_day_night = frame1.get('day_night') == frame2.get('day_night')
        
        # Similarité de description
        desc_similarity = self.calculate_description_similarity(
            frame1.get('description', ''),
            frame2.get('description', '')
        )
        
        return same_day_night and desc_similarity > 0.3
    
    def check_dialogue_continuity(self, prev_frame, curr_frame):
        """Vérifie si un dialogue est continu entre deux frames"""
        prev_dialogue = prev_frame.get('dialogue_fr', '').strip()
        curr_dialogue = curr_frame.get('dialogue_fr', '').strip()
        
        if not prev_dialogue or not curr_dialogue:
            return {'is_continuous': False, 'reason': 'Pas de dialogue'}
        
        # 1. Même phrase qui continue (mots en commun significatifs)
        prev_words = set(prev_dialogue.lower().split())
        curr_words = set(curr_dialogue.lower().split())
        
        common_words = prev_words.intersection(curr_words)
        # Filtrer les mots vides
        meaningful_common = [w for w in common_words if len(w) > 3 and w not in ['avec', 'dans', 'pour', 'elle', 'mais', 'vous', 'nous']]
        
        if len(meaningful_common) >= 2:  # Au moins 2 mots significatifs en commun
            return {
                'is_continuous': True, 
                'reason': f'Mots communs: {meaningful_common}'
            }
        
        # 2. Suite logique de phrase (détection basique)
        if self.is_sentence_continuation(prev_dialogue, curr_dialogue):
            return {
                'is_continuous': True,
                'reason': 'Suite de phrase détectée'
            }
        
        return {'is_continuous': False, 'reason': 'Pas de continuité'}
    
    def is_sentence_continuation(self, prev_text, curr_text):
        """Détecte si curr_text continue prev_text"""
        # Indicateurs de continuation
        continuation_patterns = [
            # Prev se termine par une virgule, curr continue
            (prev_text.endswith(','), not curr_text[0].isupper() if curr_text else False),
            # Même thème technique/académique
            (self.has_academic_context(prev_text), self.has_academic_context(curr_text)),
            # Pronoms de reprise
            (True, curr_text.lower().startswith(('et ', 'mais ', 'donc ', 'car ', 'puis ')))
        ]
        
        return any(prev_cond and curr_cond for prev_cond, curr_cond in continuation_patterns)
    
    def has_academic_context(self, text):
        """Détecte un contexte académique avec analyse hybride simplifiée"""
        # Version allégée pour la continuité de dialogue
        text_embedding = self.sentence_model.encode(text)
        academic_similarity = np.dot(text_embedding, self.reference_contexts['academic']) / (
            np.linalg.norm(text_embedding) * np.linalg.norm(self.reference_contexts['academic'])
        )
        return academic_similarity > 0.4  # Seuil pour contexte académique
    
    def detect_major_location_change(self, prev_location, curr_location):
        """Détecte les changements majeurs INT/EXT"""
        # Mapping basique INT/EXT
        interior_locations = ['CUISINE', 'SALON', 'CHAMBRE', 'BUREAU', 'ECOLE', 'HOPITAL', 'BAR', 'RESTAURANT']
        exterior_locations = ['RUE', 'PARC', 'PLAGE', 'MONTAGNE', 'FORET']
        
        def get_location_type(location):
            base_loc = self.extract_base_location(location)
            if base_loc in interior_locations:
                return 'INT'
            elif base_loc in exterior_locations:
                return 'EXT'
            return 'UNKNOWN'
        
        prev_type = get_location_type(prev_location)
        curr_type = get_location_type(curr_location)
        
        # Changement INT/EXT = changement majeur
        return prev_type != curr_type and prev_type != 'UNKNOWN' and curr_type != 'UNKNOWN'
    
    def save_as_text(self, results, filename):
        """Sauvegarde les résultats avec analyse audio-visuelle complète"""
        with open(filename, 'w', encoding='utf-8') as f:
            # CHRONOLOGIE EN PREMIER
            f.write("=" * 80 + "\n")
            f.write("CHRONOLOGIE\n")
            f.write("=" * 80 + "\n")
            
            for scene in results['scenes']:
                f.write(f"\n{'='*40}\n")
                f.write(f"\nSCÈNE #{scene['scene_number']} ({scene['duration']})\n")
                f.write(f"{scene['day_night']} - {scene['location']} | {scene['start_timecode']} à {scene['end_timecode']}\n")
                f.write(f"{'='*40}\n")
                
                # HYPER-RÉSUMÉ
                hyper_summary = scene.get('hyper_summary', '')
                if hyper_summary and hyper_summary != 'Action non identifiée':
                    f.write(f"Hyper-résumé : {hyper_summary}\n\n")
                
                # RÉSUMÉ (sans redondance de lieu)
                enriched_summary = scene.get('enriched_summary', scene.get('description', ''))
                # Nettoyer le résumé des redondances de lieu
                location_base = self.extract_base_location(scene['location']).replace('DÉCOR NON IDENTIFIÉ', '')
                if location_base and location_base in enriched_summary:
                    enriched_summary = enriched_summary.replace(f"Scène en {location_base}", "").strip()
                    enriched_summary = enriched_summary.replace(f"en {location_base}", "").strip()
                    if enriched_summary.startswith("(nuit) – ") or enriched_summary.startswith("(jour) – "):
                        enriched_summary = enriched_summary[8:].strip()
                
                # Ne pas afficher le résumé s'il est identique au hyper-résumé ou vide
                if enriched_summary and enriched_summary != hyper_summary and len(enriched_summary) > 10:
                    f.write(f"Résumé : {enriched_summary}\n")
                
                # DIALOGUES (seulement si présents)
                organized_dialogues = scene.get('organized_dialogues', {})
                if organized_dialogues.get('original_block') and organized_dialogues['original_block'].strip():
                    f.write(f"\nDIALOGUE :\n")
                    f.write(f"{organized_dialogues['original_block']}\n")
                    
                    if organized_dialogues.get('french_block') and organized_dialogues['french_block'].strip():
                        f.write(f"\nTRADUCTION :\n")
                        f.write(f"{organized_dialogues['french_block']}\n")
            
            # STATISTIQUES EN FIN
            f.write("\n" + "=" * 80 + "\n")
            f.write(f"ANALYSE AUDIO-VISUELLE DU FILM : {results['film']}\n")
            f.write("=" * 80 + "\n")
            f.write(f"Nombre total de scènes détectées : {results['total_scenes']}\n")
            f.write(f"Nombre de types de scènes identifiés : {results['clusters']}\n")
            
            scenes_with_dialogue = sum(1 for s in results['scenes'] if s.get('has_dialogue', False))
            f.write(f"Scènes avec dialogue : {scenes_with_dialogue}/{results['total_scenes']}\n")
            
            if 'audio_language' in results:
                f.write(f"Langue détectée : {results['audio_language']}\n")
            
            # TYPES DE SCÈNES
            f.write("\n" + "=" * 80 + "\n")
            f.write("TYPES DE SCÈNES IDENTIFIÉS :\n")
            f.write("=" * 80 + "\n")
            
            for scene_type, data in results['scene_types'].items():
                f.write(f"\n{scene_type} ({data['occurrences']} occurrences) :\n")
                f.write(f"  Résumé : {data['representative_description']}\n")
                f.write(f"  Exemple à : {data['representative_timecode']}\n")
                f.write(f"  Périodes : {', '.join(data['timecodes'])}\n")
        
        print(f"\nRésultats sauvegardés dans: {filename}")

    def print_results(self, results):
        """Affiche un résumé des résultats (détails dans le fichier)"""
        print(f"\n{'='*70}")
        print(f"ANALYSE TERMINÉE: {results['film']}")
        print(f"{'='*70}")
        print(f"Nombre total de scènes détectées: {results['total_scenes']}")
        print(f"Nombre de types de scènes identifiés: {results['clusters']}")
        print(f"\nConsultez le fichier texte pour les détails complets.")

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

def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_film.py <video> [options]")
        print("Options:")
        print("  --interval SECONDS    : intervalle entre frames (défaut: 30)")
        print("  --start HH:MM:SS      : temps de début (défaut: 00:00:00)")
        print("  --end HH:MM:SS        : temps de fin (défaut: fin du film)")
        print("  --output FICHIER      : fichier de sortie (.txt par défaut, .json si spécifié)")
        print("\nExemples:")
        print("  python analyze_film.py film.mp4")
        print("  python analyze_film.py film.mp4 --start 10:00 --end 1:30:00 --interval 60")
        sys.exit(1)
    
    video_path = sys.argv[1]
    interval_seconds = 30
    start_time_s = 0
    end_time_s = None
    output_file = None
    
    # Parse des arguments
    i = 2
    while i < len(sys.argv):
        if sys.argv[i] == '--interval' and i + 1 < len(sys.argv):
            interval_seconds = int(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--start' and i + 1 < len(sys.argv):
            start_time_s = parse_time(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--end' and i + 1 < len(sys.argv):
            end_time_s = parse_time(sys.argv[i + 1])
            i += 2
        elif sys.argv[i] == '--output' and i + 1 < len(sys.argv):
            output_file = sys.argv[i + 1]
            i += 2
        else:
            print(f"Option inconnue: {sys.argv[i]}")
            sys.exit(1)
    
    analyzer = FilmAnalyzer()
    results = analyzer.analyze_film(video_path, interval_seconds, start_time_s, end_time_s, output_file)
    
    if results:
        pass

if __name__ == "__main__":
    main()
