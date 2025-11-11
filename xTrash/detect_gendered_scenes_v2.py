#!/usr/bin/env python3

import cv2
import torch
from PIL import Image
import numpy as np
import sys
import os
from datetime import timedelta, datetime
from transformers import BlipProcessor, BlipForConditionalGeneration
from deep_translator import GoogleTranslator
import json
import re
from collections import defaultdict
from faster_whisper import WhisperModel
import argparse
import csv

# Supprimer le warning tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def _hms(seconds: int) -> str:
    """Convertit secondes en HH:MM:SS"""
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"

def _ms_to_hms(ms: int) -> str:
    """Convertit millisecondes en HH:MM:SS"""
    return _hms(int(round(ms / 1000)))

def _grab_frame_at_ms(video_path: str, time_ms: int):
    """Extrait une frame à un timestamp donné"""
    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_MSEC, time_ms)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        return None
    return frame

def _detect_shots(video_path: str, threshold: float = 0.5, min_shot_ms: int = 2000, start_ms: int = 0, end_ms: int = None):
    """Détecte les plans (changements de plan) par analyse d'histogrammes"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration_ms = int((total_frames / fps) * 1000)
    
    if end_ms is None or end_ms > video_duration_ms:
        end_ms = video_duration_ms

    print(f"Détection des plans de {_ms_to_hms(start_ms)} à {_ms_to_hms(end_ms)}")

    prev_hist = None
    shots = []
    shot_start_ms = start_ms

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cur_ms = int((frame_idx / fps) * 1000)
        frame_idx += 1
        
        if cur_ms < start_ms:
            continue
        if cur_ms > end_ms:
            break
            
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        hist = cv2.calcHist([gray], [0], None, [64], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        if prev_hist is not None:
            diff = cv2.compareHist(prev_hist.astype(np.float32), hist.astype(np.float32), cv2.HISTCMP_BHATTACHARYYA)
            if diff >= threshold:
                if cur_ms - shot_start_ms >= min_shot_ms:
                    mid = (shot_start_ms + cur_ms) // 2
                    shots.append((shot_start_ms, cur_ms, mid))
                    shot_start_ms = cur_ms
        prev_hist = hist

    # Dernier plan
    if end_ms - shot_start_ms >= min_shot_ms:
        mid = (shot_start_ms + end_ms) // 2
        shots.append((shot_start_ms, end_ms, mid))

    cap.release()
    if not shots:
        shots = [(start_ms, end_ms, (start_ms + end_ms)//2)]
    
    print(f"Détectés {len(shots)} plans")
    return shots

def _detect_day_night(frame_bgr) -> str:
    """Détecte JOUR/NUIT par luminosité moyenne"""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    avg = float(np.mean(gray))
    return "Jour" if avg > 90 else "Nuit"

def _detect_int_ext(frame_bgr) -> str:
    """Détecte INT/EXT par recherche de ciel (bleu) dans le tiers supérieur"""
    h, w, _ = frame_bgr.shape
    top = frame_bgr[: max(h // 3, 1), :, :]
    top_hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
    # Masque pour bleu clair (ciel)
    mask_blue = cv2.inRange(top_hsv, (90, 20, 120), (130, 80, 255))
    ratio_blue = float(np.count_nonzero(mask_blue)) / float(mask_blue.size)
    return "Ext" if ratio_blue > 0.03 else "Int"

def _count_persons_hog(frame_bgr) -> int:
    """Compte les personnes avec HOG detector (fallback robuste)"""
    try:
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        resized = cv2.resize(frame_bgr, None, fx=0.5, fy=0.5)
        rects, weights = hog.detectMultiScale(resized, winStride=(8, 8), padding=(8, 8), scale=1.05)
        return int(len(rects))
    except:
        return 0

class ScenePatternDetector:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Utilisation du device: {self.device}")
        
        # Traducteur
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
    
    def _initialize_scene_detectors(self):
        """Critères de détection basés sur les mots-clés audio (dialogues)"""
        return {
            'shopping': {
                'audio_keywords': ['buy', 'acheter', 'prix', 'price', 'store', 'magasin', 'shopping', 'boutique', 'essayer', 'try'],
                'min_duration': 15
            },
            'cuisine': {
                'audio_keywords': ['cook', 'cuisiner', 'recipe', 'recette', 'kitchen', 'cuisine', 'dinner', 'dîner', 'eat', 'manger'],
                'min_duration': 20
            },
            'soins_beaute': {
                'audio_keywords': ['hair', 'cheveux', 'makeup', 'maquillage', 'beautiful', 'belle', 'mirror', 'miroir'],
                'min_duration': 15
            },
            'emotion': {
                'audio_keywords': ['cry', 'pleure', 'sad', 'triste', 'tears', 'larmes', 'upset', 'bouleversé'],
                'min_duration': 10
            },
            'voiture': {
                'audio_keywords': ['drive', 'conduire', 'car', 'voiture', 'road', 'route', 'traffic', 'circulation'],
                'min_duration': 15
            },
            'sport_combat': {
                'audio_keywords': ['fight', 'combat', 'battle', 'bataille', 'game', 'match', 'sport', 'play'],
                'min_duration': 20
            },
            'bricolage': {
                'audio_keywords': ['fix', 'réparer', 'tool', 'outil', 'build', 'construire', 'repair', 'bricoler'],
                'min_duration': 15
            },
            'bar_alcool': {
                'audio_keywords': ['drink', 'boire', 'beer', 'bière', 'bar', 'pub', 'alcohol', 'alcool', 'wine', 'vin'],
                'min_duration': 15
            },
            'hierarchie': {
                'audio_keywords': ['boss', 'chef', 'order', 'ordre', 'command', 'commander', 'sir', 'monsieur', 'authority'],
                'min_duration': 10
            },
            'solitude': {
                'audio_keywords': ['alone', 'seul', 'lonely', 'solitude', 'quiet', 'silence', 'empty'],
                'min_duration': 20
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
            
            ffmpeg.run(out, overwrite_output=True)
            return temp_audio
            
        except Exception as e:
            print(f"Erreur extraction audio: {e}")
            return None
    
    def transcribe_audio(self, audio_path):
        """Transcrit l'audio avec Whisper"""
        if not self.whisper_model:
            return None
            
        try:
            print("Transcription audio...")
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
    
    def format_duration(self, duration_ms):
        """Formate une durée en format lisible"""
        seconds = int(duration_ms / 1000)
        if seconds < 60:
            return f"{seconds}s"
        minutes = seconds // 60
        seconds = seconds % 60
        return f"{minutes}m{seconds:02d}s"
    
    def analyze_film(self, video_path, interval_seconds=2, start_time_s=0, end_time_s=None, output_prefix=None):
        """Analyse complète d'un film avec la nouvelle pipeline"""
        if not os.path.exists(video_path):
            print(f"Erreur: Le fichier {video_path} n'existe pas")
            return None
        
        print(f"Analyse du film: {video_path}")
        
        # Extraire transcription audio
        transcription = None
        audio_path = self.extract_audio(video_path, start_time_s, end_time_s)
        if audio_path:
            transcription = self.transcribe_audio(audio_path)
        
        # Détecter les plans sur la période demandée
        start_ms = start_time_s * 1000
        end_ms = end_time_s * 1000 if end_time_s else None
        shots = _detect_shots(video_path, threshold=0.5, min_shot_ms=2000, start_ms=start_ms, end_ms=end_ms)
        
        # Construire la liste de toutes les scènes avec INT/EXT, JOUR/NUIT, personnes
        all_film_scenes = []
        for i, (s_ms, e_ms, m_ms) in enumerate(shots):
            print(f"Analyse plan {i+1}/{len(shots)} ({_ms_to_hms(s_ms)})")
            frame = _grab_frame_at_ms(video_path, m_ms)
            if frame is None:
                continue
            
            ie = _detect_int_ext(frame)
            dn = _detect_day_night(frame)
            persons = _count_persons_hog(frame)
            duration_ms = e_ms - s_ms
            
            all_film_scenes.append({
                'start_timecode': _ms_to_hms(s_ms),
                'end_timecode': _ms_to_hms(e_ms),
                'start_seconds': int(round(s_ms/1000)),
                'duration': self.format_duration(duration_ms),
                'duration_seconds': duration_ms/1000,
                'int_ext': ie,
                'day_night': dn,
                'persons': persons
            })
        
        # Détection des scènes genrées via transcription (mots-clés audio)
        detected_scenes = {scene_type: [] for scene_type in self.scene_detectors.keys()}
        
        if transcription and transcription.get('segments'):
            print("Détection des thèmes genrés via analyse audio...")
            for scene_type, criteria in self.scene_detectors.items():
                audio_kws = criteria.get('audio_keywords', [])
                min_dur = criteria.get('min_duration', 10)
                
                for sc in all_film_scenes:
                    s_sec = sc['start_seconds']
                    e_sec = s_sec + int(sc['duration_seconds'])
                    
                    # Chercher des dialogues qui matchent pendant ce plan
                    matched_keywords = []
                    for seg in transcription['segments']:
                        if seg['end'] < s_sec or seg['start'] > e_sec:
                            continue
                        txt = seg['text'].lower()
                        for kw in audio_kws:
                            if kw.lower() in txt:
                                matched_keywords.append(kw)
                    
                    if matched_keywords and sc['duration_seconds'] >= min_dur:
                        detected_scenes[scene_type].append({
                            'start_timecode': sc['start_timecode'],
                            'end_timecode': sc['end_timecode'],
                            'duration': sc['duration'],
                            'duration_seconds': sc['duration_seconds'],
                            'confidence': min(1.0, len(matched_keywords) * 0.3),
                            'keywords_found': matched_keywords
                        })
        
        # Calcul des métriques
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
        
        # Durée totale
        if all_film_scenes:
            total_duration = all_film_scenes[-1]['end_timecode']
        else:
            total_duration = "0:00:00"
        
        results = {
            'film': os.path.basename(video_path),
            'analysis_date': datetime.now().isoformat(),
            'total_duration': total_duration,
            'detected_scenes': detected_scenes,
            'metrics': metrics,
            'all_film_scenes': all_film_scenes
        }
        
        # Sauvegarder
        if output_prefix:
            self.save_results_formats(results, output_prefix, video_path)
        else:
            base_name = os.path.splitext(os.path.basename(video_path))[0]
            self.save_results_formats(results, f"{base_name}_analysis", video_path)
        
        return results
    
    def save_results_formats(self, results, output_prefix, video_path):
        """Sauvegarde en formats CSV, JSON, TXT, HTML"""
        
        # CSV
        self.save_as_csv(results, f"{output_prefix}.csv")
        
        # JSON
        self.save_as_json(results, f"{output_prefix}.json")
        
        # TXT
        self.save_as_text(results, f"{output_prefix}.txt")
        
        # HTML
        self.save_as_html(results, f"{output_prefix}.html", video_path)
        
        print(f"Résultats sauvegardés:")
        print(f"  - {output_prefix}.csv (données)")
        print(f"  - {output_prefix}.json (détails)")
        print(f"  - {output_prefix}.txt (lisible)")
        print(f"  - {output_prefix}.html (interactif)")
    
    def save_as_csv(self, results, filename):
        """Sauvegarde CSV avec toutes les scènes"""
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['timecode', 'duration', 'int_ext', 'day_night', 'persons', 'theme'])
            
            # Scènes genrées
            for scene_type, scenes in results['detected_scenes'].items():
                for scene in scenes:
                    writer.writerow([
                        scene['start_timecode'], 
                        scene['duration'], 
                        '', '', '',  # INT/EXT pas dans les genrées
                        scene_type.upper()
                    ])
            
            # Scènes non-genrées
            gendered_timecodes = set()
            for scenes in results['detected_scenes'].values():
                for scene in scenes:
                    gendered_timecodes.add(scene['start_timecode'])
            
            for scene in results['all_film_scenes']:
                if scene['start_timecode'] not in gendered_timecodes:
                    writer.writerow([
                        scene['start_timecode'],
                        scene['duration'],
                        scene['int_ext'],
                        scene['day_night'],
                        scene['persons'],
                        ''
                    ])
    
    def save_as_json(self, results, filename):
        """Sauvegarde JSON détaillé"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    
    def save_as_text(self, results, filename):
        """Sauvegarde TXT lisible"""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("=" * 60 + "\n")
            f.write(f"ANALYSE : {results['film']}\n")
            f.write("=" * 60 + "\n")
            f.write(f"Date : {results['analysis_date']}\n")
            f.write(f"Durée : {results['total_duration']}\n")
            f.write(f"Scènes détectées : {results['metrics']['total_scenes_detected']}\n\n")
            
            # Scènes genrées par thème
            for scene_type, scenes in results['detected_scenes'].items():
                if scenes:
                    f.write(f"{scene_type.upper()} - {len(scenes)} scène(s) :\n")
                    for scene in scenes:
                        f.write(f"  {scene['start_timecode']} → {scene['end_timecode']} ({scene['duration']})\n")
                    f.write("\n")
            
            # Résumé des scènes non-genrées
            non_gendered = len(results['all_film_scenes']) - results['metrics']['total_scenes_detected']
            f.write(f"Scènes non-genrées : {non_gendered}\n")
    
    def save_as_html(self, results, filename, video_path):
        """Sauvegarde HTML interactif avec CHRONOLOGIE"""
        film = os.path.basename(video_path)
        
        html = f'''<!DOCTYPE html>
<html lang="fr">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Analyse : {results['film']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; background: #f5f5f5; }}
        .header {{ background: white; padding: 20px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
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
        
        .video-container {{ text-align: center; margin-bottom: 20px; }}
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
        
        .scene-type {{ margin: 20px 0; border: 1px solid #ddd; border-radius: 4px; }}
        .scene-type h3 {{ background: #007acc; color: white; margin: 0; padding: 10px; cursor: pointer; }}
        .scene-type h3:hover {{ background: #005a99; }}
        .scene-type-empty {{ border-color: #ccc; }}
        .scene-type-empty h3 {{ background: #999; color: #666; }}
        .scene-type-all {{ border-color: #333; }}
        .scene-type-all h3 {{ background: #333; }}
        
        .scene-list {{ display: none; padding: 0; }}
        .scene-list.expanded {{ display: block; }}
        .scene-item {{ padding: 10px; border-bottom: 1px solid #eee; }}
        
        .timecode-link {{ 
            color: #007acc; font-weight: bold; cursor: pointer; 
            text-decoration: underline; margin-right: 10px;
        }}
        .timecode-link:hover {{ color: #005a99; }}
        
        .scene-duration {{ color: #666; font-style: italic; }}
        .confidence {{ color: #888; font-size: 0.9em; }}
        .no-scenes {{ color: #666; font-style: italic; padding: 20px; text-align: center; }}
        
        .gendered-scene {{ background-color: #fff3e0; border-left: 4px solid #ff9800; }}
        .natural-scene {{ background-color: #f3f3f3; border-left: 4px solid #9e9e9e; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Analyse : {results['film']}</h1>
    </div>
    
    <div class="main-container">
        <div class="video-panel">
            <div class="video-container">
                <video id="mainVideo" controls preload="metadata">
                    <source src="corpus-detection-gendered/{film}" type="video/mp4">
                    Votre navigateur ne supporte pas la balise vidéo.
                </video>
            </div>
            
            <div class="stats">
                <h2>Statistiques</h2>
                <p><strong>Date :</strong> {results['analysis_date'][:19].replace('T', ' ')}</p>
                <p><strong>Durée :</strong> {results['total_duration']}</p>
                <p><strong>Scènes détectées :</strong> {results['metrics']['total_scenes_detected']}</p>
            </div>
        </div>
        
        <div class="content-panel">
            <h2>Scènes par thème</h2>
'''
        
        # Scènes genrées par thème
        for scene_type, scenes in results['detected_scenes'].items():
            if scenes:
                count = len(scenes)
                total_duration = sum(scene.get('duration_seconds', 0) for scene in scenes)
                
                html += f'''
            <div class="scene-type">
                <h3 onclick="toggleSceneList('{scene_type}')" id="header-{scene_type}">
                    {scene_type.upper().replace('_', ' ')} ({count} scène{'s' if count > 1 else ''} - {total_duration:.1f}s) ▼
                </h3>
                <div class="scene-list" id="list-{scene_type}">
'''
                
                for scene in scenes:
                    start_seconds = self.timecode_to_seconds(scene['start_timecode'])
                    confidence = scene.get('confidence', 0)
                    keywords = ', '.join(scene.get('keywords_found', []))
                    
                    html += f'''
                    <div class="scene-item">
                        <span class="timecode-link" onclick="seekToTime({start_seconds})">{scene['start_timecode']}</span>
                        <span class="scene-duration">({scene['duration']})</span>
                        <span class="confidence">Conf: {confidence:.2f}</span>
                        {f"<br><small>Mots-clés: {keywords}</small>" if keywords else ""}
                    </div>
'''
                
                html += '''
                </div>
            </div>
'''
            else:
                html += f'''
            <div class="scene-type scene-type-empty">
                <h3>{scene_type.upper().replace('_', ' ')}</h3>
                <div class="no-scenes">Aucune scène détectée</div>
            </div>
'''
        
        # CHRONOLOGIE
        html += '''
            <div class="scene-type scene-type-all">
                <h3 onclick="toggleSceneList('chronologie')" id="header-chronologie">
                    CHRONOLOGIE ▲
                </h3>
                <div class="scene-list expanded" id="list-chronologie">
'''
        
        # Créer liste unifiée chronologique
        all_scenes_unified = []
        
        # Ajouter scènes genrées
        for scene_type, scenes in results['detected_scenes'].items():
            for scene in scenes:
                all_scenes_unified.append({
                    'start_timecode': scene['start_timecode'],
                    'start_seconds': self.timecode_to_seconds(scene['start_timecode']),
                    'duration': scene['duration'],
                    'theme': scene_type.upper().replace('_', ' '),
                    'is_gendered': True
                })
        
        # Ajouter scènes non-genrées
        gendered_timecodes = {scene['start_timecode'] for scenes in results['detected_scenes'].values() for scene in scenes}
        
        for scene in results['all_film_scenes']:
            if scene['start_timecode'] not in gendered_timecodes:
                all_scenes_unified.append({
                    'start_timecode': scene['start_timecode'],
                    'start_seconds': scene['start_seconds'],
                    'duration': scene['duration'],
                    'int_ext': scene['int_ext'],
                    'day_night': scene['day_night'],
                    'is_gendered': False
                })
        
        # Trier chronologiquement
        all_scenes_sorted = sorted(all_scenes_unified, key=lambda x: x['start_seconds'])
        
        # Afficher chronologie
        for scene in all_scenes_sorted:
            start_seconds = scene['start_seconds']
            style_class = 'gendered-scene' if scene['is_gendered'] else 'natural-scene'
            
            if scene['is_gendered']:
                display = f"{scene['theme']} ({scene['duration']})"
            else:
                display = f"{scene['int_ext']} - {scene['day_night']} ({scene['duration']})"
            
            html += f'''
                <div class="scene-item {style_class}">
                    <span class="timecode-link" onclick="seekToTime({start_seconds})">{scene['start_timecode']}</span>
                    <span class="scene-duration"> {display}</span>
                </div>
'''
        
        html += '''
                </div>
            </div>
        </div>
    </div>
    
    <script>
        const video = document.getElementById('mainVideo');
        
        function seekToTime(seconds) {
            try {
                video.currentTime = seconds;
                if (video.paused) {
                    video.play().catch(function(error) {
                        console.log('Lecture bloquée:', error);
                    });
                }
            } catch (error) {
                console.error('Erreur positionnement:', error);
            }
        }
        
        function toggleSceneList(sceneType) {
            const list = document.getElementById('list-' + sceneType);
            const header = document.getElementById('header-' + sceneType);
            
            if (!list || !header) return;
            
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
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html)
    
    def timecode_to_seconds(self, timecode):
        """Convertit HH:MM:SS en secondes"""
        parts = timecode.split(':')
        if len(parts) != 3:
            return 0
        try:
            hours, minutes, seconds = map(int, parts)
            return hours * 3600 + minutes * 60 + seconds
        except ValueError:
            return 0


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
    print(f"Début: {start_time.strftime('%d/%m/%Y %H:%M:%S')}")
    
    if not os.path.isdir(folder_path):
        print(f"Erreur: {folder_path} n'est pas un dossier")
        return
    
    video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}
    
    video_files = []
    for file in os.listdir(folder_path):
        if os.path.splitext(file.lower())[1] in video_extensions:
            video_files.append(os.path.join(folder_path, file))
    
    if not video_files:
        print(f"Aucun fichier vidéo dans {folder_path}")
        return
    
    print(f"Trouvé {len(video_files)} fichier(s):")
    for video in video_files:
        print(f"  - {os.path.basename(video)}")
    
    if output_dir is None:
        output_dir = os.path.join(folder_path, 'analyses')
    
    os.makedirs(output_dir, exist_ok=True)
    
    detector = ScenePatternDetector()
    all_results = []
    
    for i, video_path in enumerate(video_files, 1):
        print(f"\n{'='*50}")
        print(f"ANALYSE {i}/{len(video_files)}: {os.path.basename(video_path)}")
        print(f"{'='*50}")
        
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        output_prefix = os.path.join(output_dir, f"{base_name}_analysis")
        
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
                print(f"✓ Terminé: {results['metrics']['total_scenes_detected']} scènes")
            else:
                print(f"✗ Échec")
                
        except Exception as e:
            print(f"✗ Erreur: {e}")
    
    end_time = datetime.now()
    duration = end_time - start_time
    
    print(f"\n{'='*50}")
    print(f"TERMINÉ - {len(all_results)}/{len(video_files)} succès")
    print(f"Durée: {str(duration).split('.')[0]}")
    print(f"Résultats dans: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Détection de scènes genrées (v2)')
    parser.add_argument('path', help='Fichier vidéo ou dossier')
    parser.add_argument('--interval', type=int, default=4, help='Intervalle (défaut: 4s)')
    parser.add_argument('--start', type=str, help='Temps début (HH:MM:SS)')
    parser.add_argument('--end', type=str, help='Temps fin (HH:MM:SS)')
    parser.add_argument('--output', type=str, help='Préfixe/dossier sortie')
    
    args = parser.parse_args()
    
    start_time_s = 0
    end_time_s = None
    
    if args.start:
        start_time_s = parse_time(args.start)
    
    if args.end:
        end_time_s = parse_time(args.end)
    
    if os.path.isdir(args.path):
        analyze_folder(args.path, args.interval, args.output, start_time_s, end_time_s)
    elif os.path.isfile(args.path):
        detector = ScenePatternDetector()
        results = detector.analyze_film(
            args.path, 
            interval_seconds=args.interval,
            start_time_s=start_time_s, 
            end_time_s=end_time_s,
            output_prefix=args.output
        )
        
        if results:
            print(f"Analyse terminée: {results['metrics']['total_scenes_detected']} scènes")
    else:
        print(f"Erreur: {args.path} introuvable")


if __name__ == "__main__":
    main()
