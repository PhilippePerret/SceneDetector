#!/usr/bin/env python3
"""
Test rapide du multi_model_detector
"""

import sys
import os
from pathlib import Path

# Test imports
try:
    from multi_model_detector import MultiModelDetector
    print("✅ Import multi_model_detector OK")
except ImportError as e:
    print(f"❌ Erreur import: {e}")
    sys.exit(1)

# Vérifier qu'on a une vidéo de test
test_videos = [
    "_films-done/Harold and Maude (1971).mp4",
    "_films-done/HaroldEtMaude.mp4",
    "test.mp4",
    "sample.mp4"
]

video_path = None
for video in test_videos:
    if Path(video).exists():
        video_path = video
        break

if not video_path:
    print("⚠️  Aucune vidéo de test trouvée")
    print("Créez un fichier test.mp4 ou sample.mp4 dans ce dossier")
    sys.exit(1)

print(f"📹 Vidéo de test: {video_path}")

# Créer le détecteur
detector = MultiModelDetector(video_path)

# Charger les modèles
print("\n🔧 Chargement des modèles...")
detector.load_models()

# Test sur 10 secondes seulement
print("\n🎬 Test d'analyse (10 premières secondes)...")
results = detector.process_video(
    start_time=0,
    end_time=10,
    interval=5,  # Une frame toutes les 5 secondes
    output_dir="test_output"
)

if results:
    print(f"\n✅ Test réussi! {len(results)} frames analysées")
    print(f"📊 Première frame:")
    first = results[0]
    print(f"   - Timestamp: {first.get('timestamp_str')}")
    print(f"   - Personnes: {first.get('person_count', 0)}")
    print(f"   - Contexte: {first.get('context', {}).get('location', 'inconnu')}")
else:
    print("\n❌ Aucun résultat obtenu")

print("\n✨ Test terminé")
