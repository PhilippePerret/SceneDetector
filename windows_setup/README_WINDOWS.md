# Installation sur Windows

## Instructions pour Warp AI

Voici les étapes exactes pour reproduire l'installation sur Windows :

### 1. Prérequis
- Vérifier que Python est installé : `python --version`
- Si absent, installer depuis python.org

### 2. Installation automatique
```cmd
# Créer et naviguer vers le dossier
mkdir "%USERPROFILE%\Programmes\Analyses-films"
cd "%USERPROFILE%\Programmes\Analyses-films"

# Créer environnement virtuel
python -m venv venv
venv\Scripts\activate.bat

# Installer dépendances (version finale avec toutes les fonctionnalités)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install transformers accelerate
pip install opencv-python
pip install scikit-learn
pip install pillow
pip install deep-translator
pip install faster-whisper
pip install ffmpeg-python
pip install sentence-transformers
pip install spacy
pip install psutil
```

### 3. Copier le script
Le fichier `analyze_film.py` utilise maintenant BLIP simple au lieu de BLIP-2 pour éviter les erreurs de tenseurs sur CPU.

### 4. Usage
```cmd
# Activer l'environnement
venv\Scripts\activate.bat

# Utiliser le script
python analyze_film.py video.mp4 --start HH:MM:SS --end HH:MM:SS --interval SECONDS --output fichier.txt
```

### Fonctionnalités complètes (VERSION FINALE)
- **Transcription audio multilingue** avec Whisper (100+ langues)
- **Dialogues originaux + traduction française** automatique
- **Détection intelligente des scènes** basée sur audio + vidéo
- **Numérotation des lieux identiques** (HOPITAL#1, HOPITAL#2, etc.)
- **Résumés enrichis** combinant image et dialogue
- **Descriptions en français** avec reconnaissance des personnages
- **Détection JOUR/NUIT** et lieux configurables
- **Format de sortie complet** avec chronologie audio-visuelle

## Fichiers à copier
1. `analyze_film.py` (script principal)
2. `lieux_config.json` (configuration des lieux détectables)
3. `install_windows.ps1` (script d'installation PowerShell)
4. Ce README

### Alternative : Installation automatique
```powershell
# Dans PowerShell en tant qu'administrateur
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
# Puis exécuter le script d'installation
.\install_windows.ps1
```
