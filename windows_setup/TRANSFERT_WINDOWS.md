# TRANSFERT VERS WINDOWS - INSTRUCTIONS

## Fichiers à copier sur la machine Windows

1. **analyze_film.py** - Script principal  
2. **lieux_config.json** - Configuration des lieux détectables
3. **install_windows.ps1** - Installation automatique PowerShell
4. **README_WINDOWS.md** - Documentation complète

## Installation rapide sur Windows

### Méthode 1 : Script PowerShell (recommandé)
```powershell
# Dans Warp/PowerShell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\install_windows.ps1
```

### Méthode 2 : Commandes manuelles
```cmd
# Créer le dossier
mkdir "%USERPROFILE%\Programmes\Analyses-films"
cd "%USERPROFILE%\Programmes\Analyses-films"

# Copier les 4 fichiers dans ce dossier

# Créer environnement virtuel  
python -m venv venv
venv\Scripts\activate.bat

# Installer dépendances
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install transformers accelerate opencv-python scikit-learn pillow deep-translator
pip install openai-whisper ffmpeg-python
```

## Usage identique sur Windows
```cmd
# Activer l'environnement
venv\Scripts\activate.bat

# Utiliser le script (même syntaxe qu'macOS)  
python analyze_film.py video.mp4 --start 2:15 --end 12:15 --interval 4 --output analyse.txt
```

## Différences Windows/macOS
- Activation env virtuel : `venv\Scripts\activate.bat` au lieu de `source venv/bin/activate`
- Séparateurs de chemin : `\` au lieu de `/`
- Le reste est identique
