@echo off
echo ============================================================
echo Installation de l'analyseur de films sur Windows
echo ============================================================

REM Créer le dossier de travail
if not exist "%USERPROFILE%\Programmes" mkdir "%USERPROFILE%\Programmes"
cd /d "%USERPROFILE%\Programmes"
if not exist "SceneDetector" mkdir "SceneDetector"
cd "SceneDetector"

echo Répertoire de travail: %CD%

REM Vérifier Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERREUR: Python n'est pas installé ou pas dans le PATH
    echo Installez Python depuis https://python.org
    pause
    exit /b 1
)

echo Python détecté, version:
python --version

REM Créer l'environnement virtuel
echo Création de l'environnement virtuel...
python -m venv venv

REM Activer l'environnement virtuel
echo Activation de l'environnement virtuel...
call venv\Scripts\activate.bat

REM Mettre à jour pip
python -m pip install --upgrade pip

REM Installer les dépendances
echo Installation des dépendances...
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install transformers accelerate
pip install opencv-python
pip install scikit-learn
pip install pillow
pip install deep-translator

echo ============================================================
echo Installation terminée!
echo ============================================================
echo Pour utiliser l'analyseur:
echo 1. Activez l'environnement: venv\Scripts\activate.bat
echo 2. Utilisez: python analyze_film.py video.mp4 [options]
echo ============================================================
pause
