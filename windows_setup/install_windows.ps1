# Script d'installation pour l'analyseur de films sur Windows
# Compatible avec Warp Terminal

Write-Host "============================================================" -ForegroundColor Green
Write-Host "Installation de l'analyseur de films sur Windows" -ForegroundColor Green  
Write-Host "============================================================" -ForegroundColor Green

# Vérifier Python
try {
    $pythonVersion = python --version 2>$null
    Write-Host "Python détecté: $pythonVersion" -ForegroundColor Yellow
}
catch {
    Write-Host "ERREUR: Python n'est pas installé ou pas dans le PATH" -ForegroundColor Red
    Write-Host "Installez Python depuis https://python.org" -ForegroundColor Red
    Read-Host "Appuyez sur Entrée pour quitter"
    exit 1
}

# Créer le dossier de travail
$workDir = "$env:USERPROFILE\Programmes\SceneDetector"
if (-not (Test-Path $workDir)) {
    New-Item -ItemType Directory -Path $workDir -Force | Out-Null
}

Set-Location $workDir
Write-Host "Répertoire de travail: $workDir" -ForegroundColor Yellow

# Créer l'environnement virtuel
Write-Host "Création de l'environnement virtuel..." -ForegroundColor Yellow
python -m venv venv

# Activer l'environnement virtuel
Write-Host "Activation de l'environnement virtuel..." -ForegroundColor Yellow
& ".\venv\Scripts\Activate.ps1"

# Mettre à jour pip
Write-Host "Mise à jour de pip..." -ForegroundColor Yellow
python -m pip install --upgrade pip

# Installer les dépendances
Write-Host "Installation des dépendances..." -ForegroundColor Yellow
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

Write-Host "============================================================" -ForegroundColor Green
Write-Host "Installation terminée!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Pour utiliser l'analyseur:" -ForegroundColor Cyan
Write-Host "1. Activez l'environnement: .\venv\Scripts\Activate.ps1" -ForegroundColor White
Write-Host "2. Utilisez: python analyze_film.py video.mp4 [options]" -ForegroundColor White
Write-Host ""
Write-Host "Exemple:" -ForegroundColor Cyan  
Write-Host "python analyze_film.py film.mp4 --start 5:00 --end 25:00 --interval 10" -ForegroundColor White
Write-Host "============================================================" -ForegroundColor Green

Read-Host "Appuyez sur Entrée pour continuer"
