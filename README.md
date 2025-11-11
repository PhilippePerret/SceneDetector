# Scene Detector

*(« Détecteur de scène » en français)*

Projet d'application permettant de **détecter les scènes dans un film** et, partant, de pouvoir procéder à une analyse quantifiée précise sur un panel illimité de productions.

Dans l'idéal, elle doit : 

* savoir reconnaitre le début et la fin d'une scène,
* savoir en extraire les dialogues,
* savoir en extraire les caractéristiques (personnages, décors, effet — jour/nuit —, accessoires, etc.),
* savoir faire un résumé de la scène (action, objectif, personnages) en s'appuyant autant sur le visuel que sur le sonore (à commencer par les dialogues).

Bien que le « projet » soit simple en apparence, il est redoutable dans sa réalisation, eu égard aux difficultés techniques et *sémantique*. À commencer par la question « philosophique » prisedetêtique : *Qu'est-ce qu'un champ/contre-champ ?* À partir de quel moment peut-on savoir que deux plans qui s'enchainent, de nature très différente (fond différent, personnage différent, éclairage différent, etc.) forment-ils un champ/contre-champ ?

**Le champ/contre-champ est donc le premier défi à relever.** Trois jours de travail sur les premiers outils n'ont abouti à aucun résultat probant.

Pour le moment, aucun outil ne permet de le faire et les premiers essais ne sont vraiment pas concluant du tout. Toutes les IA essayées ont échoué à ce que j'appellerai le « Syndrome C/CC » (syndrome Champ/Contre-champ).

Les « débouchés » de cette application sont pourtant immenses, avec la possibilité d'analyser des milliers de films et d'en tirer des conclusions sur les structures (paradigme de Field augmenté, etc.), les scènes, les personnages. Donc possibilité d'une modélisation, par genre, par époque, etc., avec meilleur apprentissage à la clé.

## Outils approchés

Les outils approchés (liste non exhaustive) sont les suivants :

* [pillow](https://github.com/python-pillow)
* [YOLO (Ultralytics)](https://github.com/ultralytics/ultralytics)
* [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
* [ffmpeg-python](https://github.com/kkroening/ffmpeg-python)
* [sentence-transformers](https://github.com/huggingface/sentence-transformers)
* [torch torchvision](https://download.pytorch.org/whl/cpu)
* [deep-translator](https://github.com/nidhaloff/deep-translator)
* [spacy](https://github.com/explosion/spaCy)
* [opencv-python](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html)
* [transformers accelerate](https://github.com/huggingface/accelerate)
* [scikit-learn](https://www.intel.fr/content/www/fr/fr/goal/xeon-for-small-language-models.html?cid=sem&source=sa360&campid=2025_ao_emea_fr_comm_dco_dchqr_rep_awa_cons_txt_gen_broad_goog_is_intel_hq-learn-dcr-obs_fc25023&ad_group=Gen_SLM-Learn-AI-SLM-dco_b2b1-bp_Broad&intel_term=langages+de+programmation+ia&sa360id=2409405448983&gclsrc=aw.ds&gad_source=1&gad_campaignid=22372576055&gbraid=0AAAAA9YeV0yV7L7RVNNRcb6bIVhulBREH&gclid=EAIaIQobChMIgIOOs5bpkAMVaaX9BR20ehWpEAAYASAAEgKxMfD_BwE)

## Note sur le langage

L'embryon du projet fait la part belle à Python, mais c'est simplement parce que Claude (Sonnet 4) nous a entrainé dans cette voie. Un autre langage sera privilégié si on reprend sérieusement ce développement (tout simplement parce que je n'aime pas python…). 

Mais il semble que python soit incontournable car beaucoup d'outils concernant ces matières sont développés dans ce langage. Serait-ce l'occasion de se réconcilier ?…


## Pour lancer l’analyse

*(ne fonctionnera pas sans le fichier yolov8x.pt à la racine du dossier)*

*(ne produira de toutes façons rien de concluant — malgré le numéro de version trompeur…)*

~~~zsh
source venv/bin/activate
 python detect_gendered_scenes_v5.py \
 		--start 0:20:00 \
 		--end 0:25:00 \
 		--interval 4 \
 		--output ./analyses \
 		./corpus-detection-gendered
~~~

