# Analyse de film



## Pour lancer l’analyse

~~~zsh
source venv/bin/activate
 python detect_gendered_scenes_v4.py \
 		--start 0:20:00 \
 		--end 0:25:00 \
 		--interval 4 \
 		--output ./analyses \
 		./corpus-detection-gendered
~~~

