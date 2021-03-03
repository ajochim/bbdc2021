# FFT Python Skript

Das beiliegende Python3-Skript erlaubt es euch, die vorliegenden  ```.wav```
Dateien einfach in den Frequenzbereich zu transformieren und das Ergebnis als
Startpunkt für eure Features zu nutzen. Es ist nicht notwendig, das Skript zu
nutzen, und die Features sind nicht zwangsläufig ideal - wir möchten euch damit
den Einstieg in das Arbeiten mit den BBDC-2021-Daten vereinfachen.


## Transformation der Daten

1. Stellt zunächst sicher, dass python3 installiert ist oder ladet es herunter:
``` python3 --version```
https://www.python.org/downloads/

2. (Optional) Erstellt eine virtuelle Umgebung, damit die Pakete nicht global installiert werden. ```pthon3 -m venv Myvenv```
Mehr Informationen unter: https://docs.python.org/3/tutorial/venv.html  
Aktiviert diese:
    <br/>2.1 Unix: ```source Myvenv/bin/activate/```
    <br/>2.2. Windows: Benutzt ```activate.bat```

3. Abhängigkeiten installieren:
```pip install -r requirements```

4. Anpassen der Variablen im Skript:
Öffnet ```calc_fft.py``` in einem beliebigen Code Editor und passt die ```dataset_loc``` Variable an.

5. Skript starten:
```python3 calc_fft.py```


## Skript für Features nutzen

1. Python und Abhängigkeiten wie oben beschrieben installieren.

2. Importiert die ```load_and_calc_features``` Funktion aus dem ```calc_fft.py``` Skript.

3. Übergebt die zu ladenden Dateien und nutzt die beiden zurückgegeben Dictionaries als Features.  
