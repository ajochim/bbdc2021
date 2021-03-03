# Custom Evaluation Notebook

Das beiliegende Jupyter Notebook oder Python Skript erlauben es euch, eure
eigenen Development und Validation Splits innerhalb der gelabelten Daten zu
evaluieren. Dies kann beispielsweise für Parametertuning oder als
Performanzabschätzung von Interesse sein. 

Für die Evaluierung müsst ihr die ```dev``` Dateien in zwei Gruppen teilen, eine
auf der ihr trainiert und eine, auf der ihr für euch und unabhängig von dem BBDC
Upload System evaluiert (d.h. mit diesem Notebook und Python Skript verbraucht ihr
keine eurer Tokens). Schneidet zunächst eure dev-label.csv Datei an einer
beliebigen Stelle. Der erste Teil ist eure Trainingsgruppe, der zweite eure
Validierungsgruppe. Trainiert nun mit den Trainingsdateien und macht eine
Vorhersage auf den Validierungsdateien (von denen ihr ja die Grundwahrheit
kennt). Anschließend ladet eure Vorhersage und die Validierungsdatei in das
Notebook oder das Python Skript. Ihr bekommt dann euren PSDS Score ausgegeben.

## Jupyter Notebook

Eine einfache Möglichkeit ohne Installation, ist das Notebook in einem online
Service auszuführen (bspw. Binder, Kaggle Kernel, Google Colab, MS Azure
Notebooks, oder Datalore). Ladet dazu die Dateien in den jeweiligen Service und
nutzt dann die zwei Upload Felder innerhalb des Notebooks.

## Python Datei

Alternativ könnt ihr auch die Python Datei direkt lokal bei euch nutzen.
Installiert dazu analog zu dem FFT Skript die Abhängigkeiten aus der
requirements.txt. Importiert dann das Python Skript innerhalb von Python und
übergebt der evaluate Funktion den Pfad zu eurer Vorhersage- und
Grundwahrheitsdatei.  
