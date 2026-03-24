# YOLO_Nachhaltigkeit: Praesentationsgrundlage fuer NotebookLM

## 1. Projektueberblick

Dieses Projekt ist ein KI-basiertes Objekterkennungsprojekt fuer Nachhaltigkeit und Recycling. Der aktuelle Fokus liegt auf der Erkennung von Kartonverpackungen. Spaeter soll das Gesamtprojekt mit weiteren Materialklassen wie Metall und Plastik erweitert werden, damit am Ende eine gemeinsame Recycling-Loesung entsteht.

Wichtig ist: Dieses Repository ist nicht nur ein kleines Demo-Skript, sondern bereits ein strukturierter Workflow fuer Datenpruefung, Datensatzaufbereitung, Modelltraining, Inferenz und Edge-Export.

## 2. Ziel des Projekts

Das Ziel ist nicht, allgemeine Standardobjekte wie Personen, Stuehle oder Fernseher zu erkennen. Stattdessen soll ein eigenes Modell trainiert werden, das spezifische Kartonklassen aus dem eigenen Datensatz erkennt.

Der Vorteil davon ist:

- bessere Anpassung an den eigenen Anwendungsfall
- mehr Kontrolle ueber die Klassen
- spaetere Erweiterbarkeit fuer weitere Recycling-Materialien

## 3. Ausgangspunkt des Projekts

Laut Repository startete das Projekt mit einem CVAT-YOLO-Export. CVAT ist ein Tool, mit dem Bilder manuell annotiert werden koennen. Die Annotationen wurden im YOLO-Format exportiert und dann in dieses Projekt uebernommen.

Wichtige Pfade aus dem Projekt:

- Datensatzwurzel: `data/cvat_exports/caton_hause`
- Bildquelle: `data/images`
- Konfiguration: `configs/defaults.yaml`

Das bedeutet: Die Bilder liegen lokal im Projekt, und die Labels stammen aus einem exportierten CVAT-Datensatz.

## 4. Aktuelle benutzerdefinierte Klassen

Im Projekt sind aktuell vier eigene Klassen hinterlegt:

- `Milch_Karton_shokolade`
- `Miclh_Karton_Vanille`
- `Teeschachtel`
- `Cube_Karton`

Diese Klassen sind keine allgemeinen COCO-Klassen, sondern projektspezifische Kategorien. Genau deshalb reicht ein allgemeines Standardmodell nicht aus. Es braucht ein eigenes, nachtrainiertes Modell.

## 5. Warum ein Custom Model notwendig war

Ein vortrainiertes YOLO-Modell kennt meist Standardobjekte aus grossen oeffentlichen Datensaetzen. Fuer dieses Projekt reicht das nicht, weil hier sehr konkrete Verpackungsarten erkannt werden sollen.

Das Projekt loest dieses Problem mit einem Custom Model. Dabei wird nicht bei null begonnen, sondern ein vorhandenes YOLO-Modell als Basis verwendet und auf den eigenen Kartondatensatz feinabgestimmt.

Dieses Vorgehen nennt man:

- Transfer Learning
- Fine-Tuning

## 6. Welche KI-Methode wurde verwendet?

Die verwendete KI-Methode ist ueberwachtes Lernen fuer Objekterkennung mit Ultralytics YOLO.

Im Code ist sichtbar:

- als Basismodell wird `models/yolov8n.pt` verwendet
- darauf wird mit den eigenen Daten weitertrainiert
- das trainierte Ergebnis wird als bestes Modell gespeichert

Das bedeutet:

- Es wurde ein vorhandenes YOLO-Modell als Startpunkt genutzt.
- Danach wurde es mit eigenen gelabelten Beispielen fuer Kartonklassen angepasst.

## 7. Wurde KMeans verwendet?

Nein. Auf Basis des gescannten Repositorys wurde **kein KMeans** und auch kein anderer Clustering-Algorithmus als zentraler Trainingsansatz gefunden.

Wichtig fuer die Praesentation:

- Das Projekt verwendet **keine unueberwachte Clusteranalyse** als Hauptmethode.
- Das Projekt verwendet **YOLO fuer Objekterkennung**.
- Das Modell wurde durch **Fine-Tuning eines vortrainierten YOLO-Modells** erstellt.

Einfach erklaert:

- KMeans gruppiert Daten nach Aehnlichkeit.
- YOLO erkennt Objekte in Bildern und gibt Klassen, Positionen und Konfidenzen aus.

Dieses Projekt gehoert klar in die zweite Kategorie: Objekterkennung.

## 8. Datensatzpruefung und Datenqualitaet

Ein starker Teil des Projekts ist die Datensatzvalidierung. Vor dem eigentlichen Training wird geprueft, ob der Datensatz formal korrekt ist.

Laut Repository wurden folgende Punkte festgestellt:

- `train.txt` verweist auf 74 Bilder
- es existieren 65 Label-Dateien
- 9 Bilder haben keine Labels
- fuer das Training wird nur die gelabelte Teilmenge verwendet
- die Label-Dateien werden auf Struktur und Wertebereiche geprueft

Das ist sehr wichtig, weil schlechte oder unvollstaendige Daten direkt die Modellqualitaet verschlechtern koennen.

## 9. Wie die Labels geprueft werden

Die Datensatzlogik prueft unter anderem:

- ob ein Label-Ordner vorhanden ist
- ob `train.txt` oder ein Bildordner gefunden wird
- ob Labeldateien im richtigen YOLO-Format vorliegen
- ob pro Zeile genau 5 Werte vorhanden sind
- ob Klassen-IDs gueltig sind
- ob normalisierte Bounding-Box-Werte zwischen 0.0 und 1.0 liegen

Diese Validierung sorgt dafuer, dass fehlerhafte Annotationen nicht unbemerkt ins Training gehen.

## 10. Datensatzaufbereitung

Das Projekt erstellt aus dem rohen CVAT-Export einen sauberen Trainingsdatensatz.

Dabei passiert Folgendes:

1. Der Datensatz wird inspiziert.
2. Bilder ohne Labels werden ausgeschlossen.
3. Nur gueltige gelabelte Bilder werden uebernommen.
4. Es wird automatisch ein Train/Validation-Split erstellt.
5. Eine `data.yaml` fuer YOLO wird geschrieben.
6. Ein `dataset_report.yaml` wird erzeugt.

Damit entsteht ein reproduzierbarer Datensatz fuer das Training.

## 11. Train/Validation-Split

In der Konfiguration ist festgelegt:

- Validation Ratio: `0.2`
- Random Seed: `42`

Das bedeutet:

- 20 Prozent der gueltigen Daten werden fuer die Validierung genutzt
- 80 Prozent werden fuer das Training genutzt
- durch den festen Zufallswert kann die Aufteilung reproduzierbar wiederholt werden

Das ist ein wichtiger Punkt fuer wissenschaftliches und technisches Arbeiten.

## 12. Wie das Custom Model erstellt wurde

Der Trainingsablauf im Projekt sieht so aus:

1. Datensatz pruefen
2. Fehlende Labels erkennen
3. Nur gelabelte Beispiele fuer das Training verwenden
4. YOLO-kompatiblen Trainingsdatensatz erzeugen
5. Basismodell `yolov8n.pt` laden
6. Modell mit den eigenen Kartonklassen trainieren
7. Bestes Modell als Ergebnis abspeichern

Der wichtigste Output-Pfad ist:

- `runs/train/carton_detector_gpu/weights/best.pt`

Dieses `best.pt` ist das eigentliche Custom Model des Projekts.

## 13. Wichtige Trainingsparameter

Die Standardwerte aus der Konfiguration sind:

- Epochen: `50`
- Bildgroesse: `640`
- Batch Size: `8`
- Patience: `20`
- Workers: `2`
- Cache: `false`

Ausserdem gilt:

- wenn CUDA verfuegbar ist, wird bevorzugt die GPU verwendet
- sonst wird auf der CPU trainiert

Einfach erklaert:

- `epochs`: wie oft das Modell alle Trainingsdaten sieht
- `image size`: in welcher Groesse Bilder verarbeitet werden
- `batch size`: wie viele Bilder pro Trainingsschritt gemeinsam verarbeitet werden
- `patience`: wann das Training bei fehlender Verbesserung gestoppt werden kann

## 14. Inferenz: Was passiert nach dem Training?

Nach dem Training kann das Modell fuer Vorhersagen genutzt werden.

Das Projekt unterstuetzt:

- Bilder
- Videos
- Webcam
- externe Kamera

Bei der Inferenz passiert:

1. Das trainierte Modell wird geladen.
2. Ein Bild oder ein Videoframe wird verarbeitet.
3. Das Modell gibt erkannte Klassen zurueck.
4. Dazu kommen Bounding Boxes und Konfidenzwerte.
5. Das Bild kann annotiert angezeigt oder gespeichert werden.

## 15. Konfidenz und Ausgaben

Im Projekt ist standardmaessig ein Confidence Threshold von `0.25` konfiguriert.

Das bedeutet:

- nur Erkennungen mit ausreichender Sicherheit werden uebernommen
- jede Erkennung enthaelt Klassenname, Klassen-ID, Konfidenz und Bounding Box

Fuer die Praesentation kann man sagen:

"Das Modell erkennt nicht nur, was auf dem Bild ist, sondern auch, wo es ist und wie sicher es sich dabei ist."

## 16. Software-Architektur des Projekts

Das Repository ist modular aufgebaut. Wichtige Bereiche sind:

- `yolo_edge/cli.py`: Kommandozeilensteuerung
- `yolo_edge/config.py`: Laden der YAML-Konfiguration
- `yolo_edge/data/dataset_manager.py`: Datensatzpruefung und Datensatzaufbereitung
- `yolo_edge/training.py`: Trainingsworkflow
- `yolo_edge/core/detector.py`: Modell laden und Vorhersagen ausfuehren
- `yolo_edge/edge_export.py`: Export fuer Edge-Geraete
- `tests/`: automatisierte Tests

Das ist ein grosser Pluspunkt, weil das Projekt damit wartbar und erweiterbar bleibt.

## 17. Praktische Engineering-Verbesserungen

Im Repository wurden mehrere wichtige Software-Verbesserungen umgesetzt:

- saubere modulare Struktur
- Konfiguration ueber `configs/defaults.yaml`
- Subcommands fuer `predict`, `inspect-dataset`, `prepare-dataset`, `train` und `export`
- Reports fuer Datensatz und Training
- strukturiertes Logging
- Launcher-Kommandos wie `image`, `video`, `webcam`
- fruehe Abhaengigkeitspruefung beim Export

Das zeigt, dass hier nicht nur ein Modell trainiert wurde, sondern ein nutzbarer KI-Workflow aufgebaut wurde.

## 18. Edge Deployment und Raspberry Pi

Das Projekt denkt bereits an spaetere Nutzung auf Edge-Hardware, zum Beispiel auf einem Raspberry Pi.

Laut Konfiguration und Code werden Exportformate vorbereitet fuer:

- ONNX
- OpenVINO
- TFLite

Zusatzlich werden:

- eine `labels.txt`
- ein `edge_export_manifest.yaml`

erzeugt.

Das ist wichtig, wenn das Modell spaeter auf kleineren Geraeten laufen soll.

## 19. Tests im Projekt

Im Repository gibt es automatisierte Tests.

Diese pruefen unter anderem:

- CLI-Verhalten
- Datensatzvalidierung
- Ausschluss ungelabelter Bilder
- Erstellung eines vorbereiteten Datensatzes

Das ist wichtig, weil KI-Projekte nicht nur aus Modellen bestehen. Auch der Daten- und Software-Workflow muss stabil funktionieren.

## 20. Ehrliche Grenzen des aktuellen Stands

Fuer eine gute Praesentation ist Ehrlichkeit wichtig. Der aktuelle Stand hat auch Grenzen:

- der Datensatz ist noch relativ klein
- 9 Bilder haben keine Labels
- die Klassennamen sind nicht vollkommen konsistent geschrieben
- das Projekt ist aktuell auf Karton fokussiert
- im Repository ist noch kein vollstaendiger Evaluationsbefehl mit mAP, Precision, Recall und Confusion Matrix als eigener Workflow sichtbar

Das heisst: Das Projekt funktioniert, ist aber noch nicht die finale Komplettloesung.

## 21. Bedeutung fuer das Teamprojekt

Dieses Teilprojekt ist besonders wichtig, weil es als Vorlage fuer die anderen Materialbereiche dienen kann.

Wenn andere Teammitglieder Metall und Plastik bearbeiten, dann kann spaeter:

- ein gemeinsames Klassenschema definiert werden
- mehrere CVAT-Exporte vereinheitlicht werden
- ein groesserer kombinierter Recycling-Datensatz entstehen
- daraus ein gemeinsames Modell trainiert werden

Das ist auch so in den Empfehlungen des Repositorys angelegt.

## 22. Empfohlene naechste Schritte

Aus den Projektdateien lassen sich sinnvolle weitere Schritte ableiten:

- mehrere Datensaetze zusammenfuehren
- Klassennamen standardisieren
- gemeinsames Label-Schema fuer alle Teammitglieder festlegen
- Datensatzversionierung einfuehren
- Evaluationsworkflow mit Precision, Recall und mAP aufbauen
- Edge-Benchmark fuer Raspberry Pi ergaenzen
- Modell-Registry fuer verschiedene Modellversionen einfuehren

## 23. Fazit

Dieses Projekt hat bereits mehrere wichtige Dinge erfolgreich umgesetzt:

- Inspektion eines echten CVAT-Datensatzes
- automatische Datensatzbereinigung
- Nutzung nur gueltiger gelabelter Beispiele
- Fine-Tuning eines YOLO-Modells auf eigene Kartonklassen
- funktionierende Inferenz fuer Bilder, Videos und Kamera
- Vorbereitung fuer spaeteren Edge-Export

Das bedeutet: Es wurde nicht nur ein Modell ausprobiert, sondern eine solide technische Grundlage fuer ein groesseres Recycling-KI-System geschaffen.

## 24. Schlussgedanke fuer die Praesentation

Eine gute Abschlussaussage waere:

"Unser Karton-Projekt ist ein funktionierender erster Baustein fuer ein groesseres Nachhaltigkeitsprojekt. Wenn wir spaeter die Klassen fuer Metall und Plastik sauber ergaenzen und vereinheitlichen, kann daraus ein gemeinsames KI-System fuer Recycling-Erkennung entstehen."
