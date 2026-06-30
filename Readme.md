# PersonLocator – Bachelorarbeit

Multi-Kamera-System zur **3D-Lokalisierung von Personen** in Innenräumen. Mehrere Kamera-Knoten erkennen 2D-Posen lokal mit YOLO26-pose (OpenVINO bzw. ONNX). Ein zentraler Server fusioniert die Detektionen per Epipolargeometrie zu 3D-Skeletten, verfolgt Personen-IDs und visualisiert die Szene in Echtzeit.

Bachelorarbeit am **Lehrstuhl für Datenbanken und Informationssysteme** der **Heinrich-Heine-Universität Düsseldorf** (Prof. Dr. Stefan Conrad).

---

## Inhalt

- [Architektur](#architektur)
- [Tech-Stack](#tech-stack)
- [Setup](#setup)
- [Ausführung](#ausführung)
- [Projektstruktur](#projektstruktur)
- [Datenbank](#datenbank)
- [Netzwerkprotokoll](#netzwerkprotokoll)
- [Runtime-Mode-Switches](#runtime-mode-switches)
- [Evaluation](#evaluation)
- [Autor](#autor)

---

## Architektur

Zwei getrennte Prozesse kommunizieren per TCP/msgpack:

- **Server** (`personLocator/MainStarterServer.py`) – läuft auf einer Maschine mit dedizierter GPU. Empfängt 2D-Pose-Daten aller Kameras, fusioniert sie zu 3D-Skeletten, führt ID-Tracking durch, rendert die 3D-Szene und schreibt Evaluationsdaten.
- **Client** (`personLocator/MainStarterClient.py`) – läuft auf jedem Kamera-Knoten (Intel Edge Node mit OpenVINO oder GPU-Host mit ONNX). Erfasst Frames, schätzt Posen mit YOLO26-pose und schickt Keypoints, Bounding Boxes und Farbmerkmale an den Server.

### Datenfluss Server

```
Kamera-Clients
  -> ServerConnector              (TCP/msgpack, length-prefixed)
  -> ServerCommandHandler         (Message-Routing)
  -> ServerController.update_camera_view_logic()
  -> CalibrationRenderer.process_frame()
       1. EpipolarClusterer       - gruppiert 2D-Detektionen ueber Epipolargeometrie
       2. SkeletonTriangulator    - Triangulation pro Joint (WLS oder LM)
       3. PersonIDTracker         - Deep-SORT-Light (Hungarian + Color + 3D) oder Greedy
       4. SkeletonPostProcessor   - Outlier-Filter -> Kalman/1-Euro -> FABRIK-IK -> CoM-Clamp
  -> ServerDashboard              (PyQt6 GUI mit OpenGL-3D-View)
  -> CSV-Logging                  (eval_session_*.csv, eval_log_Block_*.csv)
```

### Koordinatensystem

- Alle 3D-Positionen sind in **Zentimetern**.
- Ursprung über die SQPnP-Kalibrierung an den Eckpunkten eines Welt-Rechtecks (`world_rectangles` + `rectangle_corners_3d`) definiert.
- Kamera-Weltposition: `-R^T · t`.
- Standardraum: **320 × 250 × 470 cm** (Default-Werte in `global_room`).

---

## Tech-Stack

| Komponente             | Verwendung                                                                                |
| ---------------------- | ----------------------------------------------------------------------------------------- |
| Python 3.12            | Sprache                                                                                   |
| PyQt6                  | Server-GUI, OpenGL-3D-View                                                                |
| YOLO26-pose            | 2D-Pose am Client (Ultralytics; OpenVINO **und** ONNX; Varianten `n` / `s` / `m` / `l`)   |
| OpenVINO               | Inferenz-Beschleunigung auf Intel-Hardware                                                |
| OpenCV                 | Kamerakalibrierung, Bildverarbeitung, SQPnP                                               |
| PostgreSQL + psycopg2  | Persistenz (Kalibrierung, Marker, Konfig)                                                 |
| msgpack                | binäres Wire-Format                                                                       |
| Numba                  | JIT-Beschleunigung des WLS-Solvers (`fast_wls_solve` in `SkeletonTriangulator.py`)        |
| python-dotenv          | Zugangsdaten aus `.env`                                                                   |

---

## Setup

### 1. Repository klonen

```bash
git clone https://github.com/JzNS/PersonenLocatorBachelorArbeit.git
cd PersonenLocatorBachelorArbeit
```

### 2. Virtuelle Umgebung

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
pip install -r requirements.txt
```

### 3. PostgreSQL

PostgreSQL lokal starten und Datenbank `personLocatorSystem` anlegen. Das Schema wird beim ersten Start des Servers automatisch erzeugt (`personLocator/server/core/database/SystemDatabase.py`).

### 4. `.env` anlegen

Datei: `personLocator/.env`

```env
DB_NAME=personLocatorSystem
DB_USER=postgres
DB_PASS=DEIN_PASSWORT
DB_HOST=127.0.0.1
```

Die `.env` ist in `.gitignore` ausgeschlossen.

### 5. YOLO26-Modelle

Erwartete Pfade pro Client:

- `personLocator/client/config/yolo26n-pose_openvino_model/` – OpenVINO-Variante (Intel-Edge-Node)
- `personLocator/client/config/graka/yolo26n-pose.onnx` – ONNX-Variante (GPU-Host)

Im Client-Settings-Dialog stehen zusätzlich `yolo26s-pose.onnx`, `yolo26m-pose.onnx` und `yolo26l-pose.onnx` zur Wahl.

---

## Ausführung

**Server starten:**

```bash
cd personLocator
python MainStarterServer.py
```

**Client starten** (auf jedem Kamera-Knoten):

```bash
cd personLocator
python MainStarterClient.py
```

Clients finden den Server über einen UDP-Beacon (`ServerBeacon.py`) und registrieren sich mit der Aktion `REGISTER` (`ClientCommandSender.send_register`).

---

## Projektstruktur

```
personLocator/
├── MainStarterServer.py            # Server-Einstiegspunkt
├── MainStarterClient.py            # Client-Einstiegspunkt
├── evaluate.py                     # Standalone-Auswertung der CSV-Logs
├── server/
│   ├── controllers/                # ServerController, ServerPersonTracker
│   ├── core/
│   │   ├── database/               # SystemDatabase, PersonDatabase
│   │   ├── math/                   # GeometryMath, TriangulationMath
│   │   └── tracking/
│   │       ├── Epipolarclusterer.py
│   │       ├── SkeletonTriangulator.py        # WLS (Numba) + LM
│   │       ├── Personidtracker.py             # Hungarian / Greedy
│   │       ├── SkeletonPostProcessor.py       # Outlier -> Smoothing -> IK
│   │       └── filters/
│   │           ├── JointKalmanFilter.py
│   │           ├── JointOneEuroFilter.py
│   │           ├── AcausalCurveFilter.py
│   │           └── SkeletonFABRIK.py
│   ├── network/                    # ServerConnector, ServerCommandHandler, ServerBeacon
│   ├── rendering/                  # CalibrationRenderer, Renderer3D, SceneRenderer
│   └── gui/
│       ├── ServerDashboard.py
│       ├── views/                  # Server3DView
│       └── windows/                # MasterSettings, Evaluation, DB-Viewer, QuickEval, ...
└── client/
    ├── logic/
    │   ├── ClientController.py
    │   └── sensors/                # PersonDetector, AsyncDetector, CameraSource
    ├── network/                    # ClientConnector, ClientCommandSender
    ├── config/                     # Client-Konfig, YOLO26-Modelle, Linsenprofile
    └── gui/                        # CalibrationWindow, SettingsDialog
```

---

## Datenbank

Schema wird in `SystemDatabase.py` automatisch erzeugt. Wichtige Tabellen:

| Tabelle                  | Zweck                                                                                  |
| ------------------------ | -------------------------------------------------------------------------------------- |
| `cameras`                | Registrierte Kamera-Knoten inkl. Auflösung, FPS, `active_lens_profile`, `model_path`   |
| `lens_profiles`          | `camera_matrix`, `dist_coeffs`, `reprojection_error` pro Linsenprofil                  |
| `global_room`            | Raum-Geometrie + JSONB-Spalten `master_settings`, `server_settings`, `tracking_settings` |
| `room_objects`           | 3D-Marker, Möbel, Referenzpunkte (Position, Größe, Farbe, Rotation)                    |
| `world_rectangles`       | SQPnP-Kalibrierrechtecke (z. B. `MAIN_ROOM_CALIB`)                                     |
| `rectangle_corners_3d`   | 3D-Eckpunkte je Rechteck                                                               |
| `camera_pixel_mapping`   | 2D-Pixel ↔ 3D-Eckpunkt-Zuordnung pro Kamera                                            |
| `camera_rectangles`      | Aktive Kalibrierrechtecke pro Kamera                                                   |

Verbindungspool über `SystemDatabase.get_connection()` – immer im `with`-Block verwenden.

Tracking-, Filter- und Triangulationskonfig werden **nicht** in eigenen Tabellen, sondern als JSONB-Spalten in `global_room` gehalten (`tracking_settings`, `master_settings`, `server_settings`).

---

## Netzwerkprotokoll

Binäres **msgpack** über raw TCP, **4-Byte big-endian Length Prefix**.

Payload-Form:

```python
{"action": <str>, "payload": <dict>}
```

Discovery erfolgt per UDP-Broadcast (`ServerBeacon.py`). Routing aller Aktionen in `ServerCommandHandler.py`. Gängige Actions: `REGISTER`, `PING`, `CAMERA_UPDATE`, `LOG`, `DB_UPDATE_CAMERA_SETTINGS`, `DB_SAVE_LENS_PROFILE`, `DB_UPDATE_GLOBAL_RECTANGLES`, `DB_UPDATE_GLOBAL_ROOM`, `DB_UPDATE_CAMERA_PIXELS`, `DB_REQUEST_CONFIG`.

### Clock-Drift-Korrektur

- Pro Kamera rollender Puffer von **120 Samples** der Transit-Zeit `t_recv - t_sent`.
- 5-%-Quantil wird alle **30 Frames** neu berechnet (kleiner Sicherheitsabzug von 0,5 ms) und in Sekunden als `_cam_clock_offset[camera_name]` gehalten.
- Korrigierte Netzwerklatenz: `max((apparent_net_s - clock_offset_s) * 1000, 0.3)` ms.
- Wird als `clock_offset_ms` in den Eval-CSVs mitgeloggt.

---

## Runtime-Mode-Switches

Ohne Neustart über die Master-/Tracking-Settings-UI umschaltbar:

| Schalter            | Werte                                  | Default     |
| ------------------- | -------------------------------------- | ----------- |
| Triangulation       | `wls` ↔ `lm`                           | `wls`       |
| Smoothing-Filter    | `one_euro` ↔ `kalman`                  | `one_euro`  |
| ID-Tracking         | `hungarian` (Deep-SORT-Light) ↔ `greedy` | `hungarian` |
| Inverse Kinematik   | `fabrik` ↔ `classic`                   | `fabrik`    |

Die zugehörigen Setter sitzen im `ServerController` (`set_tracking_mode`, plus Smoothing-/Triangulations-Wechsel) und wirken direkt auf den laufenden `CalibrationRenderer`.

---

## Evaluation

Zwei CSVs pro Session in `personLocator/logs/`:

- `eval_session_*.csv` – Rohdaten der einzelnen Session
- `eval_log_Block_*.csv` – kumuliert über alle Sessions eines Blocks

Die Auswertung läuft über pandas + matplotlib, eingebettet in PyQt6 (`EvaluationWindow.py`, `QuickEvalDialog.py`). Die Messreihe ist in zehn Blöcke gegliedert:

| Block | Inhalt                                                                  |
| ----- | ----------------------------------------------------------------------- |
|  1    | Inferenzlatenz (YOLO26, pro Kamera)                                     |
|  2    | Netzwerk-/Pipeline-Latenz                                               |
|  3    | Skalierbarkeit über `camera_count` (Multi-Kamera)                       |
|  4    | Kalibrierung (Linsenprofile + SQPnP-Posenfehler)                        |
|  5    | 3D-Lokalisierungsfehler (WLS vs. LM)                                    |
|  6    | Räumliche Fehlerverteilung (Heatmap)                                    |
|  7    | Filtervergleich (1-Euro vs. Kalman)                                     |
|  8    | Inverse Kinematik (Classic vs. FABRIK)                                  |
|  9    | ID-Tracking (Hungarian vs. Greedy)                                      |
| 10    | Langzeitstabilität (Health-Index)                                       |

Der Lokalisierungsfehler wird automatisch berechnet, sobald `room_objects` Einträge mit `category ∈ {marker, referenz, boden}` enthält. Marker-Positionen werden als numpy-Array `(N, 3)` keyed by `object_id` gecacht und nur invalidiert, wenn sich das Listenobjekt ändert.

---

## Autor

**Jonas Hafke**
B.Sc. Informatik · Heinrich-Heine-Universität Düsseldorf
Lehrstuhl Datenbanken und Informationssysteme (Prof. Dr. Stefan Conrad)
