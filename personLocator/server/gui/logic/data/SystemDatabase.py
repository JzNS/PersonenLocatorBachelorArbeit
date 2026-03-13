import logging
import psycopg2
from psycopg2 import pool
import os
import json
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor

load_dotenv()

class SystemDatabase:
    def __init__(self):
        self.db_pool = None
        self.log_sql_queries = False

        db_name = os.getenv("DB_NAME", "personLocatorSystem")
        db_user = os.getenv("DB_USER", "postgres")
        db_pass = os.getenv("DB_PASS", "")
        db_host = os.getenv("DB_HOST", "127.0.0.1")

        try:
            self.db_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=1, maxconn=20, dbname=db_name, user=db_user,
                password=db_pass, host=db_host, port="5432"
            )
            if self.db_pool:
                logging.info("PostgreSQL: Connection-Pool erfolgreich aufgebaut.")
                self._init_db()
        except psycopg2.Error as e:
            logging.critical(f"PostgreSQL Verbindungsfehler: {e}")

    def toggle_sql_logging(self) -> bool:
        """Schaltet den SQL-Inspektor Modus um. Wenn aktiviert, werden alle SQL-Queries im Log ausgegeben."""
        self.log_sql_queries = not self.log_sql_queries
        status = "AKTIVIERT" if self.log_sql_queries else "DEAKTIVIERT"
        logging.info(f"🛠️ SQL-Inspektor Modus wurde {status}.")
        return self.log_sql_queries

    def _log_query(self, cursor):
        """Wenn der SQL-Inspektor Modus aktiviert ist, wird die genaue SQL-Query im Log ausgegeben. Nützlich für Debugging und Entwicklung."""
        if self.log_sql_queries and cursor.query:
            exact_sql = cursor.query.decode('utf-8')
            logging.info(f"🔍 [SQL EXEC]:\n{exact_sql}\n")

    def _init_db(self) -> None:
        """Initialisiert die Datenbank, indem sie die notwendigen Tabellen erstellt und den unzerstörbaren Hauptraum (MAIN_ROOM_CALIB) anlegt, falls er nicht existiert."""
        conn = self.db_pool.getconn()
        try:
            with conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        'CREATE TABLE IF NOT EXISTS lens_profiles (id VARCHAR PRIMARY KEY, name VARCHAR NOT NULL, camera_matrix JSONB, dist_coeffs JSONB)')
                    cursor.execute(
                        'CREATE TABLE IF NOT EXISTS global_room (id INTEGER PRIMARY KEY DEFAULT 1, width_cm REAL DEFAULT 320.0, height_cm REAL DEFAULT 250.0, depth_cm REAL DEFAULT 470.0)')
                    cursor.execute('INSERT INTO global_room (id) VALUES (1) ON CONFLICT (id) DO NOTHING;')
                    cursor.execute('CREATE TABLE IF NOT EXISTS cameras (name VARCHAR PRIMARY KEY)')
                    cols = [
                        ("camera_index", "INTEGER DEFAULT 0"), ("resolution_width", "INTEGER DEFAULT 1920"),
                        ("resolution_height", "INTEGER DEFAULT 1080"), ("target_fps", "INTEGER DEFAULT 30"),
                        ("zoom_factor", "REAL DEFAULT 1.0"), ("rotation_angle", "INTEGER DEFAULT 0"),
                        ("render_capacity", "INTEGER DEFAULT 100"),
                        ("active_lens_profile", "VARCHAR DEFAULT 'default'"),
                        ("performance_mode", "VARCHAR DEFAULT '3D'"), ("view_options", "JSONB DEFAULT '{}'::jsonb"),
                        ("dead_zones", "JSONB DEFAULT '[]'::jsonb"), ("mirror_zones", "JSONB DEFAULT '[]'::jsonb"),
                        ("coordinates", "JSONB DEFAULT '{}'::jsonb")
                    ]
                    for col_name, col_def in cols:
                        cursor.execute(f"ALTER TABLE cameras ADD COLUMN IF NOT EXISTS {col_name} {col_def}")

                    cursor.execute('''
                                            INSERT INTO lens_profiles (id, name, camera_matrix, dist_coeffs) 
                                            VALUES ('default', 'Default', '[[1000,0,960],[0,1000,540],[0,0,1]]'::jsonb, '[[0,0,0,0,0]]'::jsonb)
                                            ON CONFLICT (id) DO NOTHING;
                                        ''')

                    cursor.execute('CREATE TABLE IF NOT EXISTS world_rectangles (id VARCHAR PRIMARY KEY, display_id VARCHAR, type VARCHAR, size_cm REAL, is_active BOOLEAN DEFAULT TRUE)')
                    cursor.execute('CREATE TABLE IF NOT EXISTS rectangle_corners_3d (id SERIAL PRIMARY KEY, rect_id VARCHAR, label VARCHAR, x REAL, y REAL, z REAL, FOREIGN KEY(rect_id) REFERENCES world_rectangles(id) ON DELETE CASCADE)')
                    cursor.execute('CREATE TABLE IF NOT EXISTS camera_pixel_mapping (id SERIAL PRIMARY KEY, camera_name VARCHAR, corner_3d_id INTEGER, px INTEGER, py INTEGER, FOREIGN KEY(camera_name) REFERENCES cameras(name) ON DELETE CASCADE, FOREIGN KEY(corner_3d_id) REFERENCES rectangle_corners_3d(id) ON DELETE CASCADE)')
                    cursor.execute("SELECT id FROM world_rectangles WHERE id = 'MAIN_ROOM_CALIB'")
                    if not cursor.fetchone():
                        cursor.execute(
                            "INSERT INTO world_rectangles (id, display_id, type, size_cm, is_active) VALUES ('MAIN_ROOM_CALIB', 'RAUM', 'Haupt-Kalibrierung', 0, TRUE)")
                        cursor.execute("SELECT width_cm, height_cm, depth_cm FROM global_room WHERE id = 1")
                        r = cursor.fetchone()
                        w, h, d = (r[0], r[1], r[2]) if r else (600.0, 250.0, 800.0)

                        default_corners = [
                            ("HUL", 0.0, h, 0.0), ("HUR", w, h, 0.0), ("HOL", 0.0, h, d), ("HOR", w, h, d),
                            ("VUL", 0.0, 0.0, 0.0), ("VOL", 0.0, 0.0, d), ("VUR", w, 0.0, 0.0), ("VOR", w, 0.0, d)
                        ]
                        for lbl, cx, cy, cz in default_corners:
                            cursor.execute(
                                "INSERT INTO rectangle_corners_3d (rect_id, label, x, y, z) VALUES ('MAIN_ROOM_CALIB', %s, %s, %s, %s)",
                                (lbl, cx, cy, cz))
                        logging.info(
                            "✅ Unzerstörbarer Hauptraum (MAIN_ROOM_CALIB) wurde in der Datenbank initialisiert.")
        except Exception as e:
            logging.error(f"Fehler bei der DB-Initialisierung: {e}")
        finally:
            self.db_pool.putconn(conn)

    def get_camera_settings(self, camera_name: str) -> dict:
        """Ruft die Kamera-Settings aus der Datenbank ab. Wenn die Kamera nicht existiert, wird ein leeres Dictionary zurückgegeben."""
        conn = self.db_pool.getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("SELECT * FROM cameras WHERE name = %s", (camera_name,))
                row = cursor.fetchone()
                if row:
                    view_opts = row.get("view_options") or {}
                    if isinstance(view_opts, str):
                        try: view_opts = json.loads(view_opts)
                        except: view_opts = {}

                    return {
                        "camera_index": row.get("camera_index", 0),
                        "resolution": [row.get("resolution_width", 1920), row.get("resolution_height", 1080)],
                        "target_fps": row.get("target_fps", 30),
                        "zoom": row.get("zoom_factor", 1.0),
                        "rotation": row.get("rotation_angle", 0),
                        "render_capacity": row.get("render_capacity", 100),
                        "active_lens_profile": row.get("active_lens_profile", "default"),
                        "performance_mode": row.get("performance_mode") == "Performance",
                        "view_render_enabled": view_opts.get("view_render_enabled", True),
                        "view_show_real": view_opts.get("view_show_real", True),
                        "view_show_grid": view_opts.get("view_show_grid", True),
                        "view_show_floor_grid": view_opts.get("view_show_floor_grid", True),
                        "view_show_rays": view_opts.get("view_show_rays", True),
                        "view_show_skeleton": view_opts.get("view_show_skeleton", True),
                        "view_show_sightlines": view_opts.get("view_show_sightlines", True),
                        "dead_zones": row.get("dead_zones", []),
                        "mirror_zones": row.get("mirror_zones", []),
                        "coordinates": row.get("coordinates", {})
                    }
        except Exception as e:
            logging.error(f"❌ DB-Fehler bei get_camera_settings: {e}")
        finally:
            self.db_pool.putconn(conn)
        return {}

    def update_camera_settings(self, camera_name: str, settings: dict):
        """Aktualisiert die Kamera-Settings in der Datenbank.
        Alle übergebenen Werte werden direkt in die entsprechenden Spalten geschrieben.
        Fehlende Werte werden mit Standardwerten aufgefüllt."""
        conn = self.db_pool.getconn()
        try:
            with conn:
                with conn.cursor() as cursor:
                    view_options_paket = {
                        "view_render_enabled": settings.get("view_render_enabled", True),
                        "view_show_real": settings.get("view_show_real", True),
                        "view_show_grid": settings.get("view_show_grid", True),
                        "view_show_floor_grid": settings.get("view_show_floor_grid", True),
                        "view_show_rays": settings.get("view_show_rays", True),
                        "view_show_skeleton": settings.get("view_show_skeleton", True),
                        "view_show_sightlines": settings.get("view_show_sightlines", True)
                    }

                    cursor.execute("""
                        UPDATE cameras 
                        SET camera_index = %s,
                            resolution_width = %s, 
                            resolution_height = %s, 
                            target_fps = %s, 
                            zoom_factor = %s, 
                            rotation_angle = %s, 
                            render_capacity = %s,
                            active_lens_profile = %s, 
                            performance_mode = %s,
                            view_options = %s,
                            dead_zones = %s,
                            mirror_zones = %s,
                            coordinates = %s
                        WHERE name = %s
                    """, (
                        int(settings.get("camera_index", 0)),
                        settings.get("resolution", [1920, 1080])[0],
                        settings.get("resolution", [1920, 1080])[1],
                        int(settings.get("target_fps", 30)),
                        float(settings.get("zoom", 1.0)),
                        int(settings.get("rotation", 0)),
                        int(settings.get("render_capacity", 100)),
                        settings.get("active_lens_profile", "default"),
                        "Performance" if settings.get("performance_mode") else "3D",
                        json.dumps(view_options_paket),
                        json.dumps(settings.get("dead_zones", [])),
                        json.dumps(settings.get("mirror_zones", [])),
                        json.dumps(settings.get("coordinates", {})),
                        camera_name
                    ))
            logging.info(f"✅ DB: Settings für {camera_name} erfolgreich gespeichert.")
        except Exception as e:
            logging.error(f"❌ DB-Fehler bei update_camera_settings: {e}")
        finally:
            self.db_pool.putconn(conn)

    def get_all_world_rectangles(self) -> list:
        """Ruft alle definierten 3D-Vierecke aus der Datenbank ab, inklusive ihrer Ecken-Koordinaten."""
        conn = self.db_pool.getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("SELECT * FROM world_rectangles")
                rects = cursor.fetchall()
                result = []
                for r in rects:
                    cursor.execute("SELECT label, x, y, z FROM rectangle_corners_3d WHERE rect_id = %s", (r["id"],))
                    corners = cursor.fetchall()
                    result.append({
                        "internal_id": r["id"], "display_id": r["display_id"],
                        "type": r["type"], "size_cm": r["size_cm"], "is_active": r["is_active"],
                        "corners": corners
                    })
                return result
        finally:
            self.db_pool.putconn(conn)

    def get_camera_pixels(self, camera_name: str) -> dict:
        """Ruft die 2D-Pixel-Koordinaten der Klicks für die Referenz-Vierecke einer Kamera ab.
        Gibt ein Dictionary zurück, das jedem Viereck (rect_id) eine Liste von Pixeln zuordnet."""
        conn = self.db_pool.getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT r.rect_id, m.px, m.py 
                    FROM camera_pixel_mapping m
                    JOIN rectangle_corners_3d r ON m.corner_3d_id = r.id
                    WHERE m.camera_name = %s
                """, (camera_name,))
                pixel_dict = {}
                for row in cursor.fetchall():
                    rid = row["rect_id"]
                    if rid not in pixel_dict: pixel_dict[rid] = []
                    pixel_dict[rid].append({"px": row["px"], "py": row["py"]})
                return pixel_dict
        finally:
            self.db_pool.putconn(conn)

    def get_all_lens_profiles(self) -> dict:
        """Ruft alle gespeicherten Objektivprofile aus der Datenbank ab.
        Gibt ein Dictionary zurück, das jedem Profil eine ID, einen Namen,
        die Kameramatrix und die Verzerrungskoeffizienten zuordnet."""
        conn = self.db_pool.getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("SELECT id, name, camera_matrix, dist_coeffs FROM lens_profiles")
                return {row["id"]: {"name": row["name"], "camera_matrix": row["camera_matrix"], "dist_coeffs": row["dist_coeffs"]} for row in cursor.fetchall()}
        finally:
            self.db_pool.putconn(conn)

    def get_global_room(self) -> dict:
        """Ruft die globalen Raummaße aus der Datenbank ab. Gibt ein Dictionary mit Breite, Höhe und Tiefe zurück."""
        conn = self.db_pool.getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("SELECT width_cm, height_cm, depth_cm FROM global_room WHERE id = 1")
                row = cursor.fetchone()
                if row: return {"width": row["width_cm"], "height": row["height_cm"], "depth": row["depth_cm"]}
                return {"width": 320.0, "height": 250.0, "depth": 470.0}
        finally:
            self.db_pool.putconn(conn)

    def register_camera_if_not_exists(self, camera_name: str):
        """Stellt sicher, dass die Kamera mit dem gegebenen Namen in der Datenbank existiert."""
        conn = self.db_pool.getconn()
        try:
            with conn:
                with conn.cursor() as cursor:
                    cursor.execute("INSERT INTO cameras (name) VALUES (%s) ON CONFLICT DO NOTHING", (camera_name,))
        finally:
            self.db_pool.putconn(conn)

    def update_global_rectangles(self, rectangles: list) -> None:
        """Aktualisiert die 3D-Koordinaten der Referenz-Vierecke in der Datenbank."""
        conn = self.db_pool.getconn()
        try:
            with conn:
                with conn.cursor() as cursor:
                    # KOMPLETT NEU: Kein pauschales Löschen mehr!
                    for r in rectangles:
                        if r["internal_id"] == "MAIN_ROOM_CALIB":
                            logging.warning("🚨 Blockiert: Versuch, den Hauptraum zu löschen!")
                            continue
                        # 1. Explizites Löschen (Nur wenn vom Client angefordert)
                        if r.get("_delete"):
                            cursor.execute("DELETE FROM world_rectangles WHERE id = %s", (r["internal_id"],))
                            self._log_query(cursor)
                            continue

                        # 2. Normales Hinzufügen / Aktualisieren (UPSERT)
                        cursor.execute('''
                            INSERT INTO world_rectangles (id, display_id, type, size_cm, is_active)
                            VALUES (%s, %s, %s, %s, %s)
                            ON CONFLICT (id) DO UPDATE SET 
                                display_id = EXCLUDED.display_id, type = EXCLUDED.type, size_cm = EXCLUDED.size_cm, is_active = EXCLUDED.is_active
                        ''', (r["internal_id"], r.get("display_id", ""), r.get("type", ""), r.get("size_cm", 0),
                              r.get("is_active", True)))
                        self._log_query(cursor)

                        for i, c in enumerate(r.get("corners", [])):
                            cursor.execute(
                                'SELECT id FROM rectangle_corners_3d WHERE rect_id = %s ORDER BY id ASC OFFSET %s LIMIT 1',
                                (r["internal_id"], i))
                            row = cursor.fetchone()
                            if row:
                                cursor.execute(
                                    'UPDATE rectangle_corners_3d SET label = %s, x = %s, y = %s, z = %s WHERE id = %s',
                                    (c["label"], c.get("x", 0), c.get("y", 0), c.get("z", 0), row[0]))
                            else:
                                cursor.execute(
                                    'INSERT INTO rectangle_corners_3d (rect_id, label, x, y, z) VALUES (%s, %s, %s, %s, %s)',
                                    (r["internal_id"], c["label"], c.get("x", 0), c.get("y", 0), c.get("z", 0)))
                            self._log_query(cursor)
        finally:
            self.db_pool.putconn(conn)

    def update_camera_pixels(self, camera_name: str, pixel_dict: dict) -> None:
        """Aktualisiert die 2D-Pixel-Koordinaten der Klicks für die Referenz-Vierecke einer Kamera in der Datenbank."""
        if not pixel_dict: return
        conn = self.db_pool.getconn()
        try:
            with conn:
                with conn.cursor() as cursor:
                    # KOMPLETT NEU: Löscht nur die Pixel der aktuell bearbeiteten Vierecke,
                    # lässt alle anderen (unbearbeiteten) unangetastet!
                    for rect_id, corners in pixel_dict.items():
                        cursor.execute('''
                            DELETE FROM camera_pixel_mapping 
                            WHERE camera_name = %s AND corner_3d_id IN (
                                SELECT id FROM rectangle_corners_3d WHERE rect_id = %s
                            )
                        ''', (camera_name, rect_id))

                        for i, p in enumerate(corners):
                            cursor.execute(
                                'SELECT id FROM rectangle_corners_3d WHERE rect_id = %s ORDER BY id ASC OFFSET %s LIMIT 1',
                                (rect_id, i))
                            corner_row = cursor.fetchone()
                            if corner_row:
                                cursor.execute(
                                    'INSERT INTO camera_pixel_mapping (camera_name, corner_3d_id, px, py) VALUES (%s, %s, %s, %s)',
                                    (camera_name, corner_row[0], int(p.get("px", 0)), int(p.get("py", 0))))
        finally:
            self.db_pool.putconn(conn)

    def update_global_room(self, dims: dict) -> None:
        """Aktualisiert die globalen Raummaße in der Datenbank.
        Alle übergebenen Werte werden direkt in die entsprechenden Spalten geschrieben.
        Fehlende Werte werden mit Standardwerten aufgefüllt."""
        conn = self.db_pool.getconn()
        try:
            with conn:
                with conn.cursor() as cursor:
                    cursor.execute('UPDATE global_room SET width_cm = %s, height_cm = %s, depth_cm = %s WHERE id = 1', (float(dims.get("width", 320.0)), float(dims.get("height", 250.0)), float(dims.get("depth", 470.0))))
        finally:
            self.db_pool.putconn(conn)