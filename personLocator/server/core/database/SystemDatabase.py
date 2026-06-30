from server.core.logger import get_logger
import psycopg2
from psycopg2 import pool
import os
import json
from typing import Optional, Any, Dict, List, Union
from dotenv import load_dotenv
from psycopg2.extras import RealDictCursor

logger = get_logger("server.database")

load_dotenv()


class SystemDatabase:
    def __init__(self) -> None:
        self.db_pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None
        self.log_sql_queries: bool = False

        db_name: str = os.getenv("DB_NAME", "personLocatorSystem")
        db_user: str = os.getenv("DB_USER", "postgres")
        db_pass: str = os.getenv("DB_PASS", "")
        db_host: str = os.getenv("DB_HOST", "127.0.0.1")

        try:
            self.db_pool = psycopg2.pool.ThreadedConnectionPool(
                minconn=1, maxconn=20, dbname=db_name, user=db_user,
                password=db_pass, host=db_host, port="5432"
            )
            if self.db_pool:
                logger.info("PostgreSQL: Connection-Pool erfolgreich aufgebaut.")
                self._init_db()
        except psycopg2.Error as e:
            logger.critical(f"PostgreSQL Verbindungsfehler: {e}")

    def toggle_sql_logging(self) -> bool:
        self.log_sql_queries = not self.log_sql_queries
        status: str = "AKTIVIERT" if self.log_sql_queries else "DEAKTIVIERT"
        logger.info(f"🛠️ SQL-Inspektor Modus wurde {status}.")
        return self.log_sql_queries

    def _log_query(self, cursor: Any) -> None:
        if self.log_sql_queries and cursor.query:
            exact_sql: str = cursor.query.decode('utf-8')
            logger.info(f"🔍 [SQL EXEC]:\n{exact_sql}\n")

    def _init_db(self) -> None:
        if not self.db_pool:
            return
        conn = self.db_pool.getconn()
        try:
            with conn:
                with conn.cursor() as cursor:
                    cursor.execute(
                        'CREATE TABLE IF NOT EXISTS lens_profiles (id VARCHAR PRIMARY KEY, name VARCHAR NOT NULL, camera_matrix JSONB, dist_coeffs JSONB)')
                    cursor.execute(
                        'ALTER TABLE lens_profiles ADD COLUMN IF NOT EXISTS reprojection_error REAL')
                    cursor.execute(
                        'CREATE TABLE IF NOT EXISTS global_room (id INTEGER PRIMARY KEY DEFAULT 1, width_cm REAL DEFAULT 320.0, height_cm REAL DEFAULT 250.0, depth_cm REAL DEFAULT 470.0)')
                    cursor.execute('INSERT INTO global_room (id) VALUES (1) ON CONFLICT (id) DO NOTHING;')
                    cursor.execute('CREATE TABLE IF NOT EXISTS cameras (name VARCHAR PRIMARY KEY)')

                    cursor.execute(
                        'ALTER TABLE global_room ADD COLUMN IF NOT EXISTS master_settings JSONB DEFAULT \'{}\'::jsonb')
                    cursor.execute(
                        'ALTER TABLE global_room ADD COLUMN IF NOT EXISTS server_settings JSONB DEFAULT \'{}\'::jsonb')
                    cursor.execute(
                        'ALTER TABLE global_room ADD COLUMN IF NOT EXISTS tracking_settings JSONB DEFAULT \'{}\'::jsonb')

                    cols: list[tuple[str, str]] = [
                        ("camera_index", "INTEGER DEFAULT 0"), ("resolution_width", "INTEGER DEFAULT 1920"),
                        ("resolution_height", "INTEGER DEFAULT 1080"), ("target_fps", "INTEGER DEFAULT 30"),
                        ("zoom_factor", "REAL DEFAULT 1.0"), ("rotation_angle", "INTEGER DEFAULT 0"),
                        ("render_capacity", "INTEGER DEFAULT 100"),
                        ("active_lens_profile", "VARCHAR DEFAULT 'default'"),
                        ("camera_type", "VARCHAR DEFAULT 'Standard'"),
                        ("performance_mode", "VARCHAR DEFAULT '3D'"), ("view_options", "JSONB DEFAULT '{}'::jsonb"),
                        ("dead_zones", "JSONB DEFAULT '[]'::jsonb"), ("mirror_zones", "JSONB DEFAULT '[]'::jsonb"),
                        ("coordinates", "JSONB DEFAULT '{}'::jsonb"),
                        ("model_path", "VARCHAR DEFAULT 'client/config/graka/yolo26n-pose.onnx'")
                    ]
                    for col_name, col_def in cols:
                        cursor.execute(f"ALTER TABLE cameras ADD COLUMN IF NOT EXISTS {col_name} {col_def}")

                    cursor.execute('''
                        INSERT INTO lens_profiles (id, name, camera_matrix, dist_coeffs) 
                        VALUES ('default', 'Default', '[[1000,0,960],[0,1000,540],[0,0,1]]'::jsonb, '[[0,0,0,0,0]]'::jsonb)
                        ON CONFLICT (id) DO NOTHING;
                    ''')

                    cursor.execute(
                        'CREATE TABLE IF NOT EXISTS room_objects ('
                        'id SERIAL PRIMARY KEY, '
                        'name VARCHAR DEFAULT \'Objekt\', '
                        'obj_type VARCHAR DEFAULT \'box\', '
                        'pos_x REAL DEFAULT 0.0, pos_y REAL DEFAULT 0.0, pos_z REAL DEFAULT 0.0, '
                        'size_w REAL DEFAULT 50.0, size_h REAL DEFAULT 50.0, size_d REAL DEFAULT 50.0, '
                        'color_hex VARCHAR DEFAULT \'#FFAA00\', '
                        'rotation_yaw REAL DEFAULT 0.0, '
                        'is_visible BOOLEAN DEFAULT TRUE)'
                    )

                    cursor.execute(
                        'CREATE TABLE IF NOT EXISTS world_rectangles (id VARCHAR PRIMARY KEY, display_id VARCHAR, type VARCHAR, size_cm REAL, is_active BOOLEAN DEFAULT TRUE)')
                    cursor.execute(
                        'CREATE TABLE IF NOT EXISTS rectangle_corners_3d (id SERIAL PRIMARY KEY, rect_id VARCHAR, label VARCHAR, x REAL, y REAL, z REAL, FOREIGN KEY(rect_id) REFERENCES world_rectangles(id) ON DELETE CASCADE)')
                    cursor.execute(
                        'CREATE TABLE IF NOT EXISTS camera_pixel_mapping (id SERIAL PRIMARY KEY, camera_name VARCHAR, corner_3d_id INTEGER, px INTEGER, py INTEGER, FOREIGN KEY(camera_name) REFERENCES cameras(name) ON DELETE CASCADE, FOREIGN KEY(corner_3d_id) REFERENCES rectangle_corners_3d(id) ON DELETE CASCADE)')
                    cursor.execute(
                        'CREATE TABLE IF NOT EXISTS camera_rectangles (camera_name VARCHAR, rect_id VARCHAR, is_active BOOLEAN DEFAULT TRUE, PRIMARY KEY (camera_name, rect_id), FOREIGN KEY(camera_name) REFERENCES cameras(name) ON DELETE CASCADE, FOREIGN KEY(rect_id) REFERENCES world_rectangles(id) ON DELETE CASCADE)')

                    cursor.execute("SELECT id FROM world_rectangles WHERE id = 'MAIN_ROOM_CALIB'")
                    if not cursor.fetchone():
                        cursor.execute(
                            "INSERT INTO world_rectangles (id, display_id, type, size_cm, is_active) VALUES ('MAIN_ROOM_CALIB', 'RAUM', 'Haupt-Kalibrierung', 0, TRUE)")
                        cursor.execute("SELECT width_cm, height_cm, depth_cm FROM global_room WHERE id = 1")
                        r = cursor.fetchone()
                        w, h, d = (r[0], r[1], r[2]) if r else (600.0, 250.0, 800.0)

                        default_corners: list[tuple[str, float, float, float]] = [
                            ("HUL", 0.0, h, 0.0), ("HUR", w, h, 0.0), ("HOL", 0.0, h, d), ("HOR", w, h, d),
                            ("VUL", 0.0, 0.0, 0.0), ("VOL", 0.0, 0.0, d), ("VUR", w, 0.0, 0.0), ("VOR", w, 0.0, d)
                        ]
                        for lbl, cx, cy, cz in default_corners:
                            cursor.execute(
                                "INSERT INTO rectangle_corners_3d (rect_id, label, x, y, z) VALUES ('MAIN_ROOM_CALIB', %s, %s, %s, %s)",
                                (lbl, cx, cy, cz))
        except Exception as e:
            logger.error(f"Fehler bei der DB-Initialisierung: {e}")
        finally:
            if self.db_pool:
                self.db_pool.putconn(conn)

    def _get_jsonb_setting(self, column_name: str) -> dict[str, Any]:
        if not self.db_pool:
            return {}
        conn = self.db_pool.getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(f"SELECT {column_name} FROM global_room WHERE id = 1")
                row = cursor.fetchone()
                if row is not None and column_name in row:
                    data = row[column_name]
                    if data is None: return {}
                    if isinstance(data, str):
                        try:
                            return json.loads(data)
                        except Exception:
                            return {}
                    if isinstance(data, dict): return data
        except Exception as e:
            logger.error(f"❌ DB Fehler beim Laden von {column_name}: {e}")
        finally:
            self.db_pool.putconn(conn)
        return {}

    def _update_jsonb_setting(self, column_name: str, settings: dict[str, Any]) -> None:
        if not self.db_pool:
            return
        conn = self.db_pool.getconn()
        try:
            from psycopg2.extras import Json
            with conn:
                with conn.cursor() as cursor:
                    cursor.execute(f"UPDATE global_room SET {column_name} = %s WHERE id = 1", (Json(settings),))
        except Exception as e:
            logger.error(f"❌ DB Fehler beim Speichern von {column_name}: {e}")
        finally:
            self.db_pool.putconn(conn)

    def get_master_settings(self) -> dict[str, Any]:
        data: dict[str, Any] = self._get_jsonb_setting("master_settings")
        for cam, conf in data.items():
            if isinstance(conf, dict) and "joints" in conf:
                conf["joints"] = {int(k): v for k, v in conf["joints"].items()}
        return data

    def update_master_settings(self, settings: dict[str, Any]) -> None:
        self._update_jsonb_setting("master_settings", settings)

    def get_server_settings(self) -> dict[str, Any]:
        return self._get_jsonb_setting("server_settings")

    def update_server_settings(self, settings: dict[str, Any]) -> None:
        self._update_jsonb_setting("server_settings", settings)

    def get_tracking_settings(self) -> dict[str, Any]:
        return self._get_jsonb_setting("tracking_settings")

    def update_tracking_settings(self, settings: dict[str, Any]) -> None:
        self._update_jsonb_setting("tracking_settings", settings)

    def save_lens_profile(self, profile_id: str, name: str, mtx: list[Any], dist: list[Any], reprojection_error: float = None) -> None:
        if not self.db_pool:
            return
        conn = self.db_pool.getconn()
        try:
            with conn:
                with conn.cursor() as cursor:
                    from psycopg2.extras import Json
                    cursor.execute('''
                        INSERT INTO lens_profiles (id, name, camera_matrix, dist_coeffs, reprojection_error)
                        VALUES (%s, %s, %s, %s, %s)
                        ON CONFLICT (id) DO UPDATE SET 
                            name = EXCLUDED.name, camera_matrix = EXCLUDED.camera_matrix,
                            dist_coeffs = EXCLUDED.dist_coeffs, reprojection_error = EXCLUDED.reprojection_error
                    ''', (profile_id, name, Json(mtx), Json(dist), reprojection_error))
        except Exception as e:
            logger.error(f"❌ Fehler beim Speichern des globalen Linsenprofils: {e}")
        finally:
            self.db_pool.putconn(conn)

    def get_camera_settings(self, camera_name: str) -> dict[str, Any]:
        if not self.db_pool:
            return {}
        conn = self.db_pool.getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT c.*, l.camera_matrix, l.dist_coeffs
                    FROM cameras c
                    LEFT JOIN lens_profiles l ON c.active_lens_profile = l.id
                    WHERE c.name = %s
                """, (camera_name,))
                row = cursor.fetchone()
                if not row: return {}

                def _parse_json(val: Any, default: Any) -> Any:
                    if val is None: return default
                    if isinstance(val, (dict, list)): return val
                    if isinstance(val, str):
                        try:
                            return json.loads(val)
                        except Exception:
                            return default
                    return default

                view_opts: dict[str, Any] = _parse_json(row.get("view_options"), {})
                perf_str: str = str(row.get("performance_mode", "3D"))

                return {
                    "camera_index": int(row.get("camera_index", 0) or 0),
                    "resolution": [int(row.get("resolution_width", 1920) or 1920),
                                   int(row.get("resolution_height", 1080) or 1080)],
                    "target_fps": int(row.get("target_fps", 30) or 30),
                    "zoom": float(row.get("zoom_factor", 1.0) or 1.0),
                    "rotation": int(row.get("rotation_angle", 0) or 0),
                    "render_capacity": int(row.get("render_capacity", 100) or 100),
                    "active_lens_profile": row.get("active_lens_profile", "default"),
                    "camera_type": str(row.get("camera_type", "Standard")),
                    "performance_mode": True if perf_str.lower().startswith("perf") else False,
                    "view_render_enabled": bool(view_opts.get("view_render_enabled", True)),
                    "view_show_real": bool(view_opts.get("view_show_real", True)),
                    "view_show_grid": bool(view_opts.get("view_show_grid", True)),
                    "view_show_floor_grid": bool(view_opts.get("view_show_floor_grid", True)),
                    "view_show_rays": bool(view_opts.get("view_show_rays", True)),
                    "view_show_skeleton": bool(view_opts.get("view_show_skeleton", True)),
                    "view_show_sightlines": bool(view_opts.get("view_show_sightlines", True)),
                    "dead_zones": _parse_json(row.get("dead_zones"), []),
                    "mirror_zones": _parse_json(row.get("mirror_zones"), []),
                    "coordinates": _parse_json(row.get("coordinates"), {}),
                    "model_path": row.get("model_path", "client/config/graka/yolo26n-pose.onnx"),
                    "camera_matrix": _parse_json(row.get("camera_matrix"), None),
                    "dist_coeffs": _parse_json(row.get("dist_coeffs"), None),
                }
        except Exception as e:
            logger.error(f"❌ DB-Fehler bei get_camera_settings: {e}")
            return {}
        finally:
            self.db_pool.putconn(conn)

    def get_all_cameras(self) -> list[str]:
        if not self.db_pool:
            return []
        conn = self.db_pool.getconn()
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT name FROM cameras")
                rows = cursor.fetchall()
                return [str(r[0]) for r in rows]
        finally:
            self.db_pool.putconn(conn)

    def get_all_lens_profile_ids(self) -> list[str]:
        if not self.db_pool:
            return ["default"]
        conn = self.db_pool.getconn()
        try:
            with conn.cursor() as cursor:
                cursor.execute("SELECT id FROM lens_profiles")
                rows = cursor.fetchall()
                if not rows: return ["default"]
                return [str(r[0]) for r in rows]
        finally:
            self.db_pool.putconn(conn)

    def update_camera_settings(self, camera_name: str, settings: dict[str, Any]) -> None:
        if not self.db_pool:
            return
        conn = self.db_pool.getconn()
        try:
            with conn:
                with conn.cursor() as cursor:
                    view_opts: dict[str, Any] = {
                        "view_render_enabled": settings.get("view_render_enabled", True),
                        "view_show_real": settings.get("view_show_real", True),
                        "view_show_grid": settings.get("view_show_grid", True),
                        "view_show_floor_grid": settings.get("view_show_floor_grid", True),
                        "view_show_rays": settings.get("view_show_rays", True),
                        "view_show_skeleton": settings.get("view_show_skeleton", True),
                        "view_show_sightlines": settings.get("view_show_sightlines", True)
                    }

                    perf_mode_raw = settings.get("performance_mode")
                    perf_mode_str = "Performance" if (perf_mode_raw is True or str(perf_mode_raw).lower() == "true") else "3D"

                    cursor.execute("""
                    INSERT INTO cameras (name, camera_index, resolution_width, resolution_height, target_fps, zoom_factor, rotation_angle, render_capacity, active_lens_profile, camera_type, performance_mode, view_options, dead_zones, mirror_zones, coordinates, model_path)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (name) DO UPDATE SET
                        camera_index = EXCLUDED.camera_index, resolution_width = EXCLUDED.resolution_width, resolution_height = EXCLUDED.resolution_height, target_fps = EXCLUDED.target_fps,
                        zoom_factor = EXCLUDED.zoom_factor, rotation_angle = EXCLUDED.rotation_angle, render_capacity = EXCLUDED.render_capacity, active_lens_profile = EXCLUDED.active_lens_profile,
                        camera_type = EXCLUDED.camera_type, performance_mode = EXCLUDED.performance_mode, view_options = EXCLUDED.view_options, dead_zones = EXCLUDED.dead_zones, mirror_zones = EXCLUDED.mirror_zones,
                        coordinates = EXCLUDED.coordinates, model_path = EXCLUDED.model_path
                    """, (
                        camera_name, int(settings.get("camera_index", 0)), settings.get("resolution", [1920, 1080])[0],
                        settings.get("resolution", [1920, 1080])[1],
                        int(settings.get("target_fps", 30)), float(settings.get("zoom", 1.0)),
                        int(settings.get("rotation", 0)), int(settings.get("render_capacity", 100)),
                        settings.get("active_lens_profile", "default"),
                        str(settings.get("camera_type", "Standard")),
                        perf_mode_str,
                        json.dumps(view_opts), json.dumps(settings.get("dead_zones", [])),
                        json.dumps(settings.get("mirror_zones", [])),
                        json.dumps(settings.get("coordinates", {})),
                        settings.get("model_path", "client/config/graka/yolo26n-pose.onnx")
                    ))
        finally:
            self.db_pool.putconn(conn)

    def get_all_world_rectangles(self) -> list[dict[str, Any]]:
        if not self.db_pool:
            return []
        conn = self.db_pool.getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("SELECT * FROM world_rectangles")
                rects = cursor.fetchall()
                result: list[dict[str, Any]] = []
                for r in rects:
                    cursor.execute("SELECT label, x, y, z FROM rectangle_corners_3d WHERE rect_id = %s", (r["id"],))
                    result.append({"internal_id": r["id"], "display_id": r["display_id"], "type": r["type"],
                                   "size_cm": r["size_cm"], "is_active": r["is_active"], "corners": cursor.fetchall()})
                return result
        finally:
            self.db_pool.putconn(conn)

    def get_camera_pixels(self, camera_name: str) -> dict[str, Any]:
        if not self.db_pool:
            return {}
        conn = self.db_pool.getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("""
                    SELECT r.rect_id, m.px, m.py, cr.is_active 
                    FROM camera_pixel_mapping m JOIN rectangle_corners_3d r ON m.corner_3d_id = r.id LEFT JOIN camera_rectangles cr ON cr.camera_name = m.camera_name AND cr.rect_id = r.rect_id
                    WHERE m.camera_name = %s ORDER BY r.id ASC
                """, (camera_name,))
                pixel_dict: dict[str, Any] = {}
                for row in cursor.fetchall():
                    rid: str = str(row["rect_id"])
                    if rid not in pixel_dict: pixel_dict[rid] = {
                        "is_active": row["is_active"] if row["is_active"] is not None else True, "pixels": []}
                    pixel_dict[rid]["pixels"].append({"px": row["px"], "py": row["py"]})
                return pixel_dict
        finally:
            self.db_pool.putconn(conn)

    def get_all_lens_profiles(self) -> dict[str, dict[str, Any]]:
        if not self.db_pool:
            return {}
        conn = self.db_pool.getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("SELECT id, name, camera_matrix, dist_coeffs, reprojection_error FROM lens_profiles")
                return {str(row["id"]): {"name": row["name"], "camera_matrix": row["camera_matrix"],
                                    "dist_coeffs": row["dist_coeffs"],
                                    "reprojection_error": row.get("reprojection_error")} for row in cursor.fetchall()}
        finally:
            self.db_pool.putconn(conn)

    def get_global_room(self) -> dict[str, float]:
        if not self.db_pool:
            return {"width": 320.0, "height": 250.0, "depth": 470.0}
        conn = self.db_pool.getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute("SELECT width_cm, height_cm, depth_cm FROM global_room WHERE id = 1")
                row = cursor.fetchone()
                if row: return {"width": float(row["width_cm"]), "height": float(row["height_cm"]), "depth": float(row["depth_cm"])}
                return {"width": 320.0, "height": 250.0, "depth": 470.0}
        finally:
            self.db_pool.putconn(conn)

    def register_camera_if_not_exists(self, camera_name: str) -> None:
        if not self.db_pool:
            return
        conn = self.db_pool.getconn()
        try:
            with conn:
                with conn.cursor() as cursor:
                    cursor.execute("INSERT INTO cameras (name) VALUES (%s) ON CONFLICT DO NOTHING", (camera_name,))
        finally:
            self.db_pool.putconn(conn)

    def update_global_rectangles(self, rectangles: list[dict[str, Any]]) -> None:
        if not self.db_pool:
            return
        conn = self.db_pool.getconn()
        try:
            with conn:
                with conn.cursor() as cursor:
                    for r in rectangles:
                        if r["internal_id"] == "MAIN_ROOM_CALIB": continue
                        if r.get("_delete"):
                            cursor.execute("DELETE FROM world_rectangles WHERE id = %s", (r["internal_id"],))
                            continue
                        cursor.execute('''
                            INSERT INTO world_rectangles (id, display_id, type, size_cm, is_active) VALUES (%s, %s, %s, %s, %s)
                            ON CONFLICT (id) DO UPDATE SET display_id = EXCLUDED.display_id, type = EXCLUDED.type, size_cm = EXCLUDED.size_cm, is_active = EXCLUDED.is_active
                        ''', (r["internal_id"], r.get("display_id", ""), r.get("type", ""), r.get("size_cm", 0),
                              r.get("is_active", True)))
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
        finally:
            self.db_pool.putconn(conn)

    def update_camera_pixels(self, camera_name: str, pixel_dict: dict[str, Any]) -> None:
        if not pixel_dict or not self.db_pool: return
        conn = self.db_pool.getconn()
        try:
            with conn:
                with conn.cursor() as cursor:
                    for rect_id, data in pixel_dict.items():
                        is_active: bool = data.get("is_active", True) if isinstance(data, dict) else True
                        corner_pixels: list[dict[str, Any]] = data.get("pixels", []) if isinstance(data, dict) else data
                        cursor.execute('''
                            INSERT INTO camera_rectangles (camera_name, rect_id, is_active) VALUES (%s, %s, %s)
                            ON CONFLICT (camera_name, rect_id) DO UPDATE SET is_active = EXCLUDED.is_active
                        ''', (camera_name, rect_id, is_active))
                        cursor.execute(
                            'DELETE FROM camera_pixel_mapping WHERE camera_name = %s AND corner_3d_id IN (SELECT id FROM rectangle_corners_3d WHERE rect_id = %s)',
                            (camera_name, rect_id))
                        for i, p in enumerate(corner_pixels):
                            cursor.execute(
                                'SELECT id FROM rectangle_corners_3d WHERE rect_id = %s ORDER BY id ASC OFFSET %s LIMIT 1',
                                (rect_id, i))
                            corner_row = cursor.fetchone()
                            if corner_row: cursor.execute(
                                'INSERT INTO camera_pixel_mapping (camera_name, corner_3d_id, px, py) VALUES (%s, %s, %s, %s)',
                                (camera_name, corner_row[0], int(p.get("px", 0)), int(p.get("py", 0))))
        finally:
            self.db_pool.putconn(conn)

    def update_global_room(self, dims: dict[str, Any]) -> None:
        if not self.db_pool:
            return
        conn = self.db_pool.getconn()
        try:
            with conn:
                with conn.cursor() as cursor:
                    cursor.execute('UPDATE global_room SET width_cm = %s, height_cm = %s, depth_cm = %s WHERE id = 1', (
                    float(dims.get("width", 320.0)), float(dims.get("height", 250.0)), float(dims.get("depth", 470.0))))
        finally:
            self.db_pool.putconn(conn)

    def get_all_room_objects(self) -> list[dict[str, Any]]:
        if not self.db_pool:
            return []
        conn = self.db_pool.getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(
                    "SELECT id, name, obj_type, pos_x, pos_y, pos_z, size_w, size_h, size_d, "
                    "color_hex, rotation_yaw, is_visible FROM room_objects ORDER BY id ASC"
                )
                rows = cursor.fetchall()
                return [
                    {
                        "id": int(r["id"]),
                        "name": str(r["name"] or ""),
                        "obj_type": str(r["obj_type"] or "box"),
                        "pos_x": float(r["pos_x"] or 0.0),
                        "pos_y": float(r["pos_y"] or 0.0),
                        "pos_z": float(r["pos_z"] or 0.0),
                        "size_w": float(r["size_w"] or 50.0),
                        "size_h": float(r["size_h"] or 50.0),
                        "size_d": float(r["size_d"] or 50.0),
                        "color_hex": str(r["color_hex"] or "#FFAA00"),
                        "rotation_yaw": float(r["rotation_yaw"] or 0.0),
                        "is_visible": bool(r["is_visible"]) if r["is_visible"] is not None else True,
                    }
                    for r in rows
                ]
        except Exception as e:
            logger.error(f"❌ DB Fehler bei get_all_room_objects: {e}")
            return []
        finally:
            self.db_pool.putconn(conn)

    def save_room_object(self, obj: dict[str, Any]) -> int:
        """Insert oder Update. Gibt die DB-id zurück. obj['id'] = 0/None ⇒ Insert."""
        if not self.db_pool:
            return -1
        conn = self.db_pool.getconn()
        try:
            with conn:
                with conn.cursor() as cursor:
                    obj_id = obj.get("id")
                    params = (
                        str(obj.get("name", "Objekt")),
                        str(obj.get("obj_type", "box")),
                        float(obj.get("pos_x", 0.0)),
                        float(obj.get("pos_y", 0.0)),
                        float(obj.get("pos_z", 0.0)),
                        float(obj.get("size_w", 50.0)),
                        float(obj.get("size_h", 50.0)),
                        float(obj.get("size_d", 50.0)),
                        str(obj.get("color_hex", "#FFAA00")),
                        float(obj.get("rotation_yaw", 0.0)),
                        bool(obj.get("is_visible", True)),
                    )
                    if obj_id and int(obj_id) > 0:
                        cursor.execute(
                            "UPDATE room_objects SET name=%s, obj_type=%s, pos_x=%s, pos_y=%s, pos_z=%s, "
                            "size_w=%s, size_h=%s, size_d=%s, color_hex=%s, rotation_yaw=%s, is_visible=%s "
                            "WHERE id=%s RETURNING id",
                            (*params, int(obj_id)),
                        )
                    else:
                        cursor.execute(
                            "INSERT INTO room_objects (name, obj_type, pos_x, pos_y, pos_z, size_w, size_h, "
                            "size_d, color_hex, rotation_yaw, is_visible) "
                            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
                            params,
                        )
                    row = cursor.fetchone()
                    return int(row[0]) if row else -1
        except Exception as e:
            logger.error(f"❌ DB Fehler bei save_room_object: {e}")
            return -1
        finally:
            self.db_pool.putconn(conn)

    def delete_room_object(self, obj_id: int) -> None:
        if not self.db_pool:
            return
        conn = self.db_pool.getconn()
        try:
            with conn:
                with conn.cursor() as cursor:
                    cursor.execute("DELETE FROM room_objects WHERE id = %s", (int(obj_id),))
        except Exception as e:
            logger.error(f"❌ DB Fehler bei delete_room_object: {e}")
        finally:
            self.db_pool.putconn(conn)