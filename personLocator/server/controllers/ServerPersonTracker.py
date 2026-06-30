import time
import numpy as np
from typing import Dict, List, Any, Optional, Type, Union

from server.core.math.TriangulationMath import TriangulationMath
from server.models.GlobalPerson import GlobalPerson
from server.core.database.PersonDatabase import PersonDatabase


class ServerPersonTracker:
    """
    Verwaltet die Fusionierung.
    Extrem robuster Fallback: Überschreibt niemals gute 3D-Skelette mit leeren Daten!
    """

    def __init__(self, person_database: PersonDatabase, person_class_ref: Type[GlobalPerson]) -> None:
        self.db: PersonDatabase = person_database
        self.PersonClass: Type[GlobalPerson] = person_class_ref

        self.controller: Any = None
        self.global_persons: List[GlobalPerson] = []
        self.next_id: int = 1

        self.input_buffer: Dict[str, Dict[str, Any]] = {}

    def set_controller(self, controller: Any) -> None:
        self.controller = controller

    def update_camera_data(self, cam_name: str, person_list: list[dict[str, Any]]) -> None:
        self.input_buffer[cam_name] = {
            "time": time.time(),
            "persons": person_list
        }
        self.__process_fusion()

    def get_raw_tracks(self) -> Dict[str, list[dict[str, Any]]]:
        return {cam: data["persons"] for cam, data in self.input_buffer.items()}

    def force_merge_and_calibrate(self, name: str, assignments: List[dict[str, Any]]) -> str:
        heights: list[float] = [float(a.get('height', 175.0)) for a in assignments]
        avg_height: float = sum(heights) / len(heights) if heights else 175.0

        self.db.register_person(name, avg_height, [])
        return f"Zuweisung für '{name}' gespeichert."

    def _get_camera_projection_matrix(self, cam_name: str) -> Optional[np.ndarray]:
        if not self.controller or not hasattr(self.controller, 'config_cache'):
            return None

        cam_conf: dict[str, Any] = self.controller.config_cache.get(cam_name, {})
        cam_3d_data: Optional[dict[str, Any]] = cam_conf.get("cam_3d_data")

        if cam_3d_data and "P_matrix" in cam_3d_data:
            return np.array(cam_3d_data["P_matrix"], dtype=np.float32)

        return None

    def __process_fusion(self) -> None:
        now: float = time.time()
        self.global_persons = [p for p in self.global_persons if now - p.last_update < 3.0]

        unassigned_detections: list[dict[str, Any]] = []
        for cam_name, cam_data in self.input_buffer.items():
            if now - cam_data["time"] > 0.100:
                continue

            for p_data in cam_data["persons"]:
                if 'pos' not in p_data:
                    continue
                unassigned_detections.append({'cam': cam_name, 'data': p_data})

        for gp in self.global_persons:
            matched_dets: list[dict[str, Any]] = []
            for det in unassigned_detections[:]:
                raw_pos: np.ndarray = np.array(det['data']['pos'], dtype=np.float32)

                if float(np.linalg.norm(gp.pos - raw_pos)) < 250.0:
                    matched_dets.append(det)
                    unassigned_detections.remove(det)

            if not matched_dets:
                continue

            if len(matched_dets) == 1:
                det: dict[str, Any] = matched_dets[0]
                p_data: dict[str, Any] = det['data']
                raw_pos_upd: np.ndarray = np.array(p_data['pos'], dtype=np.float32)

                gp.update(
                    new_pos=raw_pos_upd,
                    new_height=float(p_data.get('height', gp.height)),
                    cam_name=str(det['cam']),
                    client_id=int(p_data.get('id', 0)),
                    raw_pos=raw_pos_upd,
                    keypoints=p_data.get('keypoints', []),
                    raw_data=p_data
                )

                new_skel: dict[int, np.ndarray] = p_data.get('metrics', {}).get('skeleton_3d', {})
                if new_skel:
                    gp.skeleton_3d = new_skel
                    gp.fusion_skel = new_skel
            else:
                def count_kps(d: dict[str, Any]) -> int:
                    k: Any = d['data'].get('keypoints')
                    if isinstance(k, list):
                        return len(k)
                    if isinstance(k, dict):
                        return len(k.keys())
                    return 0

                matched_dets.sort(key=count_kps, reverse=True)

                best_skel: dict[int, np.ndarray] = {}
                best_pos: np.ndarray = np.array(matched_dets[0]['data']['pos'], dtype=np.float32)

                for i in range(len(matched_dets)):
                    for j in range(i + 1, len(matched_dets)):
                        d1: dict[str, Any] = matched_dets[i]
                        d2: dict[str, Any] = matched_dets[j]

                        p1_matrix: Optional[np.ndarray] = self._get_camera_projection_matrix(str(d1['cam']))
                        p2_matrix: Optional[np.ndarray] = self._get_camera_projection_matrix(str(d2['cam']))

                        if p1_matrix is not None and p2_matrix is not None:
                            b1: list[float] = d1['data'].get('bbox', [0.0, 0.0, 0.0, 0.0])
                            b2: list[float] = d2['data'].get('bbox', [0.0, 0.0, 0.0, 0.0])
                            pt1_coord: tuple[float, float] = (float(b1[0] + b1[2]) / 2.0, float(b1[3]))
                            pt2_coord: tuple[float, float] = (float(b2[0] + b2[2]) / 2.0, float(b2[3]))

                            try:
                                t_pos: np.ndarray = TriangulationMath.triangulate_point(p1_matrix, p2_matrix, pt1_coord, pt2_coord)
                            except Exception:
                                t_pos = best_pos

                            kps1: list[dict[str, Any]] = d1['data'].get('keypoints', [])
                            kps2: list[dict[str, Any]] = d2['data'].get('keypoints', [])

                            try:
                                t_skel: dict[int, np.ndarray] = TriangulationMath.triangulate_skeleton(p1_matrix, p2_matrix, kps1, kps2)
                            except Exception:
                                t_skel = {}

                            if len(t_skel) > len(best_skel):
                                best_skel = t_skel
                                best_pos = t_pos
                            elif len(best_skel) == 0:
                                best_pos = t_pos

                heights_list: list[float] = [float(d['data'].get('height', 175)) for d in matched_dets]
                avg_height_upd: float = sum(heights_list) / len(heights_list) if heights_list else 175.0

                for det_upd in matched_dets:
                    c_name: str = str(det_upd['cam'])
                    c_data: dict[str, Any] = det_upd['data']
                    c_raw_pos: np.ndarray = np.array(c_data['pos'], dtype=np.float32)

                    gp.update(
                        new_pos=best_pos,
                        new_height=c_data.get('height', gp.height),
                        cam_name=c_name,
                        client_id=int(c_data.get('id', 0)),
                        raw_pos=c_raw_pos,
                        keypoints=c_data.get('keypoints', []),
                        raw_data=c_data
                    )

                gp.pos = best_pos
                gp.height = avg_height_upd

                if best_skel:
                    gp.skeleton_3d = best_skel
                    gp.fusion_skel = best_skel

                s_left: Optional[Any] = gp.skeleton_3d.get(5, gp.skeleton_3d.get("5"))
                s_right: Optional[Any] = gp.skeleton_3d.get(6, gp.skeleton_3d.get("6"))

                if s_left is not None and s_right is not None:
                    def get_coord(j_val: Any, axis: str) -> float:
                        if isinstance(j_val, dict):
                            return float(j_val.get(axis, 0.0))
                        idx_map: dict[str, int] = {'x': 0, 'y': 1, 'z': 2}
                        return float(j_val[idx_map[axis]])

                    dx_coord: float = get_coord(s_left, 'x') - get_coord(s_right, 'x')
                    dz_coord: float = get_coord(s_left, 'z') - get_coord(s_right, 'z')
                    norm_val_coord: float = float(np.linalg.norm([dz_coord, -dx_coord]))

                    if norm_val_coord > 0.001:
                        gp.forward_vec = np.array([dz_coord / norm_val_coord, 0.0, -dx_coord / norm_val_coord], dtype=np.float32)

        for det_new in unassigned_detections:
            p_data_new: dict[str, Any] = det_new['data']
            raw_pos_new: np.ndarray = np.array(p_data_new['pos'], dtype=np.float32)
            height_new: float = float(p_data_new.get('height', 175.0))

            new_gp: GlobalPerson = self.PersonClass(self.next_id, raw_pos_new, height_new)
            new_gp.set_database(self.db)

            known_name: Optional[str] = self.db.match_person(height_new)
            if known_name:
                new_gp.load_identity(known_name)

            new_gp.update(raw_pos_new, height_new, str(det_new['cam']), int(p_data_new.get('id', 0)), raw_pos_new, p_data_new.get('keypoints', []),
                          raw_data=p_data_new)

            new_gp.skeleton_3d = p_data_new.get('metrics', {}).get('skeleton_3d', {})
            new_gp.fusion_skel = new_gp.skeleton_3d

            self.global_persons.append(new_gp)
            self.next_id += 1