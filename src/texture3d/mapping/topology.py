"""Stage 1: incidence, adjacency and smoothed normals of the mesh."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from texture3d.mapping.types import NO_ID, Mesh


@dataclass
class Topology:
    #: CSR: the faces incident to vertex ``v`` are
    #: ``vertex_face_list[vertex_face_offsets[v]:vertex_face_offsets[v + 1]]``, ascending.
    vertex_face_offsets: np.ndarray
    vertex_face_list: np.ndarray
    #: Per face, the face across edge ``(face[v], face[(v + 1) % 3])`` for ``v``
    #: in 0, 1, 2, or NO_ID at a boundary edge.
    face_faces: np.ndarray
    #: Unit face normals after three smoothing passes.
    face_normals: np.ndarray

    def faces_of_vertex(self, v: int) -> np.ndarray:
        return self.vertex_face_list[self.vertex_face_offsets[v] : self.vertex_face_offsets[v + 1]]


def cv_normalize(v: np.ndarray) -> np.ndarray:
    """Unit-length vectors: the norm in double, the zero vector left at
    zero, each component scaled in double and narrowed to float32."""
    d = np.asarray(v, dtype=np.float64)
    n = np.sqrt((d * d).sum(axis=-1, keepdims=True))
    scale = np.where(n != 0, 1.0 / np.where(n != 0, n, 1.0), 0.0)
    return (d * scale).astype(np.float32)


def build(mesh: Mesh) -> Topology:
    faces = mesh.faces
    face_count = len(faces)
    vertex_count = len(mesh.positions)

    # Incident faces, CSR, ascending per vertex.
    flat = faces.ravel()
    order = np.argsort(flat, kind="stable")
    vertex_face_list = (order // 3).astype(np.int64)
    counts = np.bincount(flat, minlength=vertex_count)
    vertex_face_offsets = np.zeros(vertex_count + 1, dtype=np.int64)
    np.cumsum(counts, out=vertex_face_offsets[1:])

    # Face adjacency across each edge slot. On a manifold edge the neighbour
    # is the one other face; on an edge shared by more faces it is the
    # lowest-numbered other face.
    a = faces
    b = np.roll(faces, -1, axis=1)
    lo = np.minimum(a, b).ravel()
    hi = np.maximum(a, b).ravel()
    key = lo * vertex_count + hi
    slot_face = np.repeat(np.arange(face_count, dtype=np.int64), 3)
    order = np.lexsort((slot_face, key))
    key_sorted = key[order]
    face_sorted = slot_face[order]
    group_start = np.ones(len(order), dtype=bool)
    group_start[1:] = key_sorted[1:] != key_sorted[:-1]
    group_id = np.cumsum(group_start) - 1
    starts = np.flatnonzero(group_start)
    sizes = np.diff(np.append(starts, len(order)))
    first = face_sorted[starts]
    second = np.where(sizes >= 2, face_sorted[np.minimum(starts + 1, len(order) - 1)], NO_ID)
    own = face_sorted
    neighbour = np.where(own == first[group_id], second[group_id], first[group_id])
    neighbour = np.where(sizes[group_id] >= 2, neighbour, NO_ID)
    face_faces = np.full(face_count * 3, NO_ID, dtype=np.int64)
    face_faces[order] = neighbour
    face_faces = face_faces.reshape(face_count, 3)

    # Face normals, then three passes of averaging each normal with its edge
    # neighbours within 25 degrees, half original and half average.
    p = mesh.positions
    v0 = p[faces[:, 0]]
    v1 = p[faces[:, 1]]
    v2 = p[faces[:, 2]]
    normals = cv_normalize(np.cross(v1 - v0, v2 - v0))
    cos_max_gradient = np.float32(np.cos(np.deg2rad(25.0)))
    valid = face_faces != NO_ID
    safe = np.where(valid, face_faces, 0)
    for _ in range(3):
        neighbour_normals = normals[safe]  # (F, 3, 3)
        cosine = np.clip((normals[:, None, :] * neighbour_normals).sum(axis=-1), -1, 1)
        take = valid & (cosine >= cos_max_gradient)
        summed = (neighbour_normals * take[..., None].astype(np.float32)).sum(axis=1)
        average = cv_normalize(summed)
        normals = cv_normalize(normals * np.float32(0.5) + average * np.float32(0.5))

    return Topology(
        vertex_face_offsets=vertex_face_offsets,
        vertex_face_list=vertex_face_list,
        face_faces=face_faces,
        face_normals=normals,
    )
