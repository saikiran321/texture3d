"""Stage 4: connected same-view faces become patches with a rectangle in
their view, and the seam vertices between patches are catalogued."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from texture3d.mapping import labeling
from texture3d.mapping.topology import Topology
from texture3d.mapping.types import NO_ID, FaceViews, Mesh, Rect, View

PATCH_BORDER = 2


@dataclass
class TexturePatch:
    #: View index, or NO_ID for the patch of untextured faces.
    label: int
    faces: np.ndarray
    #: Bounding rectangle of the faces' projections in the view, with a
    #: two-pixel border.
    rect: Rect


@dataclass
class PatchSet:
    #: The last patch holds the untextured faces.
    patches: List[TexturePatch]
    face_patch: np.ndarray
    #: (F, 3, 2), relative to ``patches[face_patch[f]].rect``.
    face_texcoords: np.ndarray
    seam_edges: np.ndarray

    @property
    def patch_count(self) -> int:
        return len(self.patches) - 1


@dataclass
class SeamData:
    """A mesh vertex where two or more patches meet, with, per patch, its
    projection in that patch's view and the seam edges leaving it."""

    #: Per seam vertex, the mesh vertex.
    vertex: np.ndarray
    #: CSR over seam vertices of (patch, projection) entries, in the order
    #: the seam edges registered them.
    entry_offsets: np.ndarray
    entry_patch: np.ndarray
    entry_proj: np.ndarray  # (P, 2) float32, image coordinates of the view
    #: CSR over entries of the seam edges leaving the vertex in that patch:
    #: the seam vertex at the other end and the face along the edge.
    edge_offsets: np.ndarray
    edge_other: np.ndarray
    edge_face: np.ndarray
    #: (seam vertex, patch) -> entry index.
    lookup: Dict[Tuple[int, int], int] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.vertex)

    def entries_of(self, sv: int) -> range:
        return range(int(self.entry_offsets[sv]), int(self.entry_offsets[sv + 1]))

    def entry_count(self, sv: int) -> int:
        return int(self.entry_offsets[sv + 1] - self.entry_offsets[sv])

    def edges_of(self, entry: int) -> range:
        return range(int(self.edge_offsets[entry]), int(self.edge_offsets[entry + 1]))

    @property
    def entry_sv(self) -> np.ndarray:
        return np.repeat(np.arange(self.count), np.diff(self.entry_offsets))


def build(mesh: Mesh, topology: Topology, face_views: FaceViews, outcome: labeling.Outcome, views: List[View]) -> PatchSet:
    """Connected components of same-view faces become patches, numbered in
    order of their lowest face, untextured faces collected last."""
    face_count = mesh.face_count
    labels = outcome.labels
    textured = labels != NO_ID

    # Adjacent faces with the same view join one component. Faces with no
    # view never join anything; they all end in the empty patch regardless.
    ff = topology.face_faces
    f_idx = np.broadcast_to(np.arange(face_count)[:, None], ff.shape)
    same = (ff != NO_ID) & (f_idx < ff) & textured[f_idx] & (labels[f_idx] == labels[np.where(ff != NO_ID, ff, 0)])
    graph = coo_matrix(
        (np.ones(int(same.sum()), dtype=np.int8), (f_idx[same], ff[same])), shape=(face_count, face_count)
    )
    n_components, component = connected_components(graph, directed=False)
    textured_components = np.unique(component[textured])
    patch_of_component = np.full(n_components, -1, dtype=np.int64)
    patch_of_component[textured_components] = np.arange(len(textured_components))
    num_patches = len(textured_components)
    face_patch = np.where(textured, patch_of_component[component], num_patches)

    order = np.argsort(face_patch, kind="stable")
    counts = np.bincount(face_patch, minlength=num_patches + 1)
    starts = np.concatenate([[0], np.cumsum(counts)])
    patches: List[TexturePatch] = []
    for p in range(num_patches):
        faces = order[starts[p] : starts[p + 1]]
        patches.append(TexturePatch(label=int(labels[faces[0]]), faces=faces, rect=Rect(0, 0, 0, 0)))
    empty_faces = order[starts[num_patches] : starts[num_patches + 1]]
    patches.append(TexturePatch(label=NO_ID, faces=empty_faces, rect=Rect(0, 0, 5, 5)))

    # Project every patch into its view, take the bounding rectangle with a
    # two-pixel border, and make the coordinates relative.
    texcoords = np.zeros((face_count, 3, 2), dtype=np.float32)
    for v in np.unique(labels[textured]):
        faces_v = np.flatnonzero(labels == v)
        proj = views[int(v)].camera.project(mesh.positions)
        texcoords[faces_v] = proj[mesh.faces[faces_v]]
    textured_sorted = order[: starts[num_patches]]
    if num_patches:
        corners = texcoords[textured_sorted].reshape(-1, 2)
        corner_starts = starts[:num_patches] * 3
        lo = np.minimum.reduceat(corners, corner_starts, axis=0)
        hi = np.maximum.reduceat(corners, corner_starts, axis=0)
        rx = np.floor(lo[:, 0]).astype(np.int64) - PATCH_BORDER
        ry = np.floor(lo[:, 1]).astype(np.int64) - PATCH_BORDER
        rw = np.ceil(hi[:, 0] - lo[:, 0]).astype(np.int64) + PATCH_BORDER * 2
        rh = np.ceil(hi[:, 1] - lo[:, 1]).astype(np.int64) + PATCH_BORDER * 2
        for p in range(num_patches):
            patches[p].rect = Rect(int(rx[p]), int(ry[p]), int(rw[p]), int(rh[p]))
        offset = np.stack([rx, ry], axis=-1).astype(np.float32)
        texcoords[textured] -= offset[face_patch[textured]][:, None, :]
    texcoords[empty_faces] = np.float32(0.5)
    return PatchSet(patches=patches, face_patch=face_patch, face_texcoords=texcoords, seam_edges=outcome.seam_edges)


def create_seam_vertices(mesh: Mesh, patch_set: PatchSet) -> SeamData:
    """Every seam edge registers its two vertices with the patches on both
    sides, remembering the face and the other seam vertex the edge leads to."""
    num_patches = patch_set.patch_count
    edges = patch_set.seam_edges
    faces = mesh.faces
    tex = patch_set.face_texcoords
    f0 = edges[:, 0]
    f1 = edges[:, 1]
    p0 = patch_set.face_patch[f0]
    p1 = patch_set.face_patch[f1]
    fa = faces[f0]
    fb = faces[f1]
    shared = fa[:, :, None] == fb[:, None, :]  # (E, 3, 3)
    in_f0 = shared.any(axis=2)
    keep = (p0 != p1) & (in_f0.sum(axis=1) == 2)
    e_idx = np.flatnonzero(keep)
    slots0 = np.argsort(~in_f0[e_idx], axis=1, kind="stable")[:, :2]
    slots1 = shared[e_idx][np.arange(len(e_idx))[:, None], slots0].argmax(axis=2)
    vs = fa[e_idx][np.arange(len(e_idx))[:, None], slots0]
    rect_x = np.array([p.rect.x for p in patch_set.patches], dtype=np.float32)
    rect_y = np.array([p.rect.y for p in patch_set.patches], dtype=np.float32)

    slot_of_vertex: Dict[int, int] = {}
    sv_vertex: List[int] = []
    sv_entries: List[List[int]] = []
    entry_patch: List[int] = []
    entry_proj: List[Tuple[float, float]] = []
    entry_edges: List[List[Tuple[int, int]]] = []
    lookup: Dict[Tuple[int, int], int] = {}

    def seam_vertex(v: int) -> int:
        s = slot_of_vertex.get(v)
        if s is None:
            s = len(sv_vertex)
            slot_of_vertex[v] = s
            sv_vertex.append(v)
            sv_entries.append([])
        return s

    def entry(sv: int, patch: int) -> int:
        e = lookup.get((sv, patch))
        if e is None:
            e = len(entry_patch)
            lookup[(sv, patch)] = e
            entry_patch.append(patch)
            entry_proj.append((0.0, 0.0))
            entry_edges.append([])
            sv_entries[sv].append(e)
        return e

    for i, e in enumerate(e_idx):
        v0, v1 = int(vs[i, 0]), int(vs[i, 1])
        s0 = seam_vertex(v0)
        s1 = seam_vertex(v1)
        for patch, face, slots in ((int(p0[e]), int(f0[e]), slots0[i]), (int(p1[e]), int(f1[e]), slots1[i])):
            if patch >= num_patches:
                continue
            ox, oy = float(rect_x[patch]), float(rect_y[patch])
            a = entry(s0, patch)
            entry_edges[a].append((s1, face))
            t = tex[face, int(slots[0])]
            entry_proj[a] = (float(t[0]) + ox, float(t[1]) + oy)
            b = entry(s1, patch)
            entry_edges[b].append((s0, face))
            t = tex[face, int(slots[1])]
            entry_proj[b] = (float(t[0]) + ox, float(t[1]) + oy)

    # Flatten: entries per seam vertex in registration order, edges per entry.
    entry_order = [e for entries in sv_entries for e in entries]
    new_index = {old: new for new, old in enumerate(entry_order)}
    entry_offsets = np.zeros(len(sv_vertex) + 1, dtype=np.int64)
    np.cumsum([len(entries) for entries in sv_entries], out=entry_offsets[1:])
    patch_arr = np.array([entry_patch[e] for e in entry_order], dtype=np.int64)
    proj_arr = np.array([entry_proj[e] for e in entry_order], dtype=np.float32).reshape(-1, 2)
    edge_counts = [len(entry_edges[e]) for e in entry_order]
    edge_offsets = np.zeros(len(entry_order) + 1, dtype=np.int64)
    np.cumsum(edge_counts, out=edge_offsets[1:])
    edge_other = np.array([o for e in entry_order for o, _ in entry_edges[e]], dtype=np.int64)
    edge_face = np.array([f for e in entry_order for _, f in entry_edges[e]], dtype=np.int64)
    lookup_new = {key: new_index[old] for key, old in lookup.items()}
    return SeamData(
        vertex=np.array(sv_vertex, dtype=np.int64),
        entry_offsets=entry_offsets,
        entry_patch=patch_arr,
        entry_proj=proj_arr,
        edge_offsets=edge_offsets,
        edge_other=edge_other,
        edge_face=edge_face,
        lookup=lookup_new,
    )


def merge_contained(patch_set: PatchSet) -> None:
    """Fold a patch whose rectangle lies inside another in the same view.

    Every patch joins the lowest-numbered surviving patch of its view whose
    rectangle contains its own.
    """
    patches = patch_set.patches
    num_patches = patch_set.patch_count
    if num_patches < 2:
        return
    labels = np.array([p.label for p in patches[:num_patches]])
    x = np.array([p.rect.x for p in patches[:num_patches]])
    y = np.array([p.rect.y for p in patches[:num_patches]])
    w = np.array([p.rect.width for p in patches[:num_patches]])
    h = np.array([p.rect.height for p in patches[:num_patches]])
    target = np.arange(num_patches)
    for label in np.unique(labels):
        ids = np.flatnonzero(labels == label)
        if len(ids) < 2:
            continue
        xi, yi, wi, hi = x[ids], y[ids], w[ids], h[ids]
        # contained[j, i]: rect j lies inside rect i.
        contained = (
            (xi[:, None] >= xi[None, :])
            & (yi[:, None] >= yi[None, :])
            & (xi[:, None] + wi[:, None] <= xi[None, :] + wi[None, :])
            & (yi[:, None] + hi[:, None] <= yi[None, :] + hi[None, :])
        )
        identical = contained & contained.T
        np.fill_diagonal(contained, False)
        # Of two identical rectangles the earlier absorbs the later.
        later = np.arange(len(ids))[None, :] > np.arange(len(ids))[:, None]
        contained &= ~(identical & later)
        has = contained.any(axis=1)
        first = np.argmax(contained, axis=1)
        target[ids[has]] = ids[first[has]]
    # Follow chains: a patch absorbed into one that is itself absorbed ends in
    # the final absorber, with the offsets adding up.
    for _ in range(num_patches):
        nxt = target[target]
        if np.array_equal(nxt, target):
            break
        target = nxt
    merged = target != np.arange(num_patches)
    if not merged.any():
        return
    for j in np.flatnonzero(merged):
        i = int(target[j])
        small, big = patches[j], patches[i]
        offset = np.array([small.rect.x - big.rect.x, small.rect.y - big.rect.y], dtype=np.float32)
        patch_set.face_texcoords[small.faces] += offset
        big.faces = np.concatenate([big.faces, small.faces])
    survivors = [patches[p] for p in range(num_patches) if not merged[p]] + [patches[num_patches]]
    patch_set.patches = survivors
    for p, patch in enumerate(survivors):
        patch_set.face_patch[patch.faces] = p
