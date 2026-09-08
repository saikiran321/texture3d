"""Stage 3: one view per face.

The face graph (faces adjacent across an edge, both seen by at least one view)
is an MRF whose labels are the views plus "none". A face's data cost for a
view falls with the view's quality, normalised by the 95th percentile of all
qualities so a few very sharp faces do not flatten the rest; the Potts
smoothness term makes neighbours prefer the same view, which is what grows
the patches. Solved by synchronous loopy belief propagation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from texture3d.mapping.topology import Topology
from texture3d.mapping.types import NO_ID, FaceViews, Mesh, View

MAX_ENERGY = np.float32(1000)


@dataclass
class Outcome:
    #: Per face: view index, or NO_ID for no view.
    labels: np.ndarray
    #: The 95th-percentile quality the data costs were normalised by.
    norm_quality: float
    energy: float
    iterations: int
    #: Face pairs across an edge where exactly one side has no view, in face
    #: order, then pairs whose labels differ, in the graph's edge order. The
    #: seam leveling walks these.
    seam_edges: np.ndarray
    #: The face graph's undirected edges, smaller face first, in face order.
    graph_edges: np.ndarray
    #: Faces no view rasterised that took a neighbour's view (see ``fill_unseen``).
    filled: Optional[np.ndarray] = None


def approximate_permille(values: np.ndarray, permille: float = 0.95) -> float:
    """Approximate ``permille`` of ``values`` with a 1000-bin histogram.

    Zero when every quality is zero: the data costs then read every view as
    as good as the best and the smoothness term decides.
    """
    v = np.asarray(values, dtype=np.float32)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return 0.0
    max_quality = float(v.max())
    if max_quality <= 0:
        return 0.0
    bins = 1000
    bin_interval = np.float32(bins) / np.float32(max_quality)
    i = (v * bin_interval).astype(np.int64)
    freq = np.bincount(i[(i >= 0) & (i < bins)], minlength=bins)
    total = int(freq.sum())
    num = 0
    upper = 0.0
    for b in range(bins):
        if num / total > permille:
            return upper
        num += int(freq[b])
        upper = float((np.float32(b) * np.float32(max_quality)) / np.float32(bins - 1))
    return max_quality


class BeliefPropagation:
    """The flattened MRF and its synchronous sweeps.

    Labels per node in the order they were set; messages per directed edge
    with one entry per label of the receiving node.
    """

    def __init__(
        self,
        label_offset: np.ndarray,
        labels_flat: np.ndarray,
        costs_flat: np.ndarray,
        edge_from: np.ndarray,
        edge_to: np.ndarray,
    ):
        self.label_offset = label_offset
        self.labels_flat = labels_flat
        self.costs_flat = costs_flat.astype(np.float32)
        self.edge_from = edge_from
        self.edge_to = edge_to
        node_count = len(label_offset) - 1
        n_labels = np.diff(label_offset)
        self.node_of_label = np.repeat(np.arange(node_count), n_labels)

        edge_count = len(edge_from)
        per_edge = n_labels[edge_to]
        self.msg_offset = np.zeros(edge_count + 1, dtype=np.int64)
        np.cumsum(per_edge, out=self.msg_offset[1:])
        total = int(self.msg_offset[-1])
        self.slot_edge = np.repeat(np.arange(edge_count), per_edge)
        slot_j = np.arange(total) - self.msg_offset[self.slot_edge]
        # Message slot -> the (node, label) it speaks about.
        self.s_idx = label_offset[edge_to[self.slot_edge]] + slot_j
        # A message on edge e (n1 -> n2) for n2's label j takes the minimum over
        # n1's labels k of n1's belief without n2's message plus the Potts term.
        # That belief lives in the slots of the reverse edge (its messages are
        # indexed by n1's labels), so the matching label's slot is found once.
        label_max = int(labels_flat.max()) + 1 if len(labels_flat) else 1
        node_key = self.node_of_label * label_max + labels_flat
        n1 = edge_from[self.slot_edge]
        label_j = labels_flat[self.s_idx]
        query = n1 * label_max + label_j
        pos = np.searchsorted(node_key, query)
        pos_c = np.minimum(pos, len(node_key) - 1)
        found = (node_key[pos_c] == query) & (label_j != 0)
        k = pos_c - label_offset[n1]
        self.match = np.where(found, self.msg_offset[self.slot_edge ^ 1] + k, -1)
        self.reverse_of_slot = self.slot_edge ^ 1
        self.old = np.zeros(total, dtype=np.float32)
        self.new = np.zeros(total, dtype=np.float32)
        self.label, self.data_cost = self._first_min(self.costs_flat)

    def _first_min(self, energy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Per node the first label of least energy, as (label, cost)."""
        node_min = np.minimum.reduceat(energy, self.label_offset[:-1])
        candidates = np.flatnonzero(energy == node_min[self.node_of_label])
        _, first = np.unique(self.node_of_label[candidates], return_index=True)
        chosen = candidates[first]
        return self.labels_flat[chosen], self.costs_flat[chosen]

    def compute_energy(self) -> float:
        lf = self.label[self.edge_from]
        lt = self.label[self.edge_to]
        agree = (lf == lt) & (lf != 0)
        return float(self.data_cost.sum(dtype=np.float64)) + float(MAX_ENERGY) * float((~agree).sum())

    def sweep(self) -> None:
        total_labels = len(self.labels_flat)
        incoming = np.bincount(self.s_idx, weights=self.old, minlength=total_labels).astype(np.float32)
        belief = self.costs_flat[self.s_idx] + incoming[self.s_idx] - self.old
        segment_min = np.minimum.reduceat(belief, self.msg_offset[:-1])
        floor = segment_min[self.reverse_of_slot] + MAX_ENERGY
        matched = np.where(self.match >= 0, belief[np.maximum(self.match, 0)], np.float32(np.inf))
        np.minimum(floor, matched, out=self.new)
        # Normalise each edge's messages to a minimum of zero.
        edge_min = np.minimum.reduceat(self.new, self.msg_offset[:-1])
        self.new -= edge_min[self.slot_edge]
        self.old, self.new = self.new, self.old
        incoming = np.bincount(self.s_idx, weights=self.old, minlength=total_labels).astype(np.float32)
        self.label, self.data_cost = self._first_min(self.costs_flat + incoming)

    def optimize(self, log: Optional[Callable[[str], None]] = None) -> Tuple[float, int]:
        """Sweep until the energy stops falling, tolerating a few increases,
        at most 100 sweeps."""
        energy = self.compute_energy()
        i = 0
        increases = 0
        total_increases = 0
        while True:
            last = energy
            self.sweep()
            energy = self.compute_energy()
            diff = last - energy
            i += 1
            if log:
                log(f"  sweep {i}: energy {energy:.0f}")
            if i > 100 or diff == 0:
                break
            if diff < 0:
                total_increases += 1
                increases += 1
                if increases > 3:
                    break
            else:
                if total_increases > 5:
                    break
                increases = 0
        return energy, i


def fill_unseen(
    mesh: Mesh, topology: Topology, labels: np.ndarray, views: List[View], passes: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """Give a face no view rasterised the view of a textured neighbour.

    On a dense mesh a face smaller than a pixel in every keyframe is never
    scored. It takes the view of the first edge neighbour that has one,
    provided its three vertices project in front of that view's camera and
    inside its image. Repeated ``passes`` times so small clusters fill from
    their rim. Without this such faces land in the empty patch and speckle
    the surface with the empty colour. Returns the new labels and the mask
    of faces that were filled.
    """
    labels = labels.copy()
    filled = np.zeros(len(labels), dtype=bool)
    ff = topology.face_faces
    for _ in range(passes):
        unlabelled = labels == NO_ID
        neighbour_label = np.where(ff != NO_ID, labels[np.maximum(ff, 0)], NO_ID)
        has = neighbour_label != NO_ID
        candidates = np.flatnonzero(unlabelled & has.any(axis=1))
        if len(candidates) == 0:
            break
        chosen = neighbour_label[candidates, np.argmax(has[candidates], axis=1)]
        changed = False
        for v in np.unique(chosen):
            faces = candidates[chosen == v]
            view = views[int(v)]
            cam = view.camera.to_camera(mesh.positions[mesh.faces[faces].ravel()])
            z = cam[:, 2]
            safe = cam.copy()
            safe[z <= 0, 2] = 1.0
            px = view.camera.to_image(safe)
            ok = (z > 0) & (px[:, 0] >= 0) & (px[:, 1] >= 0) & (px[:, 0] <= view.width - 1) & (px[:, 1] <= view.height - 1)
            accept = faces[ok.reshape(-1, 3).all(axis=1)]
            if len(accept):
                labels[accept] = v
                filled[accept] = True
                changed = True
        if not changed:
            break
    return labels, filled


def assign(
    mesh: Mesh,
    topology: Topology,
    face_views: FaceViews,
    data_smoothness_ratio: float = 0.1,
    log: Optional[Callable[[str], None]] = None,
    views: Optional[List[View]] = None,
    fill: bool = False,
) -> Outcome:
    face_count = mesh.face_count
    ff = topology.face_faces
    f_idx = np.broadcast_to(np.arange(face_count)[:, None], ff.shape)
    has = (ff != NO_ID) & (f_idx < ff)
    visible = face_views.visible
    vis_f = visible[f_idx]
    vis_g = visible[np.where(ff != NO_ID, ff, 0)]
    graph_mask = has & vis_f & vis_g
    seam_mask = has & (vis_f != vis_g)
    graph_edges = np.stack([f_idx[graph_mask], ff[graph_mask]], axis=-1)
    seam_first = np.stack([f_idx[seam_mask], ff[seam_mask]], axis=-1)

    norm_quality = approximate_permille(face_views.quality)
    max_energy = np.float32(data_smoothness_ratio) * MAX_ENERGY

    # Labels per node: 0 ("no view") first, then the face's views plus one.
    counts = face_views.counts
    n_labels = counts + 1
    label_offset = np.zeros(face_count + 1, dtype=np.int64)
    np.cumsum(n_labels, out=label_offset[1:])
    total = int(label_offset[-1])
    labels_flat = np.zeros(total, dtype=np.int64)
    costs_flat = np.full(total, max_energy, dtype=np.float32)
    is_view = np.ones(total, dtype=bool)
    is_view[label_offset[:-1]] = False
    labels_flat[is_view] = face_views.view + 1
    q = face_views.quality
    if norm_quality > 0:
        normalized = np.where(q >= norm_quality, np.float32(1), q / np.float32(norm_quality))
    else:
        normalized = np.ones_like(q)
    costs_flat[is_view] = (np.float32(1) - normalized) * max_energy

    edge_from = np.stack([graph_edges[:, 0], graph_edges[:, 1]], axis=-1).ravel()
    edge_to = np.stack([graph_edges[:, 1], graph_edges[:, 0]], axis=-1).ravel()
    inference = BeliefPropagation(label_offset, labels_flat, costs_flat, edge_from, edge_to)
    energy, iterations = inference.optimize(log)

    labels = np.where(inference.label > 0, inference.label - 1, NO_ID)
    filled = np.zeros(face_count, dtype=bool)
    if fill and views is not None:
        labels, filled = fill_unseen(mesh, topology, labels, views)
        if log and filled.any():
            log(f"  {int(filled.sum())} unseen faces took a neighbour's view")
        # Filled faces count as seen for the seams, which are then recomputed
        # over every edge: the same two blocks as below, in the same order.
        seen = visible | filled
        seen_f = seen[f_idx]
        seen_g = seen[np.where(ff != NO_ID, ff, 0)]
        seam_mask = has & (seen_f != seen_g)
        seam_first = np.stack([f_idx[seam_mask], ff[seam_mask]], axis=-1)
        both = has & seen_f & seen_g
        differ = both & (labels[f_idx] != labels[np.where(ff != NO_ID, ff, 0)])
        seam_second = np.stack([f_idx[differ], ff[differ]], axis=-1)
    else:
        seam_second = graph_edges[labels[graph_edges[:, 0]] != labels[graph_edges[:, 1]]]
    seam_edges = np.concatenate([seam_first, seam_second]) if len(seam_first) or len(seam_second) else np.zeros((0, 2), np.int64)
    return Outcome(
        labels=labels,
        norm_quality=norm_quality,
        energy=energy,
        iterations=iterations,
        seam_edges=seam_edges,
        graph_edges=graph_edges,
        filled=filled,
    )
