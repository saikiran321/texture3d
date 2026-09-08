"""The texture mapping's pure pieces on inputs small enough to reason about,
and an end-to-end bake on the synthetic checker scene."""

import numpy as np
import trimesh

from texture3d import BakeConfig, PinholeCamera, RGBDFrame, bake
from texture3d.mapping import Camera, Mesh, View, atlas, labeling, leveling, raster, topology
from texture3d.mapping.types import NO_ID, Rect, rgb2ycbcr, ycbcr2rgb

# ── The synthetic scene ──────────────────────────────────────────────────────
# A checkerboard plane at z = 2 m observed by three pinhole cameras with known
# poses; RGB images are rendered analytically (each pixel's colour is the
# checker colour of the world point it sees) and depth is exactly 2 m. Baking
# must reproduce the checker pattern in the texture.

HALF = 0.6  # plane half-extent in meters
Z = 2.0  # plane depth in meters
CELL = 0.3  # checker cell size
COLOR_A = np.array([200, 60, 60], dtype=np.uint8)
COLOR_B = np.array([60, 60, 200], dtype=np.uint8)
CAMERA = PinholeCamera(width=64, height=48, fx=50.0, fy=50.0, cx=32.0, cy=24.0)


def checker(x, y):
    """Checker color at world (x, y)."""
    idx = np.floor(x / CELL) + np.floor(y / CELL)
    a = np.asarray(idx, dtype=np.int64) % 2 == 0
    return np.where(a[..., None], COLOR_A, COLOR_B).astype(np.uint8)


def make_plane_mesh(n: int = 8, with_uv: bool = False) -> trimesh.Trimesh:
    """Grid plane at z=Z, winding so face normals point toward the cameras (-z)."""
    lin = np.linspace(-HALF, HALF, n + 1)
    xx, yy = np.meshgrid(lin, lin)
    verts = np.stack([xx.ravel(), yy.ravel(), np.full(xx.size, Z)], axis=-1)
    faces = []
    for r in range(n):
        for c in range(n):
            i = r * (n + 1) + c
            j = i + n + 1
            faces.append([i, j, i + 1])  # cross(v1-v0, v2-v0) ~ -z
            faces.append([i + 1, j, j + 1])
    mesh = trimesh.Trimesh(vertices=verts, faces=np.array(faces), process=False)
    assert (mesh.face_normals[:, 2] < 0).all()
    if with_uv:
        uv = (verts[:, :2] + HALF) / (2 * HALF) * 0.9 + 0.05
        mesh.visual = trimesh.visual.TextureVisuals(uv=uv)
    return mesh


def make_frames() -> list:
    frames = []
    for dx in (-0.2, 0.0, 0.2):
        pose = np.eye(4)
        pose[0, 3] = dx
        i, j = np.meshgrid(np.arange(CAMERA.width), np.arange(CAMERA.height))
        x = (i - CAMERA.cx) * Z / CAMERA.fx + dx
        y = (j - CAMERA.cy) * Z / CAMERA.fy
        rgb = checker(x, y)
        depth = np.full((CAMERA.height, CAMERA.width), Z, dtype=np.float32)
        frames.append(RGBDFrame(pose_c2w=pose, camera=CAMERA, rgb=rgb, depth=depth))
    return frames


def sample_texture(result, uv: np.ndarray) -> np.ndarray:
    """Sample the baked texture at normalized UVs with OBJ semantics (v up)."""
    s = result.texture.shape[0]
    px = np.clip((uv[:, 0] * s).astype(int), 0, s - 1)
    py = np.clip(((1.0 - uv[:, 1]) * s).astype(int), 0, s - 1)
    return result.texture[py, px].astype(np.float64)



# ── Topology ─────────────────────────────────────────────────────────────────


def test_topology_box_adjacency_and_normals():
    box = trimesh.creation.box(extents=[1.0, 1.0, 1.0])
    mesh = Mesh(positions=box.vertices, faces=box.faces)
    topo = topology.build(mesh)
    # A closed box: every edge has a neighbour, and the neighbour relation is symmetric.
    assert (topo.face_faces != NO_ID).all()
    for f in range(mesh.face_count):
        for g in topo.face_faces[f]:
            assert f in topo.face_faces[g]
    # Every vertex of a box touches at least three faces.
    assert (np.diff(topo.vertex_face_offsets) >= 3).all()
    # Coplanar neighbours are within 25 degrees; face normals across a box
    # edge are not, so smoothing leaves the box's normals as they were.
    assert np.allclose(np.linalg.norm(topo.face_normals, axis=1), 1, atol=1e-6)
    assert np.allclose(topo.face_normals, box.face_normals, atol=1e-5)


# ── Rasterisers ──────────────────────────────────────────────────────────────


def _brute_bary(tri, width, height):
    pixels = set()
    for y in range(height):
        for x in range(width):
            p = np.array([x, y], dtype=np.float32)
            b1 = raster.edge(tri[1], tri[2], p)
            b2 = raster.edge(tri[2], tri[0], p)
            b3 = raster.edge(tri[0], tri[1], p)
            area = raster.edge(tri[0], tri[1], tri[2])
            if area > 0 and b1 / area >= 0 and b2 / area >= 0 and b3 / area >= 0:
                pixels.add((x, y))
    return pixels


def test_triangles_bary_matches_brute_force():
    rng = np.random.default_rng(1)
    tris = rng.uniform(-5, 40, size=(30, 3, 2)).astype(np.float32)
    got = {}
    for ti, x, y, bary in raster.triangles_bary(tris, 32, 24, cull=True):
        for t, px, py, b in zip(ti, x, y, bary):
            got.setdefault(int(t), set()).add((int(px), int(py)))
            assert (b >= 0).all()
            assert abs(b.sum() - 1) < 1e-4
    for t in range(len(tris)):
        assert got.get(t, set()) == _brute_bary(tris[t], 32, 24)


def test_triangles_fixed_two_triangles_share_no_pixel():
    # The top-left fill convention: the two front-facing triangles of a square
    # cover it exactly once between them, and a back-facing one draws nothing.
    square = np.array(
        [[[2.0, 2.0], [12.0, 12.0], [12.0, 2.0]], [[2.0, 2.0], [2.0, 12.0], [12.0, 12.0]]], dtype=np.float32
    )
    assert (raster.edge(square[:, 0], square[:, 1], square[:, 2]) > 0).all()
    ti, x, y = raster.triangles_fixed(square, 16, 16)
    pixels = list(zip(x.tolist(), y.tolist()))
    assert len(pixels) == len(set(pixels))
    assert len(pixels) == 100
    assert set(pixels) == {(px, py) for px in range(2, 12) for py in range(2, 12)}
    _, bx, _ = raster.triangles_fixed(square[:1, ::-1].copy(), 16, 16)
    assert len(bx) == 0


def test_lines_are_eight_connected_and_end_exclusive():
    p1 = np.array([[0.0, 0.0], [5.0, 1.0], [3.0, 3.0]], dtype=np.float32)
    p2 = np.array([[6.0, 3.0], [0.0, 8.0], [3.0, 3.0]], dtype=np.float32)
    li, x, y = raster.lines(p1, p2)
    assert (li == 2).sum() == 0  # a zero-length line draws nothing
    for line in (0, 1):
        pts = np.stack([x[li == line], y[li == line]], axis=-1)
        steps = np.abs(np.diff(pts, axis=0)).max(axis=1)
        assert (steps == 1).all()
        assert len(pts) == 6 if line == 0 else 5


# ── Labelling ────────────────────────────────────────────────────────────────


def test_approximate_permille_edge_cases():
    assert labeling.approximate_permille(np.array([], dtype=np.float32)) == 0
    assert labeling.approximate_permille(np.zeros(10, dtype=np.float32)) == 0
    assert labeling.approximate_permille(np.array([1.0, np.nan, 2.0], dtype=np.float32)) > 0
    values = np.linspace(0, 100, 10001).astype(np.float32)
    assert abs(labeling.approximate_permille(values) - 95) < 0.5


def test_belief_propagation_prefers_one_label_along_a_chain():
    # Four nodes in a chain, each sees views 1 and 2 with almost equal costs;
    # the Potts term makes them agree on one view instead of alternating.
    n = 4
    label_offset = np.arange(0, 3 * (n + 1), 3)
    labels = np.tile([0, 1, 2], n)
    costs = np.tile([100.0, 40.0, 41.0], n).astype(np.float32)
    costs[1 * 3 + 1], costs[1 * 3 + 2] = 41.0, 40.0  # node 1 slightly prefers view 2
    edge_from = np.array([0, 1, 1, 2, 2, 3])
    edge_to = np.array([1, 0, 2, 1, 3, 2])
    bp = labeling.BeliefPropagation(label_offset, labels, costs, edge_from, edge_to)
    energy, iterations = bp.optimize()
    assert len(set(bp.label.tolist())) == 1
    assert bp.label[0] in (1, 2)
    assert energy <= 4 * 41 + 1e-3
    assert iterations >= 1


def test_fill_unseen_takes_a_neighbours_view_when_it_projects():
    # Two triangles sharing an edge, one labelled with view 0, one unseen. The
    # camera at the origin looks down +z at both, so the unseen one fills.
    positions = np.array([[0, 0, 2], [1, 0, 2], [0, 1, 2], [1, 1, 2]], dtype=np.float32)
    faces = np.array([[0, 2, 1], [1, 2, 3]])
    mesh = Mesh(positions=positions, faces=faces)
    topo = topology.build(mesh)
    camera = Camera(fx=20.0, fy=20.0, cx=32.0, cy=24.0, R=np.eye(3), C=np.zeros(3))
    view = View(bgr=np.zeros((48, 64, 3), dtype=np.uint8), camera=camera)
    labels, filled = labeling.fill_unseen(mesh, topo, np.array([0, NO_ID]), [view])
    assert labels.tolist() == [0, 0] and filled.tolist() == [False, True]
    # Behind the camera nothing fills.
    behind = Camera(fx=20.0, fy=20.0, cx=32.0, cy=24.0, R=np.eye(3), C=np.array([0.0, 0.0, 5.0]))
    view_behind = View(bgr=np.zeros((48, 64, 3), dtype=np.uint8), camera=behind)
    labels, filled = labeling.fill_unseen(mesh, topo, np.array([0, NO_ID]), [view_behind])
    assert labels.tolist() == [0, NO_ID] and not filled.any()


# ── Atlas ────────────────────────────────────────────────────────────────────


def test_compute_texture_size():
    rects = [Rect(0, 0, 100, 100)] * 4
    assert atlas.compute_texture_size(rects, 0) == 256  # ceil(sqrt(40000 / 0.9)) = 211 -> 256
    assert atlas.compute_texture_size(rects, 32) == 224
    assert atlas.compute_texture_size([Rect(0, 0, 300, 10)], 0) == 512  # at least the largest side


def test_maxrects_places_without_overlap():
    rng = np.random.default_rng(3)
    sizes = rng.integers(3, 40, size=(60, 2))
    pack = atlas.MaxRectsBinPack(256, 256)
    placed, remaining = pack.insert(sizes)
    assert len(remaining) == 0
    grid = np.zeros((256, 256), dtype=int)
    for rect, i in placed:
        assert rect.x >= 0 and rect.y >= 0 and rect.x + rect.width <= 256 and rect.y + rect.height <= 256
        assert {rect.width, rect.height} == {int(sizes[i, 0]), int(sizes[i, 1])}
        grid[rect.y : rect.y + rect.height, rect.x : rect.x + rect.width] += 1
    assert grid.max() == 1
    # The first placement sits at the bottom-left origin.
    assert (placed[0][0].x, placed[0][0].y) == (0, 0)


def test_shelf_pack_places_without_overlap_and_reports_unplaced():
    rng = np.random.default_rng(4)
    sizes = rng.integers(3, 30, size=(500, 2))
    placed, remaining = atlas.shelf_pack(sizes, 256, 256)
    grid = np.zeros((256, 256), dtype=int)
    for rect, i in placed:
        assert rect.x >= 0 and rect.y >= 0 and rect.x + rect.width <= 256 and rect.y + rect.height <= 256
        assert {rect.width, rect.height} == {int(sizes[i, 0]), int(sizes[i, 1])}
        grid[rect.y : rect.y + rect.height, rect.x : rect.x + rect.width] += 1
    assert grid.max() == 1
    assert len(placed) + len(remaining) == len(sizes)
    assert len(remaining) > 0  # 500 rects of that size do not fit a 256 square
    placed, remaining = atlas.shelf_pack(sizes, 1024, 1024)
    assert len(remaining) == 0
    # Over the exact-packing limit the dispatcher takes the shelf packer.
    placed_fast, _ = atlas.pack_rects(sizes, 1024, exact_limit=100)
    assert len(placed_fast) == len(sizes)


# ── Leveling ─────────────────────────────────────────────────────────────────


def test_color_round_trip():
    rng = np.random.default_rng(0)
    bgr = rng.uniform(0, 255, size=(50, 3)).astype(np.float32)
    assert np.allclose(ycbcr2rgb(rgb2ycbcr(bgr)), bgr, atol=0.05)


def test_process_mask_keeps_a_strip_inside_the_border():
    mask = np.zeros((40, 40), dtype=np.uint8)
    mask[5:35, 5:35] = leveling.MASK_INTERIOR
    leveling.process_mask(mask, strip_width=3)
    interior = mask == leveling.MASK_INTERIOR
    border = mask == leveling.MASK_BORDER
    # The core more than three rings inside the edge is left alone (empty in
    # the mask), the ring around it is the fixed border, and the outer strip
    # is what gets blended.
    assert mask[20, 20] == leveling.MASK_EMPTY
    assert border[8, 8] and border[8, 20]
    assert interior[6, 20] and interior[7, 20]
    assert mask[4, 20] == leveling.MASK_EMPTY


def test_poisson_blend_reproduces_source_with_matching_border():
    rng = np.random.default_rng(2)
    source = rng.uniform(0, 1, size=(20, 20, 3)).astype(np.float32)
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[3:17, 3:17] = leveling.MASK_INTERIOR
    mask[3, 3:17] = mask[16, 3:17] = mask[3:17, 3] = mask[3:17, 16] = leveling.MASK_BORDER
    dst = np.zeros_like(source)
    dst[mask == leveling.MASK_BORDER] = source[mask == leveling.MASK_BORDER]
    leveling.poisson_blend(source, dst, mask)
    inside = mask == leveling.MASK_INTERIOR
    assert np.allclose(dst[inside], source[inside], atol=1e-3)


# ── End to end ───────────────────────────────────────────────────────────────


def _mapping_config():
    cfg = BakeConfig()
    cfg.keyframes.rgbd_frame_gap = 1
    cfg.keyframes.closest_pose_translation = 0.0
    cfg.keyframes.closest_pose_rotation_angle = 0.0
    # The unsharp mask overshoots at checker edges on faces four pixels wide,
    # which is what the tolerance below would otherwise measure.
    cfg.mapping.sharpness_weight = 0.0
    return cfg


def test_bake_mapping_reproduces_checker(tmp_path):
    mesh = make_plane_mesh(with_uv=False)
    result = bake(mesh, make_frames(), _mapping_config(), verbose=False)
    assert result.stats["textured_faces"] == len(mesh.faces)
    assert result.stats["num_patches"] >= 1
    out_mesh = result.mesh
    uv = np.asarray(out_mesh.visual.uv)
    cent_uv = uv[out_mesh.faces].mean(axis=1)
    cent_xy = out_mesh.triangles_center[:, :2]
    fx = np.abs((cent_xy / CELL) - np.round(cent_xy / CELL))
    interior = (fx > 0.08).all(axis=1)
    assert interior.sum() > 20
    got = sample_texture(result, cent_uv[interior])
    want = checker(cent_xy[interior, 0], cent_xy[interior, 1]).astype(np.float64)
    assert (np.abs(got - want).max(axis=1) < 30).all()
    out = result.export(tmp_path / "mapped.obj")
    reloaded = trimesh.load(out, process=False, force="mesh")
    assert reloaded.visual.uv is not None
    assert len(reloaded.faces) == len(mesh.faces)


def test_bake_mapping_ignores_existing_uv():
    mesh = make_plane_mesh(with_uv=True)
    result = bake(mesh, make_frames(), _mapping_config(), verbose=False)
    # The mapping's own layout splits vertices along patch borders.
    assert len(result.mesh.vertices) >= len(mesh.vertices)
