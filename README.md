# 3D Texturing — RGBD Reconstruction + Texturing

RGBD reconstruction (Open3D TSDF fusion + OpenCV preprocessing) and mesh
texturing over continuous frame subsets extracted from the BundleFusion
`apt0` sequence.

## Data

Three continuous 1000-frame subsets created by `create_rgbd_dataset.py`:

| Subset | Source frames |
|---|---|
| `apt0_seq0` | 0–999 |
| `apt0_seq1` | 3780–4779 |
| `apt0_seq2` | 7560–8559 |

Each frame folder holds `rgb.jpg`, `depth.png` (uint16 mm), `pose.txt`
(4x4 camera-to-world), `meta.json`. Subset root has shared `intrinsics.txt`
and a `dataset.json` index.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt  # reconstruction deps (open3d etc.)
.venv/bin/pip install -e .                 # the texture3d texturing library + CLI
```

## Run

```bash
# Full reconstruction of one subset
.venv/bin/python run_reconstruction.py apt0_seq0

# All subsets, faster preview
.venv/bin/python run_reconstruction.py apt0_seq0 apt0_seq1 apt0_seq2 --stride 3 --voxel 0.02
```

Outputs land in `reconstructions/<subset>/`:

- `mesh.ply` — colored TSDF mesh (cleaned, normals computed)
- `point_cloud.ply` — downsampled point cloud
- `trajectory_tum.txt` — camera trajectory (TUM format)
- `config.json`, `summary.json` — run parameters and stats

## Texturing — the `texture3d` library

Installable library (`pip install -e .`) that maps posed RGB-D frames onto any
aligned triangle mesh — ours, or an untextured STL/OBJ from another
reconstruction engine. Built on **trimesh** (stable vertex/face ordering,
`process=False` everywhere); Open3D is not used by the library.

Each face is textured from a single keyframe — the view that sees it best —
so photograph detail is kept instead of being averaged away by slightly
misregistered frames. Connected same-view faces become patches; a global
colour-offset solve and a local Poisson blend hide the seams, then the
patches are packed into one atlas (`src/texture3d/mapping/`).

```python
import texture3d

frames = texture3d.load_dataset("apt0_seq0")     # or build RGBDFrame objects
result = texture3d.bake("reconstructions/apt0_seq0/mesh.ply", frames)
result.export("textured/apt0_seq0/textured_mesh.obj")   # or .glb
print(result.stats)                              # patches, coverage, stage timings
```

```bash
# Same thing from the CLI
.venv/bin/texture3d apt0_seq0 --mesh reconstructions/apt0_seq0/mesh.ply

# Larger patches, no sharpening, wider blend strip
.venv/bin/texture3d apt0_seq0 --data-ratio 0.05 --sharpness 0 --strip-width 30
```

Any UVs the input mesh carries are ignored: the mapping's UVs follow the
labelling. The output mesh splits vertices along patch borders (a vertex has
one UV per patch it belongs to) and is a plain single-material textured mesh,
exportable to OBJ or GLB.

### Stages (`src/texture3d/mapping/`)

0. **Keyframes** (`keyframes.py`) — the least blurry frame of every `--gap`
   window (Crete no-reference blur metric), skipping near-duplicate poses
   (<5 cm / <0.09 rad).
1. **Topology** (`topology.py`) — incident faces, face adjacency across
   edges, and face normals smoothed over three passes with neighbours within
   25°.
2. **View scoring** (`scoring.py`) — per keyframe the mesh is rasterised into a
   face map with a depth test (perspective-correct barycentrics, 3 px image
   border), so a pixel belongs to the nearest face and the mesh's own back
   side never textures anything. A face's quality in a view is the blurred
   Sobel gradient magnitude summed over its pixels times the squared cosine
   to the smoothed normal. Views whose mean face colour (YCbCr) falls below a
   Gaussian likelihood of 0.06 against the others are dropped for that face.
3. **Labelling** (`labeling.py`) — the face graph is an MRF whose labels are
   the views plus "none"; data cost `(1 - quality / q95) · 100`, Potts
   smoothness 1000, solved by synchronous loopy belief propagation.
4. **Patches** (`patches.py`) — connected same-view faces become patches with
   a 2 px bordered rectangle in their view; seam vertices catalogue where
   patches meet.
5. **Seam leveling** (`leveling.py`) — *global*: one colour offset per
   (vertex, patch) from a Tikhonov-regularised least-squares solve (λ = 0.1,
   conjugate gradient) so both sides of every seam agree, interpolated over
   each patch and added in YCbCr; *local*: a 20 px strip along each patch
   border is Poisson-blended so the border takes the mean of both sides while
   the interior keeps the photograph's gradients.
6. **Atlas** (`atlas.py`) — MaxRects bottom-left packing with rotation into
   the smallest power-of-two square (≤ 8192), the levelled pixels copied in,
   per-corner texture coordinates written, and an unsharp mask (σ 1.5,
   weight 0.5) applied.

Faces no keyframe sees take the empty colour (white by default); on a TSDF
mesh that is the back-side shell from the truncation band, which no camera
ever saw. `result.stats["stage_seconds"]` reports each stage's time.

Two options, both on by default:

- **Unseen-face fill** (`--no-fill-unseen`): a face smaller than a pixel in
  every keyframe is never rasterised. It takes the view of a textured edge
  neighbour when its vertices project inside that view (on apt0_seq0, 157k of
  977k faces), instead of speckling the surface with the empty colour.
- **Shelf packing above 4,000 patches** (`exact_packing_limit`): MaxRects is
  quadratic in the patch count; a dense mesh seen by many views yields tens
  of thousands of one-face patches (apt0_seq0: 94k patches, 640 s to pack).
  Beyond the limit a next-fit decreasing-height shelf packer runs instead
  (1 s), at the cost of a looser atlas.

Measured on `apt0_seq0` (977k faces, 58 keyframes at 640x480, 18-core Mac):
scoring 8 s, labelling 9 s (53 sweeps), global leveling 14 s, local leveling
78 s, atlas 1 s; 117 s end to end, 4.4 GB peak, 85% of faces textured (the
rest is the TSDF back shell) in 60k patches.

## Code layout

```
rgbd_recon/
  config.py         # dataclass configs (depth filter, fusion, mesh, pipeline)
  dataset.py        # RGBDSequence/Frame loaders for the dataset.json layout
  preprocessing.py  # OpenCV depth cleanup (range clip, bilateral, edge mask)
  fusion.py         # ScalableTSDFVolume wrapper
  pipeline.py       # end-to-end orchestration + mesh postprocess + exports
src/texture3d/        # installable texturing library (trimesh-based)
  camera.py         # PinholeCamera + RGBDFrame (dataset-agnostic inputs)
  io.py             # load_dataset() for the dataset.json layout
  config.py         # BakeConfig (keyframes + mapping)
  blur.py           # Crete blur metric
  keyframes.py      # least-blurry-per-window keyframes
  mapping/          # one-keyframe-per-face mapping + seam leveling
    types.py        #   Mesh/Camera/View/Result, colour helpers, conventions
    topology.py     #   incidence, adjacency, smoothed normals
    imageops.py     #   gradient image, invalid-region mask, bilinear sampling, sharpen
    raster.py       #   barycentric / fixed-point triangle rasterisers, Bresenham lines
    scoring.py      #   depth-tested face maps, per-face view quality, outlier test
    labeling.py     #   MRF + loopy belief propagation, unseen-face fill
    patches.py      #   patches, seam vertices, contained-patch merge
    leveling.py     #   global (least squares) and local (Poisson) seam leveling
    atlas.py        #   MaxRects / shelf packing, atlas assembly
    mapper.py       #   run(): the stages in order, with timings
  bake.py           # bake(): keyframes -> views -> mapping -> textured trimesh
  cli.py            # `texture3d` console command
tests/              # unit tests + a synthetic end-to-end bake (pytest)
run_reconstruction.py  # CLI
run_texturing.py       # back-compat shim for the texture3d CLI
```

The dataset loader is format-agnostic beyond `dataset.json`, so future
sequences exported by `create_rgbd_dataset.py` work unchanged. Fusion,
preprocessing, mesh postprocessing, and all texturing stages are independently
configurable for reuse (e.g. estimated SLAM poses instead of ground truth, or
texturing a mesh produced by a different engine).
