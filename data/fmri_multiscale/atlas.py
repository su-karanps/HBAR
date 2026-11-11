import math
import csv
from dataclasses import dataclass
from typing import Optional, Tuple

import nibabel as nib
import numpy as np
from nilearn import datasets, image


@dataclass
class AtlasAssignments:
    voxel_mask_flat_indices: np.ndarray
    voxel_coords: np.ndarray
    voxel_to_parcel: np.ndarray
    parcel_to_network: np.ndarray
    n_parcels: int
    n_networks: int


def _maybe_int(value: str) -> Optional[int]:
    try:
        return int(float(str(value).strip()))
    except (TypeError, ValueError):
        return None


# Canonical Yeo-17 network order used by `load_parcel_network_mapping_yeo18`.
# The DiFuMo labels CSVs carry 18 distinct `Yeo_networks17` values: these 17
# names plus "No network found" for components that no Yeo network claims.
# Ids 0..16 are the Yeo-17 networks in the order below; id 17 is the
# unassigned bucket.
YEO17_NETWORK_ORDER = (
    "VisCent",
    "VisPeri",
    "SomMotA",
    "SomMotB",
    "DorsAttnA",
    "DorsAttnB",
    "SalVentAttnA",
    "SalVentAttnB",
    "LimbicA",
    "LimbicB",
    "ContA",
    "ContB",
    "ContC",
    "DefaultA",
    "DefaultB",
    "DefaultC",
    "TempPar",
)
YEO18_UNASSIGNED_LABEL = "No network found"
YEO18_NETWORK_ORDER = YEO17_NETWORK_ORDER + (YEO18_UNASSIGNED_LABEL,)
YEO18_N_NETWORKS = len(YEO18_NETWORK_ORDER)


def load_parcel_network_mapping_yeo18(csv_path: str, n_parcels: int) -> np.ndarray:
    """
    Load parcel->network from a DiFuMo labels CSV using the fixed 18-id order in
    `YEO18_NETWORK_ORDER` (Yeo-17 networks as ids 0..16, "No network found" as
    id 17).

    Ids do not depend on the row order of the CSV. Raises on an unknown label, a
    missing/blank label, a duplicate parcel id, or any parcel left unresolved.
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = [c.strip() for c in (reader.fieldnames or [])]
        rows = list(reader)
    if not rows:
        raise ValueError(f"Empty CSV file: {csv_path}")

    col_lookup = {c.lower(): c for c in fieldnames}
    network_col = None
    for key in ["yeo_networks17", "yeo_17", "yeo17", "network_17", "network"]:
        if key in col_lookup:
            network_col = col_lookup[key]
            break
    if network_col is None:
        raise ValueError(f"Could not find Yeo-17 mapping column in {csv_path}. Found columns: {fieldnames}")

    id_col = None
    for key in ["id", "index", "component", "map", "maps"]:
        if key in col_lookup:
            id_col = col_lookup[key]
            break

    label_to_id = {label: i for i, label in enumerate(YEO18_NETWORK_ORDER)}
    parcel_to_network = np.full((n_parcels,), -1, dtype=np.int32)

    for row_idx, row in enumerate(rows):
        label = str(row.get(network_col, "") or "").strip()
        if label not in label_to_id:
            raise ValueError(
                f"Unknown {network_col} label {label!r} at row {row_idx} of {csv_path}; "
                f"expected one of {list(YEO18_NETWORK_ORDER)}"
            )

        parcel_id = row_idx
        if id_col is not None:
            parsed = _maybe_int(row.get(id_col, ""))
            if parsed is not None:
                # Accept either 1-based or 0-based IDs.
                parcel_id = parsed - 1 if 1 <= parsed <= n_parcels else parsed
        if not (0 <= parcel_id < n_parcels):
            raise ValueError(f"Parcel id {parcel_id} out of range [0,{n_parcels}) at row {row_idx} of {csv_path}")
        if parcel_to_network[parcel_id] >= 0:
            raise ValueError(f"Duplicate parcel id {parcel_id} at row {row_idx} of {csv_path}")
        parcel_to_network[parcel_id] = label_to_id[label]

    unresolved = np.flatnonzero(parcel_to_network < 0)
    if unresolved.size > 0:
        raise ValueError(f"{unresolved.size} parcels unresolved in {csv_path}: {unresolved[:10].tolist()}...")

    return parcel_to_network


def _load_difumo_maps(dimension: int = 1024) -> nib.Nifti1Image:
    atlas = datasets.fetch_atlas_difumo(dimension=dimension, resolution_mm=2)
    return nib.load(atlas.maps)


def _resample_to_target(source_img: nib.Nifti1Image, target_img: nib.Nifti1Image, is_label: bool) -> nib.Nifti1Image:
    interpolation = "nearest" if is_label else "continuous"
    try:
        return image.resample_to_img(
            source_img,
            target_img,
            interpolation=interpolation,
            force_resample=True,
            copy_header=True,
        )
    except TypeError:
        return image.resample_to_img(source_img, target_img, interpolation=interpolation)


def build_hard_assignments(
    gm_mask_img: nib.Nifti1Image,
    difumo_labels_csv_path: str,
    difumo_maps_img: Optional[nib.Nifti1Image] = None,
    n_parcels: int = 1024,
) -> AtlasAssignments:
    """
    Build voxel->parcel (DiFuMo argmax) and parcel->network (the DiFuMo labels CSV's
    Yeo_networks17 column, 18 groups; see `load_parcel_network_mapping_yeo18`).
    """
    if difumo_maps_img is None:
        difumo_maps_img = _load_difumo_maps(dimension=n_parcels)

    gm_mask = gm_mask_img.get_fdata() > 0
    if gm_mask.ndim != 3:
        raise ValueError(f"Expected 3D gray matter mask, got shape={gm_mask.shape}")

    if difumo_maps_img.ndim != 4:
        raise ValueError(f"Expected 4D DiFuMo maps, got shape={difumo_maps_img.shape}")
    if difumo_maps_img.shape[-1] != n_parcels:
        raise ValueError(
            f"DiFuMo map count mismatch: expected {n_parcels}, got {difumo_maps_img.shape[-1]}"
        )

    flat_mask = gm_mask.reshape(-1)
    flat_mask_indices = np.flatnonzero(flat_mask)
    coords = np.column_stack(np.unravel_index(flat_mask_indices, gm_mask.shape)).astype(np.int32)

    # Memory-safe argmax over maps: avoid materializing full [X,Y,Z,1024] float array.
    same_grid = tuple(difumo_maps_img.shape[:3]) == tuple(gm_mask_img.shape[:3]) and np.allclose(
        difumo_maps_img.affine, gm_mask_img.affine, atol=1e-3
    )
    best_scores = np.full((flat_mask_indices.size,), -np.inf, dtype=np.float32)
    voxel_to_parcel = np.zeros((flat_mask_indices.size,), dtype=np.int32)
    difumo_proxy = difumo_maps_img.dataobj
    batch = 16
    for start in range(0, n_parcels, batch):
        stop = min(start + batch, n_parcels)
        block = np.asarray(difumo_proxy[..., start:stop], dtype=np.float32)
        if not same_grid:
            block_img = nib.Nifti1Image(block, difumo_maps_img.affine)
            block = _resample_to_target(block_img, gm_mask_img, is_label=False).get_fdata().astype(np.float32)
        vals = block.reshape(-1, stop - start)[flat_mask_indices]  # [V, b]
        local_best_idx = np.argmax(vals, axis=1)
        local_best_score = vals[np.arange(vals.shape[0]), local_best_idx]
        update = local_best_score > best_scores
        voxel_to_parcel[update] = start + local_best_idx[update]
        best_scores[update] = local_best_score[update]

    parcel_to_network = load_parcel_network_mapping_yeo18(difumo_labels_csv_path, n_parcels)

    return AtlasAssignments(
        voxel_mask_flat_indices=flat_mask_indices.astype(np.int64),
        voxel_coords=coords,
        voxel_to_parcel=voxel_to_parcel,
        parcel_to_network=parcel_to_network,
        n_parcels=n_parcels,
        n_networks=YEO18_N_NETWORKS,
    )


def spatial_chunk_tokens(
    voxel_coords: np.ndarray,
    voxel_to_parcel: np.ndarray,
    chunk_size: int,
    n_parcels: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split each parcel's voxels into fixed-size spatially coherent chunks.
    Returns:
      - chunk_indices: [N0, chunk_size] in masked voxel index space (-1 padded)
      - chunk_parcels: [N0] parcel id per chunk
    """
    all_chunks = []
    all_chunk_parcels = []

    for parcel_id in range(n_parcels):
        idx = np.flatnonzero(voxel_to_parcel == parcel_id)
        if idx.size == 0:
            continue

        xyz = voxel_coords[idx].astype(np.float32)
        mins = xyz.min(axis=0, keepdims=True)
        spans = np.maximum(xyz.max(axis=0, keepdims=True) - mins, 1.0)
        xyz_norm = (xyz - mins) / spans

        # Space-filling ordering to keep neighboring voxels near each other.
        key = xyz_norm[:, 0] * 1.0 + xyz_norm[:, 1] * 1e-2 + xyz_norm[:, 2] * 1e-4
        ordered = idx[np.argsort(key)]

        n_chunks = int(math.ceil(ordered.size / chunk_size))
        for c in range(n_chunks):
            part = ordered[c * chunk_size : (c + 1) * chunk_size]
            chunk = np.full((chunk_size,), -1, dtype=np.int64)
            chunk[: part.size] = part
            all_chunks.append(chunk)
            all_chunk_parcels.append(parcel_id)

    chunk_indices = np.stack(all_chunks, axis=0) if all_chunks else np.zeros((0, chunk_size), dtype=np.int64)
    chunk_parcels = np.asarray(all_chunk_parcels, dtype=np.int32)
    return chunk_indices, chunk_parcels
