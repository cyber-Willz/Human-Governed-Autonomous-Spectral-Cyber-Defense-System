#!/usr/bin/env python3
"""Turn a folder of per-template cortical surface maps (GIFTI .func.gii / .shape.gii, one value per
vertex) plus an inflated surface into the --csv inputs of connectome_rnn.

usage: prepare_gifti_templates.py OUTDIR SURF.surf.gii K MAXLEN_MM NAME_REGEX GROUP:GLOB [GROUP:GLOB ...]
  e.g. prepare_gifti_templates.py out sunet.ico-6_very_inflated.surf.gii 8 150 'NODE\\d+\\.curv\\.MSM_HT_mean\\.func\\.gii$' \\
        frontal:'concate_HT/frontal/*.func.gii' parietal:'concate_HT/parietal/*.func.gii'

Writes weights.csv (Pearson r over vertices non-zero in every map, top-K positive per node, symmetric),
lengths.csv (distances between curvature-weighted centroids of each map's top 5% vertices, rescaled so
the longest equals MAXLEN_MM; raw distances are tiny, which would make every delay one step) and
labels.csv (name,hemi,system=group,x,y,z). The result is a structural-covariance graph, not wiring.
Requires nibabel, numpy.
"""
import glob, re, sys
import numpy as np
import nibabel as nib

def main(a):
    out, surf, k, maxlen, rx, *groups = a
    k, maxlen = int(k), float(maxlen)
    xyz = nib.load(surf).darrays[0].data.astype(float)
    names, lobes, maps = [], [], []
    for g in groups:
        lobe, pat = g.split(":", 1)
        for f in sorted(glob.glob(pat)):
            m = re.search(rx, f)
            if not m:
                continue
            maps.append(nib.load(f).darrays[0].data.astype(float))
            tag = re.search(r"NODE\d+", f)
            names.append(lobe[:3] + "_" + (tag.group(0) if tag else str(len(names))))
            lobes.append(lobe)
    A = np.stack(maps); n = len(names)
    mask = (A != 0).all(0)
    X = A[:, mask]; X = X - X.mean(1, keepdims=True)
    C = np.corrcoef(X); np.fill_diagonal(C, 0)
    W = np.zeros_like(C)
    for i in range(n):
        idx = np.argsort(-C[i])[:k]; idx = idx[C[i, idx] > 0]; W[i, idx] = C[i, idx]
    W = np.maximum(W, W.T)
    pos = []
    for v in maps:
        th = np.quantile(v[mask], 0.95); sel = (v >= th) & mask
        w = v[sel] - th + 1e-9; pos.append((xyz[sel] * w[:, None]).sum(0) / w.sum())
    pos = np.array(pos)
    L = np.linalg.norm(pos[:, None] - pos[None], axis=2)
    L *= maxlen / L.max()
    np.savetxt(f"{out}/weights.csv", W, delimiter=",", fmt="%.6f")
    np.savetxt(f"{out}/lengths.csv", L, delimiter=",", fmt="%.3f")
    with open(f"{out}/labels.csv", "w") as o:
        o.write("name,hemi,system,x,y,z\n")
        for nm, lb, p in zip(names, lobes, pos):
            o.write(f"{nm},L,{lb.capitalize()},{p[0]:.2f},{p[1]:.2f},{p[2]:.2f}\n")
    print(f"{n} regions, density {(W > 0).mean():.3f}, wrote {out}")

if __name__ == "__main__":
    if len(sys.argv) < 7:
        sys.exit(__doc__)
    main(sys.argv[1:])
