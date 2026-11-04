#!/usr/bin/env python3
"""Turn parcellated CIFTI maps (.pscalar.nii) + Glasser border files + inflated surfaces
into the two files `connectome_rnn --features` needs.

usage: prepare_pscalar_inputs.py OUTDIR L_SURF.surf.gii R_SURF.surf.gii L.border R.border a.pscalar.nii [b.pscalar.nii ...]

  OUTDIR/features.csv   parcels x maps (row order = pscalar parcel order)
  OUTDIR/labels.csv     name,hemi,system,x,y,z   (system left blank: Glasser names are auto-mapped)

Positions are the mean of each parcel's border vertices on the supplied (very-)inflated
surface, so lengths are geometric distances on that surface, not fibre lengths.
Requires: nibabel, numpy. Check the data-use terms of whatever you feed in.
"""
import re, sys
import numpy as np
import nibabel as nib

def main(a):
    out, lsurf, rsurf, lb, rb, *ps = a
    xyz = {"L": nib.load(lsurf).darrays[0].data.astype(float), "R": nib.load(rsurf).darrays[0].data.astype(float)}
    cent = {}
    for h, path in (("L", lb), ("R", rb)):
        s = open(path).read()
        for m in re.finditer(r'<Border Name="([^"]+)".*?<Vertices>(.*?)</Vertices>', s, re.S):
            v = np.unique(np.array(m.group(2).split(), dtype=int))
            cent[m.group(1)] = xyz[h][v].mean(0)
    names = [str(n) for n in nib.load(ps[0]).header.get_axis(1).name]
    cols = []
    for p in ps:
        img = nib.load(p)
        assert [str(n) for n in img.header.get_axis(1).name] == names, f"parcel order differs in {p}"
        cols.append(np.asarray(img.get_fdata()).squeeze())
    np.savetxt(f"{out}/features.csv", np.stack(cols, 1), delimiter=",", fmt="%.6f")
    with open(f"{out}/labels.csv", "w") as o:
        o.write("name,hemi,system,x,y,z\n")
        for n in names:
            p = cent[n]
            o.write(f"{n},{n[0]},,{p[0]:.2f},{p[1]:.2f},{p[2]:.2f}\n")
    print(f"{len(names)} parcels x {len(cols)} maps written to {out}")

if __name__ == "__main__":
    if len(sys.argv) < 7:
        sys.exit(__doc__)
    main(sys.argv[1:])
