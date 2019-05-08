# -*- coding: utf-8 -*-

import numpy as np

#===============================================================================
def append_samples(samples, kinds, atom2kind, coords, xblocks):
    for iatom, kind_name in enumerate(atom2kind):
        sample = {}
        sample['rel_coords'] = coords - coords[iatom,:]
        sample['xblock'] = xblocks[iatom]
        samples[kind_name] = samples.get(kind_name, [])
        samples[kind_name].append(sample)

#===============================================================================
def parse_pao_file(fn):
    # abusing dicts as 1-based array
    kinds = {}
    atom2kind = []
    coords = []
    xblocks = []
    ikind2name = {}

    for line in open(fn).readlines():
        parts = line.split()
        if parts[0] == "Parametrization":
            assert parts[1] == "DIRECT"

        elif parts[0] == "Kind":
            ikind = int(parts[1])
            ikind2name[ikind] = parts[2]
            kinds[ikind2name[ikind]] = {'atomic_number': int(parts[3])}

        elif parts[0] == "NParams":
            ikind = int(parts[1])
            kinds[ikind2name[ikind]]['nparams'] = int(parts[2])

        elif parts[0] == "PrimBasis":
            ikind = int(parts[1])
            kinds[ikind2name[ikind]]['prim_basis_size'] = int(parts[2])
            kinds[ikind2name[ikind]]['prim_basis_name'] = parts[3]

        elif parts[0] == "PaoBasis":
            ikind = int(parts[1])
            kinds[ikind2name[ikind]]['pao_basis_size'] = int(parts[2])

        elif parts[0] == "Atom":
            atom2kind.append(parts[2])
            coords.append(parts[3:])

        elif parts[0] == "Xblock":
            xblocks.append(np.array(parts[2:], float))

    coords = np.array(coords, float)

    for iatom, kind_name in enumerate(atom2kind):
        n = kinds[kind_name]['prim_basis_size']
        m = kinds[kind_name]['pao_basis_size']
        xblocks[iatom] = xblocks[iatom].reshape(m, n)

    return kinds, atom2kind, coords, xblocks

#===============================================================================
def write_pao_file(filename, kinds, atom2kind, coords, xblocks):
    natoms = coords.shape[0]
    assert coords.shape[1] == 3
    len(xblocks) == natoms

    output = []
    output.append("Version 4")
    output.append("Parametrization DIRECT")
    output.append("Nkinds {}".format(len(kinds)))
    for ikind, (kind_name, kind) in enumerate(kinds.items()):
        output.append("Kind {} {} {}".format(ikind+1, kind_name, kind['atomic_number']))
        output.append("NParams {} {}".format(ikind+1, kind['nparams']))
        output.append("PrimBasis {} {} {}".format(ikind+1, kind['prim_basis_size'], kind['prim_basis_name']))
        output.append("PaoBasis {} {}".format(ikind+1, kind['pao_basis_size']))
        output.append("NPaoPotentials {} 0".format(ikind+1))

    output.append("Cell 8.0 0.0 0.0   0.0 8.0 0.0   0.0 0.0 8.0")
    output.append("Natoms {}".format(natoms))

    for iatom in range(natoms):
        output.append("Atom {} {} {} {} {}".format(iatom+1, atom2kind[iatom], coords[iatom, 0], coords[iatom, 1], coords[iatom, 2]))
    for iatom in range(natoms):
        kind = kinds[atom2kind[iatom]]
        assert len(xblocks[iatom].shape) == 2
        assert xblocks[iatom].shape[0] == kind['pao_basis_size']
        assert xblocks[iatom].shape[1] == kind['prim_basis_size']
        x = xblocks[iatom].flatten()
        y = " ".join(["%f"%i for i in x])
        output.append("Xblock {} {}".format(iatom+1, y))
    output.append("THE_END")

    with open(filename, "w") as f:
        f.write("\n".join(output))

#EOF
