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

#EOF
