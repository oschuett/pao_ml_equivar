# author: Ole Schuett


import torch
import e3nn
import warnings
from typing import List


# ======================================================================================
class PaoModel(torch.nn.Module):
    def __init__(self, prim_basis_irreps, pao_basis_size, num_kinds):
        super().__init__()
        self.prim_basis_irreps = prim_basis_irreps
        self.prim_basis_size = prim_basis_irreps.dim
        self.pao_basis_size = pao_basis_size
        self.matrix = SymmetricMatrix(self.prim_basis_irreps)

        # Irreps of input features, i.e. the descriptor.
        self.features_irreps = num_kinds * e3nn.o3.Irrep("0e")

        # Irreps of Spherical Harmonics used for sensing neighbors.
        self.sensor_irreps = e3nn.o3.Irreps.spherical_harmonics(
            lmax=self.matrix.input_irreps.lmax
        )

        # Tensor Product
        # TODO: fix warning
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            self.tensor_product = e3nn.o3.FullyConnectedTensorProduct(
                irreps_in1=self.features_irreps,
                irreps_in2=self.sensor_irreps,
                irreps_out=self.matrix.input_irreps,
                shared_weights=False,
            )

        # hyperparams # TODO tune
        self.num_distances = 10
        self.num_layers = 16
        self.max_radius = 6.0
        self.num_neighbors = 6

        # Perceptron
        # Note ReLu does not work well because many of the distance buckets from soft_one_hot_linspace are zero.
        self.net = e3nn.nn.FullyConnectedNet(
            hs=[self.num_distances, self.num_layers, self.tensor_product.weight_numel],
            act=torch.sigmoid,
        )

        # CP2K uses the yzx convention, while e3nn uses xyz.
        # https://docs.e3nn.org/en/stable/guide/change_of_basis.html
        yzx_to_xyz = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        self.D_yzx_to_xyz = self.prim_basis_irreps.D_from_matrix(yzx_to_xyz)

    # ----------------------------------------------------------------------------------
    def forward(self, neighbors, features):
        assert neighbors.shape[-2] == self.num_neighbors and neighbors.shape[-1] == 3
        assert features.shape[-2] == self.num_neighbors
        assert features.shape[-1] == self.features_irreps.dim

        sensors = e3nn.o3.spherical_harmonics(
            self.sensor_irreps, neighbors, normalize=True, normalization="component"
        )
        distance_embedding = e3nn.math.soft_one_hot_linspace(
            x=neighbors.norm(dim=-1),
            start=0.0,
            end=self.max_radius,
            number=self.num_distances,
            basis="smooth_finite",
            cutoff=True,
        ).mul(self.num_distances**0.5)
        weights = self.net(distance_embedding)
        vec_per_neighbor = self.tensor_product(x=features, y=sensors, weight=weights)
        h_aux_vec = vec_per_neighbor.sum(dim=-2).div(self.num_neighbors**0.5)
        h_aux_matrix = self.matrix(h_aux_vec)
        u_matrix = torch.linalg.eigh(h_aux_matrix)[1]
        xblock = u_matrix[..., : self.pao_basis_size].transpose(-2, -1)
        return xblock @ self.D_yzx_to_xyz


# ======================================================================================
def flatten_irreps(irreps: e3nn.o3.Irreps) -> List[e3nn.o3.Irrep]:
    "Helper function to turn an Irreps object into a list of individual Irrep objects."
    result = []
    for mul, ir in irreps:
        result += mul * [ir]
    return result


# ======================================================================================
class SymmetricMatrix(torch.nn.Module):
    def __init__(self, basis_irreps):
        super().__init__()
        self.basis_irreps = flatten_irreps(basis_irreps)

        # Compute irreps required to represent a matrix
        self.input_irreps = e3nn.o3.Irreps()
        for i, a in enumerate(self.basis_irreps):
            for j, b in enumerate(self.basis_irreps):
                if j > i:
                    continue  # skip upper triangle
                self.input_irreps += a * b

    # ----------------------------------------------------------------------------------
    def forward(self, vector):
        assert vector.shape[-1] == self.input_irreps.dim
        basis_size = sum(ir.dim for ir in self.basis_irreps)
        matrix = torch.zeros(vector.shape[:-1] + (basis_size, basis_size))
        matrix[..., :, :] = torch.eye(basis_size)
        pos_c = 0
        for i, a in enumerate(self.basis_irreps):
            pos_a = sum(ir.dim for ir in self.basis_irreps[:i])
            for j, b in enumerate(self.basis_irreps):
                if j > i:
                    continue  # skip upper triangle
                pos_b = sum(ir.dim for ir in self.basis_irreps[:j])
                flat_irreps_prod = flatten_irreps(e3nn.o3.Irreps(a * b))
                for c in flat_irreps_prod:
                    if c.l < abs(a.l - b.l) or a.l + b.l < c.l:
                        continue
                    # TODO the wigner blocks are mostly zeros - not sure pytorch takes advantage of that.
                    wigner_block = e3nn.o3.wigner_3j(a.l, b.l, c.l)
                    coeffs = vector[..., pos_c : pos_c + c.dim]
                    block = torch.tensordot(coeffs, wigner_block, dims=[[-1], [-1]])
                    matrix[..., pos_a : pos_a + a.dim, pos_b : pos_b + b.dim] += block
                    pos_c += c.dim

        return matrix


# EOF
