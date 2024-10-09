# author: Ole Schuett

import torch
import e3nn
import warnings
from typing import Dict, List


# ======================================================================================
class PaoModel(torch.nn.Module):
    def __init__(
        self,
        prim_basis_irreps,
        pao_basis_size,
        num_feature_kinds,
        num_neighbors,
        num_distances,
        num_layers,
        cutoff,
    ):
        super().__init__()
        self.prim_basis_irreps = prim_basis_irreps
        self.pao_basis_size = pao_basis_size

        # hyper-parameters
        self.num_feature_kinds = num_feature_kinds
        self.num_neighbors = num_neighbors
        self.num_distances = num_distances
        self.num_layers = num_layers
        self.cutoff = cutoff

        # auxiliary Hamiltonian
        self.matrix = SymmetricMatrix(self.prim_basis_irreps)

        # Irreps of input features, i.e. the descriptor.
        self.features_irreps = num_feature_kinds * e3nn.o3.Irrep("0e")
        self.features_irreps_dim = self.features_irreps.dim

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

        # Spherical Harmonics
        self.spherical_harmonics = e3nn.o3.SphericalHarmonics(
            self.sensor_irreps,
            normalize=True,
            normalization="component",
        )

        # Used for distance embedding.
        self.distance_buckets = torch.linspace(0.0, self.cutoff, self.num_distances)

    # ----------------------------------------------------------------------------------
    def forward(self, inputs: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        neighbors_relpos = inputs["neighbors_relpos"]
        neighbors_features = inputs["neighbors_features"]

        assert neighbors_relpos.shape[-2] == self.num_neighbors
        assert neighbors_relpos.shape[-1] == 3
        assert neighbors_features.shape[-2] == self.num_neighbors
        assert neighbors_features.shape[-1] == self.features_irreps_dim

        neighbors_distances = torch.norm(neighbors_relpos, dim=-1)

        # Corresponds to e3nn.math.soft_one_hot_linspace(neighb_dists, basis="gaussian")
        bucket_width = self.distance_buckets[1] - self.distance_buckets[0]
        diff = (neighbors_distances[..., None] - self.distance_buckets) / bucket_width
        distance_embedding = diff.pow(2).neg().exp().div(1.12)
        weights = self.net(distance_embedding.mul(self.num_distances**0.5))
        sensors = self.spherical_harmonics(neighbors_relpos)
        vec_per_neighbor = self.tensor_product(
            x=neighbors_features.mul(self.num_neighbors**0.5), y=sensors, weight=weights
        )
        h_aux_vec = vec_per_neighbor.sum(dim=-2).div(self.num_neighbors**0.5)
        h_aux_matrix = self.matrix(h_aux_vec)
        u_matrix = torch.linalg.eigh(h_aux_matrix)[1]
        xblock = u_matrix[..., : self.pao_basis_size].transpose(-2, -1)
        outputs = {"xblock": xblock @ self.D_yzx_to_xyz}
        return outputs


# ======================================================================================
def flatten_irreps(irreps: e3nn.o3.Irreps) -> List[e3nn.o3.Irrep]:
    "Helper function to turn an Irreps object into a list of individual Irrep objects."
    result = []
    for mul, ir in irreps:
        result += mul * [ir]
    return result


# ======================================================================================
def dim(l: int) -> int:
    return 2 * l + 1


# ======================================================================================
class SymmetricMatrix(torch.nn.Module):
    def __init__(self, basis_irreps):
        super().__init__()
        self.basis_irreps = basis_irreps
        self.basis_irreps_ls: List[int] = basis_irreps.ls

        # Compute irreps required to represent a matrix
        self.input_irreps = e3nn.o3.Irreps()
        self.wigner_blocks = []
        for i, a in enumerate(flatten_irreps(basis_irreps)):
            for j, b in enumerate(flatten_irreps(basis_irreps)):
                if j > i:
                    continue  # skip upper triangle
                self.input_irreps += a * b

                # Pre-compute wigner blocks
                for lk in range(abs(a.l - b.l), a.l + b.l + 1):
                    self.wigner_blocks.append(e3nn.o3.wigner_3j(a.l, b.l, lk))

        self.input_irreps_ls = self.input_irreps.ls

    # ----------------------------------------------------------------------------------
    def forward(self, vector: torch.Tensor) -> torch.Tensor:
        assert vector.shape[-1] == sum(dim(l) for l in self.input_irreps_ls)
        basis_size = sum(dim(l) for l in self.basis_irreps_ls)
        matrix = torch.zeros(vector.shape[:-1] + (basis_size, basis_size))
        matrix[..., :, :] = torch.eye(basis_size)  # ensure matrix is diagonalizable
        c = 0  # position in vector
        z = 0  # position in self.wigner_blocks
        for i, li in enumerate(self.basis_irreps_ls):
            a = sum(dim(l) for l in self.basis_irreps_ls[:i])  # first matrix row
            for j, lj in enumerate(self.basis_irreps_ls):
                if j > i:
                    continue  # skip upper triangle
                b = sum(dim(l) for l in self.basis_irreps_ls[:j])  # first matrix col
                for lk in range(abs(li - lj), li + lj + 1):
                    # TODO the wigner blocks are mostly zeros - not sure pytorch takes advantage of that.
                    wigner_block = self.wigner_blocks[z]
                    coeffs = vector[..., c : c + dim(lk)]
                    block = torch.tensordot(coeffs, wigner_block, dims=[[-1], [-1]])
                    matrix[..., a : a + dim(li), b : b + dim(lj)] += block
                    c += dim(lk)
                    z += 1

        return matrix


# EOF
