"""Tests for distributed (multi-device) Galerkin assembly.

All tests use CPU fallback mode so they can run without multiple GPUs.
"""

import sys
sys.path.append("../..")

import torch
import pytest

from tensormesh import Mesh
from tensormesh.assemble import (
    LaplaceElementAssembler,
    MassElementAssembler,
    const_node_assembler,
)
from tensormesh.distributed import (
    DistributedMesh,
    distributed_element_assemble,
    distributed_element_assemble_to_sparse,
    distributed_node_assemble,
)


# ─── Helpers ────────────────────────────────────────────────────────

def _cpu_devices(n):
    return [torch.device('cpu')] * n


# ─── DistributedMesh ────────────────────────────────────────────────

class TestDistributedMesh:
    """Tests for DistributedMesh partitioning."""

    def test_partition_creates_submeshes(self):
        mesh = Mesh.gen_rectangle(chara_length=0.2, element_type="tri")
        dmesh = DistributedMesh(mesh, num_partitions=2, devices=_cpu_devices(2))

        assert dmesh.num_partitions == 2
        assert dmesh.n_global_points == mesh.n_points
        assert len(dmesh.submeshes) == 2

        # Each submesh should have orig_nid mapping
        for sub in dmesh.submeshes:
            if sub is not None:
                assert 'orig_nid' in list(sub.point_data.keys())
                orig_nid = sub.point_data['orig_nid']
                assert orig_nid.max() < mesh.n_points

    def test_partition_covers_all_points(self):
        """All global points should appear in at least one partition."""
        mesh = Mesh.gen_rectangle(chara_length=0.2, element_type="tri")
        dmesh = DistributedMesh(mesh, num_partitions=3, devices=_cpu_devices(3))

        all_global_ids = set()
        for sub in dmesh.submeshes:
            if sub is not None:
                ids = sub.point_data['orig_nid'].tolist()
                all_global_ids.update(ids)

        assert all_global_ids == set(range(mesh.n_points))

    def test_repr(self):
        mesh = Mesh.gen_rectangle(chara_length=0.2, element_type="tri")
        dmesh = DistributedMesh(mesh, num_partitions=2, devices=_cpu_devices(2))
        r = repr(dmesh)
        assert "DistributedMesh" in r
        assert "partitions=2" in r


# ─── Element Assembly ───────────────────────────────────────────────

class TestDistributedElementAssembly:
    """Verify distributed element assembly matches single-device assembly."""

    def test_laplace_matvec_2_partitions(self):
        """K_dist @ x should match K_ref @ x for Laplace."""
        mesh = Mesh.gen_rectangle(chara_length=0.2, element_type="tri")
        dmesh = DistributedMesh(mesh, num_partitions=2, devices=_cpu_devices(2))

        # Reference: single-device assembly
        K_ref = LaplaceElementAssembler.from_mesh(mesh)()

        # Distributed assembly → DSparseTensor
        K_dist = distributed_element_assemble(
            LaplaceElementAssembler, dmesh, quadrature_order=2
        )

        x = torch.randn(mesh.n_points, dtype=K_ref.dtype)
        y_ref = K_ref @ x
        y_dist = K_dist @ x

        assert torch.allclose(y_ref, y_dist, atol=1e-8), \
            f"Max diff: {(y_ref - y_dist).abs().max().item():.2e}"

    def test_laplace_matvec_4_partitions(self):
        """Test with more partitions than 2."""
        mesh = Mesh.gen_rectangle(chara_length=0.15, element_type="tri")
        dmesh = DistributedMesh(mesh, num_partitions=4, devices=_cpu_devices(4))

        K_ref = LaplaceElementAssembler.from_mesh(mesh)()
        K_dist = distributed_element_assemble(
            LaplaceElementAssembler, dmesh, quadrature_order=2
        )

        x = torch.randn(mesh.n_points, dtype=K_ref.dtype)
        y_ref = K_ref @ x
        y_dist = K_dist @ x

        assert torch.allclose(y_ref, y_dist, atol=1e-8), \
            f"Max diff: {(y_ref - y_dist).abs().max().item():.2e}"

    def test_mass_matvec(self):
        """Mass matrix distributed assembly."""
        mesh = Mesh.gen_rectangle(chara_length=0.2, element_type="tri")
        dmesh = DistributedMesh(mesh, num_partitions=2, devices=_cpu_devices(2))

        K_ref = MassElementAssembler.from_mesh(mesh)()
        K_dist = distributed_element_assemble(
            MassElementAssembler, dmesh, quadrature_order=2
        )

        x = torch.randn(mesh.n_points, dtype=K_ref.dtype)
        y_ref = K_ref @ x
        y_dist = K_dist @ x

        assert torch.allclose(y_ref, y_dist, atol=1e-8), \
            f"Max diff: {(y_ref - y_dist).abs().max().item():.2e}"

    def test_to_sparse_matrix(self):
        """distributed_element_assemble_to_sparse returns a SparseMatrix."""
        mesh = Mesh.gen_rectangle(chara_length=0.2, element_type="tri")
        dmesh = DistributedMesh(mesh, num_partitions=2, devices=_cpu_devices(2))

        K_ref = LaplaceElementAssembler.from_mesh(mesh)()
        K_dist = distributed_element_assemble_to_sparse(
            LaplaceElementAssembler, dmesh, quadrature_order=2
        )

        from tensormesh.sparse import SparseMatrix
        assert isinstance(K_dist, SparseMatrix)
        assert K_dist.shape == K_ref.shape

        x = torch.randn(mesh.n_points, dtype=K_ref.dtype)
        y_ref = K_ref @ x
        y_dist = K_dist @ x

        assert torch.allclose(y_ref, y_dist, atol=1e-8)

    def test_quad_mesh(self):
        """Test with quadrilateral mesh."""
        mesh = Mesh.gen_rectangle(chara_length=0.2, element_type="quad")
        dmesh = DistributedMesh(mesh, num_partitions=2, devices=_cpu_devices(2))

        K_ref = LaplaceElementAssembler.from_mesh(mesh, quadrature_order=3)()
        K_dist = distributed_element_assemble(
            LaplaceElementAssembler, dmesh, quadrature_order=3
        )

        x = torch.randn(mesh.n_points, dtype=K_ref.dtype)
        y_ref = K_ref @ x
        y_dist = K_dist @ x

        assert torch.allclose(y_ref, y_dist, atol=1e-8), \
            f"Max diff: {(y_ref - y_dist).abs().max().item():.2e}"


# ─── Node Assembly ──────────────────────────────────────────────────

class TestDistributedNodeAssembly:
    """Verify distributed node assembly matches single-device assembly."""

    def test_const_node_assembler(self):
        """Constant load vector: distributed vs single."""
        mesh = Mesh.gen_rectangle(chara_length=0.2, element_type="tri")
        dmesh = DistributedMesh(mesh, num_partitions=2, devices=_cpu_devices(2))

        # const_node_assembler() returns a CLASS, not an instance
        ConstLoad = const_node_assembler()
        asm_ref = ConstLoad.from_mesh(mesh)
        f_ref = asm_ref()

        # Distributed
        f_dist = distributed_node_assemble(ConstLoad, dmesh, quadrature_order=2)

        assert torch.allclose(f_ref.cpu(), f_dist, atol=1e-8), \
            f"Max diff: {(f_ref.cpu() - f_dist).abs().max().item():.2e}"


# ─── Distributed Solve (integration test) ───────────────────────────

class TestDistributedSolve:
    """End-to-end distributed solve test."""

    def test_poisson_solve(self):
        """Solve Poisson equation: -Δu = 1, u|∂Ω = 0, compare solutions."""
        from tensormesh import Condenser

        mesh = Mesh.gen_rectangle(chara_length=0.15, element_type="tri")
        boundary_mask = mesh.boundary_mask

        # --- Reference (single-device) ---
        K_ref = LaplaceElementAssembler.from_mesh(mesh)()
        ConstLoad = const_node_assembler()
        f_ref = ConstLoad.from_mesh(mesh)()

        condenser = Condenser(boundary_mask)
        K_c, f_c = condenser(K_ref, f_ref)
        u_ref = condenser.recover(K_c.solve(f_c))

        # --- Distributed assembly → SparseMatrix solve ---
        dmesh = DistributedMesh(mesh, num_partitions=2, devices=_cpu_devices(2))
        K_sparse = distributed_element_assemble_to_sparse(
            LaplaceElementAssembler, dmesh, quadrature_order=2
        )
        f_dist = distributed_node_assemble(ConstLoad, dmesh, quadrature_order=2)

        condenser2 = Condenser(boundary_mask)
        K_c2, f_c2 = condenser2(K_sparse, f_dist)
        u_dist = condenser2.recover(K_c2.solve(f_c2))

        assert torch.allclose(u_ref, u_dist, atol=1e-6), \
            f"Max diff: {(u_ref - u_dist).abs().max().item():.2e}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
