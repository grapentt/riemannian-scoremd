"""
Riemannian geometry of protein shape space: the w^delta manifold.

Implements the quotient manifold P(d,n)/E(d) equipped with the energy-informed
w^delta metric of Diepeveen (2024, arXiv:2308.07818). This is the geometric
backbone of the riemannian-scoremd generative model.

The w^delta metric is defined via pairwise inter-residue log-distances plus a
gyration-tensor correction weighted by delta (alpha in code):

  w^delta([X],[Y])^2 = (1/2) sum_{ij} (log(||xi-xj||^2 / ||yi-yj||^2))^2
                       + delta * (log(det G_X / det G_Y))^2

This metric is energy-aware: conformations that differ in shape (gyration) are
penalised more than rigid-body transforms, which are quotiented out exactly.

Tensor convention throughout: (N, M, n, d)
  N = batch of independent problems
  M = ensemble of conformations per problem
  n = number of Cα residues
  d = ambient dimension (3)

Manifold dimension = d*n - d*(d+1)/2  (horizontal DOF after quotienting SE(d))
"""

import jax
import jax.numpy as jnp
import numpy as np
from functools import partial


class ShapeManifold:
    """
    The quotient manifold P(d,n)/E(d) with the w^delta Riemannian metric.

    This is the configuration space for protein backbone conformations: n Cα
    positions in R^d, modulo global translations and rotations. The metric
    encodes both shape (pairwise distances) and size (gyration tensor), making
    geodesics physically meaningful — they interpolate through conformational
    space while preserving the internal geometry of the protein.

    Core operations:
      s_distance  — w^delta distance between conformations
      s_log       — approximate log map (separation, third-order accurate)
      s_exp       — exponential map (iterative geodesic doubling)
      s_geodesic  — geodesic interpolation
      metric_tensor — Riemannian metric H at a point
      horizontal_projection_tvector — project to shape tangent space

    Coordinate normalisation:
      gyration_scale  — sqrt(tr(G)/n), the natural length scale of a conformation
      normalize       — divide coordinates by gyration_scale → Rg ≈ 1, K ≈ 1 in s_exp
      denormalize     — reverse the normalisation (multiply by scale)
    """

    def __init__(self, dim: int, numpoints: int, base=None, alpha: float = 0.1):
        """
        :param dim: ambient dimension d (typically 3 for Cα backbone)
        :param numpoints: number of Cα atoms n
        :param base: optional (n, d) reference conformation for alignment
        :param alpha: gyration weight delta (delta in the w^delta metric).
                      Controls how much size variation is penalised relative
                      to shape variation. alpha=1.0 used for adenylate kinase.
        """
        assert numpoints >= dim + 1
        self.d = dim
        self.n = numpoints
        self.alpha = alpha
        self.manifold_dimension = int(self.d * self.n - self.d * (self.d + 1) / 2)
        self.vert_space_dimension = int(self.d * (self.d + 1) / 2)

        if base is None:
            self.has_base_point = False
            self.base_point = None
        else:
            base = jnp.asarray(base)
            assert base.ndim == 2
            self.has_base_point = True
            self.base_point = self.center_mpoint(base[None, None]).squeeze()

    # ------------------------------------------------------------------
    # Layer 0: elementary point-cloud operations
    # ------------------------------------------------------------------

    def translate_mpoint(self, x, t):
        """
        :param x: (N, M, n, d)
        :param t: (N, M, d)
        :return: (N, M, n, d)
        """
        return x + t[:, :, None, :]

    # ------------------------------------------------------------------
    # Coordinate normalisation
    # ------------------------------------------------------------------

    def gyration_scale(self, x):
        """
        Natural length scale: sqrt(tr(G(x)) / n).

        Normalising by this factor maps the gyration radius to ≈ 1, which
        reduces K in s_exp from O(20) to O(1) on typical protein data.
        The w^delta metric is NOT scale-invariant, so scale must be stored
        and restored for any downstream physical quantity.

        :param x: (N, M, n, d)
        :return: (N, M) scale per conformation
        """
        G = self.gyration_matrix(x)                             # (N, M, d, d)
        return jnp.sqrt(jnp.trace(G, axis1=-2, axis2=-1) / self.n)  # (N, M)

    def normalize(self, x):
        """
        Scale x so that its gyration radius ≈ 1. Returns (x_norm, scale).

        :param x: (N, M, n, d)
        :return: (x_norm, scale)  where scale has shape (N, M)
        """
        scale = self.gyration_scale(x)                          # (N, M)
        x_norm = x / scale[:, :, None, None]
        return x_norm, scale

    def denormalize(self, x_norm, scale):
        """
        Reverse normalisation: x = x_norm * scale.

        :param x_norm: (N, M, n, d)
        :param scale: (N, M)
        :return: (N, M, n, d)
        """
        return x_norm * scale[:, :, None, None]

    def center_mpoint(self, x):
        """
        :param x: (N, M, n, d)
        :return: (N, M, n, d)  — centroid at origin
        """
        t = jnp.mean(x, axis=2)        # (N, M, d)
        return self.translate_mpoint(x, -t)

    def pairwise_distances(self, x):
        """
        Squared pairwise distances ||xi - xj||^2.
        :param x: (N, M, n, d)
        :return: (N, M, n, n)
        """
        gram = jnp.einsum("NMia,NMja->NMij", x, x)         # (N, M, n, n)
        diag = jnp.diagonal(gram, axis1=2, axis2=3)         # (N, M, n)
        diag_mat = jnp.einsum("NMi,j->NMij", diag, jnp.ones(self.n))
        return diag_mat - 2 * gram + jnp.swapaxes(diag_mat, 2, 3)

    # ------------------------------------------------------------------
    # Layer 1: gyration & alignment
    # ------------------------------------------------------------------

    def gyration_matrix(self, x):
        """
        G = X_c^T X_c  (sum over residues)
        :param x: (N, M, n, d)
        :return: (N, M, d, d)
        """
        xc = self.center_mpoint(x)
        return jnp.einsum("NMia,NMib->NMab", xc, xc)

    def orthogonal_transform_mpoint(self, x, O):
        """
        Apply rotation O to every residue: x_i -> O x_i
        :param x: (N, M, n, d)
        :param O: (N, M, d, d)
        :return: (N, M, n, d)
        """
        return jnp.einsum("NMba,NMia->NMib", O, x)

    def least_orthogonal(self, x, base=None):
        """
        Kabsch: find O = argmin sum_i ||base_i - O x_i||^2
        :param x: (N, M, n, d)
        :param base: (n, d)
        :return: (N, M, d, d)
        """
        if base is None:
            assert self.base_point is not None
            base = self.base_point
        inertia = jnp.einsum("NMia,ib->NMab", x, base) / self.n   # (N,M,d,d)
        # jnp.linalg.svd returns (U, S, Vh) where Vh = V^T
        # torch.svd returned (U, S, V); torch used O = V U^T
        # so O = V U^T = Vh^T U^T = (U Vh)^T
        U, _, Vh = jnp.linalg.svd(inertia)
        V = jnp.swapaxes(Vh, -1, -2)                               # (N,M,d,d)
        O = jnp.einsum("NMcb,NMab->NMca", V, U)
        return O

    def align_mpoint(self, x, base=None):
        """
        Center x then rotate it to minimise RMSD to base.
        :param x: (N, M, n, d)
        :param base: (n, d)
        :return: (N, M, n, d)
        """
        if base is None:
            assert self.base_point is not None
            base = self.base_point
        base_ = self.center_mpoint(base[None, None]).squeeze()
        xc = self.center_mpoint(x)
        O = self.least_orthogonal(xc, base=base_)
        return self.orthogonal_transform_mpoint(xc, O)

    # ------------------------------------------------------------------
    # Layer 2: metric tensor
    # ------------------------------------------------------------------

    def metric_tensor(self, x, asmatrix: bool = False):
        """
        H = A + alpha * B  (w^delta Riemannian metric)
        :param x: (N, M, n, d)
        :param asmatrix: if True return (N, M, nd, nd); else (N, M, n, n, d, d)
        :return: see above
        """
        N, M = x.shape[0], x.shape[1]

        pw = self.pairwise_distances(x)
        pw = pw + jnp.eye(self.n)                   # numerical stability

        # A term: -xixj / pw^2, with diagonal correction
        xixj = x[:, :, :, None, :] - x[:, :, None, :, :]  # (N,M,n,n,d)
        xij = jnp.einsum("NMija,NMijb->NMijab", xixj, xixj)  # (N,M,n,n,d,d)
        A = -xij / pw[:, :, :, :, None, None] ** 2       # (N,M,n,n,d,d)

        # Fix diagonal of A: A_ii = -sum_{j!=i} A_ij
        Adiag = -jnp.sum(A, axis=3)                       # (N,M,n,d,d)
        # diag_embed over axis 2,3 (the two n dimensions)
        A = A + jnp.einsum("NMiab,ij->NMijab",
                           Adiag, jnp.eye(self.n))

        # B term: 4 * yi ⊗ yj  where y = G^{-1} xc
        xc = self.center_mpoint(x)
        G = self.gyration_matrix(x)                        # (N,M,d,d)
        L, Q = jnp.linalg.eigh(G)                          # ascending eigs
        yi = jnp.einsum("NMab,NMb,NMcb,NMic->NMia",
                        Q, 1.0 / L, Q, xc)                 # (N,M,n,d)
        B = 4 * jnp.einsum("NMia,NMjb->NMijab", yi, yi)   # (N,M,n,n,d,d)

        H = A + self.alpha * B                             # (N,M,n,n,d,d)

        if asmatrix:
            # permute (N,M,n,n,d,d) -> (N,M,n,d,n,d) then reshape to (N,M,nd,nd)
            H = H.transpose(0, 1, 2, 5, 3, 4)             # (N,M,n,d,n,d)
            H = H.reshape(N, M, self.n * self.d, self.n * self.d)
        return H

    # ------------------------------------------------------------------
    # Layer 3: distance, prelog, log
    # ------------------------------------------------------------------

    def s_distance(self, x, y):
        """
        w^delta distance.
        :param x: (N, M, n, d)
        :param y: (N, M', n, d)
        :return: (N, M, M')
        """
        pw_x = self.pairwise_distances(x) + jnp.eye(self.n)  # (N,M,n,n)
        pw_y = self.pairwise_distances(y) + jnp.eye(self.n)  # (N,M',n,n)

        predists = (0.5 * jnp.log(
            pw_x[:, :, None, :, :] / pw_y[:, None, :, :, :]
        )) ** 2                                               # (N,M,M',n,n)

        Gx = self.gyration_matrix(x)                          # (N,M,d,d)
        Gy = self.gyration_matrix(y)                          # (N,M',d,d)
        det_x = jnp.linalg.det(Gx)                           # (N,M)
        det_y = jnp.linalg.det(Gy)                           # (N,M')
        corrections = jnp.log(
            det_x[:, :, None] / det_y[:, None, :]
        ) ** 2                                                # (N,M,M')

        return jnp.sqrt(
            0.5 * jnp.sum(predists, axis=(3, 4)) + self.alpha * corrections
        )

    def s_prelog(self, x, y, asvector: bool = False):
        """
        Gradient -1/2 * grad_x w(x,y)^2  (separation pre-log).
        :param x: (N, M, n, d)
        :param y: (N, M', n, d)
        :param asvector: if True return (N, M, M', n*d)
        :return: (N, M, M', n, d) or (N, M, M', n*d)
        """
        N, M = x.shape[0], x.shape[1]
        MM = y.shape[1]

        pw_x = self.pairwise_distances(x) + jnp.eye(self.n)  # (N,M,n,n)
        pw_y = self.pairwise_distances(y) + jnp.eye(self.n)  # (N,M',n,n)

        predists = 0.5 * jnp.log(
            pw_x[:, :, None, :, :] / pw_y[:, None, :, :, :]
        )                                                     # (N,M,M',n,n)

        xixj = x[:, :, None, :, None, :] - x[:, :, None, None, :, :]  # (N,M,1,n,n,d)
        # predists: (N,M,M',n,n), xixj: (N,M,1,n,n,d), pw_x: (N,M,1,n,n)
        prelogs = (
            -predists[:, :, :, :, :, None]
            * xixj
            / pw_x[:, :, None, :, :, None]
        )                                                     # (N,M,M',n,n,d)
        prelog = jnp.sum(prelogs, axis=4)                    # (N,M,M',n,d)

        # alpha * gyration correction
        Gx = self.gyration_matrix(x)                          # (N,M,d,d)
        Gy = self.gyration_matrix(y)                          # (N,M',d,d)
        det_x = jnp.linalg.det(Gx)                           # (N,M)
        det_y = jnp.linalg.det(Gy)                           # (N,M')
        precorrections = jnp.log(
            det_x[:, :, None] / det_y[:, None, :]
        )                                                     # (N,M,M')

        xc = self.center_mpoint(x)
        Lv, Qv = jnp.linalg.eigh(Gx)
        gxi = jnp.einsum("NMab,NMb,NMcb,NMic->NMia",
                         Qv, 1.0 / Lv, Qv, xc)               # (N,M,n,d)

        prelogcorrections = (
            -2 * gxi[:, :, None]
            * precorrections[:, :, :, None, None]
        )                                                     # (N,M,M',n,d)

        result = prelog + self.alpha * prelogcorrections      # (N,M,M',n,d)
        if asvector:
            return result.reshape(N, M, MM, self.n * self.d)
        return result

    def s_log(self, x, y, asvector: bool = False):
        """
        Approximate Riemannian log map: H^{-1} prelog  (restricted to horizontal space).
        :param x: (N, M, n, d)
        :param y: (N, M', n, d)
        :param asvector: if True return (N, M, M', n*d)
        :return: (N, M, M', n, d) or (N, M, M', n*d)
        """
        N, M = x.shape[0], x.shape[1]
        MM = y.shape[1]
        vert_dim = self.vert_space_dimension

        prelog = self.s_prelog(x, y, asvector=True)          # (N,M,M',nd)
        H = self.metric_tensor(x, asmatrix=True)             # (N,M,nd,nd)
        L, Q = jnp.linalg.eigh(H)                           # ascending eigs

        # Solve H * log = prelog in horizontal space only (skip bottom vert_dim eigenvectors)
        log = jnp.einsum(
            "NMxy,NMy,NMzy,NMLz->NMLx",
            Q[:, :, :, vert_dim:],
            1.0 / L[:, :, vert_dim:],
            Q[:, :, :, vert_dim:],
            prelog,
        )                                                     # (N,M,M',nd)

        if asvector:
            return log
        return log.reshape(N, M, MM, self.n, self.d)

    def inner(self, x, X, Y):
        """
        g-inner product of tangent vectors.
        :param x: (N, M, n, d)
        :param X: (N, M, L, n, d)
        :param Y: (N, M, K, n, d)
        :return: (N, M, L, K)
        """
        H = self.metric_tensor(x)                            # (N,M,n,n,d,d)
        return jnp.einsum("NMijab,NMLia,NMKjb->NMLK", H, X, Y)

    def orthonormal_basis(self, x, asvector: bool = False):
        """
        Orthonormal basis for the horizontal tangent space.
        :param x: (N, M, n, d)
        :param asvector: if True return (N, M, L, n*d)
        :return: (N, M, L, n, d) with L = manifold_dimension
        """
        N, M = x.shape[0], x.shape[1]
        vert_dim = self.vert_space_dimension

        H = self.metric_tensor(x, asmatrix=True)             # (N,M,nd,nd)
        L, Q = jnp.linalg.eigh(H)

        horizontal_vectors = jnp.swapaxes(Q, -1, -2)[:, :, vert_dim:]   # (N,M,L,nd)
        rescaling = 1.0 / jnp.sqrt(L[:, :, vert_dim:])                   # (N,M,L)
        horizontal_vectors = rescaling[:, :, :, None] * horizontal_vectors

        if asvector:
            return horizontal_vectors
        return horizontal_vectors.reshape(N, M, self.manifold_dimension, self.n, self.d)

    def horizontal_projection_tvector(self, x, X):
        """
        Project tangent vectors onto horizontal subspace (remove rigid-body component).
        :param x: (N, M, n, d)
        :param X: (N, M, M', n, d)
        :return: (N, M, M', n, d)
        """
        N, M = x.shape[0], x.shape[1]
        vert_dim = self.vert_space_dimension

        xc = self.center_mpoint(x)
        G = self.gyration_matrix(x)                          # (N,M,d,d)
        L, Q = jnp.linalg.eigh(G)                           # (N,M,d), (N,M,d,d)

        # Build vertical basis vectors (same construction as PyTorch)
        # Translation part: e_i / sqrt(n) for i in range(d)
        # Rotation part: (L_i*L_j/(L_i+L_j))^{1/2} * Q L^{-1/2} G_{ij} L^{-1/2} Q^T * xc

        # Translation basis: d vectors of shape (N,M,n,d)
        e = jnp.eye(self.d)                                   # (d, d)
        trans_basis = jnp.broadcast_to(
            e[None, None, None, :, :] / self.n ** 0.5,
            (N, M, self.n, self.d, self.d),
        )  # (N,M,n,d,d) — axis -2 = basis index, axis -1 = spatial
        # Reshape to (N,M,d,n,d): [N,M,basis_idx,n,d]
        trans_basis = trans_basis.transpose(0, 1, 3, 2, 4)    # (N,M,d,n,d)

        # Rotation basis: d*(d-1)/2 vectors
        rot_parts = []
        Ld = jnp.einsum("ab,NMa->NMab", jnp.eye(self.d), L ** (-0.5))  # (N,M,d,d)
        for i in range(self.d):
            for j in range(i + 1, self.d):
                Gij = jnp.zeros((self.d, self.d))
                Gij = Gij.at[i, j].set(1.0)
                Gij = Gij.at[j, i].set(-1.0)
                QLGijLQt = Q @ Ld @ Gij[None, None] @ Ld @ jnp.swapaxes(Q, -1, -2)
                norm_factor = jnp.sqrt(
                    (L[:, :, i] * L[:, :, j]) / (L[:, :, i] + L[:, :, j])
                )[:, :, None, None]                            # (N,M,1,1)
                vij = jnp.einsum(
                    "NMab,NMib->NMia", norm_factor * QLGijLQt, xc
                )                                              # (N,M,n,d)
                rot_parts.append(vij)

        rot_basis = jnp.stack(rot_parts, axis=2)              # (N,M,d*(d-1)/2,n,d)

        # Full vertical basis: (N,M,vert_dim,n,d)
        vertical_basis = jnp.concatenate([trans_basis, rot_basis], axis=2)

        # Project X onto vertical basis, subtract
        VX_inner = jnp.einsum("NMVia,NMLia->NMVL", vertical_basis, X)  # (N,M,V,L)
        Vproj_X = jnp.einsum("NMVL,NMVia->NMLia", VX_inner, vertical_basis)

        return X - Vproj_X

    # ------------------------------------------------------------------
    # Layer 4: norm (vmap)
    # ------------------------------------------------------------------

    def norm(self, x, X):
        """
        g-norm of a batch of tangent vectors.
        :param x: (N, M, n, d)
        :param X: (N, M, L, n, d)
        :return: (N, M, L)
        """
        # inner expects (N,M,L,n,d), (N,M,K,n,d) and returns (N,M,L,K)
        inner_val = self.inner(x, X, X)                      # (N,M,L,L)
        # We only need the diagonal (L == K, l==k)
        return jnp.sqrt(jnp.diagonal(inner_val, axis1=2, axis2=3))  # (N,M,L)

    # ------------------------------------------------------------------
    # Layer 5: iterative methods (geodesic, mean, exp)
    # ------------------------------------------------------------------

    def s_geodesic(self, x, y, tau, base=None,
                   step_size: float = 1.0, max_iter: int = 100, tol: float = 1e-3,
                   use_separation_grad: bool = True, z_init=None,
                   return_z: bool = False):
        """
        Geodesic interpolation: find z(tau) s.t. s_distance(z,x)/(s_distance(z,x)+s_distance(z,y)) ≈ tau.
        Uses jax.lax.while_loop for JIT compatibility.

        :param x: (N, 1, n, d)
        :param y: (N, 1, n, d)
        :param tau: (M,) interpolation parameters in [0,1]
        :param base: (n, d) alignment reference
        :param use_separation_grad: if True (default), use Euclidean gradient of w² (s_prelog,
            O(n²d) per step, no eigh). If False, use Riemannian gradient (s_log, O((nd)³) per step).
            Both converge to the same minimum; True is ~2800× cheaper per iteration for n=214.
        :param z_init: (N, M, n, d) optional warm-start initial guess for z.
            If None, initialises z = y (cold start). Pass the converged z from a nearby
            s_geodesic call (e.g. the previous doubling step in s_exp) to reduce iterations.
        :param return_z: if True, return (aligned_z, z_converged) instead of just aligned_z.
            z_converged is the raw (pre-alignment) converged iterate, useful as z_init for the
            next call. Default False for backward compatibility.
        :return: (N, M, n, d)  or  ((N, M, n, d), (N, M, n, d)) if return_z=True
        """
        if base is None:
            assert self.base_point is not None
            base = self.base_point

        tau = jnp.asarray(tau)                               # (M,)
        N = x.shape[0]
        M = tau.shape[0]

        error0 = jnp.max(self.s_distance(x, y)) + 1e-6      # scalar

        if z_init is None:
            z = jnp.ones((M,))[None, :, None, None] * y      # (N,M,n,d) cold start
        else:
            z = z_init                                        # (N,M,n,d) warm start

        def cond_fn(carry):
            z, relerror, k, error0 = carry
            return jnp.logical_and(relerror > tol, k <= max_iter)

        if use_separation_grad:
            # Fast path: use flat gradient of w² (s_prelog = -1/2 grad_z w(z,·)²)
            # grad_z f = -(1-tau)*s_prelog(z,x) - tau*s_prelog(z,y)
            # step: z -= step_size * grad_z f  =>  z += step_size * [(1-tau)*prelog_x + tau*prelog_y]
            # Convergence: Euclidean norm of the step direction (no eigh needed)
            def body_fn(carry):
                z, relerror, k, error0 = carry
                prelog_x = self.s_prelog(z, x)[:, :, 0]     # (N,M,n,d)
                prelog_y = self.s_prelog(z, y)[:, :, 0]     # (N,M,n,d)
                grad_Wz = (
                    -(1 - tau[None, :, None, None]) * prelog_x
                    - tau[None, :, None, None] * prelog_y
                )
                z = z - step_size * grad_Wz
                # Euclidean norm of step — no eigh, cheap O(nd)
                error = jnp.max(jnp.sqrt(jnp.sum(grad_Wz ** 2, axis=(-2, -1))))
                relerror = error / error0
                return z, relerror, k + 1, error0
        else:
            # Exact path: Riemannian gradient (H⁻¹ prelog), requires eigh(nd×nd) per step
            def body_fn(carry):
                z, relerror, k, error0 = carry
                grad_Wzx = -self.s_log(z, x)[:, :, 0]       # (N,M,n,d)
                grad_Wzy = -self.s_log(z, y)[:, :, 0]       # (N,M,n,d)
                grad_Wz = (
                    (1 - tau[None, :, None, None]) * grad_Wzx
                    + tau[None, :, None, None] * grad_Wzy
                )
                z = z - step_size * grad_Wz
                error = jnp.max(self.norm(z, grad_Wz[:, :, None]))
                relerror = error / error0
                return z, relerror, k + 1, error0

        init = (z, jnp.array(1.0), jnp.array(1), error0)
        z_conv, _, _, _ = jax.lax.while_loop(cond_fn, body_fn, init)

        aligned = self.align_mpoint(z_conv, base=base)
        if return_z:
            return aligned, z_conv
        return aligned

    def s_mean(self, x, x0=None, base=None,
               step_size: float = 1.0, max_iter: int = 100, tol: float = 1e-3):
        """
        Fréchet mean on the manifold.
        Uses jax.lax.while_loop for JIT compatibility.
        :param x: (N, M, n, d)
        :param x0: (N, 1, n, d) optional initial guess
        :param base: (n, d) alignment reference
        :return: (N, 1, n, d)
        """
        if base is None:
            assert self.base_point is not None
            base = self.base_point

        if x0 is not None:
            z = x0
            pws_mat = self.s_distance(x, z) ** 2
            error0 = jnp.sqrt(jnp.max(jnp.min(jnp.sum(pws_mat, 1), axis=1))) + 1e-6
        else:
            pws_mat = self.s_distance(x, x) ** 2
            error0 = jnp.sqrt(jnp.max(jnp.min(jnp.sum(pws_mat, 1), axis=1))) + 1e-6
            best_idx = jnp.argmin(jnp.sum(pws_mat, axis=2), axis=1)   # (N,)
            # Gather: z[n] = x[n, best_idx[n], :, :]
            z = x[jnp.arange(x.shape[0]), best_idx][:, None]          # (N,1,n,d)

        def cond_fn(carry):
            z, relerror, k, error0 = carry
            return jnp.logical_and(relerror > tol, k <= max_iter)

        def body_fn(carry):
            z, relerror, k, error0 = carry
            grad_Wz = -jnp.mean(self.s_log(z, x), axis=2)   # (N,1,n,d)
            z = z - step_size * grad_Wz
            error = jnp.max(self.norm(z, grad_Wz[:, :, None]))
            relerror = error / error0
            return z, relerror, k + 1, error0

        init = (z, jnp.array(1.0), jnp.array(1), error0)
        z, _, _, _ = jax.lax.while_loop(cond_fn, body_fn, init)

        return self.align_mpoint(z, base=base)

    def _s_exp_fixed_K(self, x, X, K: int,
                        c: float = 0.25,
                        step_size: float = 1.0, max_iter: int = 100, tol: float = 1e-3,
                        use_separation_grad: bool = True):
        """
        Pure-JAX exponential map with K fixed as a Python constant.

        Identical to s_exp but skips the data-dependent K computation
        (K = int(c * max(nrm)) + 1).  Because K is a Python literal here,
        this function contains no concrete-value dependencies and can be
        safely vmapped or JIT-compiled as part of a larger computation.

        Use this (via ManifoldVP.marginal_prob(fixed_K=1)) when you want to
        vmap prepare_batch over the batch dimension for GPU-parallel training.

        K=1 is valid for BBA (n=28) and chignolin (n=10) data — confirmed
        empirically from the precomputed dataset (all frames had K=1).
        For larger proteins (e.g. adenylate kinase n=214), verify K first.

        :param x: (N, 1, n, d)
        :param X: (N, 1, n, d) tangent vector
        :param K: Python int — number of doubling steps (fixed, not computed from data)
        :return: (N, 1, n, d)
        """
        base = self.base_point
        x0 = x
        x1 = x + (1.0 / K) * X
        z_init_first = 2.0 * x1 - x0

        cache_key = (K, use_separation_grad, tol, max_iter, step_size)
        if not hasattr(self, '_doubling_cache'):
            self._doubling_cache = {}
        if cache_key not in self._doubling_cache:
            self._doubling_cache[cache_key] = self._build_doubling_fn(
                K, use_separation_grad, tol, max_iter, step_size, base
            )
        run_doubling = self._doubling_cache[cache_key]
        xK = run_doubling(x0, x1, z_init_first)
        return self.align_mpoint(xK, base=base)

    def s_exp(self, x, X, c: float = 0.25, base=None,
              step_size: float = 1.0, max_iter: int = 100, tol: float = 1e-3,
              use_separation_grad: bool = True):
        """
        Geodesic exponential map via iterative geodesic doubling.

        K = int(c * norm(x, X).max()) + 1 is computed as a Python int outside JIT.
        The doubling loop is compiled via a cached jax.jit function keyed on
        (K, use_separation_grad, tol, max_iter, step_size) so repeated calls with
        the same parameters reuse the compiled XLA program (no Python dispatch
        overhead per iteration).

        Warm-starting: each doubling step initialises z = 2*x_k - x_{k-1}
        (linear extrapolation, exact in flat geometry for tau=2). This reduces
        s_geodesic iterations from ~47 (cold start) to ~4 on n=214.

        :param x: (N, 1, n, d)
        :param X: (N, 1, n, d)  tangent vector
        :param use_separation_grad: if True (default), use s_prelog gradient in
            s_geodesic inner loop (no eigh, ~2800× cheaper per step for n=214).
        :return: (N, 1, n, d)
        """
        if base is None:
            assert self.base_point is not None
            base = self.base_point

        # K is a Python int — computed outside JIT
        nrm = self.norm(x, X[:, :, None])                   # (N, 1, 1)
        K = int(c * float(jnp.max(nrm))) + 1

        x0 = x
        x1 = x + (1.0 / K) * X
        z_init_first = 2.0 * x1 - x0                        # linear extrapolation warm-start

        # Retrieve (or compile) a JIT-compiled doubling function for this K/parameter combo.
        # _doubling_cache is a dict on self, keyed by hashable params. base is not in the
        # key because it's always self.base_point (stable for the lifetime of ShapeManifold).
        cache_key = (K, use_separation_grad, tol, max_iter, step_size)
        if not hasattr(self, '_doubling_cache'):
            self._doubling_cache = {}
        if cache_key not in self._doubling_cache:
            self._doubling_cache[cache_key] = self._build_doubling_fn(
                K, use_separation_grad, tol, max_iter, step_size, base
            )
        run_doubling = self._doubling_cache[cache_key]
        xK = run_doubling(x0, x1, z_init_first)
        return self.align_mpoint(xK, base=base)

    def _build_doubling_fn(self, K, use_separation_grad, tol, max_iter, step_size, base):
        """
        Build and return a jax.jit-compiled function that runs K geodesic doubling
        steps with warm-starting. Called once per unique (K, params) combination;
        the result is cached in self._doubling_cache.

        Signature: run_doubling(x0, x1, z_init_first) -> xK
        All inputs/output are (N, 1, n, d).
        """
        tau_two = jnp.array([2.0])

        @jax.jit
        def run_doubling(x0, x1, z_init_first):
            def body_fn(k, carry):
                xkk, xk, z_init = carry
                # Inline s_geodesic to avoid Python dispatch overhead
                error0 = jnp.max(self.s_distance(xkk, xk)) + 1e-6

                if use_separation_grad:
                    def cond(c): z, rel, it = c; return jnp.logical_and(rel > tol, it <= max_iter)
                    def body(c):
                        z, rel, it = c
                        px = self.s_prelog(z, xkk)[:, :, 0]
                        py = self.s_prelog(z, xk)[:, :, 0]
                        grad = (-(1 - tau_two[None, :, None, None]) * px
                                - tau_two[None, :, None, None] * py)
                        z = z - step_size * grad
                        err = jnp.max(jnp.sqrt(jnp.sum(grad ** 2, axis=(-2, -1))))
                        return z, err / error0, it + 1
                else:
                    def cond(c): z, rel, it = c; return jnp.logical_and(rel > tol, it <= max_iter)
                    def body(c):
                        z, rel, it = c
                        gx = -self.s_log(z, xkk)[:, :, 0]
                        gy = -self.s_log(z, xk)[:, :, 0]
                        grad = ((1 - tau_two[None, :, None, None]) * gx
                                + tau_two[None, :, None, None] * gy)
                        z = z - step_size * grad
                        err = jnp.max(self.norm(z, grad[:, :, None]))
                        return z, err / error0, it + 1

                z_conv, _, _ = jax.lax.while_loop(
                    cond, body, (z_init, jnp.array(1.0), jnp.array(1))
                )
                x_new = self.align_mpoint(z_conv, base=base)
                z_next = 2.0 * x_new - xk
                return xk, x_new, z_next

            _, xK, _ = jax.lax.fori_loop(0, K, body_fn, (x0, x1, z_init_first))
            return xK

        return run_doubling
