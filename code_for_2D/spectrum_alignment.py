"""Functions for spectrum alignment."""
import os
import sys

import numpy as np
import scipy
import torch
from scipy import sparse

from shape_library import load_mesh, prepare_mesh, resample

DEFAULT_DEVICE = torch.device("cuda")


class OptimizationParams:
    def __init__(self):
        """Class that holds the hyperparamters."""
        # Training
        self.steps = 5000
        self.checkpoint = 100
        self.remesh_step = 500

        # Number of eigenvalues to align
        self.evals = [20]

        # Early stopping
        self.min_eval_loss = 0.05

        # Adam optimizer
        self.learning_rate = 0.005

        # Regularizer coefficients
        self.decay_target = 0.05
        self.bound_reg = 2e1
        self.inner_reg = 1e0
        self.flip_penalty_reg = 1e10


def tf_calc_lap(mesh, VERT, device=DEFAULT_DEVICE):
    """Compute the Laplacian."""
    # Move the mesh to the target device
    mesh_tensor = []
    for i in range(len(mesh)):
        mesh_tensor.append(torch.as_tensor(mesh[i]).to(device))

    # Unpack the mesh
    [
        _,
        TRIV,
        n,
        m,
        Ik,
        Ih,
        Ik_k,
        Ih_k,
        Tpi,
        Txi,
        Tni,
        iM,
        Windices,
        Ael,
        Bary,
        bound_edges,
        ord_list,
    ] = mesh_tensor

    # Set the data type
    dtype = "float32"
    if VERT.dtype == "float64":
        dtype = "float64"
    if VERT.dtype == "float16":
        dtype = "float16"

    # Move the embedding to the target device
    VERT = torch.as_tensor(VERT).to(device)

    # Compute the edge lengths
    L2 = torch.unsqueeze(torch.sum(torch.mm(iM, VERT) ** 2, dim=1), dim=1)
    L = torch.sqrt(L2)

    def fAk(Ik, Ik_k):  # Ik: 1 if (edg1, edg2) in same tri, -1 if same edge
        Ikp = torch.abs(Ik)
        Sk = torch.mm(Ikp, L) / 2  # Perimeter of associated tri for each edge (m, )
        SkL = Sk - L
        Ak = (
            Sk
            * (torch.mm(Ik_k[:, :, 0], Sk) - torch.mm(Ik_k[:, :, 0], L))
            * (torch.mm(Ik_k[:, :, 0], Sk) - torch.mm(Ik_k[:, :, 1], L))
            * (torch.mm(Ik_k[:, :, 0], Sk) - torch.mm(Ik_k[:, :, 2], L))
        )
        return torch.sqrt(torch.abs(Ak) + 1e-20)

    Ak = fAk(Ik, Ik_k)  # (m, )
    Ah = fAk(Ih, Ih_k)  # (m, )

    # Sparse representation of the Laplacian matrix
    W = -torch.mm(Ik, L2) / (8 * Ak) - torch.mm(Ih, L2) / (8 * Ah)  # (m, )

    # Compute indices to build the dense Laplacian matrix
    if dtype == "float32":
        Windtf = torch.sparse.FloatTensor(
            torch.tensor(
                Windices.type(torch.long), dtype=torch.long, device=device
            ).t(),  #
            torch.tensor(-np.ones((m), dtype), dtype=torch.float, device=device),
            torch.Size([n * n, m]),
        )
    elif dtype == "float64":
        Windtf = torch.sparse.DoubleTensor(
            torch.cuda.LongTensor(Windices.type(torch.long), device=device).t(),
            torch.cuda.DoubleTensor(-np.ones((m), dtype), device=device),
            torch.Size([n * n, m]),
        )
    elif dtype == "float16":
        Windtf = torch.sparse.HalfTensor(
            torch.cuda.LongTensor(Windices.type(torch.long), device=device).t(),
            torch.cuda.HalfTensor(-np.ones((m), dtype), device=device),
            torch.Size([n * n, m]),
        )
    Wfull = -torch.reshape(torch.mm(Windtf, W), (n, n))
    Wfull = Wfull + torch.t(Wfull)

    # Compute the actual Laplacian
    Lx = Wfull - torch.diag(torch.sum(Wfull, dim=1))  # (n, n)
    S = (torch.mm(Ael, Ak) + torch.mm(Ael, Ah)) / 6  # (n, )

    return Lx, S, L, Ak


def calc_evals(VERT, TRIV):
    """Compute the eigenvalue sequence."""
    mesh = prepare_mesh(VERT, TRIV)
    Lx, S, _, _ = tf_calc_lap(mesh, mesh[0])
    Si = torch.diag(torch.sqrt(1 / S[:, 0]))
    Lap = torch.mm(Si, torch.mm(Lx, Si))
    evals, _ = torch.symeig(Lap)
    return evals


def initialize(mesh, step=1.0, params=OptimizationParams(), device=DEFAULT_DEVICE):
    """Initialize the model."""
    # Namespace
    graph = lambda: None
    graph.global_step = torch.as_tensor(step + 1.0, dtype=torch.float32)

    # Unpack the mesh
    [
        Xori,
        TRIV,
        n,
        m,
        Ik,
        Ih,
        Ik_k,
        Ih_k,
        Tpi,
        Txi,
        Tni,
        iM,
        Windices,
        Ael,
        Bary,
        bound_edges,
        ord_list,
    ] = mesh

    # Set datatype
    graph.dtype = "float32"
    if Xori.dtype == "float64":
        graph.dtype = "float64"
    elif Xori.dtype == "float16":
        graph.dtype = "float16"

    # Model the shape deformation as a displacement vector field
    graph.dXb = torch.zeros(Xori.shape, requires_grad=True, device=device)
    graph.dXi = torch.zeros(Xori.shape, requires_grad=True, device=device)

    # The optimizers
    graph.optim_dXb = torch.optim.Adam([graph.dXb], lr=params.learning_rate)
    graph.optim_dXi = torch.optim.Adam([graph.dXi], lr=params.learning_rate)

    return graph

def l2_loss(t):
    """Return the l2 loss."""
    return 0.5 * torch.sum(t ** 2)

def forward(
    costType,
    mode,
    graph,
    mesh,
    target_evals,
    nevals,
    step=1.0,
    params=OptimizationParams(),
    device=DEFAULT_DEVICE
):
    """Perform a forward pass."""
    # Unpack the mesh
    [
        Xori,
        TRIV,
        n,
        m,
        Ik,
        Ih,
        Ik_k,
        Ih_k,
        Tpi,
        Txi,
        Tni,
        iM,
        Windices,
        Ael,
        Bary,
        bound_edges,
        ord_list,
    ] = mesh

    # Cosine decay for the regularizers
    cosine_decay = 0.5 * (
        1
        + np.cos(
            3.14
            * np.minimum(params.steps / 2.0, graph.global_step)
            / (params.steps / 2.0)
        )
    )
    decay = (1 - params.decay_target) * cosine_decay + params.decay_target
    decay = np.float(decay)

    scaleX = 1  # not used in shape alignment

    # Model the shape deformation as a displacement vector field
    bound_vert = np.zeros((n, 1), graph.dtype)
    bound_vert[ord_list] = 1

    def to_device(t):
        return torch.as_tensor(t).to(device)

    bound_vert = to_device(bound_vert)
    X = (to_device(Xori) + graph.dXb * bound_vert + graph.dXi * (1 - bound_vert)) * scaleX

    Lx, S, L, Ak = tf_calc_lap(mesh, X)

    # Normalized Laplacian
    Si = torch.diag(torch.sqrt(1 / S[:, 0]))
    Lap = torch.mm(Si, torch.mm(Lx, Si))

    # Spectral decomposition
    [evals, v] = torch.symeig(Lap, eigenvectors=True)
    cost_evals = 1e1 * l2_loss(
        (evals[0:nevals] - target_evals[0:nevals])
        * (
            1
            / torch.as_tensor(np.asarray(range(1, nevals + 1), graph.dtype)).to(device)
        )
    )

    # Triangle flip penalty
    Tpi = to_device(Tpi)
    Txi = to_device(Txi)
    Tni = to_device(Tni)
    tp = torch.mm(Tpi[:, :], X)
    tx = torch.mm(Txi[:, :], X)
    tn = torch.mm(Tni[:, :], X)
    Rot = to_device(np.asarray([[0, 1], [-1, 0]], graph.dtype))
    cp = torch.sum(torch.mm(tn, Rot) * (tx - tp), dim=1)
    cp = cp - 1e-4
    flip_cost = params.flip_penalty_reg * l2_loss(cp - torch.abs(cp))

    # Inner points regularizer
    varA = torch.std(Ak, dim=[0])
    inner_reg_cost = params.inner_reg * (l2_loss(L) + l2_loss(varA))

    # Boundary points regularizer
    bound_reg_cost = params.bound_reg * decay * torch.sum(L[bound_edges[:, 0], :])

    # Inner and outer points cost functions
    cost_bound = cost_evals + flip_cost + bound_reg_cost
    cost_inner = inner_reg_cost + flip_cost

    def to_numpy(a):
        o = []
        for ai in a:
            oi = ai.cpu().detach().numpy()
            o.append(oi)
        return o

    if mode == "train":
        if costType == "bound":
            graph.optim_dXb.zero_grad()
            cost_bound.backward()
            graph.dXb.grad.data.clamp_(-0.0001, 0.0001)
            graph.optim_dXb.step()
            outList = [cost_bound, cost_evals, X]
            return to_numpy(outList)
        if costType == "inner":
            graph.optim_dXi.zero_grad()
            cost_inner.backward()
            graph.dXi.grad.data.clamp_(-0.0001, 0.0001)
            graph.optim_dXi.step()
            outList = [cost_inner, cost_evals, X]
            return to_numpy(outList)
    elif mode == "eval":
        outList1 = [cost_bound, cost_evals, inner_reg_cost, bound_reg_cost]
        outList1 = to_numpy(outList1)
        outList2 = [cp, evals]
        outList2 = to_numpy(outList2)
        return outList1 + [decay] + outList2


def run_optimization(mesh, target_evals, out_path, params=OptimizationParams()):
    """Run the optimization."""
    # Create the output directories
    os.makedirs(f"{out_path}/ply", exist_ok=True)
    os.makedirs(f"{out_path}/txt", exist_ok=True)

    # Unpack the mesh
    [
        Xopt,
        TRIV,
        n,
        m,
        Ik,
        Ih,
        Ik_k,
        Ih_k,
        Tpi,
        Txi,
        Tni,
        iM,
        Windices,
        Ael,
        Bary,
        bound_edges,
        ord_list,
    ] = mesh

    # Save the initial embedding
    save_ply(Xopt, TRIV, "%s/ply/initial.ply" % out_path)

    # Save the target eigenvalue sequence
    np.savetxt("%s/txt/target.txt" % out_path, target_evals.cpu().detach().numpy())

    iterations = []
    for nevals in params.evals:

        step = 0

        while step < params.steps - 1:
            # Prepare the mesh
            mesh = prepare_mesh(Xopt, TRIV)

            # Unpack the mesh
            [
                Xori,
                TRIV,
                n,
                m,
                Ik,
                Ih,
                Ik_k,
                Ih_k,
                Tpi,
                Txi,
                Tni,
                iM,
                Windices,
                Ael,
                Bary,
                bound_edges,
                ord_list,
            ] = mesh

            # Initialize the model
            graph = initialize(mesh, step=step)

            tic()

            # Start iteration
            for step in range(step + 1, params.steps):

                # Recompute triangulation
                if step % params.remesh_step == 0:
                    print("RECOMPUTING TRIANGULATION at step %d" % step)
                    break

                try:
                    # Alternate optimization of inner and boundary vertices
                    if int(step / 10) % 2 == 0:
                        # Optimize over inner points
                        er, ee, Xopt_t = forward(
                            "inner",
                            "train",
                            graph,
                            mesh,
                            target_evals,
                            nevals,
                            step,
                            params,
                        )
                    else:
                        # Optimize over boundary points
                        er, ee, Xopt_t = forward(
                            "bound",
                            "train",
                            graph,
                            mesh,
                            target_evals,
                            nevals,
                            step,
                            params,
                        )

                    iterations.append((step, nevals, er, ee, int(step / 10) % 2))

                    if (
                        step % params.checkpoint == 0
                        or step == params.steps - 1
                        or step == 1
                    ):
                        toc()
                        tic()

                        # Perform a forward pass in eval mode
                        (
                            cost,
                            cost_evals,
                            cost_vcL,
                            cost_vcW,
                            decay,
                            flip,
                            evout,
                        ) = forward(
                            "bound", "eval", graph, mesh, target_evals, nevals, step
                        )

                        print(
                            "Iter %f, cost: %f(evals cost: %f (%f) (%f), smoothness weight: %f). Flip: %d"
                            % (
                                step,
                                cost,
                                cost_evals,
                                cost_vcL,
                                cost_vcW,
                                decay,
                                np.sum(flip < 0),
                            )
                        )

                        # Save the current embedding
                        save_ply(
                            Xopt,
                            TRIV,
                            "%s/ply/evals_%d_iter_%06d.ply" % (out_path, nevals, step),
                        )

                        # Save the current eigenvalue sequence
                        np.savetxt(
                            "%s/txt/evals_%d_iter_%06d.txt" % (out_path, nevals, step),
                            evout,
                        )

                        # Save the training progress statistics
                        np.savetxt("%s/iterations.txt" % (out_path), iterations)

                        # Early stopping
                        if ee < params.min_eval_loss:
                            step = params.steps
                            print("Minimum eigenvalues loss reached")
                            break

                except KeyboardInterrupt:
                    step = params.steps
                    break

                except:
                    print(sys.exc_info())
                    ee = float("nan")

                if ee != ee:
                    # If nan (something went wrong) with the spectral decomposition,
                    # perturbate the last valid state and start over
                    print("iter %d. Perturbating initial condition" % step)
                    Xopt = (
                        Xopt
                        + (np.random.rand(np.shape(Xopt)[0], np.shape(Xopt)[1]) - 0.5)
                        * 1e-3
                    )
                    graph.global_step = step

                else:
                    Xopt = Xopt_t
                    graph.global_step += 1

            if step < params.steps - 1:
                [Xopt, TRIV] = resample(Xopt, TRIV)
