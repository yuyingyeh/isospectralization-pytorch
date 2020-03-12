"""Functions for spectrum alignment."""
import os
import sys

import numpy as np
import scipy
import torch
from scipy import sparse

from shape_library import *

DEFAULT_DEVICE = torch.device("cuda")


class OptimizationParams:
    """Class that holds the hyperparamters."""

    def __init__(self, smoothing="displacement"):
        # Training
        self.steps = 2000
        self.checkpoint_steps = 100
        self.eval_steps = 100

        # Number of eigenvalues to align
        self.evals = [10, 20, 30]

        # Early stopping
        self.min_eval_loss = 0.05

        # Adam optimizer
        self.learning_rate = 0.005  # 0.00025
        self.beta1 = 0.9
        self.beta2 = 0.999

        # Smoothing mode
        self.smoothing = smoothing
        if smoothing == "displacement":
            self.curvature_reg = 2e3
            self.smoothness_reg = 2e3
        elif smoothing == "absolute":
            self.curvature_reg = 1e5
            self.smoothness_reg = 5e4
        else:
            raise ValueError(f"Unrecognized smoothing mode (got {smoothing})")

        # Regularizer coefficients
        self.volume_reg = 1e3  # 1e1
        self.l2_reg = 2e6
        self.decay_target = 0.05  # 0.01


def tf_calc_lap(mesh, VERT, device=DEFAULT_DEVICE):
    """Compute the Laplacian."""
    # Move the mesh to the target device
    mesh_tensor = []
    for i in range(len(mesh)):
        mesh_tensor.append(torch.as_tensor(mesh[i]).to(device))

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
    ] = mesh_tensor

    # Set the data type
    dtype = "float32"
    if VERT.dtype == "float64":
        dtype = "float64"
    elif VERT.dtype == "float16":
        dtype = "float16"

    # Move the embedding to the target device
    VERT = torch.as_tensor(VERT).to(device)

    # Compute the edge lengths
    L2 = torch.sum(torch.mm(iM, VERT) ** 2, dim=1, keepdim=True)
    L = torch.sqrt(L2)

    def fAk(Ik, Ik_k):
        Ikp = torch.abs(Ik)
        Sk = torch.mm(Ikp, L) / 2
        SkL = Sk - L
        Ak = (
            Sk
            * (torch.mm(Ik_k[:, :, 0], Sk) - torch.mm(Ik_k[:, :, 0], L))
            * (torch.mm(Ik_k[:, :, 0], Sk) - torch.mm(Ik_k[:, :, 1], L))
            * (torch.mm(Ik_k[:, :, 0], Sk) - torch.mm(Ik_k[:, :, 2], L))
        )
        return torch.sqrt(torch.abs(Ak) + 1e-20)

    Ak = fAk(Ik, Ik_k)
    Ah = fAk(Ih, Ih_k)

    # Sparse representation of the Laplacian matrix
    W = -torch.mm(Ik, L2) / (8 * Ak) - torch.mm(Ih, L2) / (8 * Ah)

    # Compute indices to build the dense Laplacian matrix
    if dtype == "float32":
        col_dtype = torch.float
    elif dtype == "float64":
        col_dtype = torch.double
    elif dtype == "float16":
        col_dtype = torch.half
    else:
        raise TypeError(f"Unrecognized dtype (got {dtype})")

    Windtf = torch.sparse.FloatTensor(
        torch.tensor(Windices.type(torch.long), dtype=torch.long, device=device).t(),
        torch.tensor(-np.ones((m), dtype), dtype=col_dtype, device=device),
        torch.Size([n * n, m]),
    )
    Wfull = -torch.reshape(torch.mm(Windtf, W), (n, n))
    Wfull = Wfull + torch.t(Wfull)

    # Compute the actual Laplacian
    Lx = Wfull - torch.diag(torch.sum(Wfull, dim=1))
    S = (torch.mm(Ael, Ak) + torch.mm(Ael, Ah)) / 6

    return Lx, S, L, Ak


def calc_evals(VERT, TRIV):
    """Compute the eigenvalue sequence."""
    mesh = prepare_mesh(VERT, TRIV, "float32")
    Lx, S, _, _ = tf_calc_lap(mesh, mesh[0])
    Si = torch.diag(torch.sqrt(1 / S[:, 0]))
    Lap = torch.mm(Si, torch.mm(Lx, Si))
    evals, _ = torch.symeig(Lap)
    return evals


def initialize(mesh, step=1.0, params=OptimizationParams(), device=DEFAULT_DEVICE):
    """Initialize the model."""
    # Namespace
    graph = lambda: None
    graph.is_training = None
    graph.global_step = torch.tensor(
        step + 1.0, dtype=torch.float32, requires_grad=False
    )


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
    ] = mesh

    # Set datatype
    if Xori.dtype == "float32":
        graph.dtype = torch.float
    elif Xori.dtype == "float64":
        graph.dtype = torch.double
    elif Xori.dtype == "float16":
        graph.dtype = torch.half
    else:
        raise TypeError(f"Unsupported dtype (got {Xori.dtype})")
    graph.np_dtype = Xori.dtype

    # Model the shape deformation as a displacement vector field
    graph.dX = torch.zeros(
        Xori.shape, dtype=graph.dtype, requires_grad=True, device=device
    )

    graph.scaleX = torch.tensor(
        1.0, dtype=graph.dtype, requires_grad=True, device=device
    )

    # The optimizer
    graph.optim = torch.optim.Adam(
        [graph.dX], lr=params.learning_rate, betas=(params.beta1, params.beta2)
    )

    return graph


def l2_loss(t):
    """Return the l2 loss."""
    return 0.5 * torch.sum(t ** 2)


def forward(
    graph,
    mesh,
    target_evals,
    nevals,
    nfix,
    step=1.0,
    params=OptimizationParams(),
    device=DEFAULT_DEVICE,
):
    """Perform a forward pass. Update the parameters if in train mode."""

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
    ] = mesh
    Bary = torch.as_tensor(Bary).to(device)
    TRIV = torch.as_tensor(TRIV, dtype=torch.long).to(device)

    graph.X = torch.as_tensor(Xori).to(device) * graph.scaleX + graph.dX

    Lx, S, L, Ak = tf_calc_lap(mesh, graph.X)

    # Normalized Laplacian
    Si = torch.diag(torch.sqrt(1 / S[:, 0]))
    Lap = torch.mm(Si, torch.mm(Lx, Si))

    # Spectral decomposition approach
    s_, v = torch.symeig(Lap, eigenvectors=True)
    graph.cost_evals_f1 = (
        1e2
        * l2_loss(
            (s_[0:nevals] - target_evals[0:nevals])
            * (
                1
                / torch.as_tensor(np.asarray(range(1, nevals + 1), graph.np_dtype)).to(
                    device
                )
            )
        )
        / nevals
    )

    # Cosine decay for the regularizers
    cosine_decay = 0.5 * (
        1
        + np.cos(
            3.14
            * np.minimum(params.steps / 2.0, graph.global_step)
            / (params.steps / 2.0)
        )
    )
    graph.decay = (1 - params.decay_target) * cosine_decay + params.decay_target
    graph.decay = np.float(graph.decay)

    if params.smoothing == "displacement":
        # Regularizers for displacement-based formulation
        graph.vcL = (
            params.curvature_reg
            * graph.decay
            * l2_loss(torch.mm(Bary.type(graph.dtype), graph.dX)[nfix:, :])
        )
        graph.vcW = (
            params.smoothness_reg
            * graph.decay
            * l2_loss(torch.mm(Lx, graph.dX)[nfix:, :])
        )

    elif params.smoothing == "absolute":
        # Regularizers for absolute-coordinate-based formulation
        graph.vcL = (
            params.curvature_reg
            * graph.decay
            * tf.nn.l2_loss(tf.matmul(Bary.astype(dtype), S * graph.X)[nfix:, :])
        )
        # [Herman] Seems incorrect to have ** instead of *
        graph.vcW = params.smoothness_reg ** graph.decay * tf.nn.l2_loss(
            tf.matmul(Lx, graph.X)[nfix:, :]
        )

    # Compute the volume
    T1 = graph.X[TRIV[:, 0]]
    T2 = graph.X[TRIV[:, 1]]
    T3 = graph.X[TRIV[:, 2]]
    XP = torch.cross(T2 - T1, T3 - T2)
    T_C = (T1 + T2 + T3) / 3

    # Volume regularizer
    graph.Volume = params.volume_reg * graph.decay * torch.sum(XP * T_C / 2) / 3

    # L2 regularizer on total displacement weighted by area elements
    graph.l2_reg = params.l2_reg * l2_loss(S * graph.dX)

    # Total loss
    graph.cost_spectral = (
        graph.cost_evals_f1 + graph.vcW + graph.vcL - graph.Volume + graph.l2_reg
    )

    if graph.is_training:
        # Update the parameters if in train mode
        graph.optim.zero_grad()
        graph.cost_spectral.backward()
        graph.dX.grad.data.clamp_(-0.0001, 0.0001)  # Gradient clipping
        graph.optim.step()

    else:
        # Compute the eigenvalue sequence of the Laplacian
        graph.s_, _ = torch.symeig(Lap)

    return graph


def to_numpy(a):
    """Convert tensors to numpy arrays."""
    return [x.cpu().detach().numpy() for x in a]


def run_optimization(mesh, target_evals, out_path, params=OptimizationParams()):
    """Run the optimization."""
    # Create the output directories
    os.makedirs(f"{out_path}/ply", exist_ok=True)
    os.makedirs(f"{out_path}/txt", exist_ok=True)

    # Unpack the mesh
    [
        VERT,
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
    ] = mesh

    # Initialize variables
    pstart = 0
    Xori = VERT[:, 0:3]
    Xopt = VERT[:, 0:3]

    # Save the initial embedding
    save_ply(Xori, TRIV, "%s/ply/initial.ply" % out_path)
    # Save the target eigenvalue sequence
    np.savetxt("%s/txt/target.txt" % out_path, target_evals.cpu().detach().numpy())

    # Optimize the shape increasing the number of eigenvalue to be taken into account
    iterations = []
    for nevals in params.evals:

        step = 0

        # Initialize the model
        graph = initialize(mesh, step, params)

        while step < params.steps - 1:
            tic()

            for step in range(step, params.steps):
                try:
                    # Optimization step
                    graph.is_training = True
                    forward(graph, mesh, target_evals, nevals, pstart, step, params)
                    er, ee, Xopt_t = to_numpy(
                        [graph.cost_spectral, graph.cost_evals_f1, graph.X]
                    )
                    iterations.append((step, nevals, er, ee))

                    # Evaluation
                    if step % params.eval_steps == 0 or step == params.steps - 1:
                        toc()
                        tic()

                        # Perform a forward pass in eval mode
                        graph.is_training = False
                        forward(graph, mesh, target_evals, nevals, pstart, step, params)
                        er, erE, ervcL, evout, errcW, vol, l2reg = to_numpy(
                            [
                                graph.cost_spectral,
                                graph.cost_evals_f1,
                                graph.vcL,
                                graph.s_,
                                graph.vcW,
                                graph.Volume,
                                graph.l2_reg,
                            ]
                        )
                        print(
                            "Iter %f, cost: %f(e %f, l %f, w %f - vol: %f + l2reg: %f)"
                            % (int(step), er, erE, ervcL, errcW, vol, l2reg)
                        )

                        if (
                            step % params.checkpoint_steps == 0
                            or step == params.steps - 1
                        ):
                            # Save the current embedding
                            save_ply(
                                Xopt,
                                TRIV,
                                "%s/ply/evals_%d_iter_%06d.ply"
                                % (out_path, nevals, step),
                            )

                        # Save the current eigenvalue sequence
                        np.savetxt(
                            "%s/txt/evals_%d_iter_%06d.txt" % (out_path, nevals, step),
                            evout,
                        )

                        # Save the training progress statistics
                        np.savetxt("%s/iterations.txt" % (out_path), iterations)

                        # Early stop
                        if erE < params.min_eval_loss:
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
                    print("iter %d: Perturbating vertices position" % step)
                    Xopt = (
                        Xopt
                        + (np.random.rand(np.shape(Xopt)[0], np.shape(Xopt)[1]) - 0.5)
                        * 1e-3
                    )
                    graph.global_step = step

                else:
                    Xopt = Xopt_t
                    graph.global_step += 1
