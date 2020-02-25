from scipy import sparse
#import matplotlib.pyplot as plt
import os
import sys
# import tensorflow as tf
import torch
from torch.autograd import Variable
import torch.optim as optim
import torchvision.utils as vutils
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import numpy as np
import scipy
# from IPython.display import clear_output

from shape_library import *

DEVICE = torch.device('cuda:3')

class OptimizationParams:
    def __init__(self, smoothing='displacement'):
        # self.checkpoint = 10
        self.checkpoint_steps = 100
        self.eval_steps = 100

        self.numsteps = 2000
        self.evals = [10, 20, 30]
        self.smoothing = smoothing

        if smoothing == 'displacement':
            self.curvature_reg = 2e3
            self.smoothness_reg = 2e3
        else:
            self.curvature_reg = 1e5
            self.smoothness_reg = 5e4

        self.volume_reg = 1e1
        self.l2_reg = 2e6

        self.min_eval_loss = 0.05

        # self.opt_step = 0.00025
        # self.decay_target = 0.01
        self.learning_rate = 0.005
        self.decay_target = 0.05
        self.beta1 = 0.9
        self.beta2 = 0.999


def tf_calc_lap(mesh, VERT):
    # [Xori,TRIV,n, m, Ik, Ih, Ik_k, Ih_k, Tpi, Txi, Tni, iM, Windices, Ael, Bary] = mesh
    meshTensor = []
    for i in range(len(mesh)):
        meshTensor.append(torch.as_tensor(mesh[i]).to(DEVICE))
    [Xori, TRIV, n, m, Ik, Ih, Ik_k, Ih_k, Tpi, Txi, Tni, iM, Windices, Ael, Bary] = meshTensor

    dtype = 'float32'
    if VERT.dtype == 'float64':
        dtype = 'float64'
    elif VERT.dtype == 'float16':
        dtype = 'float16'

    VERT = torch.as_tensor(VERT).to(DEVICE)

    # L2 = tf.expand_dims(tf.reduce_sum(tf.matmul(iM,VERT)**2,axis=1),axis=1)
    # L=tf.sqrt(L2)
    L2 = torch.sum(torch.mm(iM, VERT)**2, dim=1, keepdim=True)
    L = torch.sqrt(L2)

    def fAk(Ik, Ik_k):
        # Ikp = np.abs(Ik)
        Ikp = torch.abs(Ik)
        # Sk = tf.matmul(Ikp,L)/2
        Sk = torch.mm(Ikp, L) / 2
        SkL = Sk - L
        # Ak = Sk*(tf.matmul(Ik_k[:,:,0],Sk)-tf.matmul(Ik_k[:,:,0],L))\
        #                *(tf.matmul(Ik_k[:,:,0],Sk)-tf.matmul(Ik_k[:,:,1],L))\
        #                *(tf.matmul(Ik_k[:,:,0],Sk)-tf.matmul(Ik_k[:,:,2],L))
        Ak = Sk * (torch.mm(Ik_k[:, :, 0], Sk) - torch.mm(Ik_k[:, :, 0], L)) \
                * (torch.mm(Ik_k[:, :, 0], Sk) - torch.mm(Ik_k[:, :, 1], L)) \
                * (torch.mm(Ik_k[:, :, 0], Sk) - torch.mm(Ik_k[:, :, 2], L))
        # return tf.sqrt(tf.abs(Ak)+1e-20)
        return torch.sqrt(torch.abs(Ak) + 1e-20)

    Ak = fAk(Ik, Ik_k)
    Ah = fAk(Ih, Ih_k)

    #sparse representation of the Laplacian matrix
    # W = -tf.matmul(Ik,L2)/(8*Ak)-tf.matmul(Ih,L2)/(8*Ah)
    W = -torch.mm(Ik, L2) / (8 * Ak) - torch.mm(Ih, L2) / (8 * Ah)

    #compute indices to build the dense Laplacian matrix
    # Windtf = tf.SparseTensor(indices=Windices, values=-np.ones((m),dtype), dense_shape=[n*n, m])
    if dtype == 'float32':
        col_dtype = torch.float
    elif dtype == 'float64':
        col_dtype = torch.double
    elif dtype == 'float16':
        col_dtype = torch.half
    else:
        raise TypeError(f"Unrecognized dtype (got {dtype})")
    Windtf = torch.sparse.FloatTensor(
        torch.tensor(Windices.type(torch.long), dtype=torch.long, device=DEVICE).t(),
        torch.tensor(-np.ones((m), dtype), dtype=col_dtype, device=DEVICE),
        torch.Size([n * n, m]))
    # Wfull  = -tf.reshape(tf.sparse_tensor_dense_matmul(Windtf,W),(n,n))
    Wfull = -torch.reshape(torch.mm(Windtf, W), (n, n))
    # Wfull = (Wfull + tf.transpose(Wfull))
    Wfull = (Wfull + torch.t(Wfull))

    #actual Laplacian
    # Lx = Wfull-tf.diag(tf.reduce_sum(Wfull,axis=1))
    # S = (tf.matmul(Ael,Ak)+tf.matmul(Ael,Ah))/6
    Lx = Wfull - torch.diag(torch.sum(Wfull, dim=1))
    S = (torch.mm(Ael, Ak) + torch.mm(Ael, Ah)) / 6

    return Lx, S, L, Ak


def calc_evals(VERT, TRIV):
    mesh = prepare_mesh(VERT, TRIV, 'float64')
    # Lx, S, L, Ak = tf_calc_lap(mesh, mesh[0])
    Lx, S, _, _ = tf_calc_lap(mesh, mesh[0])
    # Si = tf.diag(tf.sqrt(1 / S[:, 0]))
    Si = torch.diag(torch.sqrt(1 / S[:, 0]))
    # Lap = tf.matmul(Si, tf.matmul(Lx, Si))
    Lap = torch.mm(Si, torch.mm(Lx, Si))
    # [evals, evecs]  = tf.self_adjoint_eig(Lap)
    evals, _ = torch.symeig(Lap)
    return evals


# def build_graph(mesh, evals, nevals, nfix, step=1.0, params=OptimizationParams()): #smoothing='absolute', numsteps=40000):
#     """Build the tensorflow graph

#     Input arguments:
#     - mesh: structure representing the triangulated mesh
#     - nevals: number of eigenvalues to optimize
#     - nfix: number of vertices to keep fixed (for partial shape optimization, 0 otherwise)
#     """
def initialize(mesh, step=1.0, params=OptimizationParams()):
    """Initialize the model."""
    # Namespace
    graph = lambda: None

    [Xori, TRIV, n, m, Ik, Ih, Ik_k, Ih_k, Tpi, Txi, Tni, iM, Windices, Ael, Bary] = mesh

    if Xori.dtype == 'float32':
        graph.dtype = torch.float
    elif Xori.dtype == 'float64':
        graph.dtype = torch.double
    elif Xori.dtype == 'float16':
        graph.dtype = torch.half
    else:
        raise TypeError(f"Unsupported dtype (got {dtype})")
    graph.np_dtype = Xori.dtype

    #model the shape deformation as a displacement vector field
    # dX = tf.Variable((0 * Xori).astype(dtype))
    graph.dX = torch.zeros(Xori.shape, dtype=graph.dtype, requires_grad=True, device=DEVICE)

    # scaleX = tf.Variable(1, dtype=dtype) #not used in shape alignment
    graph.scaleX = torch.tensor(1.0, dtype=graph.dtype, requires_grad=True, device=DEVICE)

    # graph.global_step = tf.Variable(step+1.0, name='global_step',trainable=False, dtype=dtype)
    # graph.global_step_val = tf.placeholder(dtype)
    # graph.set_global_step = tf.assign(graph.global_step, graph.global_step_val).op
    graph.global_step = torch.tensor(step + 1.0, dtype=torch.float32, requires_grad=False)

    graph.is_training = None

    graph.optim = optim.Adam([graph.dX], lr=params.learning_rate, betas=(params.beta1, params.beta2))

    return graph

def forward(graph, mesh, target_evals, nevals, nfix, step=1.0, params=OptimizationParams()):

    [Xori, TRIV, n, m, Ik, Ih, Ik_k, Ih_k, Tpi, Txi, Tni, iM, Windices, Ael, Bary] = mesh
    Bary = torch.as_tensor(Bary).to(DEVICE)
    TRIV = torch.as_tensor(TRIV, dtype=torch.long).to(DEVICE)


    # graph.input_X = tf.placeholder(shape=dX.shape, dtype=dtype)
    # graph.assign_X = tf.assign(dX, graph.input_X - Xori * scaleX).op

    # graph.X = Xori * scaleX + dX
    graph.X = torch.as_tensor(Xori).to(DEVICE) * graph.scaleX + graph.dX

    Lx, S, L, Ak = tf_calc_lap(mesh, graph.X)

    #Normalized Laplacian
    # Si = tf.diag(tf.sqrt(1/S[:,0]))
    # Lap =  tf.matmul(Si,tf.matmul(Lx,Si))
    Si = torch.diag(torch.sqrt(1 / S[:, 0]))
    Lap = torch.mm(Si, torch.mm(Lx, Si))

    def l2_loss(t):
        return 0.5 * torch.sum(t**2)

    #Spectral decomposition approach
    # [s_,v]  = tf.self_adjoint_eig( Lap )
    # graph.cost_evals_f1 = 1e2*tf.nn.l2_loss( (s_[0:nevals]-evals[0:nevals])* (1/np.asarray(range(1,nevals+1),dtype)) )/nevals # \
    s_, v = torch.symeig(Lap, eigenvectors=True)
    graph.cost_evals_f1 = 1e2 * l2_loss((s_[0:nevals] - target_evals[0:nevals]) * (1 / torch.as_tensor(np.asarray(range(1, nevals + 1), graph.np_dtype)).to(DEVICE))) / nevals

    #Approach avoiding spectral decomposition - NOT USED
    # [_,EigsOpt,lap] = tfeig(Lap)
    # v = tf.Variable(EigsOpt[:,0:nevals].astype(dtype) )
    # cost_evals_a = 1e3*tf.nn.l2_loss(tf.matmul(tf.transpose(v),v)-tf.eye(nevals,dtype=dtype))
    # cost_evals_b = 1e1*tf.nn.l2_loss( (tf.matmul(Lap,v) - tf.matmul(v,np.diag(evals[0:nevals]).astype(dtype))) )/nevals
    # graph.cost_evals_f2 = cost_evals_a + cost_evals_b

    # [Herman] Seems not used
    # meanA, varA = tf.nn.moments(Ak, axes=[0])
    # meanL, varL = tf.nn.moments(L, axes=[0])

    # Regularizers decay factor
    # cosine_decay = 0.5 * (1 + tf.cos(3.14 * tf.minimum(np.asarray(params.numsteps/2.0,dtype=dtype),graph.global_step) / (params.numsteps/2.0)))
    cosine_decay = 0.5 * (1 + np.cos(3.14 * np.minimum(params.numsteps / 2.0, graph.global_step) / (params.numsteps / 2.0)))
    graph.decay = (1 - params.decay_target) * cosine_decay + params.decay_target
    graph.decay = np.float(graph.decay)

    if params.smoothing == 'displacement':
        # graph.vcL = params.curvature_reg * graph.decay * tf.nn.l2_loss(tf.matmul(Bary.astype(dtype), dX)[nfix:, :])
        # graph.vcW = params.smoothness_reg * graph.decay * tf.nn.l2_loss(tf.matmul(Lx, dX)[nfix:, :])
        graph.vcL = params.curvature_reg * graph.decay * l2_loss(torch.mm(Bary.type(graph.dtype), graph.dX)[nfix:, :])
        graph.vcW = params.smoothness_reg * graph.decay * l2_loss(torch.mm(Lx, graph.dX)[nfix:, :])
    # elif params.smoothing == 'absolute':
    #     graph.vcL = params.curvature_reg * graph.decay * tf.nn.l2_loss(tf.matmul(Bary.astype(dtype), S * graph.X)[nfix:, :])
    #     # [Herman] Seems incorrect to have ** instead of *
    #     graph.vcW = params.smoothness_reg ** graph.decay * tf.nn.l2_loss(tf.matmul(Lx, graph.X)[nfix:,:])

    # Volume compuation
    # T1 =  tf.gather(graph.X, TRIV[:, 0])
    # T2 =  tf.gather(graph.X, TRIV[:, 1])
    # T3 =  tf.gather(graph.X, TRIV[:, 2])
    T1 = graph.X[TRIV[:, 0]]
    T2 = graph.X[TRIV[:, 1]]
    T3 = graph.X[TRIV[:, 2]]
    # XP = tf.cross(T2 - T1, T3 - T2)
    XP = torch.cross(T2 - T1, T3 - T2)
    T_C = (T1 + T2 + T3) / 3

    # graph.Volume = params.volume_reg * graph.decay * tf.reduce_sum(XP * T_C / 2) / 3
    graph.Volume = params.volume_reg * graph.decay * torch.sum(XP * T_C / 2) / 3

    #L2 regularizer on total displacement weighted by area elements
    # graph.l2_reg = params.l2_reg * tf.nn.l2_loss(S * dX)
    graph.l2_reg = params.l2_reg * l2_loss(S * graph.dX)

    graph.cost_spectral = graph.cost_evals_f1 + graph.vcW + graph.vcL - graph.Volume + graph.l2_reg

    # optimizer = tf.train.AdamOptimizer(params.opt_step)

    #gradient clipping
    # gvs = optimizer.compute_gradients(graph.cost_spectral)
    # capped_gvs = [(tf.clip_by_value(grad, -0.0001, 0.0001), var) for grad, var in gvs if grad!=None]
    # graph.train_op_spectral = optimizer.apply_gradients(capped_gvs, global_step=graph.global_step)

    if graph.is_training:
        graph.optim.zero_grad()
        graph.cost_spectral.backward()
        graph.dX.grad.data.clamp_(-0.0001, 0.0001)
        graph.optim.step()

    # [graph.s_, v]  = tf.self_adjoint_eig(Lap)
    graph.s_, _ = torch.symeig(Lap)

    return graph

def toNumpy(a):
    return [x.cpu().detach().numpy() for x in a]


def run_optimization(mesh, target_evals, out_path, params=OptimizationParams()):

    # gpu_options = tf.GPUOptions(allow_growth = True)
    # config = tf.ConfigProto(device_count={'CPU': 1, 'GPU': 1}, allow_soft_placement = False, gpu_options=gpu_options)
    # config.gpu_options.allow_growth=True

    try:
        os.makedirs(f'{out_path}/ply')
        os.makedirs(f'{out_path}/txt')
    except OSError:
        pass

    [VERT, TRIV, n, m, Ik, Ih, Ik_k, Ih_k, Tpi, Txi, Tni, iM, Windices, Ael, Bary] = mesh
    pstart = 0
    # Xori = VERT[:, 0:3]
    Xopt = VERT[:, 0:3]

    #Optimize the shape increasing the number of eigenvalue to be taken into account
    iterations = []
    for nevals in params.evals:

        step=0
        # tf.reset_default_graph()

        graph = initialize(mesh, step, params)

        # with tf.Session(config=config) as session:
            # tf.global_variables_initializer().run()

            # _ = session.run(graph.assign_X,feed_dict = {graph.input_X: Xopt})

        while(step < params.numsteps - 1):
            tic()

            for step in range(step, params.numsteps):
                try:
                    # Optimization step
                    graph.is_training = True
                    forward(graph, mesh, target_evals, nevals, pstart, step, params)
                    # _, er, ee, Xopt_t = session.run([graph.train_op_spectral, graph.cost_spectral, graph.cost_evals_f1, graph.X])
                    er, ee, Xopt_t = toNumpy([graph.cost_spectral, graph.cost_evals_f1, graph.X])
                    iterations.append((step, nevals, er, ee))

                    if step % params.eval_steps == 0 or step == params.numsteps - 1:
                        toc()
                        tic()
                        graph.is_training = False
                        # graph.cost_spectral, graph.cost_evals_f1, decay, evals
                        forward(graph, mesh, target_evals, nevals, pstart, step, params)
                        er, erE, ervcL, evout, errcW, vol, l2reg = toNumpy([
                            graph.cost_spectral, graph.cost_evals_f1, graph.vcL, graph.s_,
                             graph.vcW, graph.Volume, graph.l2_reg])
                        # er, erE, ervcL, Xopt2, evout, errcW, vol, l2reg = session.run(
                        #     [graph.cost_spectral, graph.cost_evals_f1, graph.vcL, graph.X, graph.s_, graph.vcW, graph.Volume, graph.l2_reg])
                        print('Iter %f, cost: %f(e %f, l %f, w %f - vol: %f + l2reg: %f)' % (int(step), er, erE,  ervcL, errcW, vol, l2reg))

                        if step % params.checkpoint_steps == 0 or step == params.numsteps - 1:
                            save_ply(Xopt, TRIV, '%s/ply/evals_%d_iter%d.ply' % (out_path, nevals, step))

                        np.savetxt('%s/txt/evals_%d_iter%d.txt' % (out_path, nevals, step), evout)
                        np.savetxt('%s/iterations.txt' % (out_path), iterations)

                        # Early stop
                        if erE < params.min_eval_loss:
                            step = params.numsteps
                            print('Minimum eighenvalues loss reached')
                            break

                except KeyboardInterrupt:
                    step = params.numsteps
                    break

                except:
                    print(sys.exc_info())
                    ee = float('nan')

                # If something went wrong with the spectral decomposition perturbate the last valid state and start over
                if ee != ee: # Check nan
                    print('iter %d: Perturbating vertices position' % step)
                    # tf.global_variables_initializer().run()
                    Xopt = Xopt + (np.random.rand(np.shape(Xopt)[0], np.shape(Xopt)[1]) - 0.5) * 1e-3
                    graph.global_step = step
                    # _ = session.run(graph.assign_X,feed_dict = {graph.input_X: Xopt})
                    # _ = session.run(graph.set_global_step, feed_dict = {graph.global_step_val: step})
                    # session.run(tf.variables_initializer(optimizer.variables()))
                else:
                    Xopt = Xopt_t
                    graph.global_step += 1
