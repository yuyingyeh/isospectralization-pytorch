"""Align spectrum of a shape to another."""

from scipy import sparse

import os
import sys

import torch
from torch.autograd import Variable
import torch.optim as optim
import torchvision.utils as vutils
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F

import numpy as np
import scipy
from shape_library import *

DEVICE = torch.device('cuda')

class OptimizationParams:
    def __init__(self, smoothing='displacement'):

        self.checkpoint = 100
        self.plot=False

        self.evals = [20]
        self.numsteps = 5000
        self.remesh_step = 500

        self.decay_target = 0.05
        self.learning_rate = 0.005
        self.min_eval_loss = 0.05

        self.flip_penalty_reg = 1e10
        self.inner_reg = 1e0
        self.bound_reg = 2e1

def tf_calc_lap(mesh,VERT):
    meshTensor = []
    for i in range(len(mesh)):
        meshTensor.append(torch.as_tensor(mesh[i]).to(DEVICE))
    [_,TRIV,n, m, Ik, Ih, Ik_k, Ih_k, Tpi, Txi, Tni, iM, Windices, Ael, Bary, bound_edges, ord_list] = meshTensor
    dtype='float32'
    if(VERT.dtype=='float64'):
        dtype='float64'
    if(VERT.dtype=='float16'):
        dtype='float16'

    VERT = torch.as_tensor(VERT).to(DEVICE)
    L2 = torch.unsqueeze(torch.sum(torch.mm(iM,VERT)**2,dim=1),dim=1)
    L=torch.sqrt(L2) # Dist of edges (m, 1)

    def  fAk(Ik,Ik_k): # Ik: 1 if (edg1, edg2) in same tri, -1 if same edge
        Ikp=torch.abs(Ik)
        Sk = torch.mm(Ikp,L)/2 # Perimeter of associated tri for each edge (m, )
        SkL = Sk-L
        Ak = Sk*(torch.mm(Ik_k[:,:,0],Sk)-torch.mm(Ik_k[:,:,0],L))\
                       *(torch.mm(Ik_k[:,:,0],Sk)-torch.mm(Ik_k[:,:,1],L))\
                       *(torch.mm(Ik_k[:,:,0],Sk)-torch.mm(Ik_k[:,:,2],L))
        return torch.sqrt(torch.abs(Ak)+1e-20)

    Ak = fAk(Ik,Ik_k) # (m, )
    Ah = fAk(Ih,Ih_k) # (m, )

    # Sparse representation of the Laplacian matrix
    W = -torch.mm(Ik,L2)/(8*Ak)-torch.mm(Ih,L2)/(8*Ah) # (m, )


    # Compute indices to build the dense Laplacian matrix
    if dtype == 'float32':
        Windtf = torch.sparse.FloatTensor(
            torch.tensor(Windices.type(torch.long), dtype=torch.long, device=DEVICE).t(), #
            torch.tensor(-np.ones((m), dtype), dtype=torch.float, device=DEVICE),
            torch.Size([n*n, m]))
    if dtype == 'float64':
        Windtf = torch.sparse.DoubleTensor(torch.cuda.LongTensor(Windices.type(torch.long), device=DEVICE).t(), \
                                      torch.cuda.DoubleTensor(-np.ones((m), dtype), device=DEVICE), \
                                      torch.Size([n*n, m]))
    if dtype == 'float16':
        Windtf = torch.sparse.HalfTensor(torch.cuda.LongTensor(Windices.type(torch.long), device=DEVICE).t(), \
                                      torch.cuda.HalfTensor(-np.ones((m), dtype), device=DEVICE), \
                                      torch.Size([n*n, m]))
    Wfull  = -torch.reshape(torch.mm(Windtf,W),(n,n))
    Wfull = (Wfull + torch.t(Wfull))

    # Compute the actual Laplacian
    Lx = Wfull-torch.diag(torch.sum(Wfull,dim=1)) # (n, n)
    S = (torch.mm(Ael,Ak)+torch.mm(Ael,Ah))/6 # (n, )

    return Lx,S,L,Ak


def calc_evals(VERT,TRIV):
    mesh = prepare_mesh(VERT,TRIV)
    Lx,S,_,_ = tf_calc_lap(mesh,mesh[0])
    Si = torch.diag(torch.sqrt(1/S[:,0]))
    Lap =  torch.mm(Si,torch.mm(Lx,Si))
    [evals, _] = torch.symeig( Lap )
    return evals

def initialize(mesh, step=1.0, params=OptimizationParams()):
    graph = lambda: None

    [Xori,TRIV,n, m, Ik, Ih, Ik_k, Ih_k, Tpi, Txi, Tni, iM, Windices, Ael, Bary, bound_edges, ord_list] = mesh
    dtype='float32'
    if(Xori.dtype=='float64'):
        dtype='float64'
    if(Xori.dtype=='float16'):
        dtype='float16'

    graph.dtype = dtype

    graph.dXb = torch.zeros(Xori.shape, requires_grad=True, device=DEVICE)
    graph.dXi = torch.zeros(Xori.shape, requires_grad=True, device=DEVICE)

    graph.global_step = torch.as_tensor(step+1.0, dtype=torch.float32)

    graph.optim_dXb = optim.Adam([graph.dXb], lr=params.learning_rate)
    graph.optim_dXi = optim.Adam([graph.dXi], lr=params.learning_rate)

    return graph

def forward(costType, mode, graph, mesh, target_evals, nevals, step=1.0, params=OptimizationParams()):

    [Xori,TRIV,n, m, Ik, Ih, Ik_k, Ih_k, Tpi, Txi, Tni, iM, Windices, Ael, Bary, bound_edges, ord_list] = mesh

    # Setup cosine decay
    cosine_decay = 0.5 * (1 + np.cos(3.14 * np.minimum(params.numsteps/2.0, graph.global_step) / (params.numsteps/2.0)))
    decay= (1 - params.decay_target) * cosine_decay + params.decay_target
    decay = np.float(decay)

    scaleX = 1 #not used in shape alignment

    # Model the shape deformation as a displacement vector field
    bound_vert = np.zeros((n,1),graph.dtype)
    bound_vert[ord_list] = 1

    def toGPUt(t):
        return torch.as_tensor(t).to(DEVICE)
    bound_vert = toGPUt(bound_vert)
    X=(toGPUt(Xori) + graph.dXb*bound_vert + graph.dXi*(1-bound_vert))*scaleX

    Lx,S,L,Ak = tf_calc_lap(mesh,X)

    # Normalized Laplacian
    Si = torch.diag(torch.sqrt(1/S[:,0]))
    Lap =  torch.mm(Si,torch.mm(Lx,Si))

    def l2_loss(t):
        return 0.5*torch.sum(t**2)

    # Spectral decomposition
    [evals,v]  = torch.symeig( Lap, eigenvectors=True )
    cost_evals = 1e1*l2_loss( (evals[0:nevals]-target_evals[0:nevals]) * (1/torch.as_tensor(np.asarray(range(1,nevals+1),graph.dtype) ).to(DEVICE)) ) # \

    # Triangle flip penalty
    Tpi = toGPUt(Tpi)
    Txi = toGPUt(Txi)
    Tni = toGPUt(Tni)
    tp = torch.mm(Tpi[:, :], X)
    tx = torch.mm(Txi[:, :], X)
    tn = torch.mm(Tni[:, :], X)
    Rot = toGPUt(np.asarray([[0, 1],[-1, 0]], graph.dtype))
    cp = torch.sum(torch.mm(tn,Rot) * (tx-tp), dim=1)
    cp = cp - 1e-4
    flip_cost =  params.flip_penalty_reg*l2_loss(cp - torch.abs(cp))

    # Inner points regularizer
    varA = torch.std(Ak, dim=[0])
    inner_reg_cost = params.inner_reg*(l2_loss(L) + l2_loss(varA))
    # Boundary points regularizer
    bound_reg_cost = params.bound_reg*decay* torch.sum(L[bound_edges[:,0],:])

    # Inner and outer points cost functions
    cost_bound = cost_evals + flip_cost + bound_reg_cost
    cost_inner = inner_reg_cost + flip_cost

    def clipped_grad_minimize(cost, variables):
        op = optim.Adam(variables, lr=params.learning_rate)
        op.zero_grad()
        cost.backward()
        variables[0].grad.data.clamp_(-0.0001, 0.0001)
        op.step()

    def toNumpy(a):
        o = []
        for ai in a:
            oi = ai.cpu().detach().numpy()
            o.append(oi)
        return o

    if mode == 'train':
        if costType == 'bound':
            graph.optim_dXb.zero_grad()
            cost_bound.backward()
            graph.dXb.grad.data.clamp_(-0.0001, 0.0001)
            graph.optim_dXb.step()
            outList = [cost_bound, cost_evals, X]
            return toNumpy(outList)
        if costType == 'inner':
            graph.optim_dXi.zero_grad()
            cost_inner.backward()
            graph.dXi.grad.data.clamp_(-0.0001, 0.0001)
            graph.optim_dXi.step()
            outList = [cost_inner, cost_evals, X]
            return toNumpy(outList)
    elif mode == 'eval':
        outList1 = [cost_bound, cost_evals, inner_reg_cost, bound_reg_cost]
        outList1 = toNumpy(outList1)
        outList2 = [cp, evals]
        outList2 = toNumpy(outList2)
        return outList1 + [decay] + outList2

def run_optimization(mesh, target_evals, out_path, params = OptimizationParams() ):

    # GPU options
    '''
    gpu_options = tf.GPUOptions(allow_growth = True)
    config = tf.ConfigProto(device_count={'CPU': 1, 'GPU': 1}, allow_soft_placement = False, gpu_options=gpu_options)
    config.gpu_options.allow_growth=True
    '''

    try:
        os.makedirs(f'{out_path}/ply')
        os.makedirs(f'{out_path}/txt')
    except OSError:
        pass

    [Xopt,TRIV,n, m, Ik, Ih, Ik_k, Ih_k, Tpi, Txi, Tni, iM, Windices, Ael, Bary, bound_edges, ord_list] = mesh
    save_ply(Xopt,TRIV,'%s/ply/initial.ply' % out_path)
    np.savetxt('%s/txt/target.txt' % out_path, target_evals.cpu().detach().numpy())

    iterations = []
    for nevals in params.evals:

        step=0
        while(step<params.numsteps-1):
            mesh = prepare_mesh(Xopt,TRIV)

            [Xori,TRIV,n, m, Ik, Ih, Ik_k, Ih_k, Tpi, Txi, Tni, iM, Windices, Ael, Bary, bound_edges, ord_list] = mesh

            # Initialize the model
            graph = initialize(mesh, step=step)
            tic()
            for step in range(step+1,params.numsteps):

                if((step)%params.remesh_step==0):
                    print("RECOMPUTING TRIANGULATION at step %d" % step)
                    break

                try:
                    feed_dict = {}

                    # Alternate optimization of inner and boundary vertices
                    if(int(step/10)%2==0):
                        er, ee, Xopt_t = forward('inner', 'train', graph, mesh, target_evals, nevals, step, params)
                    else:
                        er, ee, Xopt_t = forward('bound', 'train', graph, mesh, target_evals, nevals, step, params)

                    iterations.append((step, nevals, er, ee,int(step/10)%2))


                    if ( (step) % params.checkpoint == 0 or step==(params.numsteps-1) or step==1):
                        toc()
                        tic()

                        cost, cost_evals, cost_vcL, cost_vcW, decay, flip, evout = forward('bound', 'eval', graph, mesh, target_evals, nevals, step)

                        print('Iter %f, cost: %f(evals cost: %f (%f) (%f), smoothness weight: %f). Flip: %d' %
                            (step, cost, cost_evals, cost_vcL, cost_vcW, decay, np.sum(flip<0)))

                        save_ply(Xopt,TRIV,'%s/ply/evals_%d_iter_%06d.ply' % (out_path,nevals,step))
                        np.savetxt('%s/txt/evals_%d_iter_%06d.txt' % (out_path,nevals,step),evout)

                        np.savetxt('%s/iterations.txt' % (out_path),iterations)
                        #early stop
                        if(ee<params.min_eval_loss):
                            step=params.numsteps
                            print('Minimum eigenvalues loss reached')
                            break

                except KeyboardInterrupt:
                    step = params.numsteps
                    break
                except:
                    print(sys.exc_info())
                    ee=float('nan')

                # If something went wrong with the spectral decomposition perturbate the last valid state and start over
                if(ee!=ee):
                    print('iter %d. Perturbating initial condition' % step)
                    Xopt=Xopt+(np.random.rand(np.shape(Xopt)[0],np.shape(Xopt)[1])-0.5)*1e-3
                    graph.global_step = step
                else:
                    Xopt=Xopt_t
                    graph.global_step += 1
            if step < params.numsteps-1:
                [Xopt,TRIV] = resample(Xopt, TRIV)
