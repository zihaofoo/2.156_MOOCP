from __future__ import division
from lib2to3.pgen2 import driver
from math import sin, cos, acos, pi
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pymoo
from pymoo.util.display.display import Display
from tqdm.autonotebook import trange
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import multiprocessing as mp
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
import csv
from pymoo.indicators.hv import HV
import os
import pickle
from io import StringIO
import requests
import xml.etree.ElementTree as etree
from svgpath2mpl import parse_path
from scipy.spatial.distance import pdist,squareform
import torch
from scipy.spatial import KDTree, distance

def get_oriented(curve):
    """ Orient a curve
    Parameters
    ----------
    curve:  numpy array [n_points,2]
            The curve to be oriented
    Returns
    -------
    out:    numpy array [n_points,2]
            The oriented curve
    """

    ds = squareform(pdist(curve))
    pi,t = np.unravel_index(np.argmax(ds),ds.shape)

    d = curve[pi] - curve[t]

    if d[1] == 0:
        theta = 0
    else:
        d = d * np.sign(d[1])
        theta = -np.arctan(d[1]/d[0])

    rot = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    out = np.matmul(rot,curve.T).T
    out = out - np.min(out,0)

    rot2 = np.array([[np.cos(theta+np.pi),-np.sin(theta+np.pi)],[np.sin(theta+np.pi),np.cos(theta+np.pi)]])
    out2 = np.matmul(rot2,curve.T).T
    out2 = out2 - np.min(out2,0)

    if out2[:,0].sum() > out[:,0].sum():
        return out2

    return out

def get_oriented_both(curve):
    """ Orient a curve
    Parameters
    ----------
    curve:  numpy array [n_points,2]
            The curve to be oriented
    Returns
    -------
    out:    numpy array [n_points,2]
            The oriented curve
    """

    ds = squareform(pdist(curve))
    pi,t = np.unravel_index(np.argmax(ds),ds.shape)

    d = curve[pi] - curve[t]

    if d[1] == 0:
        theta = 0
    else:
        d = d * np.sign(d[1])
        theta = -np.arctan(d[1]/d[0])

    rot = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    out = np.matmul(rot,curve.T).T
    out = out - np.min(out,0)

    rot2 = np.array([[np.cos(theta+np.pi),-np.sin(theta+np.pi)],[np.sin(theta+np.pi),np.cos(theta+np.pi)]])
    out2 = np.matmul(rot2,curve.T).T
    out2 = out2 - np.min(out2,0)

    return out, out2

def get_oriented_angle(curve):
    """ Calculate the orientation of a curve
    Parameters
    ----------
    curve:  numpy array [n_points,2]
            The curve to be oriented
    Returns
    -------
    theta:  float
            The orientation of the curve
    """

    ds = squareform(pdist(curve))
    pi,t = np.unravel_index(np.argmax(ds),ds.shape)

    d = curve[pi] - curve[t]

    if d[1] == 0:
        theta = 0
    else:
        d = d * np.sign(d[1])
        theta = -np.arctan(d[1]/d[0])

    rot = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    out = np.matmul(rot,curve.T).T
    out = out - np.min(out,0)

    rot2 = np.array([[np.cos(theta+np.pi),-np.sin(theta+np.pi)],[np.sin(theta+np.pi),np.cos(theta+np.pi)]])
    out2 = np.matmul(rot2,curve.T).T
    out2 = out2 - np.min(out2,0)

    if out2[:,0].sum() > out[:,0].sum():
        return theta+np.pi

    return theta

def solve_rev_vectorized_batch_wds(As,x0s,node_types,thetas):
    """ Solve a vectorized batch of mechanisms
    Parameters
    ----------
    As:     torch tensor [batch_size,N,N]
            Adjacency/Conncetivity matrix describing the structure of the palanar linkage mechanism.
    x0s:    torch tensor [batch_size,N,2]
            The initial positions of the nodes
    node_types: torch tensor [batch_size,N,1]
            A list of the nodes that are grounded/fixed.
    thetas: torch tensor [n_thetas]
            The angles to solve the mechanism at.
    Returns
    -------
    x:     torch tensor [batch_size,N,n_thetas,2]
            The positions of the nodes at each angle.

    """
    Gs = torch.cdist(x0s,x0s)

    x = torch.zeros([x0s.shape[0],x0s.shape[1],thetas.shape[0],2]).to(As.device)

    x = x + torch.unsqueeze(node_types * x0s,2)

    m = x[:,0] + torch.tile(torch.unsqueeze(torch.transpose(torch.cat([torch.unsqueeze(torch.cos(thetas),0),torch.unsqueeze(torch.sin(thetas),0)],0),0,1),0),[x0s.shape[0],1,1]) * torch.unsqueeze(torch.unsqueeze(Gs[:,0,1],-1),-1)

    x[:,1,:,:] = m

    cos_list = []

    for k in range(3,x0s.shape[1]):
        inds = torch.argsort(As[:,k,0:k])[:,-2:].numpy()

        l_ijs = torch.linalg.norm(x[np.arange(x0s.shape[0]),inds[:,0]] - x[np.arange(x0s.shape[0]),inds[:,1]], dim=-1)

        gik = torch.unsqueeze(Gs[np.arange(x0s.shape[0]),inds[:,0],np.ones(shape=[x0s.shape[0]])*k],-1)
        gjk = torch.unsqueeze(Gs[np.arange(x0s.shape[0]),inds[:,1],np.ones(shape=[x0s.shape[0]])*k],-1)

        cosphis = (torch.square(l_ijs) + torch.square(gik) - torch.square(gjk))/(2 * l_ijs * gik)

        cosphis = torch.where(torch.tile(node_types[:,k],[1,thetas.shape[0]])==0.0,cosphis,torch.zeros_like(cosphis))

        cos_list.append(cosphis.unsqueeze(1))

        x0i1 = x0s[np.arange(x0s.shape[0]),inds[:,0],np.ones(shape=[x0s.shape[0]]).astype(np.int32)]
        x0i0 = x0s[np.arange(x0s.shape[0]),inds[:,0],np.zeros(shape=[x0s.shape[0]]).astype(np.int32)]

        x0j1 = x0s[np.arange(x0s.shape[0]),inds[:,1],np.ones(shape=[x0s.shape[0]]).astype(np.int32)]
        x0j0 = x0s[np.arange(x0s.shape[0]),inds[:,1],np.zeros(shape=[x0s.shape[0]]).astype(np.int32)]

        x0k1 = x0s[:,k,1]
        x0k0 = x0s[:,k,0]

        s = torch.unsqueeze(torch.sign((x0i1-x0k1)*(x0i0-x0j0) - (x0i1-x0j1)*(x0i0-x0k0)),-1)


        phi = s * torch.arccos(cosphis)

        a = torch.permute(torch.cat([torch.unsqueeze(torch.cos(phi),0),torch.unsqueeze(-torch.sin(phi),0)],0),dims=[1,2,0])
        b = torch.permute(torch.cat([torch.unsqueeze(torch.sin(phi),0),torch.unsqueeze(torch.cos(phi),0)],0),dims=[1,2,0])

        R = torch.einsum("ijk...->jki...", torch.cat([torch.unsqueeze(a,0),torch.unsqueeze(b,0)],0))

        xi = x[np.arange(x0s.shape[0]),inds[:,0]]
        xj = x[np.arange(x0s.shape[0]),inds[:,1]]

        scaled_ij = (xj-xi)/torch.unsqueeze(l_ijs,-1) * torch.unsqueeze(gik,-1)

        x_k = torch.squeeze(torch.matmul(R, torch.unsqueeze(scaled_ij,-1))) + xi
        x_k = torch.where(torch.tile(torch.unsqueeze(node_types[:,k],-1),[1,thetas.shape[0],2])==0.0,x_k,torch.zeros_like(x_k))
        x[:,k,:,:] += x_k
    return x, torch.cat(cos_list,dim=1)

def Transformation(curve):
    """Get transformation tuple for curve
    Parameters
    ----------
    curve:  numpy array [n_points,2]
            The curve to be transformed
    Returns
    -------
    tr:     tuple
            Transformation tuple (theta,scale,translation)
    """

    theta = -get_oriented_angle(curve)
    or_c = get_oriented(curve)
    scale = or_c.max()
    # scale = total_length(np.array([or_c]))[0]
    translation = curve.min(0)
    # translation = curve.mean(0)

    return theta,scale,translation

def apply_transformation(tr,curve):
    """ Apply transformation to curve
    Parameters
    ----------
    tr:     tuple
            Transformation tuple (theta,scale,translation)
    curve:  numpy array [n_points,2]
            The curve to be transformed
    Returns
    -------
    out:    numpy array [n_points,2]
            The transformed curve
    """

    theta,scale,translation = tr

    rot = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
    out = np.matmul(rot,curve.T).T
    out = scale * out
    out = out - np.min(out,0)
    # out = out - out.mean(0)
    out = out + translation

    return out

def batch_chamfer_distance(c1,c2):
    """Chamfer distance between two point clouds
    Parameters
    ----------
    c1: torch tensor [n_points_x, n_dims]
        first point cloud
    c2: torch tensor [n_points_y, n_dims]
        second point cloud
    Returns
    -------
    chamfer_dist: float
        computed bidirectional Chamfer distance:
            sum_{x_i \in x}{\min_{y_j \in y}{||x_i-y_j||**2}} + sum_{y_j \in y}{\min_{x_i \in x}{||x_i-y_j||**2}}
    """
    d = torch.cdist(c1,c2)
    d1 = d.min(dim=1)[0].mean(dim=1)
    d2 = d.min(dim=2)[0].mean(dim=1)
    chamfer_dist = d1+d2
    return chamfer_dist

def get_mat(x0, C):
    """Get total link length of mechanism
    Parameters
    ----------
    x0:    torch tensor [N,2]
        The initial positions of the
    C:     torch tensor [N,N]
              Adjacency/Conncetivity matrix describing the structure of the palanar linkage mechanism.

    Returns
    -------
    total:     float
              total length of all links in mechanism.
    """

    N = C.shape[0]
    G = torch.zeros_like(C, dtype=float)
    for i in range(N):
        for j in range(i):
            if C[i,j]:
                G[i,j] = G[j,i] = torch.norm(x0[i]-x0[j])
            else:
                G[i,j] = G[j,i] = 0
    total = torch.sum(G)
    return total

def sort_mech(A, x0, motor,fixed_nodes):
    if motor[1] in fixed_nodes:
        return None
    nt = np.zeros([A.shape[0],1])
    nt[fixed_nodes] = 1 


    if motor[-1] in fixed_nodes:
        t = motor[0]
        motor[0] = motor[1]
        motor[1] = t

    motor_first_order = np.arange(A.shape[0])
    motor_first_order = motor_first_order[motor_first_order!=motor[0]]
    motor_first_order = motor_first_order[motor_first_order!=motor[1]]
    motor_first_order = np.concatenate([motor,motor_first_order])

 

    A = A[motor_first_order,:][:,motor_first_order]
    x0 = x0[motor_first_order]
    nt = nt[motor_first_order]
    p,check = find_path(A,[0,1],np.where(nt)[0])

 

    if check:
        fixed_nodes = np.where(nt[1:])[0]+1
        sorted_order = np.concatenate([[0,1],fixed_nodes,p[:,0]])
        A = A[sorted_order,:][:,sorted_order]
        x0 = x0[sorted_order]
        nt = nt[sorted_order]
        return A,x0,np.where(nt)[0], motor_first_order[sorted_order]
    else:
        return None


def functions_and_gradients(C,x0,fixed_nodes,target_pc, motor, idx=None,device='cpu',timesteps=2000):
    """Simulate, then return functions and gradients for the mechanism
    Parameters
    ----------
    C:     numpy array [N,N]
            Adjacency/Conncetivity matrix describing the structure of the palanar linkage mechanism.
    x0:    numpy array [N,2]
            The positions of the nodes
    fixed_nodes: numpy array [n_fixed_nodes]
            A list of the nodes that are grounded/fixed.
    target_pc: numpy array [n_points,2]
            The target point cloud that the mechanism should match.

    Returns
    -------
    validity: bool
            bool indicating whether the mechanism is valid
    CD_fn:     function
            function that returns the Chamfer distance between the target point cloud and the mechanism.
    mat_fn:     function
            function that returns the total length of the mechanism.
    CD_grad:     function
            function that returns the gradient of the Chamfer distance between the target point cloud and the mechanism.
    mat_grad:     function
            function that returns the gradient of the total length of the mechanism.
    matched_curve: numpy array [n_points,2]
            The curve that the mechanism matches to the target point cloud.
    
    """
    # target_pc = target_pc/500
    target_pc = get_oriented(target_pc)
    scale = target_pc.max()

    if idx is None:
        idx = C.shape[0]-1

    xc = x0.copy()
    res = sort_mech(C, x0, motor,fixed_nodes)
    if res: 
        C, x0, fixed_nodes, sorted_order = res
        
        inverse_order = np.argsort(sorted_order)
    else:
        return False, None, None, None, None, None

    A = torch.Tensor(np.expand_dims(C,0)).to(device)
    X = torch.Tensor(np.expand_dims(x0,0)).to(device)
    node_types = np.zeros([1,C.shape[0],1])
    node_types[0,fixed_nodes,:] = 1
    node_types = torch.Tensor(node_types).to(device)
    thetas = torch.Tensor(np.linspace(0,np.pi*2,timesteps+1)[0:timesteps]).to(device)

    x_sol,cos = solve_rev_vectorized_batch_wds(A,X,node_types,thetas)
    best_match = x_sol[0,idx].detach().cpu().numpy()
    best_tree = KDTree(best_match)

    tr = Transformation(best_match)
    matched_curve = apply_transformation(tr,target_pc/scale)
    matched_curve_180 = apply_transformation((tr[0]+np.pi,tr[1],tr[2]),target_pc/scale)

    m_tree = KDTree(matched_curve)
    m180_tree = KDTree(matched_curve_180)

    multiplier = scale/tr[1]

    cd_tr = np.array([best_tree.query(matched_curve)[0].mean()+m_tree.query(best_match)[0].mean(),best_tree.query(matched_curve_180)[0].mean()+m180_tree.query(best_match)[0].mean()])

    if cd_tr[1]<cd_tr[0]:
        tr = (tr[0]+np.pi,tr[1],tr[2])
        matched_curve = matched_curve_180

    target = torch.Tensor(matched_curve.astype(np.float32)).to(device).unsqueeze(0)

    def CD_fn(x0_inp):
        x0_in = np.reshape(x0_inp,x0.shape)[sorted_order]/multiplier

        current_x0 = torch.nn.Parameter(torch.Tensor(np.expand_dims(x0_in,0)),requires_grad = True).to(device)
        with torch.no_grad():


            sol,cos = solve_rev_vectorized_batch_wds(A,current_x0,node_types,thetas)

            ds = torch.square(torch.log(torch.pow(1 - torch.square(cos[0]),0.25))).mean()

            current_sol = sol[0,idx]
            CD = batch_chamfer_distance(current_sol.unsqueeze(0)/tr[1],target/tr[1])[0]

        # if torch.isnan(current_sol).sum()>0 or CD<=thCD:
        #     break

        # final_loss = CD + pen*ds

        if torch.isnan(CD):
            return 1e6

        return CD.detach().cpu().numpy()*multiplier

    def mat_fn(x0_inp):
        x0_in = np.reshape(x0_inp,x0.shape)[sorted_order]
        x0_in = torch.from_numpy(x0_in)
        material = get_mat(x0_in, A[0])
        return material.detach().cpu().numpy()

    def CD_grad(x0_inp):
        x0_in = np.reshape(x0_inp,x0.shape)[sorted_order]/multiplier
        current_x0 = torch.nn.Parameter(torch.Tensor(np.expand_dims(x0_in,0)),requires_grad = True).to(device)

        sol,cos = solve_rev_vectorized_batch_wds(A,current_x0,node_types,thetas)

        ds = torch.square(torch.log(torch.pow(1 - torch.square(cos[0]),0.25))).mean()

        current_sol = sol[0,idx]
        CD = batch_chamfer_distance(current_sol.unsqueeze(0)/tr[1],target/tr[1])[0]

        CD.backward()

        if torch.isnan(CD):
            return np.zeros_like(x0_inp)
        return current_x0.grad.detach().cpu().numpy()[:,inverse_order].reshape(x0_inp.shape)*multiplier

    def mat_grad(x0_inp):
        x0_in = np.reshape(x0_inp,x0.shape)[sorted_order]
        current_x0 = torch.nn.Parameter(torch.Tensor(x0_in),requires_grad = True).to(device)
        material = get_mat(current_x0, A[0])
        material.backward()
        if torch.isnan(material):
            return np.zeros_like(x0_inp)

        return current_x0.grad.detach().cpu().numpy()[:,inverse_order].reshape(x0_inp.shape)

    return True, CD_fn, mat_fn, CD_grad, mat_grad, matched_curve*multiplier


def evaluate_mechanism(C,x0,fixed_nodes,target_pc, motor, idx=None,device='cpu',timesteps=2000):
    target_pc = get_oriented(target_pc)

    if idx is None:
        idx = C.shape[0]-1

    xc = x0.copy()
    res = sort_mech(C, x0, motor,fixed_nodes)
    if res: 
        C, x0, fixed_nodes, sorted_order = res
        
        inverse_order = np.argsort(sorted_order)
    else:
        return False, None, None, None

    A = torch.Tensor(np.expand_dims(C,0)).to(device)
    X = torch.Tensor(np.expand_dims(x0,0)).to(device)
    node_types = np.zeros([1,C.shape[0],1])
    node_types[0,fixed_nodes,:] = 1
    node_types = torch.Tensor(node_types).to(device)
    thetas = torch.Tensor(np.linspace(0,np.pi*2,timesteps+1)[0:timesteps]).to(device)

    current_x0 = torch.Tensor(np.expand_dims(x0,0)).to(device)
    sol,cos = solve_rev_vectorized_batch_wds(A,current_x0,node_types,thetas)
    if torch.isnan(sol).any():
        return False, None, None, None
    else:
        sol = sol.detach().numpy()[0,idx,:,:]
        sol1, sol2 = get_oriented_both(sol)
        CD1 = batch_chamfer_distance(torch.tensor(sol1, dtype = float).unsqueeze(0),torch.tensor(target_pc, dtype = float).unsqueeze(0))[0]
        CD2 = batch_chamfer_distance(torch.tensor(sol2, dtype = float).unsqueeze(0),torch.tensor(target_pc, dtype = float).unsqueeze(0))[0]
        if CD1<CD2:
            CD = CD1
            sol = sol1
        else:
            CD = CD2
            sol = sol2
        material = get_mat(torch.Tensor(x0), A[0])
        return True, CD, material, sol

def draw_mechanism(C,x0,fixed_nodes,motor):
    """Draw and simulate 2D planar mechanism and plot the traces for all nodes
    Parameters
    ----------
    C:     numpy array [N,N]
            Adjacency/Conncetivity matrix describing the structure of the palanar linkage mechanism.
    x0:    numpy array [N,2]
            The initial positions of the 
    fixed_nodes: numpy array [n_fixed_nodes]
            A list of the nodes that are grounded/fixed.
    motor: numpy array [2]
            Start and end nodes of the driven linkage (Note: this linkage must be connected to ground on one end). 
    """
    
    C,x0,fixed_nodes,motor = np.array(C),np.array(x0),np.array(fixed_nodes),np.array(motor)
    
    x = x0
    fig = plt.figure(figsize=(12,12))
    N = C.shape[0]
    for i in range(N):
        if i in fixed_nodes:
            plt.scatter(x[i,0],x[i,1],color="Black",s=100,zorder=10)
        else:
            plt.scatter(x[i,0],x[i,1],color="Grey",s=100,zorder=10)
        
        for j in range(0,i):
            if C[i,j]:
                plt.plot([x[i,0],x[j,0]],[x[i,1],x[j,1]],color="Black",linewidth=3.5)
    solver = mechanism_solver()
    xo,f1,f2 = solver.solve_rev(192,x,C,motor,fixed_nodes,False)
    if not f1 and not f2:
        for i in range(C.shape[0]):
            if not i in fixed_nodes:
                plt.plot(xo[:,i,0],xo[:,i,1])
    else:
        plt.text(0.5, 0.5, 'Locking Or Under Defined', color='red', horizontalalignment='center', verticalalignment='center')
        
    plt.axis('equal')
    plt.xlim([-0.1,1.1])
    plt.ylim([-0.1,1.1])
    
def draw_mechanism_on_ax(C,x,fixed_nodes,motor,ax):
    """Draw and simulate 2D planar mechanism and plot the traces for all nodes
    Parameters
    ----------
    C:     numpy array [N,N]
            Adjacency/Conncetivity matrix describing the structure of the palanar linkage mechanism.
    x0:    numpy array [N,2]
            The initial positions of the 
    fixed_nodes: numpy array [n_fixed_nodes]
            A list of the nodes that are grounded/fixed.
    motor: numpy array [2]
            Start and end nodes of the driven linkage (Note: this linkage must be connected to ground on one end). 
    ax:    matplotlib axis object
            in case part of a subplot use this and pass the axis object.
    """
    N = C.shape[0]
    for i in range(N):
        if i in fixed_nodes:
            ax.scatter(x[i,0],x[i,1],color="Black",s=100,zorder=10)
        else:
            ax.scatter(x[i,0],x[i,1],color="Grey",s=100,zorder=10)
        
        for j in range(i+1,N):
            if C[i,j]:
                ax.plot([x[i,0],x[j,0]],[x[i,1],x[j,1]],color="Black",linewidth=3.5)
    solver = mechanism_solver()
    xo,f1,f2 = solver.solve_rev(192,x,C,motor,fixed_nodes,False)
    if not f1 and not f2:
        for i in range(C.shape[0]):
            if not i in fixed_nodes:
                ax.plot(xo[:,i,0],xo[:,i,1])
    else:
        ax.text(0.5, 0.5, 'Locking Or Under Defined', color='red', horizontalalignment='center', verticalalignment='center')
        
    ax.axis('equal')
    ax.set_xlim([-0.1,1.1])
    ax.set_ylim([-0.1,1.1])





















def line_segment(start, end):
    """Bresenham's Line Algorithm
    Parameters
    ----------
    start: numpy array or list with size [2]
            The start of the line segment being rasterized. 
    end:   numpy array or list with size [2]
            The end of the line segment being rasterized.
    
    Returns
    -------
    points: numpy array [n_point,2].
        computed coordinates of the pixels needed to display the line segment inputted.
    """
    # Setup initial conditions
    x1, y1 = start[0], start[1]
    x2, y2 = end[0], end[1]
    dx = x2 - x1
    dy = y2 - y1
 
    # Determine how steep the line is
    is_steep = np.abs(dy) > np.abs(dx)
 
    # Rotate line
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2
 
    # Swap start and end points if necessary and store swap state
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True
 
    # Recalculate differentials
    dx = x2 - x1
    dy = y2 - y1
 
    # Calculate error
    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1
 
    # Iterate over bounding box generating points between start and end
    y = y1
    points = []
    for x in range(x1, x2 + 1):
        coord = (y, x) if is_steep else (x, y)
        points.append(coord)
        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx
 
    # Reverse the list if the coordinates were swapped
    if swapped:
        points.reverse()
    return points

def rasterized_curve_coords(curve, res):
    """Rasterize curves into point clouds standardized into a grid of specified resolution
    Parameters
    ----------
    curve:  numpy array [n_point,2].
                Coordinates of a curve that fits within a 1x1 box (i.e., coordinates normalized to the range [0,1])
    res:    int
                Resolution of the grid the curve is being rasterized to (e.g. res=500 means rasterize to a grid of 500x500)
    Returns
    -------
    points: numpy array [n,2].
                computed coordinates of the pixels needed to display the curve within the specified resolution.
    """
    c = np.minimum(res-1,np.floor(curve*res)).astype(np.int32)
    ps = []
    for i in range(curve.shape[0]-1):
        ps += line_segment(c[i],c[i+1])

    out = np.array(list(set(ps)))
    
    return out



class mechanism_solver():
    def __init__(self):
        """Instance of a solver which outputs the path taken by nodes of a mechanims
        """
        pass
    
    def find_neighbors(self, index, C):
        """Find neighbors of a node (i.e., nodes that are connected to the node of interest)
        Parameters
        ----------
        index: int
                Index of the node neighbours are needed for.
        C:     numpy array [N,N]
                 Adjacency/Conncetivity matrix describing the structure of the palanar linkage mechanism.
        Returns
        -------
        neigbours: numpy array [n_neighbours].
                 List of the neigbours for the input node.
        """
        return np.where(C[index])[0]
    
    def find_unvisited_neighbors(self, index, C, visited_list):
        """Find neighbors of a node that are not visited (i.e., neighbours that are not solved yet)
        Parameters
        ----------
        index: int
                Index of the node neighbours are needed for.
        C:     numpy array [N,N]
                 Adjacency/Conncetivity matrix describing the structure of the palanar linkage mechanism.
        visited_list: numpy array [N]
                 A list of booleans indicated wether the node with the index of the element in this array has been visited.
        Returns
        -------
        visted neigbours: numpy array [n_neighbours].
                 List of the visited neigbours for the input node.
        """
        return np.where(np.logical_and(C[index],np.logical_not(visited_list)))[0]
    
    def get_G(self, x0, C):
        """Get distance matrix for a mechanism
        Parameters
        ----------
        x0:    numpy array [N,2]
            The initial positions of the 
        C:     numpy array [N,N]
                 Adjacency/Conncetivity matrix describing the structure of the palanar linkage mechanism.
        
        Returns
        -------
        G:     numpy array [N,N].
                 distance between nodes that are connected and infinity if nodes are not connected constructed similar to C.
        """
        N = C.shape[0]
        G = np.zeros_like(C, dtype=float)
        for i in range(N):
            for j in range(i):
                if C[i,j]:
                    G[i,j] = G[j,i] = np.linalg.norm(x0[i]-x0[j])
                else:
                    G[i,j] = G[j,i] = np.inf
                    
        return G
    
    def get_path(self, x0, C, G, motor, fixed_nodes=[0, 1], show_msg=False):
        """Get Path to solution
        Parameters
        ----------
        x0:    numpy array [N,2]
                The initial positions of the 
        C:     numpy array [N,N]
                Adjacency/Conncetivity matrix describing the structure of the palanar linkage mechanism.
        G:     numpy array [N,N].
                Distance matrix of the palanar linkage mechanism.
        motor: numpy array [2]
                Start and end nodes of the driven linkage (Note: this linkage must be connected to ground on one end). 
        fixed_nodes: numpy array [n_fixed_nodes]
                A list of the nodes that are grounded/fixed.
        show_msg: boolean
                Will show messages if True. Default: False.
        Returns
        -------
        Path:  numpy array [n_steps].
                Order of solving nodes.
        op:
               numpy array [n_steps,2]
                Nodes used to solve for specific step.
        """
        
        C,x0,fixed_nodes,motor,G = np.array(C),np.array(x0),np.array(fixed_nodes),np.array(motor),np.array(G)
        theta = 0.0
        
        path = []
        op = []
        
        N = C.shape[0]
        x = np.zeros((N, 2))
        visited_list = np.zeros(N, dtype=bool)
        active_list = []
        
        visited_list[fixed_nodes] = True
        
        if (C != C.T).any():
            if show_msg:
                print('Asymmetric Connectivity Matrix.')
            return None, -1
        
        if motor[1] in fixed_nodes.tolist():
            t = motor[0]
            motor[0] = motor[1]
            motor[1] = t
        elif not motor[0] in fixed_nodes.tolist():
            if show_msg:
                print('Incorrect motor linkage.')
            return None, -2
        if motor[0] not in fixed_nodes.tolist() and motor[1] not in fixed_nodes.tolist():
            if show_msg:
                print('Incorrect motor linkage.')
            return None, -2
        for item in fixed_nodes:
            if item != motor[0]:
                if C[motor[1], item]:
                    print([motor[1], item])
                    if show_msg:
                        print('Driven node linked to ground.')
                    return None, -2

        if (~C.any(axis=0)).any():
            if show_msg:
                print('Disconnected node.')
            return None, 0


        
        visited_list[motor[1]] = True
        
        x[fixed_nodes] = x0[fixed_nodes]
        
        dx = x0[motor[1],0] - x0[motor[0],0]
        dy = x0[motor[1],1] - x0[motor[0],1]
        
        theta_0 = np.math.atan2(dy,dx)
        theta = theta_0 + theta
        
        x[motor[1], 0] = x[motor[0],0] + G[motor[0],motor[1]] * np.cos(theta)
        x[motor[1], 1] = x[motor[0],1] + G[motor[0],motor[1]] * np.sin(theta)
        
        neighbors = self.find_neighbors(motor[1], C)
        vn = neighbors[visited_list[neighbors]]
        if vn.shape[0]>2:
            if show_msg:
                print('Redundant or overdefined system.')
            return None, -2

        hanging_nodes = np.where(C.sum(0)==0)[0]
        for hn in hanging_nodes:
            if not hn in fixed_nodes.tolist():
                if show_msg:
                    print('DOF larger than 1.')
                return None, 0
        
        for i in np.where(visited_list)[0]:            
            active_list += list(self.find_unvisited_neighbors(i, C, visited_list))
            
        active_list = list(set(active_list))
        
        counter = 0
        
        while len(active_list)>0:
            k = active_list.pop(0)
            neighbors = self.find_neighbors(k, C)
            vn = neighbors[visited_list[neighbors]]
            if vn.shape[0]>1:
                if vn.shape[0]>2:
                    if show_msg:
                        print('Redundant or overdefined system.')
                    return None, -2
                i = vn[0]
                j = vn[1]
                l_ij = np.linalg.norm(x[j]-x[i])
                s = np.sign((x0[i,1]-x0[k,1])*(x0[i,0]-x0[j,0]) - (x0[i,1]-x0[j,1])*(x0[i,0]-x0[k,0]))
                cosphi = (l_ij**2+G[i,k]**2-G[j,k]**2)/(2*l_ij*G[i,k])
                if cosphi >= -1.0 and cosphi <= 1.0:
                    phi = s * acos(cosphi)
                    R = np.array([[cos(phi), -sin(phi)],
                                  [sin(phi), cos(phi)]])
                    scaled_ij = (x[j]-x[i])/l_ij * G[i,k]
                    x[k] = np.matmul(R, scaled_ij.reshape(2,1)).flatten() + x[i]
                    path.append(k)
                    op.append([i,j,s])
                else:
                    if show_msg:
                        print('Locking or degenerate linkage!')
                    return None, -1
                
                visited_list[k] = True
                active_list += list(self.find_unvisited_neighbors(k, C, visited_list))
                active_list = list(set(active_list))
                counter = 0
            else:
                counter += 1
                active_list.append(k)
            
            if counter > len(active_list):
                if show_msg:
                    print('DOF larger than 1.')
                return None, 0
        return path, op
    
    def position_from_path(self, path, op, theta, x0, C, G, motor, fixed_nodes=[0, 1], show_msg=False):
        """Get position at specific angle to solution
        Parameters
        ----------
        Path:  numpy array [n_steps].
                Order of solving nodes.
        op:
               numpy array [n_steps,2]
                Nodes used to solve for specific step.
        theta: float
                Deviation from initial position angle at the motor.
        x0:    numpy array [N,2]
                The initial positions of the 
        C:     numpy array [N,N]
                Adjacency/Conncetivity matrix describing the structure of the palanar linkage mechanism.
        G:     numpy array [N,N].
                Distance matrix of the palanar linkage mechanism.
        motor: numpy array [2]
                Start and end nodes of the driven linkage (Note: this linkage must be connected to ground on one end). 
        fixed_nodes: numpy array [n_fixed_nodes]
                A list of the nodes that are grounded/fixed.
        show_msg: boolean
                Will show messages if True. Default: False.
        Returns
        -------
        x:     numpy array [N,2].
                Positions of nodes at given angle.
        """
        
        C,x0,fixed_nodes,motor,G = np.array(C),np.array(x0),np.array(fixed_nodes),np.array(motor),np.array(G)
        N = C.shape[0]
        x = np.zeros((N, 2))
        visited_list = np.zeros(N, dtype=bool)
        active_list = []
        
        visited_list[fixed_nodes] = True
        
        
        if motor[1] in fixed_nodes.tolist():
            t = motor[0]
            motor[0] = motor[1]
            motor[1] = t
        elif not motor[0] in fixed_nodes.tolist():
            if show_msg:
                print('Incorrect motor linkage.')
            return None, -2
        
        for item in fixed_nodes:
            if item != motor[0]:
                if C[motor[1], item]:
                    print([motor[1], item])
                    if show_msg:
                        print('Incorrect motor linkage.')
                    return None, -2

        visited_list[motor[1]] = True
        
        x[fixed_nodes] = x0[fixed_nodes]
        
        dx = x0[motor[1],0] - x0[motor[0],0]
        dy = x0[motor[1],1] - x0[motor[0],1]
        
        theta_0 = np.math.atan2(dy,dx)
        theta = theta_0 + theta
        
        x[motor[1], 0] = x[motor[0],0] + G[motor[0],motor[1]] * np.cos(theta)
        x[motor[1], 1] = x[motor[0],1] + G[motor[0],motor[1]] * np.sin(theta)
        
        
        for l,step in enumerate(path):
            i = op[l][0]
            j = op[l][1]
            k = step
            
            l_ij = np.linalg.norm(x[j]-x[i])
            cosphi = (l_ij**2+G[i,k]**2-G[j,k]**2)/(2*l_ij*G[i,k])
            if cosphi >= -1.0 and cosphi <= 1.0:
                phi = op[l][2] * acos(cosphi)
                R = np.array([[cos(phi), -sin(phi)],
                              [sin(phi), cos(phi)]])
                scaled_ij = (x[j]-x[i])/l_ij * G[i,k]
                x[k] = np.matmul(R, scaled_ij.reshape(2,1)).flatten() + x[i]
            else:
                if show_msg:
                    print('Locking or degenerate linkage!')
                return np.abs(cosphi)
        return x
    
    def solve_rev(self, n_steps, x0, C, motor, fixed_nodes=[0, 1], show_msg=False):
        """Get path traced by mechanism nodes for one revolution of the motor for a number of descrete steps.
        Parameters
        ----------
        n_steps:    int
                Number of descrete steps to solve the mechanism at for the entire revolution of motor.
        x0:    numpy array [N,2]
                The initial positions of the 
        C:     numpy array [N,N]
                Adjacency/Conncetivity matrix describing the structure of the palanar linkage mechanism.
        G:     numpy array [N,N].
                Distance matrix of the palanar linkage mechanism.
        motor: numpy array [2]
                Start and end nodes of the driven linkage (Note: this linkage must be connected to ground on one end). 
        fixed_nodes: numpy array [n_fixed_nodes]
                A list of the nodes that are grounded/fixed.
        show_msg: boolean
                Will show messages if True. Default: False.
        Returns
        -------
        x:     numpy array [n_steps,N,2].
                Positions of nodes at each angle steps for the entire revolution.
        """
        
        
        C,x0,fixed_nodes,motor = np.array(C),np.array(x0),np.array(fixed_nodes),np.array(motor)
        
        for j in fixed_nodes:
            if j!=motor[1]:
                if C[j, motor[0]]==1 or C[motor[0], j]==1:
                    return np.zeros([n_steps,C.shape[0],2]), True, False
        if motor[0] in fixed_nodes:
            return np.zeros([n_steps,C.shape[0],2]), True, False


        pop = find_path(C,motor,fixed_nodes)
        

        if not pop[1]:
            return np.zeros([n_steps,C.shape[0],2]), False, True
        
        out = solve_rev_vectorized(pop[0],x0,get_G(x0),motor,fixed_nodes,np.linspace(0, 2*np.pi, n_steps))[0]
        out = np.transpose(out,axes=[1,0,2])

        if np.isnan(out).sum() > 0:
            return np.zeros([n_steps,C.shape[0],2]), True, False
        
        return np.array(out), False, False
    
    def check(self, n_steps, x0, C, motor, fixed_nodes=[0, 1], show_msg=False):
        """check validity of mechanism (used for random generator).
        Parameters
        ----------
        n_steps:    int
                Number of descrete steps to solve the mechanism at for the entire revolution of motor.
        x0:    numpy array [N,2]
                The initial positions of the 
        C:     numpy array [N,N]
                Adjacency/Conncetivity matrix describing the structure of the palanar linkage mechanism.
        G:     numpy array [N,N].
                Distance matrix of the palanar linkage mechanism.
        motor: numpy array [2]
                Start and end nodes of the driven linkage (Note: this linkage must be connected to ground on one end). 
        fixed_nodes: numpy array [n_fixed_nodes]
                A list of the nodes that are grounded/fixed.
        show_msg: boolean
                Will show messages if True. Default: False.
        Returns
        -------
        x:     boolean.
                whether mechanism is valid or not.
        """
        
        C,x0,fixed_nodes,motor = np.array(C),np.array(x0),np.array(fixed_nodes),np.array(motor)
        G = self.get_G(x0,C)
        
        pop = self.get_path(x0, C, G, motor, fixed_nodes, show_msg)

        if not pop[0]:
            if pop[1] == 0:
                return False
            elif pop[1] == -1:
                return False
            elif pop[1] == -2:
                return False
            
        func = lambda t: self.position_from_path(pop[0],pop[1],t, x0, C, G, motor, fixed_nodes, show_msg)
        
        ts = np.linspace(0, 2*np.pi, n_steps)
        
        out = []
        
        for t in ts:
            x_temp = func(t)
            if np.array(x_temp).size == 1:
                if x_temp:
                    return False
                else:
                    return False
            else:
                out.append(x_temp)
        
        if np.max(out)>1.0 or np.min(out)<0.0 :
            return False
        else:
            return True
    
    def material(self,x0, C):
        G = self.get_G(x0,C)
        return np.sum(G[np.logical_not(np.isinf(G))])    

def batch_random_generator(N, g_prob = 0.15, n=None, N_min=8, N_max=20, strategy='rand', show_progress=True):
    """Fast generate a batch of random mechanisms that are not locking or invalid.
    Parameters
    ----------
    N:      int
            The number of mechanims to generate
    g_prob: float
            Probability of a node being assigned as ground. Default: 0.15
    n:     int
            Size of mechanism. Default: None (variable size)
    N_min: int
            Minimum size of mechanims if n is not set. Default: 8
    N_max: int
            Maximum size of mechanims if n is not set. Default: 20
    strategy: str
            Either rand or srand. 'rand': fully random, 'srand': sequentially built random. Default: 'rand'
    show_progress: Boolean
            If true will display progress bar. Deafault: True
    
    Returns
    -------
    list of: [N,...]
    
        C:     numpy array [n,n]
                Adjacency/Conncetivity matrix describing the structure of the palanar linkage mechanism.
        x0:    numpy array [n,2]
                    The initial positions of the 
        fixed_nodes: numpy array [n_fixed_nodes]
                A list of the nodes that are grounded/fixed.
        motor: numpy array [2]
                Start and end nodes of the driven linkage (Note: this linkage must be connected to ground on one end). 
    """
    
    
    args = [(g_prob, n, N_min, N_max, strategy)]*N
    return run_imap_multiprocessing(auxilary, args, show_prog = show_progress)

# Auxiary function for batch generator
def auxilary(intermidiate):
    np.random.seed(int.from_bytes(os.urandom(4), byteorder='little'))
    return random_generator_ns(g_prob = intermidiate[0], n=intermidiate[1], N_min=intermidiate[2], N_max=intermidiate[3], strategy=intermidiate[4])
    
    
def random_generator_ns(g_prob = 0.15, n=None, N_min=8, N_max=20, strategy='rand'):
    """Generate random mechanism that is not locking or invalid.
    Parameters
    ----------
    g_prob: float
            Probability of a node being assigned as ground. Default: 0.15
    n:     int
            Size of mechanism. Default: None (variable size)
    N_min: int
            Minimum size of mechanims if n is not set. Default: 8
    N_max: int
            Maximum size of mechanims if n is not set. Default: 20
    strategy: str
            Either rand or srand. 'rand': fully random, 'srand': sequentially built random. Default: 'rand'
            
    Returns
    -------
    C:     numpy array [n,n]
            Adjacency/Conncetivity matrix describing the structure of the palanar linkage mechanism.
    x0:    numpy array [n,2]
                The initial positions of the 
    fixed_nodes: numpy array [n_fixed_nodes]
            A list of the nodes that are grounded/fixed.
    motor: numpy array [2]
            Start and end nodes of the driven linkage (Note: this linkage must be connected to ground on one end). 
    """
    if not n:
        n = int(np.round(np.random.uniform(low=N_min,high=N_max)))
    
    edges = [[0,1],[1,3],[2,3]]
    
    fixed_nodes = [0,2]
    motor = [1,0]
    
    node_types = np.random.binomial(1,g_prob,n-4)
    
    for i in range(4,n):
        if node_types[i-4] == 1:
            fixed_nodes.append(i)
        else:
            picks = np.random.choice(i,size=2,replace=False)
            
            while picks[0] in fixed_nodes and picks[1] in fixed_nodes:
                picks = np.random.choice(i,size=2,replace=False)
            
            edges.append([picks[0],i])
            edges.append([picks[1],i])
    
    C = np.zeros([n,n], dtype=bool)
    
    for edge in edges:
        C[edge[0],edge[1]] = True
        C[edge[1],edge[0]] = True
    
    
    fixed_nodes = np.array(fixed_nodes)
    
    solver = mechanism_solver()
    
    if strategy == 'srand':
        x = np.random.uniform(low=0.0,high=1.0,size=[n,2])

        for i in range(4,n+1):

            sub_size = i
            invalid = not solver.check(100,x[0:sub_size],C[0:sub_size,0:sub_size],motor,fixed_nodes[np.where(fixed_nodes<sub_size)],False)

            co = 0

            while invalid:
                if sub_size > 4:
                    x[sub_size-1:sub_size] = np.random.uniform(low=0.0,high=1.0,size=[1,2])
                else:
                    x[0:sub_size] = np.random.uniform(low=0.0,high=1.0,size=[sub_size,2])

                invalid = not solver.check(50,x[0:sub_size],C[0:sub_size,0:sub_size],motor,fixed_nodes[np.where(fixed_nodes<sub_size)],False)

                co+=1

                if co == 100:
                    neighbours = np.where(C[sub_size-1])[0]
                    relavent_neighbours = neighbours[np.where(neighbours<sub_size-1)]
                    C[relavent_neighbours[0],sub_size-1] = 0
                    C[relavent_neighbours[1],sub_size-1] = 0
                    C[sub_size-1,relavent_neighbours[0]] = 0
                    C[sub_size-1,relavent_neighbours[1]] = 0
                    picks = np.random.choice(sub_size-1,size=2,replace=False)
                    C[picks[0],sub_size-1] = True
                    C[picks[1],sub_size-1] = True
                    C[sub_size-1,picks[0]] = True
                    C[sub_size-1,picks[1]] = True
                    co = 0
    else:
        co = 0
        x = np.random.uniform(low=0.05,high=0.95,size=[n,2])
        invalid = not solver.check(50,x,C,motor,fixed_nodes,False)

        res = sort_mech(C, x, motor,fixed_nodes)
        if res: 
            C, x, fixed_nodes, sorted_order = res
            
            inverse_order = np.argsort(sorted_order)
        else:
            invalid = True
        while invalid:
            x = np.random.uniform(low=0.1+0.25*co/1000,high=0.9-0.25*co/1000,size=[n,2])
            invalid = not solver.check(50,x,C,motor,fixed_nodes,False)
            co += 1
            
            if co>=1000:
                return random_generator_ns(g_prob, n, N_min, N_max, strategy)

    return C,x,fixed_nodes,[0,1]

class curve_normalizer():
    def __init__(self, scale=True):
        """Intance of curve rotation and scale normalizer.
        Parameters
        ----------
        scale: boolean
                If true curves will be oriented and scaled to the range of [0,1]. Default: True.
        """
        self.scale = scale
        self.vfunc = np.vectorize(lambda c: self.get_oriented(c),signature='(n,m)->(n,m)')
        
    def get_oriented(self, curve):
        """Orient and scale(if enabled on initialization) the curve to the normalized configuration
        Parameters
        ----------
        curve: [n_point,2]
                Point coordinates describing the curve.

        Returns
        -------
        output curve: [n_point,2]
                Point coordinates of the curve oriented such that the maximum length is parallel to the x-axis and 
                scaled to have exactly a width of 1.0 on the x-axis is scale is enabled. Curve position is also 
                normalized to be at x=0 for the left most point and y=0 for the bottom most point.
        """
        ci = 0
        t = curve.shape[0]
        pi = t
        
        while pi != ci:
            pi = t
            t = ci
            ci = np.argmax(np.linalg.norm(curve-curve[ci],2,1))
        
        d = curve[pi] - curve[t]
        
        if d[1] == 0:
            theta = 0
        else:
            d = d * np.sign(d[1])
            theta = -np.arctan(d[1]/d[0])
        
        rot = np.array([[np.cos(theta),-np.sin(theta)],[np.sin(theta),np.cos(theta)]])
        out = np.matmul(rot,curve.T).T
        out = out - np.min(out,0)
        
        if self.scale:
            out = out/np.max(out,0)[0]

        if np.isnan(np.sum(out)):
            out = np.zeros(out.shape)

        out = out - np.min(out,0)
        # out = out + (1-np.max(out,0))/2
                    
        return out
    
    def __call__(self, curves):
        """Orient and scale(if enabled on initialization) the batch of curve to the normalized configuration
        Parameters
        ----------
        curve: [N,n_point,2]
                batch of point coordinates describing the curves.

        Returns
        -------
        output curve: [N,n_point,2]
                Batch of point coordinates of the curve oriented such that the maximum length is parallel to the x-axis and 
                scaled to have exactly a width of 1.0 on the x-axis is scale is enabled. Curve position is also 
                normalized to be at x=0 for the left most point and y=0 for the bottom most point.
        """
        return self.vfunc(curves)
    
def run_imap_multiprocessing(func, argument_list, show_prog = True):
    """Run function in parallel
    Parameters
    ----------
    func:          function
                    Python function to run in parallel.
    argument_list: list [N]
                    List of arguments to be passed to the function in each parallel run.
            
    show_prog:     boolean
                    If true a progress bas will be displayed to show progress. Default: True.
    Returns
    -------
    output:        list [N,]
                    outputs of the function for the given arguments.
    """
    pool = mp.Pool(processes=mp.cpu_count())
    
    if show_prog:            
        result_list_tqdm = []
        for result in tqdm(pool.imap(func=func, iterable=argument_list), total=len(argument_list),position=0, leave=True):
            result_list_tqdm.append(result)
    else:
        result_list_tqdm = []
        for result in pool.imap(func=func, iterable=argument_list):
            result_list_tqdm.append(result)

    return result_list_tqdm


class best(Display):
    """Pymoo display to show the best performance of each generation during optimization
    """
    
    def _do(self, problem, evaluator, algorithm):
        super()._do(problem, evaluator, algorithm)
        lowest = np.min(algorithm.pop.get("F"),0)
        for i in range(lowest.shape[0]):
            self.output.append("Lowest Memeber for Objective %i" % (i), lowest[i])
            
def PolyArea(x,y):
    """Area of polygon
    Parameters
    ----------
    x:   numpy array [N]
           list of the x coordinates of the points of the polygon
    y:   numpy array [N]
           list of the y coordinates of the points of the polygon
           
    Returns
    -------
    Area:  float
             Area of the polygon.
    """
    return 0.5*np.abs(np.dot(x,np.roll(y,1))-np.dot(y,np.roll(x,1)))


def hyper_volume(F,ref):
    """Get Hypervolume of population
    Parameters
    ----------
    F:          numpy array [N,m]
                    Perfromance of the paretor front/population.
    ref:        numpy array [m]
                    Reference point for hypervolume calculations.

    Returns
    -------
    hypervolume:  float
                    Hyper volume of the selected population with respect to the reference point.
    """
    ind = HV(ref_point=np.array(ref))
    return ind(F)


def evaluate_submission():
    """Evaluate CSVs in the results folder
    """
    # Get a solver instance
    solver = mechanism_solver()
    
    # Get a solver normalizer
    normalizer = curve_normalizer(scale=False)
    
    scores = []
    
    
    target_curves = []

    # Read every file separately and append to the list
    for i in range(6):
        if not os.path.exists('./data/%i.csv'%(i)):
            raise IOError('Could not find %i.csv in the data folder'%(i))
        target_curves.append(np.loadtxt('./data/%i.csv'%(i),delimiter=','))
    
    
    for i in trange(6):
        if os.path.exists('./results/%i.csv'%i):
            
            mechanisms = get_population_csv('./results/%i.csv'%i)
            F = []
            for m in mechanisms:
                C,x0,fixed_nodes,motor,target = from_1D_representation(m)

                # Solve
                x_sol,f1,f2 = solver.solve_rev(200,x0,C,motor,fixed_nodes,False)
                
                if not f1 and not f2:
                    # Normalize
                    x_norm = normalizer.get_oriented(x_sol[:,target,:])

                    # Step 4: Rasterize
                    out_pc = rasterized_curve_coords(x_norm,500)

                    # Step 5: Compare
                    cd = chamfer_distance(out_pc,target_curves[i],subsample=False)
                    material = solver.material(x0,C)
                    
                    if cd<=0.2 and material<=5.0:
                        F.append([cd,material])
            if len(F):            
                if len(F)>1000:
                    print("Over 1000 linkages submitted! Truncating submission to first 1000.")
                    F=F[:1000]
                scores.append(hyper_volume(np.array(F),[0.2,5.0]))
            else:
                scores.append(0)
        else:
            scores.append(0)
    
    print('Score Break Down:')
    for i in range(6):
        print('Curve %i: %f'%(i,scores[i]))
    
    print('Overall Score: %f'%(np.mean(scores)))
    
    return np.mean(scores)

def to_final_representation(C,x0,fixed_nodes,motor,target):
    """Get 1D representation of mechanism
    ----------
    C:     numpy array [N,N]
            Adjacency/Conncetivity matrix describing the structure of the palanar linkage mechanism.
    x0:    numpy array [N,2]
            The initial positions of the 
    fixed_nodes: numpy array [n_fixed_nodes]
            A list of the nodes that are grounded/fixed.
    motor: numpy array [2]
            Start and end nodes of the driven linkage (Note: this linkage must be connected to ground on one end). 
    target: int
            The index of the node which is to be evaluated as the target node.
    Returns
    -------
    1D representation: numpy array [N^2+3*N+3]
            1D representation of mechanims
    """
    C,x0,fixed_nodes,motor,target = np.array(C),np.array(x0),np.array(fixed_nodes),np.array(motor),np.array(target)
    
    # Get the size of the mechanism in this case 5
    N = C.shape[0]

    # Make the list of node types (set all to zero or ordinary)
    node_types = np.zeros([N])

    # Set the ground nodes
    node_types[fixed_nodes] = 1

    # Set the target
    target_node = target

    # Concatenate to make the final representation
    final_representation = np.concatenate([[N],C.reshape(-1),x0.reshape(-1),node_types,motor,[target_node]])
    
    return final_representation

def from_1D_representation(mechanism):
    """Get python representation of mechanism from 1D representation
    ----------
    mechanism: numpy array [N^2+3*N+3]
                1D representation of mechanims
                
    Returns
    -------
    C:     numpy array [N,N]
            Adjacency/Conncetivity matrix describing the structure of the palanar linkage mechanism.
    x0:    numpy array [N,2]
            The initial positions of the 
    fixed_nodes: numpy array [n_fixed_nodes]
            A list of the nodes that are grounded/fixed.
    motor: numpy array [2]
            Start and end nodes of the driven linkage (Note: this linkage must be connected to ground on one end). 
    target: int
            The index of the node which is to be evaluated as the target node.
    """
    mechanism = np.array(mechanism)
    
    # Get size
    N = int(mechanism[0])
    
    mechanism = mechanism[1:]
    
    # Extract mechanism components
    C = mechanism[0:N**2].reshape([N,N])
    x0 = mechanism[N**2:N**2 + 2*N].reshape([N,2])
    fixed_nodes = np.where(mechanism[N**2 + 2*N:N**2 + 3*N])[0].astype(np.int)
    motor = mechanism[N**2 + 3*N:N**2 + 3*N+2].astype(np.int)
    target = mechanism[-1].astype(np.int)
    
    return C,x0,fixed_nodes,motor,target

def save_population_csv(file_name, population):
    """Save a population of mechanims as csv for submission/evaluation
    ----------
    file_name: str
                File name and path to save the population in.
    population: list [N]
                A list of 1D mechanims representations to be saved in the CSV.
    """
    
    with open(file_name, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', lineterminator = '\n')
        for m in population:
            m = np.array(m)
            writer.writerow(m.tolist())

def get_population_csv(file_name):
    """Load a population of mechanims from csv of submission/evaluation
    ----------
    file_name: str
                File name and path to save the population in.
                
    Returns
    -------            
    population: list [N]
                A list of 1D mechanims representations from the saved CSV.
    """
    population = []
    with open(file_name,'r') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for row in csv_reader:
            population.append(np.array(row).astype(np.float32))
    return population

def is_pareto_efficient(costs, return_mask = True):
    """
    Find the pareto-efficient points
    :param costs: An (n_points, n_costs) array
    :param return_mask: True to return a mask
    :return: An array of indices of pareto-efficient points.
        If return_mask is True, this will be an (n_points, ) boolean array
        Otherwise it will be a (n_efficient_points, ) integer array of indices.
    """
    is_efficient = np.arange(costs.shape[0])
    n_points = costs.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(costs):
        nondominated_point_mask = np.any(costs<costs[next_point_index], axis=1)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        costs = costs[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    if return_mask:
        is_efficient_mask = np.zeros(n_points, dtype = bool)
        is_efficient_mask[is_efficient] = True
        return is_efficient_mask
    else:
        return is_efficient
    
    
def visualize_pareto_front(mechanisms,F,target_curve):
    """Draw Pareto Front in population and visualize mechanisms
    Parameters
    ----------
    mechanisms: numpy array [N,...]
                    List of 1D representations of the mechanims in the population.
    F:          numpy array [N,n_objectives]
                    Perfromance of the paretor population ([[chamfer distance, material]*N]).
    target_curve:  numpy array [n,2]
                    point cloud of target.

    """
    ind = is_pareto_efficient(F)
    X_p = mechanisms[ind]
    F_p = F[ind]
    
    ind = np.argsort(F_p[:,0])
    
    X_p = X_p[ind]
    F_p = F_p[ind]
    
    fig, axs = plt.subplots(X_p.shape[0], 3,figsize=(15,5*X_p.shape[0]))
    
    # Get a solver instance
    solver = mechanism_solver()
    # Get a solver normalizer
    normalizer = curve_normalizer(scale=False)
    
    for i in trange(X_p.shape[0]):
        C,x0,fixed_nodes,motor,target = from_1D_representation(X_p[i])
        draw_mechanism_on_ax(C,x0,fixed_nodes,motor,axs[i,0])

        # Solve
        x_sol, locking,over_under_defined = solver.solve_rev(200,x0,C,motor,fixed_nodes,False)

        # Normalize
        x_norm = normalizer.get_oriented(x_sol[:,target,:])

        # Step 4: Rasterize
        out_pc = rasterized_curve_coords(x_norm,500)

        # Plot
        axs[i,1].scatter(target_curve[:,0],target_curve[:,1],s=2)
        axs[i,1].scatter(out_pc[:,0],out_pc[:,1],s=2)
        axs[i,1].axis('equal')

        axs[i,1].set_title('Chamfer Distance: %f'%(chamfer_distance(out_pc,target_curve)))
        
        axs[i,2].scatter(F_p[:,1],F_p[:,0])
        axs[i,2].set_xlabel('Material Use')
        axs[i,2].set_ylabel('Chamfer Distance')
        axs[i,2].scatter([F_p[i,1]],[F_p[i,0]],color="red")
        

def find_path(A, motor = [0,1], fixed_nodes=[0, 1]):
    
    path = []
    
    A,fixed_nodes,motor = np.array(A),np.array(fixed_nodes),np.array(motor)
    
    unknowns = np.array(list(range(A.shape[0])))

    if motor[-1] in fixed_nodes:
        driven = motor[0]
    else:
        driven = motor[-1]

    knowns = np.concatenate([fixed_nodes,[driven]])
    
    unknowns = unknowns[np.logical_not(np.isin(unknowns,knowns))]
    
    counter = 0
    while unknowns.shape[0] != 0:
        if counter == unknowns.shape[0]:
            # Non dyadic or DOF larger than 1
            return [], False
        n = unknowns[counter]
        ne = np.where(A[n])[0]
        
        kne = knowns[np.isin(knowns,ne)]
#         print(kne.shape[0])
        
        if kne.shape[0] == 2:
            
            path.append([n,kne[0],kne[1]])
            counter = 0
            knowns = np.concatenate([knowns,[n]])
            unknowns = unknowns[unknowns!=n]
        elif kne.shape[0] > 2:
            #redundant or overconstraint
            return [], False
        else:
            counter += 1
    
    return np.array(path), True

def get_G(x0):
    return (np.linalg.norm(np.tile([x0],[x0.shape[0],1,1]) - np.tile(np.expand_dims(x0,1),[1,x0.shape[0],1]),axis=-1))

def solve_rev_vectorized(path,x0,G,motor,fixed_nodes,thetas):
    
    path,x0,G,motor,fixed_nodes = np.array(path),np.array(x0),np.array(G),np.array(motor),np.array(fixed_nodes)
    
    x = np.zeros([x0.shape[0],thetas.shape[0],2])
    
    x[fixed_nodes] = np.expand_dims(x0[fixed_nodes],1)

    if motor[-1] in fixed_nodes:
        driven = motor[0]
        motor_joint = motor[-1]
    else:
        driven = motor[-1]
        motor_joint = motor[0]

    x[driven] = x[motor_joint] + G[motor[0],motor[1]] * np.concatenate([[np.cos(thetas)],[np.sin(thetas)]]).T
    
    state = np.zeros(thetas.shape[0])
    flag = True
    kk = np.zeros(thetas.shape[0]) - 1.0
    
    for step in path:
        i = step[1]
        j = step[2]
        k = step[0]
        
        l_ij = np.linalg.norm(x[j]-x[i],axis=-1)
        cosphi = (l_ij ** 2 + G[i,k]**2 - G[j,k]**2)/(2 * l_ij * G[i,k])

        state += np.logical_or(cosphi<-1.0,cosphi>1.0)
        
        kk = state * k * (kk==-1.0) + kk
        
        s = np.sign((x0[i,1]-x0[k,1])*(x0[i,0]-x0[j,0]) - (x0[i,1]-x0[j,1])*(x0[i,0]-x0[k,0]))

        phi = s * np.arccos(cosphi)

        a = np.concatenate([[np.cos(phi)],[-np.sin(phi)]]).T
        b = np.concatenate([[np.sin(phi)],[np.cos(phi)]]).T

        R = np.swapaxes(np.concatenate([[a],[b]]),0,1)

        scaled_ij = (x[j]-x[i])/np.expand_dims(l_ij,-1) * G[i,k]
        x[k] = np.squeeze(R @ np.expand_dims(scaled_ij,-1)) + x[i]
        
    kk = (kk!=-1.0) + kk    
    return x, state == 0.0, kk.astype(np.int32)

def draw_mechanism(A,x0,fixed_nodes,motor, highlight=100, solve=True, thetas = np.linspace(0,np.pi*2,200), def_alpha = 1.0, h_alfa =1.0, h_c = "#f15a24"):
    
    fig = plt.figure(figsize=(12,12))

    def fetch_path():
        root = etree.parse(StringIO('<svg id="Layer_1" data-name="Layer 1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 620 338"><defs><style>.cls-1{fill:#1a1a1a;stroke:#1a1a1a;stroke-linecap:round;stroke-miterlimit:10;stroke-width:20px;}</style></defs><path class="cls-1" d="M45.5,358.5l70.71-70.71M46,287.5H644m-507.61,71,70.72-70.71M223,358.5l70.71-70.71m20.18,70.72,70.71-70.71m13.67,70.7,70.71-70.71m20.19,70.72,70.71-70.71m15.84,70.71,70.71-70.71M345,39.62A121.38,121.38,0,1,1,223.62,161,121.38,121.38,0,0,1,345,39.62Z" transform="translate(-35.5 -29.62)"/></svg>')).getroot()
        view_box = root.attrib.get('viewBox')
        if view_box is not None:
            view_box = [int(x) for x in view_box.split()]
            xlim = (view_box[0], view_box[0] + view_box[2])
            ylim = (view_box[1] + view_box[3], view_box[1])
        else:
            xlim = (0, 500)
            ylim = (500, 0)
        path_elem = root.findall('.//{http://www.w3.org/2000/svg}path')[0]
        return xlim, ylim, parse_path(path_elem.attrib['d'])
    _,_,p = fetch_path()
    p.vertices -= p.vertices.mean(axis=0)
    p.vertices = (np.array([[np.cos(np.pi), -np.sin(np.pi)],[np.sin(np.pi), np.cos(np.pi)]])@p.vertices.T).T
    


    A,x0,fixed_nodes,motor = np.array(A),np.array(x0),np.array(fixed_nodes),np.array(motor)
    
    x = x0
    
    N = A.shape[0]
    for i in range(N):
        if i in fixed_nodes:
            if i == highlight:
                plt.scatter(x[i,0],x[i,1],color=h_c,s=700,zorder=10,marker=p)
            else:
                plt.scatter(x[i,0],x[i,1],color="#1a1a1a",s=700,zorder=10,marker=p)
        else:
            if i == highlight:
                plt.scatter(x[i,0],x[i,1],color=h_c,s=100,zorder=10,facecolors=h_c,alpha=0.7)
            else:
                plt.scatter(x[i,0],x[i,1],color="#1a1a1a",s=100,zorder=10,facecolors='#ffffff',alpha=0.7)
        
        for j in range(i+1,N):
            if A[i,j]:
                if (motor[0] == i and motor[1] == j) or(motor[0] == j and motor[1] == i):
                    plt.plot([x[i,0],x[j,0]],[x[i,1],x[j,1]],color="#ffc800",linewidth=4.5)
                else:
                    plt.plot([x[i,0],x[j,0]],[x[i,1],x[j,1]],color="#1a1a1a",linewidth=4.5,alpha=0.6)
                
    if solve:
        path = find_path(A,motor,fixed_nodes)[0]
        G = get_G(x0)
        x,c,k =  solve_rev_vectorized(path.astype(np.int32), x0, G, motor, fixed_nodes,thetas)
        x = np.swapaxes(x,0,1)
        if np.sum(c) == c.shape[0]:
            for i in range(A.shape[0]):
                if not i in fixed_nodes:
                    if i == highlight:
                        plt.plot(x[:,i,0],x[:,i,1],'-',color=h_c,linewidth=4.5,alpha=h_alfa)
                    else:
                        plt.plot(x[:,i,0],x[:,i,1],'--',color="#0078a7",linewidth=1.5, alpha=def_alpha)
        else:
            for i in range(A.shape[0]):
                if not i in fixed_nodes:
                    if i == highlight:
                        plt.plot(x[:,i,0],x[:,i,1],'-',color=h_c,linewidth=4.5,alpha=h_alfa)
                    else:
                        plt.plot(x[:,i,0],x[:,i,1],'--',color="#0078a7",linewidth=1.5, alpha=def_alpha)
            plt.text(0.5, 0.5, 'Locking Or Under Defined', color='red', horizontalalignment='center', verticalalignment='center')
        
    plt.axis('equal')
    plt.axis('off')


def draw_mechanism_on_ax(A,x0,fixed_nodes,motor, ax, highlight=100, solve=True, thetas = np.linspace(0,np.pi*2,200), def_alpha = 1.0, h_alfa =1.0, h_c = "#f15a24"):
    
    # fig = plt.figure(figsize=(12,12))

    def fetch_path():
        root = etree.parse(StringIO('<svg id="Layer_1" data-name="Layer 1" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 620 338"><defs><style>.cls-1{fill:#1a1a1a;stroke:#1a1a1a;stroke-linecap:round;stroke-miterlimit:10;stroke-width:20px;}</style></defs><path class="cls-1" d="M45.5,358.5l70.71-70.71M46,287.5H644m-507.61,71,70.72-70.71M223,358.5l70.71-70.71m20.18,70.72,70.71-70.71m13.67,70.7,70.71-70.71m20.19,70.72,70.71-70.71m15.84,70.71,70.71-70.71M345,39.62A121.38,121.38,0,1,1,223.62,161,121.38,121.38,0,0,1,345,39.62Z" transform="translate(-35.5 -29.62)"/></svg>')).getroot()
        view_box = root.attrib.get('viewBox')
        if view_box is not None:
            view_box = [int(x) for x in view_box.split()]
            xlim = (view_box[0], view_box[0] + view_box[2])
            ylim = (view_box[1] + view_box[3], view_box[1])
        else:
            xlim = (0, 500)
            ylim = (500, 0)
        path_elem = root.findall('.//{http://www.w3.org/2000/svg}path')[0]
        return xlim, ylim, parse_path(path_elem.attrib['d'])
    _,_,p = fetch_path()
    p.vertices -= p.vertices.mean(axis=0)
    p.vertices = (np.array([[np.cos(np.pi), -np.sin(np.pi)],[np.sin(np.pi), np.cos(np.pi)]])@p.vertices.T).T
    


    A,x0,fixed_nodes,motor = np.array(A),np.array(x0),np.array(fixed_nodes),np.array(motor)
    
    x = x0
    
    N = A.shape[0]
    for i in range(N):
        if i in fixed_nodes:
            if i == highlight:
                ax.scatter(x[i,0],x[i,1],color=h_c,s=700,zorder=10,marker=p)
            else:
                ax.scatter(x[i,0],x[i,1],color="#1a1a1a",s=700,zorder=10,marker=p)
        else:
            if i == highlight:
                ax.scatter(x[i,0],x[i,1],color=h_c,s=100,zorder=10,facecolors=h_c,alpha=0.7)
            else:
                ax.scatter(x[i,0],x[i,1],color="#1a1a1a",s=100,zorder=10,facecolors='#ffffff',alpha=0.7)
        
        for j in range(i+1,N):
            if A[i,j]:
                if (motor[0] == i and motor[1] == j) or(motor[0] == j and motor[1] == i):
                    ax.plot([x[i,0],x[j,0]],[x[i,1],x[j,1]],color="#ffc800",linewidth=4.5)
                else:
                    ax.plot([x[i,0],x[j,0]],[x[i,1],x[j,1]],color="#1a1a1a",linewidth=4.5,alpha=0.6)
                
    if solve:
        path = find_path(A,motor,fixed_nodes)[0]
        G = get_G(x0)
        x,c,k =  solve_rev_vectorized(path.astype(np.int32), x0, G, motor, fixed_nodes,thetas)
        x = np.swapaxes(x,0,1)
        if np.sum(c) == c.shape[0]:
            for i in range(A.shape[0]):
                if not i in fixed_nodes:
                    if i == highlight:
                        ax.plot(x[:,i,0],x[:,i,1],'-',color=h_c,linewidth=4.5,alpha=h_alfa)
                    else:
                        ax.plot(x[:,i,0],x[:,i,1],'--',color="#0078a7",linewidth=1.5, alpha=def_alpha)
        else:
            for i in range(A.shape[0]):
                if not i in fixed_nodes:
                    if i == highlight:
                        ax.plot(x[:,i,0],x[:,i,1],'-',color=h_c,linewidth=4.5,alpha=h_alfa)
                    else:
                        ax.plot(x[:,i,0],x[:,i,1],'--',color="#0078a7",linewidth=1.5, alpha=def_alpha)
            ax.text(0.5, 0.5, 'Locking Or Under Defined', color='red', horizontalalignment='center', verticalalignment='center')
        
    ax.axis('equal')
    ax.axis('off')