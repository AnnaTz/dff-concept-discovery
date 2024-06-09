import torch
import math

EPSILON = 1e-7

class SeedContextManager:
    """
    Context manager to temporarily set the random seed for PyTorch operations.
    
    Parameters:
        - seed (int): The seed value to set for random number generation.
        - cuda (bool): If True, sets the seed for CUDA operations. Defaults to False.
    
    Upon entering, the current random seed is saved and the new seed is applied. 
    Upon exiting, the original seed is restored.
    """
    def __init__(self, seed, cuda=False):
        self.seed = seed
        self.cuda = cuda

    def __enter__(self):
        if self.cuda:
            self.current_seed = torch.cuda.initial_seed()
            torch.cuda.manual_seed(self.seed)
        else:
            self.current_seed = torch.initial_seed()
            torch.manual_seed(self.seed)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.cuda:
            torch.cuda.manual_seed(self.current_seed)
        else:
            torch.manual_seed(self.current_seed)


def initialize_matrices(V, k, W, H, device, scale):
    """
    Initializes W and H matrices for NMF (if not provided), and moves all matrices to 
    the given device. 
    
    Parameters:
        - V (Tensor): The input matrix V of shape (n, m), where n is the number of 
        features and m is the number of samples.
        - k (int): The number of components to use for factorization.
        - W (Tensor): The W matrix of shape (n, k). If None, a random matrix is created.
        - H (Tensor): The H matrix of shape (k, m). If None, a random matrix is created.
        - device (str): The device to move the matrices to ('cpu' or 'cuda').
        - scale (float): A scaling factor for the initial random matrices.
    
    Returns:
        - tuple: A tuple containing W and H, adjusted to have non-negative elements.
    """  
    if W is None:
        W = torch.empty(
            V.size(0), k, dtype=torch.float, device=device
        ).normal_() * scale
    else:
        W = W.to(device)
    if H is None:
        H = torch.empty(
            k, V.size(1), dtype=torch.float, device=device
        ).normal_() * scale
    else:
        H = H.to(device)
    return torch.abs(W), torch.abs(H)


def approximation_error(V, W, H):
    """
    Computes the Frobenius norm of V - WH, which represents the NMF approximation error.
    
    Parameters:
        - V (Tensor): The original matrix V of shape (n, m).
        - W (Tensor): The basis matrix W of shape (n, k).
        - H (Tensor): The coefficient matrix H of shape (k, m).
    
    Returns:
        - Tensor: A scalar tensor containing the norm.
    """
    return torch.norm(V - torch.mm(W, H))


def multiplicative_update_step(V, W, H, update_H=True, VH=None, HH=None):
    """
    Performs the multiplicative update step for matrices W and H during NMF.
    
    Parameters:
        - V (Tensor): The input matrix V.
        - W (Tensor): Current estimate of the W matrix.
        - H (Tensor): Current estimate of the H matrix.
        - update_H (bool): Whether to update H.
        - VH (Tensor): Precomputed V*H'. If None, it is computed within the function.
        - HH (Tensor): Precomputed H*H'. If None, it is computed within the function.
    
    Returns:
        - tuple: A tuple containing the final W, H, V*H' and H*H' matrices.
    """
    with torch.no_grad():
        if VH is None:
            assert HH is None
            Ht = torch.t(H)
            VH = torch.mm(V, Ht)
            HH = torch.mm(H, Ht)

        WHH = torch.mm(W, HH)
        WHH[WHH == 0] = EPSILON
        W *= VH / WHH

        if update_H:
            Wt = torch.t(W)
            WV = torch.mm(Wt, V)
            WWH = torch.mm(Wt, W).mm(H)
            WWH[WWH == 0] = EPSILON
            H *= WV / WWH
            VH, HH = None, None

    return W, H, VH, HH


def NMF(V, k, W=None, H=None, random_seed=None, max_iter=200, tol=1e-4, cuda=True):
    """
    Performs Non-negative Matrix Factorization on a matrix V.
    Parameters:
        - V (Tensor): Input matrix.
        - k (int): Number of components.
        - W, H (Tensor): Initial matrices.
        - random_seed (int): Seed for random number generator.
        - max_iter (int): Maximum number of iterations.
        - tol (float): Tolerance for convergence.
        - cuda (bool): Whether to use CUDA.
        
    Returns:
        - Tuple[Tensor, Tensor]: Factorized matrices W and H.
    """
    scale = math.sqrt(V.mean().item() / k)
    device = 'cuda' if cuda else 'cpu'
    V = V.to(device)

    update_H = H is None
    with SeedContextManager(random_seed, cuda):
        W, H = initialize_matrices(V, k, W, H, device, scale)

    previous_error = approximation_error(V, W, H)

    VH, HH = None, None
    for n_iter in range(max_iter):
        W, H, VH, HH = multiplicative_update_step(
            V, W, H, update_H=update_H, VH=VH, HH=HH
        )

        if tol > 0 and n_iter % 10 == 0:
            error = approximation_error(V, W, H)
            if (previous_error - error) / previous_error < tol:
                break
            previous_error = error

    return W, H
