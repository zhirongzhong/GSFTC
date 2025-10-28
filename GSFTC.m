function [Xhat, Ehat, info] = GSFTC(XoNoise, Omega, L_norm, opts)
% GSFTC  Graph-Structured Factorized Tensor Completion (GSFTC)
%
%   [Xhat, Ehat, info] = GSFTC(XoNoise, Omega, L_norm, opts)
%
%   This function implements the core algorithm of GSFTC, a framework for
%   high-dimensional tensor recovery via graph-regularized low-rank priors.
%
% INPUTS:
%   XoNoise : noisy or incomplete tensor (numeric array, up to 3D)
%   Omega   : binary mask tensor (1 = observed, 0 = missing)
%   L_norm  : normalized graph Laplacian matrix (n-by-n)
%   opts    : (optional) struct with parameters:
%       .lambda  - weight for tensor low-rank term (default 1)
%       .beta    - weight for graph-regularized term (default 5e-4)
%       .gamma   - weight for noise sparsity term (default 1e-1)
%       .psi     - scaling factor for proximal operators (default 1e-1)
%       .p       - Schatten-p norm power (default 0.4875)
%       .a, .b, .c - graph parameters for Mmat and Lmat (default 2, 1, 0)
%       .maxIter - maximum iterations (default 1000)
%       .tol     - stopping tolerance (default 1e-8)
%       .verbose - display progress (default true)
%
% OUTPUTS:
%   Xhat : recovered clean tensor
%   Ehat : estimated sparse residual tensor
%   info : struct with fields
%       .iter     - number of iterations performed
%       .residual - final ||X + E - Y||_F
%       .converged - boolean flag (true if tolerance reached)
%
% Example:
%   opts = struct('lambda',1,'beta',5e-4,'gamma',1e-1,'p',0.5);
%   [Xhat,Ehat,info] = GSFTC(XoNoise,Omega,L_norm,opts);
%
% Reference:
%   "GSFTC: Graph-Structured Factorized Tensor Completion",
%   Advanced Engineering Informatics, 2025.
%
% -------------------------------------------------------------------------
% Open-source implementation (c) 2025 GSFTC authors under MIT License
% -------------------------------------------------------------------------

%% --- Parameter setup ---
if nargin < 4, opts = struct(); end
lambda  = get_opt(opts, 'lambda', 1);
beta    = get_opt(opts, 'beta', 5e-4);
gamma   = get_opt(opts, 'gamma', 1e-1);
psi     = get_opt(opts, 'psi', 1e-1);
p       = get_opt(opts, 'p', 0.4875);
a       = get_opt(opts, 'a', 2);
b       = get_opt(opts, 'b', 1);
c       = get_opt(opts, 'c', 0);
maxIter = get_opt(opts, 'maxIter', 1000);
tol     = get_opt(opts, 'tol', 1e-8);
verbose = get_opt(opts, 'verbose', true);

% Derived proximal parameters
tau_1 = lambda ./ psi;
tau_2 = beta ./ psi;
tau_3 = gamma ./ psi;

Y = XoNoise;
mask = (Omega == 1);

% Graph Laplacian-related matrices
Mmat = a * eye(size(L_norm)) - b * L_norm;  % (2I - L)
Lmat = b * L_norm + c * eye(size(L_norm));
pMmat = pinv(Mmat);
InvL  = pinv(Lmat);

%% --- Initialization ---
X = Y; X(~Omega) = mean(Y(mask), 'all');
tensor_size = size(Y);
E = zeros(tensor_size);

%% --- Iterative Block Coordinate Descent ---
for iter = 1:maxIter
    % === Update X ===
    D = Y - E;  X_last = X;
    X0_sum = zeros(tensor_size);
    X1_sum = zeros(tensor_size);

    for mode = 1:3
        A = Unfold_tensor(X, mode);
        M0 = sp_proximal(A, tau_1, p);
        A_fold = Fold_tensor(A, mode, tensor_size);
        MTM = matrix_tensor_mult(A_fold, Mmat);
        B = Unfold_tensor(MTM, mode);
        M1 = sp_proximal(B, tau_2, p);
        T0 = Fold_tensor(M0, mode, tensor_size);
        M1_fold = Fold_tensor(M1, mode, tensor_size);
        Xtilde = matrix_tensor_mult(M1_fold, pMmat);
        T1 = Fold_tensor(Unfold_tensor(Xtilde, mode), mode, tensor_size);
        X0_sum = X0_sum + T0;
        X1_sum = X1_sum + T1;
    end

    X0 = X0_sum / 3;
    X1 = X1_sum / 3;
    X  = (lambda * X0 + beta * X1) / (lambda + beta);
    X(Omega == 1) = D(Omega == 1);

    % === Update E ===
    D = Y - X;  E_last = E;
    E0_sum = zeros(tensor_size);
    E1_sum = zeros(tensor_size);

    for mode = 1:3
        U = Unfold_tensor(E, mode);
        N0 = sp_proximal(U, tau_3, p);
        U_fold = Fold_tensor(U, mode, tensor_size);
        B_MTM = matrix_tensor_mult(U_fold, Lmat);
        B = Unfold_tensor(B_MTM, mode);
        N1 = sp_proximal(B, tau_3, p);
        T0 = Fold_tensor(N0, mode, tensor_size);
        N1_fold = Fold_tensor(N1, mode, tensor_size);
        Etilde = matrix_tensor_mult(N1_fold, InvL);
        T1 = Fold_tensor(Unfold_tensor(Etilde, mode), mode, tensor_size);
        E0_sum = E0_sum + T0;
        E1_sum = E1_sum + T1;
    end

    E0 = E0_sum / 3;
    E1 = E1_sum / 3;
    E  = (E0 + E1) / 2;
    E(Omega == 1) = D(Omega == 1);

    % === Check convergence ===
    relX = norm(X(:) - X_last(:)) / (norm(X_last(:)) + eps);
    relE = norm(E(:) - E_last(:)) / (norm(E_last(:)) + eps);
    if relX < tol && relE < tol
        converged = true;
        break;
    end
end

if ~exist('converged','var'), converged = false; end

%% --- Output ---
Xhat = X;
Ehat = E;
info.iter = iter;
info.residual = norm(X + E - Y, 'fro');
info.converged = converged;

if verbose
    fprintf('GSFTC done in %d iterations. ||X+E-Y||_F=%.4e\n', iter, info.residual);
end

end

%% --- Helper: get_opt ---
function val = get_opt(opts, field, default)
    if isfield(opts, field) && ~isempty(opts.(field))
        val = opts.(field);
    else
        val = default;
    end
end
