%% synthetic_demo.m
% Example use of GSFTC on a synthetic low-rank tensor
% ---------------------------------------------------
% Demonstrates the GSFTC algorithm on a simple 3D low-rank tensor
% with missing entries and additive Gaussian noise.
%
% Author: GSFTC authors
% License: MIT
% ---------------------------------------------------

clear; clc; close all; rng(1);
addpath(genpath(pwd));

%% --- Parameters ---
n = 20;           % tensor dimension (n x n x n)
r = 3;            % rank per slice
p_miss = 0.1;     % missing data ratio
noise_level = 0.1;

%% --- Generate synthetic low-rank tensor ---
X_true = zeros(n,n,n);
for i = 1:n
    A = randn(n, r);
    B = randn(r, n);
    X_true(:,:,i) = A * B; % rank-r slice
end

%% --- Create observations ---
Omega = rand(n,n,n) > p_miss;                 % observation mask
XoNoise = X_true + noise_level * randn(n,n,n); % noisy observations
XoNoise(~Omega) = 0;                          % mask missing entries

%% --- Construct normalized Laplacian (chain graph) ---
L = diag(2*ones(n,1)) - diag(ones(n-1,1),1) - diag(ones(n-1,1),-1);
L = L ./ max(eig(L));  % normalize by spectral radius

%% --- Run GSFTC ---
opts = struct('verbose', true, 'maxIter', 300, 'tol', 1e-6);
[Xhat, Ehat, info] = GSFTC(XoNoise, Omega, L, opts);



