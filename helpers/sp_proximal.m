function M_out = sp_proximal(M_in, tau, p)
% sp_proximal  Proximal operator for Schatten-p norm (0 < p <= 1)
%   M_out = sp_proximal(M_in, tau, p)
% Uses singular value shrinkage with nonconvex shrinkage for 0<p<1

if nargin < 3, p = 1; end
[U,S,V] = svd(M_in, 'econ');
s = diag(S);

if p == 1
    s_shrunk = max(s - tau, 0);
else
    % nonconvex shrinkage: iterative reweighting approximation
    s_shrunk = max(s - tau * s.^(p-1), 0);
end

M_out = U * diag(s_shrunk) * V';
end
