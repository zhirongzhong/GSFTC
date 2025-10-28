function T_out = matrix_tensor_mult(T_in, M)
% matrix_tensor_mult  Multiply tensor along its first mode by matrix M
%   T_out = matrix_tensor_mult(T_in, M)
% Applies: for each frontal slice, multiply by M.
dims = size(T_in);
if size(M,2) ~= dims(1)
    error('matrix_tensor_mult: dimension mismatch');
end
T_out = zeros([size(M,1), dims(2), dims(3)]);
for k = 1:dims(3)
    T_out(:,:,k) = M * T_in(:,:,k);
end
end
