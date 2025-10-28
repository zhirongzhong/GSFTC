function T = Fold_tensor(M, mode, tensor_size)
% Fold_tensor  Fold a matrix back to a 3D tensor.
%   T = Fold_tensor(M, mode, tensor_size)
switch mode
    case 1
        T = reshape(M, tensor_size);
    case 2
        T = ipermute(reshape(M, tensor_size([2,1,3])), [2,1,3]);
    case 3
        T = ipermute(reshape(M, tensor_size([3,1,2])), [3,1,2]);
    otherwise
        error('mode must be 1, 2, or 3');
end
end
