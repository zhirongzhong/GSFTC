function M = Unfold_tensor(T, mode)
% Unfold_tensor  Unfold a 3D tensor along a specified mode.
%   M = Unfold_tensor(T, mode)
%   mode = 1, 2, or 3
dims = size(T);
switch mode
    case 1
        M = reshape(T, dims(1), []);
    case 2
        M = reshape(permute(T, [2,1,3]), dims(2), []);
    case 3
        M = reshape(permute(T, [3,1,2]), dims(3), []);
    otherwise
        error('mode must be 1, 2, or 3');
end
end
