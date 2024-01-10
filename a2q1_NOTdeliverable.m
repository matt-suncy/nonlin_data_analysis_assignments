% f is a function with a vector input 'w' with two entries and a scalar output
syms w1 w2

% Other functions
f_1 = ((w1 + 1.21) - 2 .* (w2 - 1)).^4 + 64 .* (w1 + 1.21) .* (w2 - 1);
f_2 = 2 .* w2.^3 - 6 .* (w2).^2 + 3 .* w1^2 .* w2;
f_3 = 100 * (w2 - w1.^2).^2 + (1 - w1).^2;
f = f_1;
g = transpose(gradient(f));


% Finding stationary points
s = solve(g==[0 0], [w1, w2]);
s = [s.w1, s.w2];

% Evaluate the Hessian for each stationary point
hessian_mat = hessian(f, [w1, w2]);
for i = 1:length(s)
    hessian_at_point = subs(hessian_mat, [w1, w2], [s(i,1), s(i,2)])
    eigvh = eval(eig(hessian_at_point));
    
    fprintf('Stationary Point %d: (%f, %f)', i, s(i,1), s(i,2));
    % Show the function value at the stationary point
    fprintf('Function value at stationary point: %f\n', eval(subs(f, [w1, w2], [s(i,1), s(i,2)])));
    
    if all(eigvh > 0)
        disp('The stationary point is a local minimum');
    elseif all(eigvh < 0)
        disp('The stationary point is a local maximum');
    elseif any(eigvh < 0) && any(eigvh > 0)
        disp('The stationary point is a saddle point');
    else
        disp('Not enough information to determine the type of stationary point');
    end
    disp(' ')
end
