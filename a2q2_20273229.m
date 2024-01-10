function a2q2_20273229
% CISC371, Fall 2023, Assignment #2, Question #2: vector optimization

% Anonymous functions for objective functions, gradients, and Hessian
% matrices
% %
% % STUDENT CODE GOES HERE: REPLACE WITH WORKING CODE
% %
syms w1 w2;

% First define the function and its gradient and Hessian using symbolic math
f1_sym = ((w1 + 1.21) - 2 * (w2 - 1))^4 + 64 * (w1 + 1.21) * (w2 - 1);
g1_sym = transpose(gradient(f1_sym, [w1, w2]));
h1_sym = hessian(f1_sym, [w1, w2]);
% Convert to annonymous functions using matlabFunction()
% Remember that these annonymous functions take col object args
% Uses {...} to make the input a col vector
f1 = matlabFunction(f1_sym, 'Vars', {[w1 ; w2]});
g1 = matlabFunction(g1_sym, 'Vars', {[w1 ; w2]});
h1 = matlabFunction(h1_sym, 'Vars', {[w1 ; w2]});

% Doing the same for ther other functions
f2_sym = 2 * w2^3 - 6 * w2^2 + 3 * w1^2 * w2;
g2_sym = transpose(gradient(f2_sym, [w1, w2]));
h2_sym = hessian(f2_sym, [w1, w2]);
f2 = matlabFunction(f2_sym, 'Vars', {[w1 ; w2]});
g2 = matlabFunction(g2_sym, 'Vars', {[w1 ; w2]});
h2 = matlabFunction(h2_sym, 'Vars', {[w1 ; w2]});

f3_sym = 100 * (w2 - w1^2)^2 + (1 - w1)^2;
g3_sym = transpose(gradient(f3_sym, [w1, w2]));
h3_sym = hessian(f3_sym, [w1, w2]);
f3 = matlabFunction(f3_sym, 'Vars', {[w1 ; w2]});
g3 = matlabFunction(g3_sym, 'Vars', {[w1 ; w2]});
h3 = matlabFunction(h3_sym, 'Vars', {[w1 ; w2]});

% Unify the above functions for standard calls to optimization code
fg1  =@(w) deal((f1(w)), (g1(w)));
fg2  =@(w) deal((f2(w)), (g2(w)));
fg3  =@(w) deal((f3(w)), (g3(w)));

% Use the same start point and backtracking values
w0 = [-1.2 ; 1];
beta = 0.5;

% Stepsizes for the methods
sfixed = 0.001;
sline = 0.1;

% Minimizer, minimum, and iterations for fixed stepsize
[wvfixed1, fmfixed1, ixfixed1] = steepfixed(fg1, w0, sfixed);
[wvfixed2, fmfixed2, ixfixed2] = steepfixed(fg2, w0, sfixed);
[wvfixed3, fmfixed3, ixfixed3] = steepfixed(fg3, w0, sfixed);

% Minimizer, minimum, and iterations for line search
[wvline1, fmline1, ixline1] = steepline(fg1, w0, sline, beta);
[wvline2, fmline2, ixline2] = steepline(fg2, w0, sline, beta);
[wvline3, fmline3, ixline3] = steepline(fg3, w0, sline, beta);

% Apologies for using sline I didn't hear that 1 was reccomended until it was too late
% Minimizer, minimum, and iterations for damped Newton's method
[wvnewton1, fmnewton1, ixnewton1] = newtondamped(fg1, h1, w0, sline, beta);
[wvnewton2, fmnewton2, ixnewton2] = newtondamped(fg2, h2, w0, sline, beta);
[wvnewton3, fmnewton3, ixnewton3] = newtondamped(fg3, h3, w0, sline, beta);

disp(sprintf(' '));
disp('A2Q2(a)> Fixed stepsize, iterations and minimizer:');
disp(sprintf('f%d: %5d (% 3.2f,% 3.2f)', [1 ixfixed1 wvfixed1']));
disp(sprintf('f%d: %5d (% 3.2f,% 3.2f)', [2 ixfixed2 wvfixed2']));
disp(sprintf('f%d: %5d (% 3.2f,% 3.2f)', [3 ixfixed3 wvfixed3']));

disp(sprintf(' '));
disp('A2Q2(b)> Line search, iterations and minimizer:');
disp(sprintf('f%d: %5d (% 3.2f,% 3.2f)', [1 ixline1 wvline1']));
disp(sprintf('f%d: %5d (% 3.2f,% 3.2f)', [2 ixline2 wvline2']));
disp(sprintf('f%d: %5d (% 3.2f,% 3.2f)', [3 ixline3 wvline3']));

disp(sprintf(' '));
disp('A2Q2(c)> Damped Newton search, iterations and minimizer:');
disp(sprintf('f%d: %5d (% 3.2f,% 3.2f)', [1 ixnewton1 wvnewton1']));
disp(sprintf('f%d: %5d (% 3.2f,% 3.2f)', [2 ixnewton2 wvnewton2']));
disp(sprintf('f%d: %5d (% 3.2f,% 3.2f)', [3 ixnewton3 wvnewton3']));

end

function [tmin,fmin,ix]=steepfixed(objgradf,w0,s,imax_in,eps_in)
% [WMIN,FMIN,IX]=STEEPFIXED(OBJGRADF,W0,S,IMAX,F)
% estimates the minimum of function and gradient OBJGRADF, beginning
% at point W0 and using constant stepsize S. Optional
% arguments are the limits on the gradient norm, EPS, and the
% number of iterations, IMAX. 
%
% INPUTS:
%         OBJGRADF - Function for objective and gradient
%         W0       - initial estimate of W
%         S        - stepsize, positive scalar value
%         IMAX     - optional, limit on iterations; default is 50000
%         EPS      - optional, limit on gradient norm; default is 10^-6
% OUTPUTS:
%         WMIN     - minimizer, scalar or vector argument for OBJF
%         FMIN     - scalar value of OBJF at, or near, TMIN
%         IX       - Iterations performed

% Set convergence criteria to those supplied, if available
if nargin >= 4 & ~isempty(imax_in)
  imax = imax_in;
else
  imax = 50000;
end

if nargin >= 5 & ~isempty(eps_in)
  epsilon = eps_in;
else
  epsilon = 1e-6;
end

% Initialize: search vector, objective, gradient
tmin = w0;
[fmin gval] = objgradf(tmin);
ix = 0;
while (norm(gval)>epsilon & ix<imax)
% %
% % STUDENT CODE GOES HERE: REPLACE "BREAK" WITH WORKING CODE
% %
    % Step iteration
    tmin = tmin + s*(-gval');
    % Reassign objective and gradient
    [fmin, gval] = objgradf(tmin);
    % Increment iter counter
    ix = ix + 1;
end
end

function [wmin,fmin,ix]=steepline(objgradf,w0,s0,beta,imax_in,eps_in)
% [WMIN,FMIN]=STEEPLINE(OBJGRADF,W0,S,BETA,IMAX,EPS)
% estimates the minimum of function and gradient OBJGRADF, beginning
% at point W0 and using constant stepsize S0. Backtracking is
% controlled by reduction ratio BETA. Optional
% arguments are the limits on the gradient norm, EPS, and the
% number of iterations, IMAX. 
%
% INPUTS:
%         OBJGRADF - Function for objective and gradient
%         W0       - initial estimate of W
%         S0       - stepsize, positive scalar value
%         BETA     - backtracing hyper-parameter, 0<beta<1
%         IMAX     - optional, limit on iterations; default is 50000
%         EPS      - optional, limit on gradient norm; default is 10^-6
% OUTPUTS:
%         TMIN     - minimizer, scalar or vector argument for OBJF
%         FMIN     - scalar value of OBJF at, or near, TMIN
%         IX       - Iterations performed

% Set convergence criteria to those supplied, if available
if nargin >= 5 & ~isempty(imax_in)
  imax = imax_in;
else
  imax = 50000;
end

if nargin >= 6 & ~isempty(eps_in)
  epsilon = eps_in;
else
  epsilon = 1e-6;
end


% Limit BETA to the interval (0,1)
beta  = max(1e-6, min(1-(1e-6), beta));

% Initialize: objective, gradient, unit search vector
wmin = w0;
[fmin, gofm] = objgradf(wmin);
dvec = -gofm';
alpha = gofm/2;
ix = 0;

while (norm(gofm)>epsilon & ix<imax)

% %
% % STUDENT CODE GOES HERE: REPLACE "BREAK" WITH WORKING CODE
% %
    s = s0;
    % Armijo backtracking
    [f_estimate, ~] = objgradf(wmin + s * dvec);
    while (f_estimate >= (fmin + alpha * s * dvec))
        % Update stepsize and estimated step
        s = beta * s;
        [f_estimate, ~] = objgradf(wmin + s * dvec);
    end
    % Update current estimate
    wmin = wmin + s * dvec;
    % Update function evaluation and gradient at point
    [fmin, gofm] = objgradf(wmin);
    % Update search direction
    dvec = -gofm';
    % Update alpha parameter for backtracking
    alpha = gofm/2; 
    % Increment iter counter
    ix = ix + 1;
end
end

function [wmin,fmin,ix]=newtondamped(objgradf,hessf,w0,s0,beta,imax_in,eps_in)
% [WMIN,FMIN,IX]=NEWTONDAMPED(OBJGRADF,HESSF,W0,BETA,IMAX,EPS)
% estimates the minimum of function and gradient OBJGRADF, and Hessian
% matrix HESSF, using a damped Newton's method. It begins at point W0,
% estimates the stepsize by backtracking with ALPHA and BETA subject to
% a modified Armijo condition, and took iterations IX.
%
% Optional arguments are the limits on the gradient norm, EPS, and the
% number of iterations, IMAX.
%
% INPUTS:
%         OBJGRADF - Function for objective and gradient
%         HESSF    - Hessian function  of scalar or vector argument T
%         W0       - initial estimate of W
%         BETA     - scalar exponential back-off argument, 0<beta<1
%         IMAX     - optional, limit on iterations; default is 200
%         EPS      - optional, limit on gradient norm; default is 10^-6
% OUTPUTS:
%         WMIN     - minimizer, scalar or vector argument for OBJF
%         FMIN     - scalar value of OBJF at, or near, TMIN
%         IX       - number of iterations performed

% Set convergence criteria to those supplied, if available
if nargin >= 6 & ~isempty(imax_in)
  imax = imax_in;
else
  imax = 200;
end

if nargin >= 7 & ~isempty(eps_in)
  epsilon = eps_in;
else
  epsilon = 1e-6;
end

% Initialize: search vector, objective, gradient, Hessian, descent vector
wmin = w0;
[fmin, gofm] = objgradf(wmin);
hmat = hessf(wmin);
dvec = hmat\(-gofm');
alpha = gofm/2;
ix = 0;

while (norm(gofm)>epsilon & ix<imax) 

% %
% % STUDENT CODE GOES HERE: REPLACE "BREAK" WITH WORKING CODE
% %
    s = s0;
    % Armijo backtracking
    [f_estimate, ~] = objgradf(wmin + s * dvec);
    while (f_estimate >= (fmin +  alpha * s * dvec))
        % Update stepsize and estimated step
        s = beta * s;
        [f_estimate, ~] = objgradf(wmin + s * dvec);
    end
    % Update current estimate
    wmin = wmin + s*dvec;
    % Update function evaluation and gradient at point
    [fmin, gofm] = objgradf(wmin);
    % Update Hessian at point
    hmat = hessf(wmin);
    % Update Newton's method search direction
    dvec = hmat\(-gofm');
    % Update alpha parameter for backtracking
    alpha = gofm/2;
    % Increment iter counter
    ix = ix + 1;
end
end
