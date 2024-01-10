function a1_20273229
% CISC371, Fall 2023, Assignment #1, Question #1: scalar optimization

% Anonymous functions for objective functions and gradients
f1 = @(t) exp(3*t) + 5*(exp(-2*t));
g1 = @(t) 3*exp(3*t) - 10*exp(-2*t);
h1 = @(t) 20*exp(-2*t) + 9*exp(3*t);

f2 = @(t) (log(t)).^2 - 2 + (log(10-t)).^2 - t.^(0.2);
g2 = @(t) (2*log(t))/t + (2*log(10 - t))/(t - 10) - 1/(5*t.^(4/5));
h2 = @(t) 2/(t - 10).^2 - (2*log(10 - t))/(t - 10).^2 - (2*log(t))/t.^2 + 2/t.^2 + 4/(25*t.^(9/5));

f3 = @(t) -3 * t * sin(0.75*t) + exp(-2*t);
g3 = @(t) -2*exp(-2*t) - 3*sin((3*t)/4) - (9*t*cos((3*t)/4))/4;
h3 = @(t) 4*exp(-2*t) - (9*cos((3*t)/4))/2 + (27*t*sin((3*t)/4))/16;

% Unify objective and gradient for standard call to optimization code
fg1  =@(t) deal((f1(t)), (g1(t)));
fgh1 =@(t) deal((f1(t)), (g1(t)), (h1(t)));
fg2  =@(t) deal((f2(t)), (g2(t)));
fgh2 =@(t) deal((f2(t)), (g2(t)), (h2(t)));
fg3  =@(t) deal((f3(t)), (g3(t)));
fgh3 =@(t) deal((f3(t)), (g3(t)), (h3(t)));
fg3A  =@(t) deal((f3A(t)), (g3A(t)));
fgh3A =@(t) deal((f3A(t)), (g3A(t)), (h3A(t)));


% Compute the quadratic approximations and search estimates

% I have chosen to display my results
% Note that fx_quad_stationary is the estimated minimizer for the quadratic
% approximation

% Approximations for f1
f1_t0 = 1;
[~, f1_quad_stationary] = quadapprox(fgh1, f1_t0);
[f1_fixed_minimizer, ~, f1_fixed_numiter] = steepfixed(fg1, f1_t0, 1/100);
[f1_bt_minimizer , ~, f1_bt_numiter] = steepline(fg1, f1_t0, 1/10, 0.5, 0.5);
f1_quad_stationary
f1_fixed_minimizer
f1_fixed_numiter
f1_bt_minimizer
f1_bt_numiter


% Approximations for f2
f2_t0 = 9.9;
[~, f2_quad_stationary] = quadapprox(fgh2, f2_t0);
[f2_fixed_minimizer, ~, f2_fixed_numiter] = steepfixed(fg2, f2_t0, (9.9 - 6)/100);
[f2_bt_minimizer ,~,f2_bt_numiter] = steepline(fg2, f2_t0, (9.9 - 6)/10, 0.5, 0.5);
f2_quad_stationary
f2_fixed_minimizer
f2_fixed_numiter
f2_bt_minimizer
f2_bt_numiter


f3_t0 = 2*pi;
[~, f3_quad_stationary] = quadapprox(fgh3, f3_t0);
[f3_fixed_minimizer, ~, f3_fixed_numiter] = steepfixed(fg3, f3_t0, (2*pi)/100);
[f3_bt_minimizer, ~,f3_bt_numiter] = steepline(fg3, f3_t0, (2*pi)/10, 0.5, 0.5);
f3_quad_stationary
f3_fixed_minimizer
f3_fixed_numiter
f3_bt_minimizer
f3_bt_numiter



end

function [fcoef, tstat] = quadapprox(funfgh, t1)
% [FCOEF,TSTAT]=QUADAPPROX(FUNFGH,T1) finds the polynomial coefficients
% FCOEF of the quadratic approximation of a function at a scalar point
% T1, using the objective value, gradient, and Hessian from function FUNFGH
% at T1 to complete the approximation. FCOEF is ordered for use in POLYVAL.
% The stationary point of the quadratic is returned as TSTAT.
%
% INPUTS:
%         FUNFGH - handle to 3-output function that computed the
%                  scalar-valued function, gradient, and 2nd derivative
%         T1     - scalar argument
% OUTPUTS:
%         FCOEF  - 1x3 array of polynomial coefficients
%         TSTAT  - stationary point of the approximation
% ALGORITHM:
%     Set up and solve a 3x3 linear equation. If the points are colinear
%     then TSTAT is empty

% Initialize the outputs
fcoef = [];
tstat = [];

% Set up a linear equation
estimate_quatratic = [t1^2, t1, 1; 2*t1, 1, 0; 2, 0, 0];
% Separate outputs of funfgh
[f_evald, g_evald, h_evald] = funfgh(t1);
evald_vec = [f_evald; g_evald; h_evald];

% Solve system of equations
fcoef = (estimate_quatratic \ evald_vec)';
% Solve for stationary point
tstat = -fcoef(2)/(2*fcoef(1));

end


function [tmin,fmin,ix] = steepfixed(objgradf,t0,s,imax_in,eps_in)
% [TMIN,FMIN,IX]=STEEPFIXED(OBJGRADF,T0,S,IMAX,F)
% estimates the minimum of function and gradient OBJGRADF, beginning
% at point T0 and using constant stepsize S. Optional
% arguments are the limits on the gradient norm, EPS, and the
% number of iterations, IMAX. 
%
% INPUTS:
%         OBJGRADF - Function for objective and gradient
%         W0       - initial estimate of minimizer
%         S        - stepsize, positive scalar value
%         IMAX     - optional, limit on iterations; default is 50000
%         EPS      - optional, limit on gradient norm; default is 10^-6
% OUTPUTS:
%         TMIN     - minimizer for OBJF
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

% Initialize: search point, objective, gradient
tmin = t0;
[fmin, gval] = objgradf(tmin);
ix = 0;

while (norm(gval)>epsilon & ix<imax)
    % Step iteration
    tmin = tmin + s*(-gval);
    % Reassign objective and gradient
    [fmin, gval] = objgradf(tmin);
    % Increment iter counter
    ix = ix + 1;
end

end


function [tmin,fmin,ix]=steepline(objgradf,t0,s0,beta,alpha,imax_in,eps_in)
% [TMIN,FMIN]=STEEPLINE(OBJGRADF,T0,S,BETA,IMAX,EPS)
% estimates the minimum of function and gradient OBJGRADF, beginning
% at point T0 and using constant stepsize S. Backtracking is
% controlled by reduction ratio BETA. Optional
% arguments are the limits on the gradient norm, EPS, and the
% number of iterations, IMAX. 
%
% INPUTS:
%         OBJGRADF - Function for objective and gradient
%         T0       - initial estimate of minimizer
%         S        - stepsize, positive scalar value
%         BETA     - backtracking hyper-parameter, 0<beta<1
%         IMAX     - optional, limit on iterations; default is 50000
%         EPS      - optional, limit on gradient norm; default is 10^-6
% OUTPUTS:
%         TMIN     - minimizer, scalar or vector argument for OBJF
%         FMIN     - scalar value of OBJF at, or near, TMIN
%         IX       - Iterations performed

% Set convergence criteria to those supplied, if available
if nargin >= 6 & ~isempty(imax_in)
  imax = imax_in;
else
  imax = 50000;
end

if nargin >= 7 & ~isempty(eps_in)
  epsilon = eps_in;
else
  epsilon = 1e-6;
end


% Limit BETA to the interval (0,1)
beta  = max(1e-6, min(1-(1e-6), beta));
alpha = max(1e-6, min(1-(1e-6), alpha));


% Initialize: objective, gradient, unit search vector
tmin = t0;
[fmin, gofm] = objgradf(tmin);
dvec = -gofm;
wmat = tmin;
ix = 0;

while (abs(gofm)>epsilon & ix<imax)
% %
% % STUDENT CODE GOES HERE
% %
    s = s0;
    % Armijo backtracking
    [f_estimate, ~] = objgradf(wmat + s*dvec);
    while (f_estimate >= (fmin + alpha*gofm * s*dvec))
        % Update stepsize and estimated step
        s = beta*s;
        [f_estimate, ~] = objgradf(wmat + s*dvec);
    end
    % Update values
    wmat = wmat + s*dvec;
    [fmin, gofm] = objgradf(wmat);
    dvec = -gofm;   
    ix = ix + 1;
end
% Assign wmat to tmin at the end to return
tmin = wmat;
end