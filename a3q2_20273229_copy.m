function a3q2_20273229_copy
% A3Q2: CISC371, Assignment 3, Question 2
% CISC371, Fall 2023, A3Q2: IRIS sepal data for species I. versicolor

    % Use Fisher's iris data: sepals and I. versicolor species
    load fisheriris;
    xmat = meas(:, 3:4);
    yvec = 1*strcmp(species, 'versicolor');
    
    % Clear the Matlab data from the workspace
    clear meas species;

    % Set the size of the ANN: L layers, expanded weight vector
    Lnum = 2;
    wvecN = (Lnum+1) + Lnum*(size(xmat, 2) + 1);

    % Set the auxiliary data structures for functions and gradients
    global ANNDATA
    ANNDATA.lnum = Lnum;
    ANNDATA.xmat = [xmat ones(size(xmat, 1), 1)];
    ANNDATA.yvec = yvec;

    % Set the starting point: fixed weight vector
    w0 = [ 1 ; -1 ; 0 ; -1 ; -1 ; 3 ; -1 ; -1 ; 7];

    % Set the learning rate and related parameters
    eta   = 0.01;
    imax  = 5000;
    gnorm = 1e-3;

    % Original data
    disp('   ... doing RAW...');
    % Plot and pause
    figure(1);
    ph=gscatter(xmat(:,1), xmat(:,2), yvec, "rb", "o+", [],'on');
    set(ph, 'LineWidth', 2);
    axis('equal');
    xlabel('Sepal Length (cm)', 'interpreter', ...
        'latex', 'fontSize', 12');
    ylabel('Sepal Width (cm)', 'interpreter', ...
        'latex', 'fontSize', 12');
    legend(" other ", "I. setosa", "Location", "southeast");
    title('{\it{}I. versicolor} sepal data: labels', ...
        'interpreter', 'latex', 'fontSize', 14);
    pause(0.5);
    
    % Builtin neural network
    disp('   ... doing NET...');
    net2layer = configure(feedforwardnet(3), xmat', yvec');
    net2layer.trainParam.showWindow = 0;
    [mlnet, mltrain] = train(net2layer, xmat', yvec');
    ynet = (mlnet(xmat')>0.5)*2 - 1;
    % Plot and proceed
    figure(2)
    ph=gscatter(xmat(:,1), xmat(:,2), ynet, "mc", "o+", [],'on');
    set(ph, 'LineWidth', 2);
    axis('equal');
    xlabel('Sepal Length (cm)', 'interpreter', ...
        'latex', 'fontSize', 12');
    ylabel('Sepal Width (cm)', 'interpreter', ...
        'latex', 'fontSize', 12');
    legend(" other ", "I. setosa", "Location", "southeast");
    title('{\it{}I. versicolor} sepal data: MATLAB network', ...
        'interpreter', 'latex', 'fontSize', 14);
    pause(0.5);
    
    %{ 
    % Hard-coded logistic activation, 1 hidden layer
    disp('   ... doing ANN response...');
    [wann fann iann] = steepfixed(@annfun, ...
        w0, eta, imax, gnorm);
    % Use lsqnonlin to optimize instead
    % wann = lsqnonlin(@annfun_lsqnonlin, w0);
    yann = annclass(wann);
    cok = 100*(1 - sum(abs(ANNDATA.yvec - yann))/numel(ANNDATA.yvec));
    % Plot and pause
    disp(sprintf('ANN (%d), W is', iann));
    disp(wann');
    fprintf('Descent: %d%% correct\n', cok);
    figure(3);
    ph=gscatter(xmat(:,1), xmat(:,2), yann, "mc", "o+", [],'on');
    set(ph, 'LineWidth', 2);
    axis('equal');
    xlabel('Sepal Length (cm)', 'interpreter', ...
        'latex', 'fontSize', 12');
    ylabel('Sepal Width (cm)', 'interpreter', ...
        'latex', 'fontSize', 12');
    legend(" other ", "I. setosa", "Location", "southeast");
    title('{\it{}I. versicolor} sepal data: custom network', ...
        'interpreter', 'latex', 'fontSize', 14);
    %}
    
    
    % Display confusion matrix for the builtin network
    disp('   ... doing NET confusion...');
    % Compute confusion matrix, first convert ynet to double
    ynet = double(ynet);
    % Make ynet contain only two labels
    ynet(ynet == -1) = 0;
    confMat = confusionmat(ANNDATA.yvec, ynet);
    % Show
    disp('Confusion matrix:');
    disp(confMat);

    %{
    % Display the confusion matrix for the custom network
    disp('   ... doing ANN confusion...');
    % Compute confusion matrix, first convert yann to double
    yann = double(yann);
    confMat = confusionmat(ANNDATA.yvec, yann);
    % Show
    disp('Confusion matrix:');
    disp(confMat);
    %}
    

    % Plot the decision boundaries for the builtin network
    disp('   ... doing NET decision boundaries...');
    % Plot and pause
    figure(4);
    ph=gscatter(xmat(:,1), xmat(:,2), yvec, "rb", "o+", [],'on');
    set(ph, 'LineWidth', 2);
    axis('equal');
    xlabel('Sepal Length (cm)', 'interpreter', ...
        'latex', 'fontSize', 12');
    ylabel('Sepal Width (cm)', 'interpreter', ...
        'latex', 'fontSize', 12');
    legend(" other ", "I. setosa", "Location", "southeast");
    title('{\it{}I. versicolor} sepal data: labels', ...
        'interpreter', 'latex', 'fontSize', 14);
    hold on;
    % Plot the decision boundaries
    x1 = linspace(min(xmat(:,1)), max(xmat(:,1)), 100);
    x2 = linspace(min(xmat(:,2)), max(xmat(:,2)), 100);
    [X1, X2] = meshgrid(x1, x2);
    Y = mlnet([X1(:)'; X2(:)']);
    Y = (Y>0.5)*2 - 1;
    Y = reshape(Y, size(X1));
    contour(X1, X2, Y, [0 0], 'k', 'LineWidth', 2);
    hold off;

    % Plot the decision boundaries for the custom network
    disp('   ... doing ANN decision boundaries...');
    % Plot and pause
    figure(5);
    ph=gscatter(xmat(:,1), xmat(:,2), yvec, "rb", "o+", [],'on');
    set(ph, 'LineWidth', 2);
    axis('equal');
    xlabel('Sepal Length (cm)', 'interpreter', ...
        'latex', 'fontSize', 12');
    ylabel('Sepal Width (cm)', 'interpreter', ...
        'latex', 'fontSize', 12');
    legend(" other ", "I. setosa", "Location", "southeast");
    title('{\it{}I. versicolor} sepal data: labels', ...
        'interpreter', 'latex', 'fontSize', 14);
    hold on;
    %{
    % Plot the decision boundaries
    x1 = linspace(min(xmat(:,1)), max(xmat(:,1)), 100);
    x2 = linspace(min(xmat(:,2)), max(xmat(:,2)), 100);
    [X1, X2] = meshgrid(x1, x2);
    wann_hidden = reshape(wann((Lnum+2):end), 3, Lnum);

    % Compute and plot decision boundary for the first neuron
    Y1 = wann_hidden(1,1)*X1 + wann_hidden(2,1)*X2 + wann_hidden(3,1);
    % Y1 = (Y1 > 0) * 2 - 1;  % Binarize
    contour(X1, X2, Y1, [0 0], 'k', 'LineWidth', 2);

    % Compute and plot decision boundary for the second neuron
    Y2 = wann_hidden(1,2)*X1 + wann_hidden(2,2)*X2 + wann_hidden(3,2);
    % Y2 = (Y2 > 0) * 2 - 1;  % Binarize
    contour(X1, X2, Y2, [0 0], 'g', 'LineWidth', 2);  
    %}
    hold off;
end


function [fval, gform] = annfun(wvec)
% FUNCTION [FVAL,GFORM]=ANNFUN(WVEC) computes the response of a simple
% neural network that has 1 hidden layer of sigmoids and a linear output
% neuron. WVEC is the initial estimate of the weight vector. FVAL is the
% scalar objective evaluation a GFORM is the gradient 1-form (row vector).
% ANNDATA is a global structure containing data and labels.
%
% INPUTS:
%         WVEC    -  weight vector, output then hidden
% GLOBALS:
%         ANNDATA -  Structure containing
%                    lnum - number of hidden units to compute
%                    xmat - MxN matrix, each row is a data vector
%                    yvec - Mx1 column vector, each label is +1 or -1
% OUTPUTS:
%         FVAL    - 1xM row vector of sigmoid responses

    global ANNDATA
    % Problem size: original data, intermediate data
    [m, n] = size(ANNDATA.xmat);
    l = ANNDATA.lnum;

    % Separate output weights and hidden weights; latter go into a matrix
    wvec1= wvec(1:(l + 1));
    wvecH = reshape(wvec((l+2):end), n, l);

    % Compute the hidden responses as long row vectors, 1 per hidden neuron;
    % then, append 1's to pass the transfer functions to the next layer
    
    % %
    % % STUDENT CODE GOES HERE; REPLACE THE FOLLOWING LINE WITH 1 OR MORE
    % %
    % Compute the linear product of the data (with bias terms) and hidden weights
    lin_prod = ANNDATA.xmat * wvecH;
    phi2mat = 1./(1 + exp(-lin_prod));
    % Append a column of 1's to the hidden responses
    phi2mat(:,end+1) = 1;

    % Compute the output transfer function: linear in hidden responses
    phi2vec = phi2mat*wvec1;

    % ANN quantization is Heaviside step function of transfer function
    q2vec = phi2vec >= 0;

    % Residual is difference between label and network output
    rvec = ANNDATA.yvec - q2vec;

    % Objective is sum of squares of residual errors
    fval = 0.5*sum((rvec).^2);

    % If required, compute and return the gradient 1-form
    if nargout > 1
        % Compute the hidden differential responses, ignoring the appended 1's

        % %
        % % STUDENT CODE GOES HERE; REPLACE THE FOLLOWING LINE WITH 1 OR MORE LINES
        % %
        % Remove the appended 1's
        phi2mat_nobias = phi2mat(:,1:end-1);
        % Compute the differential responses
        psimat = phi2mat_nobias.*(1 - phi2mat_nobias);
 
        % Set up the hidden gradients, then loop through the data vectors
        hidgrad = [];
        for jx = 1:m
            % Find the product of the derivative vector and Jacobian matrix

            % %
            % % STUDENT CODE GOES HERE; REPLACE THE FOLLOWING 4 LINES WITH 4 OR MORE
            % %
            % Compute the Jacobian matrix
            thisJmat = diag(psimat(jx,:));
            % Compute the differential matrix of the linear product in respect to the weights
            thisDmat = kron(eye(l), ANNDATA.xmat(jx,:));
            % Compute the product of the Jacobian and differential matrices
            JD_product = thisJmat*thisDmat;
            % Append a row of zeros to the product
            JD_product_ncols = size(JD_product,2);
            thisJDaug = [thisJmat*thisDmat; zeros(1,JD_product_ncols)];
            % Stack the product of the Jacobian and differential matrices
            hidgrad = [hidgrad ; wvec1'*thisJDaug];
        end

        % Differentiate the residual error and scale the gradient matrix
        % Note that the rvec is this case is based on the difference between
        % the actual label and the classification value, which is slightly
        % different from the notes
        grad12mat = -diag(rvec)*[phi2mat hidgrad];

        % Net gradient is the sum of the gradients of each data vector
        gform = sum(grad12mat);
    end
end

function [rvec, xfmat] = annclass(wvec)
% FUNCTION RVEC=ANNCLASS(WVEC) computes the response of a simple neural
% network that has 1 hidden layer of logistic cells and a linear output.
% ANNDATA is a global structure containing data and labels.
%
% INPUTS:
%         WVEC  -  weight vector, output then hidden
% GLOBALS:
%         ANNDATA -  Structure containing
%                    lnum - number of hidden units to compute
%                    xmat - MxN matrix, each row is a data vector
%                    yvec - Mx1 column vector, each label is 0 or 1
% OUTPUTS:
%         RVEC  - Mx1 vector of linear responses to data
%         XFMAT - Mx(L+1) array of hidden-layer responses to data

    % Problem size: original data, intermediate data
    global ANNDATA
    [m,n] = size(ANNDATA.xmat);
    l = ANNDATA.lnum;

    % Separate output weights and hidden weights; latter go into a matrix
    wvec2 = wvec(1:(l + 1));
    wvecH = reshape(wvec((l+2):end), n, l);

    % Compute the hidden responses as long row vectors, 1 per hidden neuron;
    % then, append 1's to pass the transfer functions to the next layer
    xfmat = (1./(1+exp(-(ANNDATA.xmat*wvecH))));
    xfmat(:,end+1) = 1;

    % Compute the transfer function: linear in hidden responses
    hidxfvec = xfmat*wvec2;

    % ANN response is Heaviside step function of transfer function
    rvec = (hidxfvec >= 0);
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

function [fval, gform] = annfun_lsqnonlin(wvec)
% FUNCTION [FVAL,GFORM]=ANNFUN(WVEC) computes the response of a simple
% neural network that has 1 hidden layer of sigmoids and a linear output
% neuron. WVEC is the initial estimate of the weight vector. FVAL is the
% scalar objective evaluation a GFORM is the gradient 1-form (row vector).
% ANNDATA is a global structure containing data and labels.
%
% INPUTS:
%         WVEC    -  weight vector, output then hidden
% GLOBALS:
%         ANNDATA -  Structure containing
%                    lnum - number of hidden units to compute
%                    xmat - MxN matrix, each row is a data vector
%                    yvec - Mx1 column vector, each label is +1 or -1
% OUTPUTS:
%         FVAL    - 1xM row vector of sigmoid responses

    global ANNDATA
    % Problem size: original data, intermediate data
    [m, n] = size(ANNDATA.xmat);
    l = ANNDATA.lnum;

    % Separate output weights and hidden weights; latter go into a matrix
    wvec1= wvec(1:(l + 1));
    wvecH = reshape(wvec((l+2):end), n, l);

    % Compute the hidden responses as long row vectors, 1 per hidden neuron;
    % then, append 1's to pass the transfer functions to the next layer
    
    % %
    % % STUDENT CODE GOES HERE; REPLACE THE FOLLOWING LINE WITH 1 OR MORE
    % %
    % Compute the linear product of the data (with bias terms) and hidden weights
    lin_prod = ANNDATA.xmat * wvecH;
    phi2mat = 1./(1 + exp(-lin_prod));
    % Append a column of 1's to the hidden responses
    phi2mat(:,end+1) = 1;

    % Compute the output transfer function: linear in hidden responses
    phi2vec = phi2mat*wvec1;

    % ANN quantization is Heaviside step function of transfer function
    q2vec = phi2vec >= 0;

    % Residual is difference between label and network output
    rvec = ANNDATA.yvec - q2vec;

    % Objective is sum of squares of residual errors
    fval = rvec;

    % If required, compute and return the gradient 1-form
    if nargout > 1
        % Compute the hidden differential responses, ignoring the appended 1's

        % %
        % % STUDENT CODE GOES HERE; REPLACE THE FOLLOWING LINE WITH 1 OR MORE LINES
        % %
        % Remove the appended 1's
        phi2mat_nobias = phi2mat(:,1:end-1);
        % Compute the differential responses
        psimat = phi2mat_nobias.*(1 - phi2mat_nobias);
 
        % Set up the hidden gradients, then loop through the data vectors
        hidgrad = [];
        for jx = 1:m
            % Find the product of the derivative vector and Jacobian matrix

            % %
            % % STUDENT CODE GOES HERE; REPLACE THE FOLLOWING 4 LINES WITH 4 OR MORE
            % %
            % Compute the Jacobian matrix
            thisJmat = diag(psimat(jx,:));
            % Compute the differential matrix of the linear product in respect to the weights
            thisDmat = kron(eye(l), ANNDATA.xmat(jx,:));
            % Compute the product of the Jacobian and differential matrices
            JD_product = thisJmat*thisDmat;
            % Append a row of zeros to the product
            JD_product_ncols = size(JD_product,2);
            thisJDaug = [thisJmat*thisDmat; zeros(1,JD_product_ncols)];
            % Stack the product of the Jacobian and differential matrices
            hidgrad = [hidgrad ; wvec1'*thisJDaug];
        end

        % Differentiate the residual error and scale the gradient matrix
        % Note that the rvec is this case is based on the difference between
        % the actual label and the classification value, which is slightly
        % different from the notes
        grad12mat = -diag(rvec)*[phi2mat hidgrad];

        % Net gradient is the sum of the gradients of each data vector
        gform = sum(grad12mat);
    end
end
