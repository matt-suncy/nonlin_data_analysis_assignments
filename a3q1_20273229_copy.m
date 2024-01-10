function a3q1_20273229_copy
% Code for CISC371, Fall 2023, Assignment #3, Question #1

    % Options to silence LSQNONLIN
    optnls = optimset('Display','none');
    % Option to set max function evaluations 
    max_eval_option = optimset('MaxFunEvals', 1000, 'Display', 'iter', 'FunValCheck', 'on');
    % Option to use the Levenberg-Marquardt algorithm
    levenberg_option = optimset('Display', 'iter', 'Algorithm', 'levenberg-marquardt', 'FunValCheck', 'on');
    % Option to display the number of iterations
    iter_option = optimset('Display', 'iter');
    % Option to display function value
    fun_option = optimset('Display', 'iter', 'FunValCheck', 'on');
 
    % Load the GPS data
    satellite_locs = load('xgps.txt');
    pseudo_ranges = load('ygps.txt');

    % %
    % % STUDENT CODE GOES HERE: define as anonymous functions or 
    % %     use global variables and append their code after this function
    % %

    % Annonymous function for residual errors
    residual_err = @(w) residual_vals(w);
    % residual_err = @(w) arrayfun(@(i) norm(w - satellite_locs(i, :)) - pseudo_ranges(i), 1:size(satellite_locs, 1));
    % Helper function for doing calculations
    function res_val = residual_vals(w)
        n = size(satellite_locs, 1);
        res_val = zeros(1, n);
        for i = 1:n
            res_val(i) = norm(w - satellite_locs(i, :)) - pseudo_ranges(i);
        end
    end     
   
    % Mean location of the satellites as a possible starting point
    w0 = mean(satellite_locs, 1)';
    % Other possible starting points
    % Origin
    w0_origin = [0; 0; 0];
    % Kingston, Ontario
    w0_kingston = [lla2ecef([44.2312, -76.4810, 93], 'WGS84')]';
    starting_points = [w0, w0_origin, w0_kingston];

    % Find the receiver location
    % %
    % % STUDENT CODE GOES HERE: solve the Fermat-Weber problem for GPS
    % %

    % Make loop to do calculations for all 3 starting points
    for ix = 1:size(starting_points)
        % Extract current starting point
        w_start = starting_points(:, ix);
        % Run non-linear least squares solver
        % wopt = lsqnonlin(residual_err, w_start, [], []);
        wopt = lsqnonlin(residual_err, w_start, [], []);


        % Display starting point info
        disp('A3Q1> Starting point for LSQNONLIN is:');
        disp(w_start);

        % Display the receiver location in ECEF for lookup
        disp('A3Q1> Cartesian coordinates of the GPS receiver are:');
        fprintf('%7.1f %7.1f %7.1f\n', wopt);

        % Display function value at the solution
        disp('A3Q1> Function value at the solution is:');
        disp(norm(residual_err(wopt))^2);

        % Conver to LLA
        lla = ecef2lla(wopt', 'WGS84');
        lat = lla(1);
        lon = lla(2);
        alt = lla(3);
        % Display the receiver location in LLA for lookup
        disp('A3Q1> LLA coordinates of the GPS receiver are (latitude, longitude, altitude):');
        fprintf('%7.1f %7.1f %7.1f\n', lat, lon, alt);
        fprintf('\n END OF INFO \n')
    end

    %{ 
    % Find ECEF coordinates of Kingston, Ontario
    kingston_lla = [44.2312, -76.4810, 93];
    kingston_ecef = lla2ecef(kingston_lla, 'WGS84');
    disp('A3Q1> Cartesian coordinates of Kingston, Ontario are:');
    fprintf('%7.1f %7.1f %7.1f\n', kingston_ecef);
    %}
    
end