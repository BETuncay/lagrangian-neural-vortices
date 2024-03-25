% calculate the flow map from a velocity field given a timespan
function F = get_flowmap(derivative,domain,resolution,timespan, odeSolverOptions)

    initialPosition = initialize_ic_grid(resolution,domain);
    finalPosition = ode45_vector(@(t,y)derivative(t,y,false),timespan,initialPosition,false,odeSolverOptions);
    
    initialPosition = reshape(initialPosition,[fliplr(resolution), 2]);
    finalPosition = reshape(finalPosition,[fliplr(resolution), 2]);   
 
    F(:,:,1:2) = initialPosition;
    F(:,:,3:4) = finalPosition;


function yf = ode45_vector(odefun,tspan,y0,useEoV,options)

% Reshape m-by-2 array to column array
y0 = transpose(y0);
y0 = y0(:);

if useEoV
    coupledSize = 6;
else
    coupledSize = 2;
end

% Specify three timesteps for ode45's tspan. This has been reported to
% reduce memory usage.
tsteps = [tspan(1),tspan(1) + .5*diff(tspan),tspan(2)];
[~,yf] = ode45(odefun,tsteps,y0,options);
yf = yf(end,:);
yf = transpose(reshape(yf,coupledSize,size(yf,2)/coupledSize));



