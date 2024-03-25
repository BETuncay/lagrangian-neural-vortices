% calculates the trajectory of a velocity field within a given timespan
function trajectory = get_trajectory(derivative,domain,resolution,timespan, steps, odeSolverOptions)

    times = linspace(timespan(1), timespan(2), steps);
    initialPosition = initialize_ic_grid(resolution,domain);
    trajectory = ode45_vector(@(t,y)derivative(t,y,false),times,initialPosition,odeSolverOptions, resolution);
    


function trajectory = ode45_vector(odefun,tspan,y0,options,resolution)

y0 = transpose(y0);
y0 = y0(:);

[~,yf] = ode45(odefun,tspan,y0,options);

trajectory = single(ones(resolution(1), resolution(2), size(tspan,2)));
for i = 0:size(tspan,2) - 1  
    trajectory(:,:,2*i+1:2*i+2) = reshape(transpose(reshape(yf(i+1,:),2,[])), resolution(1), resolution(2), 2);
end