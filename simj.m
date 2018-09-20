function X = simj(fun,X0,opt,V,funHelp)
% SIMJ: Simulates time continous systems
% In:
%    fun          fhandle   returns dxdt
%    X0           E x N0    Initial points
%    opt.                   option struct
%        dt       1 x 1         Time step of  simulation (default = 0.1)
%        tol      1 x 1         tolenranz for reaching origin (default = 1)
%        maxN     1 x 1         Maximum number of simulation steps (default = 100)
%        mins     1 x 1         Spezifies minimum stepsize (default = 1)
%        maxs     1 x 1         Spezifies maximum stepsize (default = 10)
%        minh     1 x 1         absolut lower bound for step size (default = 0.01)
%    V            fhandle       Lyapunov function to be reduced
%    funHelp     fhandle       Emergency dxdt, called if trapped
% Out:
%    X     E x Nsim x N0     Trajectory
% E: Dimensionality of x
% N0: Number of starting points
% Nsim: Numer of steps done in simulation
%
% Copyright (c) by Jonas Umlauft (TUM) under BSD License
% Last modified: Jonas Umlauft, Lukas Poehler 06/2017

% Check inputs
[E, N0] = size(X0);
if ~exist('opt','var')
    opt.dt = 0.1; opt.tol = 1; opt.maxN = 1e2; opt.mins = 1; opt.maxs = 10;
    opt.minh = 1e-2;
else
    if ~isfield(opt,'dt') || isempty(opt.dt), opt.dt = 0.01; end
    if ~isfield(opt,'tol') || isempty(opt.tol), opt.tol = 1; end
    if ~isfield(opt,'maxN') || isempty(opt.maxN), opt.maxN = 1e2; end
    if ~isfield(opt,'mins') || isempty(opt.mins), opt.mins = 1-4; end
    if ~isfield(opt,'maxs') || isempty(opt.maxs), opt.maxs = 10; end
    if ~isfield(opt,'minh') || isempty(opt.minh), opt.minh = 1e-2; end
    
end

X = zeros(E,N0,opt.maxN);
X(:,:,1) = X0;
for k = 2:opt.maxN
    % Find active trajectories (which have not yet converged)
    iact = sqrt(sum(X(:,:,k-1).^2,1)) > opt.tol;
    % If all have converged, finish simulation
    if  ~any(iact), X(:,:,k) = zeros(E,N0); X = X(:,:,1:k); break;   end
    
    X(:,~iact,k) = 0;
    
    % Simulate active trajectories
    dxdt = reshape(fun(X(:,iact,k-1)),E,sum(iact));
    
    eps = 1e-3;
    normdx = sqrt(sum(dxdt.^2,1));
    % Fix if gradient is too small or zero
    iact2 = find(iact);
    if exist('funHelp','var')
        dxdt(:,normdx < eps) = funHelp(X(:,iact2(normdx<eps),k-1));
    end
    %dxdt(:,normdx==0) = - 0.5*X(:,iact2(normdx==0),k-1);
    
    % restrict step size
    normdx = sqrt(sum(dxdt.^2,1));
    h = max(min(normdx,opt.maxs/opt.dt),opt.mins/opt.dt);
    
    % Go the step
    X(:,iact,k) = X(:,iact,k-1) + opt.dt * h .* dxdt ./ normdx;
    
    % If Lyapunov function is provided  it is checked for descendants
    if exist('V','var')
        % Find increasing trajectories to avoid jumps over valleys
        iinc = V(X(:,iact,k)) - V(X(:,iact,k-1)) > 0;
        while any(iinc)
            % and reduce step size
            h(iinc)=h(iinc)/2;
            X(:,iact,k) = X(:,iact,k-1) + opt.dt*h.* dxdt./normdx;
            
            % check if still increasing
            iinc = V(X(:,iact,k)) - V(X(:,iact,k-1)) > 0;
            
            % treat cases of local minima
            ihelp = iact2(iinc & h'<opt.minh) ; %iact;
            if any(ihelp)
                if exist('funHelp','var')
                    % go in mean direction if GP is given with 1/2 average
                    % of mins and maxs
                    dxdt = funHelp(X(:,ihelp,k-1));
                    X(:,ihelp,k) = X(:,ihelp,k-1) + ((opt.mins+opt.maxs)/4) .* dxdt ./ sqrt(sum(dxdt.^2,1));
                else
                    % go in gradient direction
                    X(:,ihelp,k) = X(:,ihelp,k-1) + opt.minh .* dxdt(:,iinc) ./ normdx(:,iinc);
                end
                break;
            end
        end
    end
    
end
if k == opt.maxN
    disp('Trajectories have not converged. Try increase optSim.maxN');
end
% Bring to output format:  E x Ntraj x N0
X = permute(X,[1 3 2]);

end
