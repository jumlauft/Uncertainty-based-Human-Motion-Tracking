function [logVsim, logV0] = LyapVar(Xsim,varfun,us,psn2)
%VARLYAP Computes the Variance based Lyapunov function
% In:
%    Xsim     E  x NSim x N0 Stable simulations start
%    varfun   fhandle        Variance function   E x N -> Nx1     
%    opt.                    option struct
%        dt    1 x 1           Time step of  simulation (default = 0.1)
%        tol   1 x 1           tolenranz for reaching origin (default = 1)
%        maxN  1 x 1           Maximum number of simulation steps (default = 100)
%        mins  1 x 1           Spezifies minimum stepsize
%        maxs  1 x 1           Spezifies maximum stepsize
%        minh  1 x 1           absolut lower bound for step size
%    us        1 x 1        subsampling of integral     
%    psn2      E x 1        sample noise
% Out:
%    logVsim   Nsim x 1    log Variance Lyapunov function at Xsim
%    logV0     N0 x 1      log Variance Lyapunov function at x0
% E: Dimensionality of x
% N0: Number of starting points
% Nsim: Interation steps done in simulation
%
% Copyright (c) by Jonas Umlauft (TUM) under BSD License
% Last modified: Jonas Umlauft, 02/2017

% Check inputs
[E, NSim, N0] = size(Xsim);
if ~exist('us','var'), us = 10; end
if ~exist('psn2','var'), psn2 = zeros(E,1); end

% Append zeros
Xsim = cat(2,Xsim, zeros(E,1,N0));
Nsim = NSim+1;

% Determine intermediate steps for integration
Xint = zeros(E,(Nsim-1)*us,N0);
for e2=1:E
    Xint(e2,:,:) =  permute(interp1(1:Nsim,permute(Xsim(e2,:,:),[2 3 1]),linspace(1,Nsim,(Nsim-1)*us)),[3 1 2]);
end
Xint = reshape(Xint,E,[]);

% Compute distances between steps
diffnorm = sqrt(sum(diff(Xint,1,2).^2,1));
% Book keeping since all in one vector
diffnorm((Nsim-1)*us:(Nsim-1)*us:end) = [];
Xint(:,1:(Nsim-1)*us:end) = [];

% Identify alredy converged trajectories and set Lyap to zero
Xint = reshape(Xint,E,[]);
idrop = sum(Xint.^2,1) == 0;
s2(1:E,idrop) = 0;

% Compute Variance and multiply with distances and add up
s2(:,~idrop) = varfun(Xint(:,~idrop))+psn2;
norm_s2 = sqrt(sum(s2.^2,1));  % calculate norm of s2
Vsim = cumsum(reshape(sum(diffnorm(:)'.*norm_s2,1),[],N0),1,'reverse'); % f(s2)

% Sort out integration steps and log
logVsim = log(Vsim(1:us:end,:));

if nargout > 1
    logV0 = log(sum(reshape(sum(diffnorm(:)'.*s2,1),[],N0),1));
end

end


