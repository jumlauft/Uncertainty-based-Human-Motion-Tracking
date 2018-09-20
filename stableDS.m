function dxdt = stableDS(x,funf,funV,rho0)
%%STABLEDS: Stabilizes a continous time DS with control lyapunov (CLF)
% The system given by DEQ dxdt = fun(x) is stabilized using the control
% Lyapunov function funV. rho0 controls how fast it converges.
% In:
%    x      E x N      Current points
%    funf   fhandle    computes dxdt for multiple x, thus ExN->ExN
%    funV   fhandle    computes CLF V and derivatives dVdx for multiple x,
%                      thus ExN->[N, NxExN]
%    rho0   1 x 1      Angle pointing in from Lyapfun (default = 5)
% Out:
%    dxdt   E x N      stablized dxdt
% E: Dimensionality of x
% N: Number of data points
%
% Copyright (c) by Jonas Umlauft (TUM) under BSD License
% Last modified: Jonas Umlauft, Armin Lederer, 02/2017

% Check inputs
[E,N] = size(x);
if ~exist('rho0','var')
    rho0 = 5;
end
rho = @(x) min(sin(pi*rho0/180),sum(x.^2,1));
u = zeros(E,N);

% Compute DEQ and CLF
dx = funf(x);
if N>1e4
    dVdx = zeros(N,E);
    for n=1:N
        [~, dVdx(n,:)] = funV(x(:,n));
    end
    dVdx = dVdx';
else
    [~, dVdx] = funV(x);
    
    % Get relevant derivatives
    if size(dVdx,2) == 1
        dVdx = permute(dVdx,[3 1 2]); % stable DS expects Output as NxExN with N = 1 here
    end
    
    dVdx = permute(dVdx,[2 1 3]);
    dVdx  = dVdx(:,1:N+1:N^2);
end

% check where angle decrease towards Lyapunov function is already suficient
inc_norm = sum(dVdx.*dx,1)./(sqrt(sum(dVdx.^2,1)).*sqrt(sum(dx.^2,1))) + rho(x);
iinc = (inc_norm > 0);

% compute stabilization and execute
if any(iinc)
    inc = sum(dVdx.*dx,1)./sqrt(sum(dVdx.^2,1))+rho(x).*sqrt(sum(dx.^2,1));
    u(:,iinc) = -inc(iinc)  .*  dVdx(:,iinc)./sqrt(sum(dVdx(:,iinc).^2,1));
end

dxdt = dx+u;

end