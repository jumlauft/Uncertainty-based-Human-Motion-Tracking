function [mon, dmondx,dmondxdx]= getMonomial(x,exMat)
%%GETMONOMIAL computes all monomials for x
% In:
%   x     E  x N         input
%   ExMat Dm x E         Matrix of all combinations
% Out
%  mon      Dm x N       all monomials
%  dmondx   Dm x N x E x N      Derivative w.r.t x
%  dmondxdx Dm x N x E x N x E x N  Derivative w.r.t x (twice)
% E: Dimensionality of x
% Dm: Dimensionality of monomial
%{
clear all, close all, rng default; addpath('./mtools');
N = 3; E = 2; dSOS = 2; x = rand(E,N); exMat = getExpoMatrix(E,dSOS);
[m2, dmdx,dmdxdx]= getMonomial(x,exMat);
[n,num,ana] = checkGrad(@getMonomial,1,1,2,{x,exMat});
[n,num,ana] = checkGrad(@getMonomial,1,2,3,{x,exMat});
%}
%
% Copyright (c) by Jonas Umlauft under BSD License 
% Last modified: Jonas Umlauft, 05/2017

[E,N] = size(x);
x = x+10e-20;

Dm = size(exMat,1);

if size(exMat,2) ~=E
    error('wrong input dimensions');
end
exMatp = permute(exMat,[1 3 2]);
xp = permute(x,[3 2 1]);
monall = xp.^exMatp; % Dm x N x E
mon = prod(monall,3);

% Compute derivatives if necessary
if nargout > 1
    deriv_new = exMatp.*xp.^(exMatp-1);  % Dm x N x E
    
    iNN = 1:N+1:N^2; dmondx = zeros(Dm,E,N,N);
    
    dmondx(:,:,iNN) = permute(deriv_new.*mon./monall,[1 3 2]);
    dmondx = permute(dmondx,[1 3 2 4]);
end

% % Compute derivatives if necessary
if nargout > 2
    dmondxdx = zeros(Dm,E,E,N,N,N); iNNN =1:N^2+N+1:N^3;
    iEE = 1:E+1:E^2; niEE = setdiff(1:E^2,iEE);
    deriv2 = zeros(Dm,N,E,E);
    deriv2(:,:,iEE) =  exMatp.*(exMatp-1).*xp.^(exMatp-2);
    if E ~= 1,deriv2(:,:,niEE) = exMatp.*xp.^(exMatp-1);end
    
    %     [iE,jE] =  ind2sub(E,iEE); [niE,njE] =  ind2sub(E,niEE);
    %     dmondxdx_new(:,iE,jE,iNNN) = permute(deriv2_new(:,:,iEE).*(mon./monall(:,:,iE)),[1 3 4 2]);
    %     if E ~= 1
    %     dmondxdx_new(:,niE,njE,iNNN) = permute(deriv2_new(:,:,niE,njE).*deriv2_new(:,:,njE,niE).*(mon./monall(:,:,niE)./monall(:,:,njE)),[1 3 4 2]);
    %     end
    for e1=1:E
        for e2=1:E
            if e1 == e2
                dmondxdx(:,e1,e2,iNNN)=permute(deriv2(:,:,e1,e2).*...
                    mon./monall(:,:,e1),[1 3 4 2]);
            else
                dmondxdx(:,e1,e2,iNNN)=permute(deriv2(:,:,e2,e1).*...
                    deriv2(:,:,e1,e2).*mon./monall(:,:,e1)./monall(:,:,e2),[1 3 4 2]);
            end
        end
    end
    
    dmondxdx = permute(dmondxdx,[1 4 2 5 3 6]);
    
end

end






