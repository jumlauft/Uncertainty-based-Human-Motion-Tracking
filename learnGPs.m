function [GPm,GPs2, hyp,iKn, beta] = learnGPs(X,Y,parGP)
%%LEARNGPDM Buildes mean and variance function for GP incl. hyp opt.
% In:
%     X        D  x Ntr   Input training points
%     Y        E  x Ntr   Output training points
%     parGP    struct with following (possible) components:
%           likfunc  fhandle    likelihood function from gpml
%           covfunc  fhandle    covariance function from gpml
%           meanfun  fhandle    mean function from gpml
%           sn_init  E x 1      Initialization obeservation noise
%           optp    1 x 1      Hyp Optimization settings
%           N_rand    1 x 1    Number of random initialization
%           init_mean E+1 x1      Initialization mean of l and sf2
% Out:
%     GPm   fhandle      E x N -> D x N   mean function
%     GPs2  fhandle      E x N -> D x N   variance function
% E: Dimensionality of x
% Ntr: Number of training points
%
% Copyright (c) by Jonas Umlauft under BSD License 
% Last modified: Jonas Umlauft, 02/2017


%covfunc,meanfun,likfunc, sn_init, optp, N_rand, init_mean


% Check inputs and set defaults
[E,Ntr] = size(Y); D = size(X,1); % Needed for hyp.cov
if ~isfield(parGP,'sn_init')||isempty(parGP.sn_init), sn_init = 0.1*ones(E,1);
else sn_init = parGP.sn_init; end

if ~isfield(parGP,'likfunc')||isempty(parGP.likfunc), likfunc = @likGauss;
else likfunc = parGP.likfunc; end

if ~isfield(parGP,'covfunc')||isempty(parGP.covfunc), covfunc = @covSEard;
else covfunc = parGP.covfunc; end

if ~isfield(parGP,'meanfun')||isempty(parGP.meanfun), meanfun = @meanZero;
else meanfun = parGP.meanfun; end

if ~isfield(parGP,'optp')|| isempty(parGP.optp), optp = -100;
else optp = parGP.optp; end

if ~isfield(parGP,'Nrand')|| ~isfield(parGP,'init_mean'), N_rand = 1; init_mean = zeros(eval(feval(covfunc)),1);
else N_rand = parGP.Nrand; init_mean = parGP.init_mean; end

if size(X,2) ~= Ntr ||  numel(sn_init) ~= E
    error('wrong input dimensions');
end


% Initialize 
GPs = cell(E,1); ll_best = -inf(E,1);

for i = 1:N_rand
    for e=1:E
        % Initialize and optimize hyps - case for linear mean added
        hyp_in(e).lik=log(sn_init(e)); hyp_in(e).cov=init_mean+randn(eval(feval(covfunc)),1);
        if strcmp(func2str(meanfun),'meanLinear')
            hyp_in(e).mean = zeros(E,1);
            for d=1:D
                hyp_in(e).mean(d,1) = -1;
            end
        elseif strcmp(func2str(meanfun),'meanConst')
            hyp_in(e).mean = 0;
        end
        [hyp_new(e), nlls] = minimize(hyp_in(e), @gp, optp, @infExact, meanfun, covfunc, likfunc, X', Y(e,:)');
        ll_new = -nlls(end);
        
        % Select highest likelihood
        if ll_new > ll_best(e)
            hyp(e) = hyp_new(e);
            %hyp(e).mean = -eye(size(hyp(e).mean));
            GPs{e} = @(x)gp(hyp_new(e), @infExact, meanfun, covfunc, likfunc, X', Y(e,:)', x');
            ll_best(e) = ll_new;
        end
    end 
end



 % Build inline function

GPm = @(x)  GPmfun(x,GPs);
GPs2 = @(x) GPs2fun(x,GPs,hyp,X,covfunc);

if nargout >3
    iKn = zeros(Ntr,Ntr,E);
    beta = zeros(Ntr,E);
    for e=1:E
        K  = covfunc(hyp(e).cov,X');sn = exp(2*hyp(e).lik);
        iKn(:,:,e) = inv(K + sn*eye(Ntr));
        beta(:,e) = (K + sn*eye(Ntr))\Y(e,:)';
    end
end
end

function m = GPmfun(x,GPs)
D = length(GPs); N = size(x,2);
m = zeros(D,N);
for d=1:D
    m(d,:) = GPs{d}(x);
end
end

function [s2, ds2dx] = GPs2fun(x,GPs,hyp,Xtr,covfunc)
E = length(GPs); [D,N] = size(x);
s2 = zeros(E,N);
for d=1:E
    [~, s2(d,:)] = GPs{d}(x);
end
if nargout > 1
    ds2dx = zeros(E,N,D,N);
    if isequal(covfunc,@covLINard)
        for e = 1:E
            [k,~, dkdx] = covLINardj(hyp(e).cov,x,Xtr);
            K = covLINardj(hyp(e).cov,Xtr,Xtr);
            sn2 = exp(2*hyp(e).lik);
            l = exp(-2*hyp(e).cov);
            for n = 1:N
                ds2dx(e,n,:,n) = 2*x(:,n).*l -...
                    (2*k(n,:)/(K+sn2*eye(size(K)))*permute(dkdx(n,:,:,n),[2 3 1 4]))';
            end
        end
    else if isequal(covfunc,@covSEard)
            for e = 1:E
                [k,~, dkdx] = covSEardj(hyp(e).cov,x,Xtr);
                K = covSEardj(hyp(e).cov,Xtr,Xtr);
                sn2 = exp(2*hyp(e).lik);
                %l = exp(-2*hyp(e).cov);
                for n = 1:N
                    ds2dx(e,n,:,n) = -2*k(n,:)/(K+sn2*eye(size(K)))*permute(dkdx(n,:,:,n),[2 3 1 4]);
                end
            end
        else
            error('kernel not supported for derivative');
        end
    end
end


end