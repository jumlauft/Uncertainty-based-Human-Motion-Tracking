% Copyright (c) by Jonas Umlauft and Lukas Poehler (TUM) under BSD License 
% Last modified: Lukas Poehler 2018-09

clear; close all; clc;
addpath(genpath('gpml'));

%% Set parameters
ds = 5;               % Downsampling of training data

warning('off','MATLAB:scatteredInterpolant:DupPtsAvValuesWarnId');
warning('off','MATLAB:griddata:DuplicateDataPoints');

% GP learning
optGP.covfunc = [];
optGP.meanfun = @meanLinear;
optGP.likfunc = [];
optGP.sn_init = [];
optGP.optp = [];
optGP.Nrand = 1;   % number of random initializations
optGP.init_mean = [0.1;0.1;0.17]; % mean of initialization of lengthscales

% Var Lyapunov computation
Ngrid = 1e3;            % Number of gridpoints
gm = 5;                 % Grid margin outside of training points
Intsamp = 2;            % Sampling of integral between two simulation points
optSimLyap.dt = 0.1;    % Time step forward simulation
optSimLyap.tol = 2.5;     % Tolenrance for reaching origin
optSimLyap.maxN = 20e3; % Maximum number of simulation steps
optSimLyap.maxs = 1.5;    % Maximum step size
optSimLyap.mins = 0.2;    % Minimum step size
optSimLyap.minh = 0.1;  % Lower bound for stepsize decrease
rho = 5;                % CLF descent in degree pointing in

optLL.minP = 1e-5;   % Lower bound for EV of learned matrices
optLL.maxP = 1e8;    % Upper bound for EV of learned matrices
optLL.dSOS = 2;      % Degree of Sum of Squares
optLL.opt = optimoptions('fmincon','Display','off','GradObj','on',...
    'CheckGradients',false,'MaxFunctionEvaluations',1e8,'MaxIterations',1e3,...
    'SpecifyConstraintGradient',false);

% Simulation
rand_init = 1; % random starting point if ==1
radi_max = 3; % max radius of circle for varying starting point
optSimtraj = optSimLyap;
kc = 1;             % Scaling of gradient
eps = 1e-4;         % distance for gradient computation

% Visualization
Nte = 1e3;              % Number of testpoints in the surf grid
gmte = 5;              % Grid margin outside of training points



%% A --- Preprocessing
close all; rng default


% Get Training data
demos = load(['Data', '.mat'],'demos');  demos=demos.demos;
Ndemo = length(demos); Xtr= []; Ytr= []; x_train = [];
for ndemo = 1:Ndemo
    Xtrtemp = demos{ndemo}(:,1:ds:end); Xtr = [Xtr Xtrtemp(:,1:end-1)];
    Ytr = [Ytr Xtrtemp(:,2:end)];   % x_train(:,:,ndemo) = Xtrtemp;
end
dXtr = Ytr-Xtr;
Ytr(:,end+1) = [0,0]; Xtr(:,end+1) = [0,0]; dXtr(:,end+1) = [0,0]; % impose equilibrium point
[E, Ntr] = size(Xtr);
x0 = cell2mat(cellfun(@(v) v(:,1), demos,'UniformOutput', false));


% Generate deviating starting points for the reproduction if needed
if rand_init
    for i = 1:size(x0,2)
        radi = rand()*radi_max^2;
        beta = rand()*2*pi;
        x0(:,i) = x0(:,i) + [sqrt(radi)*cos(beta); sqrt(radi)*sin(beta)];
    end
end
Ntraj = size(x0,2);

% Define and build variance grid
Ndgrid = floor(nthroot(Ngrid,E)); %Ngrid = Ndgrid^E;
Xgrid = ndgridj(min(Xtr,[],2)-gm,max(Xtr,[],2)+gm,Ndgrid*ones(E,1));
Xgrid1 = reshape(Xgrid(1,:),Ndgrid,Ndgrid); Xgrid2 = reshape(Xgrid(2,:),Ndgrid,Ndgrid);



%% B ---  Learn GPSSM from training data
disp(['Learn GPSSM...'])
% using GPML
[~,~,hyp] = learnGPs(Xtr,dXtr,optGP);

[GPSSMm,varfun,gprMdls, psn2] = learnGPR(Xtr,dXtr,...
    'FitMethod','none','OptimizeHyperparameters','none',...
    'ConstantSigma',true,'Sigma',exp([hyp.lik]'),...
    'KernelFunction','ardsquaredexponential','KernelParameters',exp([hyp.cov]), ...
    'BasisFunction','linear','Beta',[hyp.mean]);


%% C ---  Run Vvar approach
% Learn CLF
disp(['Stabilize GPSSM...'])
[P_SOS, val_SOS] = learnSOS(Xtr,Ytr,optLL);
Vclf = @(x) SOS(x,P_SOS,optLL.dSOS);

% Learn stable dynamical system from CLF and GPSSM
dxdtfun = @(x) stableDS(x,GPSSMm,Vclf,rho);

% Evaluate Mean Trajectories on Grid Points
disp(['Simulate Mean Trajectories...'])
Xsim = simj(dxdtfun,Xgrid,optSimLyap,Vclf);

% Integrate Variance and scatter to grid
disp(['Compute Uncertainty Based Lyapunov Function....'])
logVvar = LyapVar(Xsim,varfun,Intsamp, psn2);
Vvar = scatteredInterpolant(reshape(Xsim,E,[])',exp(logVvar(:)),'linear');

% Generate Trajectories UCLD
disp(['Reproduce Paths...'])
Xtraj = simj(@(x) -kc*gradestj(@(xi) Vvar(xi'),x,eps),x0,optSimtraj,@(xi)Vvar(xi'),dxdtfun);



%% D --- Visualize
dss = 2; % density of stream slice

Ndte = floor(nthroot(Nte,E)); % Nte = Ndte^E;
Xte = ndgridj(min(Xtr,[],2)-gmte, max(Xtr,[],2)+gmte,Ndte*ones(E,1)) ;
Xte1 = reshape(Xte(1,:),Ndte,Ndte); Xte2 = reshape(Xte(2,:),Ndte,Ndte);


%% Stabilized GPSSM + GPvar
figure; hold on; axis tight;
title('Stabilized GPSSM + GPvar')
surf(Xte1,Xte2,reshape(sqrt(sum(varfun(Xte).^2,1))-1e4,Ndte,Ndte),'EdgeColor','none','FaceColor','interp');

Xte_vec = dxdtfun(Xte);
streamslice(Xte1,Xte2,reshape(Xte_vec(1,:),Ndte,Ndte),reshape(Xte_vec(2,:),Ndte,Ndte),dss);
quiver(Xtr(1,:),Xtr(2,:),dXtr(1,:),dXtr(2,:),'k','AutoScale','off');


%% UCLD approach

% plot Trajectories
figure; hold on; axis tight
title('Vvar + Trajectories')
surf(Xte1,Xte2,reshape(min(Vvar(Xte'),8e3),Ndte,Ndte)-1e4,'EdgeColor','none','FaceColor','interp');

if isnumeric(Xtraj)
    plot(squeeze(Xtraj(1,:,:)),squeeze(Xtraj(2,:,:)),'r');
else
    for i=1:length(Xtraj)
        plot(Xtraj{i}(:,1),Xtraj{i}(:,2),'r'); 
    end
end
quiver(Xtr(1,:),Xtr(2,:),dXtr(1,:),dXtr(2,:),'k','AutoScale','off');
