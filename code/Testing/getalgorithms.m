% Get standard list of algorithms to test.
function [algorithms,algorithm_params,names,colors,order] = ...
    getalgorithms(use_dtc,use_ftc,use_dtree,use_learch,use_mmp,use_me,use_cont)

algorithms = {'gpirl',...
              'gpirl',...
              'maxent',...
              'maxent',...
              'maxent',...
              'mwal',...
              'an',...
              'mmp',...
              'mmpboost',...
              'learch',...
              'learch',...
              'firl'};
names = {'GPIRL (all states)',...
         'GPIRL',...
         'MaxEnt True',...
         'MaxEnt',...
         'MaxEnt Laplace',...
         'MWAL',...
         'Abbeel & Ng',...
         'MMP',...
         'MMPBoost',...
         'LEARCH',...
         'LEARCH Non-Linear',...
         'FIRL'};
algorithm_params = {...
    struct('inducing_pts','all'),...
    struct('inducing_pts','examplesplus'),...
    struct('all_features',0,'true_features',1),...
    struct('all_features',1),...
    struct('all_features',1,'laplace_prior',1),...
    struct('all_features',1),...
    struct('all_features',1),...
    struct('all_features',1),...
    struct(),...
    struct(),...
    struct('function','linear,dtree'),...
    struct()};
colors = {...
    [0.0,0.0,0.0],...
    [0.0,0.0,0.5],...
    [1.0,0.5,0.5],...
    [0.9,0.1,0.1],...
    [0.6,0.0,0.0],...
    [0.0,1.0,0.0],...
    [0.7,0.5,0.1],...
    [0.0,0.5,0.0],...
    [0.2,0.9,0.9],...
    [0.6,0.9,0.4],...
    [0.5,0.6,0.2],...
    [0.4,0.6,0.9]};

% Set continuous parameters.
if use_cont,
    algorithm_params{1}.warp_x = 1;
    algorithm_params{2}.warp_x = 1;
    algorithm_params{11}.function = 'linear,logistic';
end;

% Remove decision tree entries.
if ~use_dtree,
    if use_learch,
        if use_cont,
            ke = 1;
        else
            ke = 2;
        end;
    else
        ke = 3;
    end;
    % Remove FIRL & non-linear LEARCH.
    for k=1:ke,
        colors(end) = [];
        algorithm_params(end) = [];
        names(end) = [];
        algorithms(end) = [];
    end;
    
    % Remove MMPBoost (#7).
    idx = 9;
    colors(idx) = [];
    algorithm_params(idx) = [];
    names(idx) = [];
    algorithms(idx) = [];
elseif ~use_learch,
    % Remove learch only.
    idx = length(colors)-3;
    colors(idx:idx+1) = [];
    algorithm_params(idx:idx+1) = [];
    names(idx:idx+1) = [];
    algorithms(idx:idx+1) = [];
end;

if ~use_mmp,
    % Remove MMP and A&N.
    if use_dtree,
        idx = 7:9;
    else
        idx = 7:8;
    end;
    colors(idx) = [];
    algorithm_params(idx) = [];
    names(idx) = [];
    algorithms(idx) = [];
end;

if ~use_me,
    % Remove MaxEnt (except for Laplace).
    idx = 3:4;
    colors(idx) = [];
    algorithm_params(idx) = [];
    names(idx) = [];
    algorithms(idx) = [];
end;

% Remove unwanted GPIRL entries.
if ~use_ftc,
    algorithms(1) = [];
    names(1) = [];
    algorithm_params(1) = [];
    colors(1) = [];
end;
if ~use_dtc,
    i1 = 2;
    i2 = 2;
    if ~use_ftc,
        i1 = 1;
        i2 = 1;
    end;
    algorithms(i1:i2) = [];
    names(i1:i2) = [];
    algorithm_params(i1:i2) = [];
    colors(i1:i2) = [];
end;

% Create order.
order = 1:length(colors);