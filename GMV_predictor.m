% ----------------------------------------------------------------------- %
%                      G M V     P R E D I C T O R                        %
% ----------------------------------------------------------------------- %
% This script predicts depressive symptoms of Achenbach questionnaire     %
% based on the 473 features of brains' gray matter volume (GMV) and       %
% combination of sleep quality and GMV features.                          %
%                                                                         %
%   Input parameters:                                                     %
%       - vol:          473 GMV features.                                 %
%       - sleep:        Sleep quality parameters.                         %
%       - depression:   Achenbach questionnaire scores.                   %
%       - family:       Family IDs for considering family structure to    %
%                       avoid siblings to be seperated in training/test   %
%                       sets.                                             %
%       - confounding:  Age,gender, ant tGMV considered as confound       %
%                       variables.                                        %
%                                                                         %
%   Output variables:                                                     %
%       - newmodel:     Models that were creared by machine learning      %
%                       algorithm for individual prediction.              %
%       - YHat:         Predicted values.                                 %
%       - ls1:          Loss-function of prediction in test sets based on %
%                       minimum mean squared error (MSE).                 %
%       - R2:           Coefficient of determination.                     %
%       - perf:         Mean absolute error.                              %
%       - ci:           95% confidence interval.                          %
%       - MSE:          Mean squared error.                               %
% ----------------------------------------------------------------------- %
%   Script information:                                                   %
%       - Version:      1.2.                                              %
%       - Author:       Mahnaz Olfati                                     %
%       - Date:         25/12/2021                                        %
% ----------------------------------------------------------------------- %
% Read data
clc,clear,close all

sleep                  % Put sleep variable here
depression             % Put depression varible here 
confounding            % Put confounding variables here
family                 % Put family ID here
anxi                   % Put anxiety variable here
vol                    % Put GMV or fALFF or ReHo variable here

%% Predictor and target variables
x = vol;               % Put predictors here 
y = depression;        % Put target here

%% Nested 10-fold cross-validation considering the family structure
kfold_o = 10;          % Number of outer folds
kfold_i = 10;          % NUmber of inner folds
[test_idx,train_outer_idx,train_inner_idx,validation_idx] = NestedCV(y,kfold_o,kfold_i,family);

%% Training models on inner training folds, saving the best models, fitting selected models on the entire training sets and prediction of test sets
feature_set = 10;     % Number of feature sets
conf_mdl = cell(kfold_o,size(x,2));
for outer = 1:kfold_o           % Outer folds

    y_test = y(test_idx{outer},1); 
    y_train = y(train_outer_idx{outer},1);
    x_test = x(test_idx{outer},:);
    x_train = x(train_outer_idx{outer},:);
    conftrain_outer = confounding(train_outer_idx{outer},:);
    conftest = confounding(test_idx{outer},:);
   
    % creation of confound removal models (conf_mdl) on training sets and applying them on test sets
    [x_train,x_test,conf_mdl] = Confound_Remove_model(outer,x_train,x_test,conftrain_outer,conftest,conf_mdl);
    
    % Ranking features in outer training folds
    rng default
    [ranks{outer},weights{outer}] = relieff(x_train,y_train,10);

    % Training models for each inner fold
    for inner = 1:kfold_i       % Inner folds
        
        % Repeating training process for 10 different feature number
        parfor loop = 1:feature_set     % Number of feature-set
        
           feature_num{loop} = 10+10*loop;      % Number of features are included 20,30,40,50,60,70,80,90,100,110 (loop = 10 sets)
            x_trainn = x(train_inner_idx{outer,inner},ranks{outer}(1:feature_num{loop}));
            x_validation = x(validation_idx{outer,inner},ranks{outer}(1:feature_num{loop}));
            y_trainn = y(train_inner_idx{outer,inner},1);
            y_validation = y(validation_idx{outer,inner},1);
            conftrain_inner = confounding(train_inner_idx{outer,inner},:);
            conf_validation = confounding(validation_idx{outer,inner},:);
            
            % regressing out the cnfound variables from inner training and validation predictors
            [x_trainn,x_validation] = RegressOut(outer,x_trainn,conftrain_inner,x_validation,conf_validation,conf_mdl,ranks,feature_num{loop})
            
            % Training models on inner training folds
            Mdl1{inner,loop}= Model(x_trainn,y_trainn)
            
            % Assessment of loss-function of validation sets to select optimum feature number and model with best performance
            ls{outer,inner,loop} = loss(Mdl1{inner,loop},x_validation,y_validation);
        end
        
    end
    
    % Selection of best models for each outer fold based on minimum loss of ineer folds
    v = [ls{outer,:,:}];
    [lost{outer},best{outer}] = min(v);
    [mj{outer},mk{outer}] = ind2sub([kfold_i,feature_set],best{outer});
    bestmodel{outer} = Mdl1{mj{outer},mk{outer}}; 
    
    % Fitting selected models on the entire training sets and prediction of test sets
    [newmodel{outer}, ls1{outer},Yhat1{outer}] = fitmodel(bestmodel{outer},x_train,y_train,x_test,y_test,ranks{outer}(1:feature_num{mk{outer}}));

end

%% Predicted values
YHat(cell2mat(test_idx')) = cell2mat(Yhat1');
YHat = YHat';

%% Plotting
reg = [YHat,y];
figure;
[R,PValue] = corrplot(reg,'testR','on')

R2 = 1 - sum((y - YHat) .^ 2) / sum((y - mean(y)) .^ 2);

e = YHat-y;
perf = mae(e);

% 95% confidence interval
ts = tinv([0.025 0.975],length(YHat)-1);
sem = std(YHat)/sqrt(length(YHat));
ci = mean(YHat)+ts*sem

MSE = mean(cell2mat(ls1))

%% Storage of models for out of cohort validation
save('model_GMV_conf.mat','newmodel','ranks','conf_mdl')
