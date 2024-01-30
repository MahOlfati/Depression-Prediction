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

data = readtable('r_without_sleepydep_ReHo.csv');
data = table2array(data);
sleep = data(:,2:21);
depression = data(:,23:37);
confounding = data(:,38:40);
family = data(:,41);
vol = readtable('ReHo.csv');
vol = table2array(vol);
%% Predictor and target values
x = vol;            % x = [sleep vol] : for prediction based on combination of GMV and sleep quality
y = depression(:,1);

%% Family IDs
[c,ia,ic] = unique(family,'stable');

%% Nested 10-fold cross-validation considering the family structure
[test_idx,train_outer_idx,train_inner_idx,validation_idx] = NestedCV(y,family)

%% Training models, saving the best models, and fitting selected models on outer folds

conf_mdl = cell(kf,size(x,2));
for h = 1:kf

    y_test = y(test_idx{h},1); 
    y_train = y(train_outer_idx{h},1);
    x_test = x(test_idx{h},:);
    x_train = x(train_outer_idx{h},:);
    conftrain_outer = confounding(train_outer_idx{h},:);
    conftest = confounding(test_idx{h},:);
   
    % creating confound removal models (conf_mdl) on training sets and applying them on test sets
        [x_train,x_test,conf_mdl] = Confound_Remove_model(h,x_train,x_test,conftrain_outer,conftest);

    
    % Ranking features in outer training folds
    rng default
    [ranks{h},weights{h}] = relieff(x_train,y_train,10);

    % Training models for each inner folds
    for i = 1:kf
        
        % Repeating training process for 10 different feature number
        parfor j = 1:kf
            feature_num{h,i,j} = 10+10*j;
            x_trainn = x(train_inner_idx{h,i},ranks{h}(1:feature_num{h,i,j}));
            x_validation = x(validation_idx{h,i},ranks{h}(1:feature_num{h,i,j}));
            y_trainn = y(train_inner_idx{h,i},1);
            y_validation = y(validation_idx{h,i},1);
            conftrain_inner = confounding(train_inner_idx{h,i},:);
            conf_validation = confounding(validation_idx{h,i},:);
            
            % regressing out the cnfound variables from inner-training/validation predictors
            [x_trainn,x_validation] = RegressOut(h,x_trainn,conftrain_inner,x_validation,conf_validation,conf_mdl,ranks,feature_num)
            
            % Training models on inner training folds
            Mdl1= ModelG(i,j,x_trainn,y_trainn)
            % Calculation of loss-function of validation sets
            ls{h,i,j} = loss(Mdl1{i,j},x_validation,y_validation);
        end
        
    end
    
    % Selection of best models for each outer fold based on minimum loss of ineer folds
    
    v = [ls{h,:,:}];
    [lost{h},best{h}] = min(v);
    [mj{h},mk{h}] = ind2sub([i,j],best{h});
    bestmodel{h} = Mdl1{mj{h},mk{h}}; 
    
    % Fitting selected model on training set of outer fold
    rng default
    tmp{h} = classreg.learning.FitTemplate.makeFromModelParams(bestmodel{h}.ModelParameters);
    newmodel{h} = fit(tmp{h},x_train(:,ranks{h}(1:feature_num{h,mj{h},mk{h}})),y_train);
    ls1{h} = loss(newmodel{h},x_test(:,ranks{h}(1:feature_num{h,mj{h},mk{h}})),y_test);
    Yhat1{h} = predict(newmodel{h},x_test(:,ranks{h}(1:feature_num{h,mj{h},mk{h}})));

end

%% Plotting
YHat(cell2mat(test_idx')) = cell2mat(Yhat1');
YHat = YHat';

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

%%
save('model_GMV_conf.mat','newmodel','ranks','conf_mdl')