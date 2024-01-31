% ----------------------------------------------------------------------- %
%            R E P L I C A T I O N     P R E D I C T I O N                %
% ----------------------------------------------------------------------- %
% This script predicts depressive symptoms of Achenbach questionnaire     % 
% in out of sample dataset using pre-trained models on HCP dataset.       %
%                                                                         %
%   Input parameters:                                                     %
%       - pre_trained:      Pre-trained model contains models and their   %
%                           features' ranks.                              %
%       - vol:              473 GMV features.                             %
%       - depression:       Achenbach questionnaire scores.               %
%       - sleep:            Sleep quality parameters.                     %
%       - confounding:      Age,gender, ant tGMV considered as confound   %
%                           variables.                                    %
%                                                                         %
%   Output variables:                                                     %
%       - YHat_nki:         Predicted values.                             %
%       - R2:               Coefficient of determination.                 %
%       - perf:             Mean absolute error.                          %
%       - ci:               95% confidence interval.                      %
%       - MSE:              Mean squared error.                           %
% ----------------------------------------------------------------------- %
%   Script information:                                                   %
%       - Version:          1.2.                                          %
%       - Author:           Mahnaz Olfati                                 %
%       - Date:             21/02/2022                                    %
% ----------------------------------------------------------------------- %
% Read data
clear,close,clc

pre_trained = load('model_sleep_conf.mat');        % model_sleep_*: for sleep, model_GMV_*: for GMV, model_sleepGMV_*: for combination

% Initializing parameters
data = xlsread('r_nki.csv');
% BDI_nki = data_nki(:,2);
sleep = data(:,3:22);
depression = data(:,23:37);
confounding = data(:,38:40);
vol = data(:,41:end);
% anx = xlsread('anxiety.csv');
% anx_nki = anx(:,2);
%% Predictor and target values
x = [sleep];            % sleep_nki: for sleep_model, vol_nki: for vol_model, [sleep_nki vol_nki]: for combination  
target = depression(:,1);
%% Mean prediction of new data based on all pre-trained models

feature_number = zeros(1,size(pre_trained.newmodel,2));
yhat_nki = zeros(size(target,1),size(pre_trained.newmodel,2));
loss_nki = zeros(1,size(pre_trained.newmodel,2));
 
% prediction of target based on all pre-trained models
for model = 1:size(pre_trained.newmodel,2)
    
    % confound removal based on pre-trained models
    predictor = zeros(size(x));
    for col = 1:size(x,2)
        predictor(:,col) = x(:,col) - predict(pre_trained.conf_mdl{model,col},confounding);
    end
    
    % prediction of targets based on pre-trained models
    feature_number(1,model) = size(pre_trained.newmodel{model}.X,2);
    yhat_nki(:,model) = predict(pre_trained.newmodel{model},predictor(:,pre_trained.ranks{model}(1:feature_number(1,model))));
    loss_nki(1,model) = loss(pre_trained.newmodel{model},predictor(:,pre_trained.ranks{model}(1:feature_number(1,model))),target);

end

% mean prediction
YHat_nki = mean(yhat_nki,2);

%% Plotting

reg = [YHat_nki,target];
figure;
[R,PValue] = corrplot(reg,'testR','on')

R2 = 1 - sum((target - YHat_nki) .^ 2) / sum((target - mean(target)) .^ 2);

% mdl1 = fitlm(target,YHat_nki);
% figure
% p = plot(mdl1);
% xlabel('Real depressive score');ylabel('Predicted depressive score');
% title('Prediction based on sleep quality');
% p(1).Marker = 'o';p(1).Color = 'black';

e = YHat_nki-target;
perf = mae(e);

% 95% confidence interval
ts = tinv([0.025 0.975],length(YHat_nki)-1);
sem = std(YHat_nki)/sqrt(length(YHat_nki));
ci = mean(YHat_nki)+ts*sem

MSE = mean(loss_nki)
%% Saving results
save('sleep_hcpaging.mat')