function Mdl1= ModelG(i,j,x_trainn,y_trainn)
rng default
Mdl1{i,j} = fitrensemble(x_trainn,y_trainn,'OptimizeHyperparameters','auto','Resample','on',...
'HyperparameterOptimizationOptions',struct('Optimizer','bayesopt','ShowPlots',false,'UseParallel',false,'AcquisitionFunctionName',...
'expected-improvement-plus','MaxObjectiveEvaluations',100,'Repartition',true));

end