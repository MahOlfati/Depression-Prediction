function [newmodel, ls1,Yhat1] = fitmodel(bestmodel,x_train,y_train,x_test,y_test,ranks)

rng default
    tmp = classreg.learning.FitTemplate.makeFromModelParams(bestmodel.ModelParameters);
    newmodel = fit(tmp,x_train(:,ranks),y_train);
    ls1 = loss(newmodel,x_test(:,ranks),y_test);
    Yhat1 = predict(newmodel,x_test(:,ranks));

end