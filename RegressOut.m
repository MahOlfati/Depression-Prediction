function [x_trainn,x_validation] = RegressOut(h,x_trainn,conftrain_inner,x_validation,conf_validation,conf_mdl,ranks,feature_num)

    for culmn = 1:feature_num
            x_trainn(:,culmn) = x_trainn(:,culmn) - predict(conf_mdl{h,ranks{h}(culmn)},conftrain_inner);
            x_validation(:,culmn) = x_validation(:,culmn) - predict(conf_mdl{h,ranks{h}(culmn)},conf_validation);
    end
    
end