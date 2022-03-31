selected_features=[];
AUCs=[];

load("ad_data.mat");

for par  = [0,0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    [w,c]=logistic_l1_train(X_train,y_train,par);
    pred=X_test*w+c;
    
    num_zeros = sum(w(:)==0);
    numberSelectedFeature=size(w,1)-num_zeros;
    
    [X,Y,T,AUC] = perfcurve(y_test,pred,1);
    selected_features=[selected_features numberSelectedFeature];
    AUCs=[AUCs AUC];
end   

disp(AUCs);
disp(selected_features);
