% data generation
L=1000;
x=rand(L,2); y=0.1*rand(L,1);
for k=1:L
    if x(k,1)+x(k,2) <= 0.5
        y(k)=y(k)+2*(0.5-x(k,1)-x(k,2));
    end
    if x(k,1)+0.5*x(k,2) >= 1
        y(k)=y(k)+2*(x(k,1)+0.5*x(k,2)-1);
    end
end
% random forest training and prediction
opts.N=L;
opts.Nt=20;
opts.L=1;
opts.Mp=2;
opts.Mpt=2;
opts.Ms=0;
opts.Mst=0;
opts.Ns=8;
rf=rf_train(x,y,opts);
yp=rf_eval(x,rf);
% plot
scatter3(x(:,1),x(:,2),y); hold on
scatter3(x(:,1),x(:,2),yp);