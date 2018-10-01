L=500; N=2*L; p1=-1.5; p2=1.5;
s1=normrnd(p1,0.05,L,1); t1=rand(L,1); v1=sin(2*pi*t1)+normrnd(0,0.05,L,1)+p1;
s2=normrnd(p2,0.05,L,1); t2=rand(L,1); v2=cos(2*pi*t2)+normrnd(0,0.05,L,1)+p2;
x=[t1,s1; t2,s2]; y=[v1; v2];
opts.N=N;
opts.Nt=50;
opts.L=1;
opts.Mp=1;
opts.Mpt=1;
opts.Ms=1;
opts.Mst=1;
opts.Ns=10;
rf=rf_train(x,y,opts);
yp=rf_eval(x,rf);
scatter(t1,v1); hold on
scatter(t2,v2);
scatter(x(1:L,1),yp(1:L)); 
scatter(x(L+1:N,1),yp(L+1:N));

xc1=0.6; l=0.15;
[y0,B]=rf_model([xc1,p1],rf);
xl=[xc1-l;xc1+l];
yl=xl*B+y0;
plot(xl,yl,'--','color','k','linewidth',2);

xc2=0.3; 
[y0,B]=rf_model([xc2,p2],rf);
xl=[xc2-l;xc2+l];
yl=xl*B+y0;
plot(xl,yl,'-.','color','k','linewidth',2);