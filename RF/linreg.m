%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Description: linear regression 
% Author: Ruigang Wang 
% Email: ray.1987.wang@gmail.com
%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
%% (y-ym)=(x-xm)B+Residual, SSE=Residual^2
function [B,SSE]=linreg(x,xm,y,ym,B0)
[m,p]=size(B0);
G0=zeros(m,p); G1=zeros(m,m); x=x-xm; y=y-ym; 
for k=1:size(x,1)
    G0=G0-2*x(k,:)'*y(k,:);
    G1=G1+2*x(k,:)'*x(k,:);
end
rho=0.9; rms_eps=1E-10; eta=0.2;
Tmax=500; Ts=10;
% Tmin=30; Tmax=2000;
B=B0; EG=0; 
for t=1:Ts
    Gt=G0+G1*B;
    EG=rho*EG+(1-rho)*sum(sum(Gt.^2));
    B=B-eta/sqrt(EG+rms_eps)*Gt;
end
R=y-x*B; SSE=sum(sum(R.^2)); SSE_1=SSE;
while (t <= Tmax)
    Gt=G0+G1*B;
    EG=rho*EG+(1-rho)*sum(sum(Gt.^2));
    dB=eta/sqrt(EG+rms_eps)*Gt;
    B=B-dB;
    t=t+1;
    if mod(t,Ts) == 0
        R=y-x*B; SSE=sum(sum(R.^2));
        dS=sqrt(abs(SSE_1-SSE));
        rS=dS/sqrt(SSE+rms_eps);
        rb=sum(sum(dB.^2))/(m*p);
        if dS <= 1E-3 || rS <= 5E-2 || rb <= 1E-4
            break;
        end
        SSE_1=SSE;
    end
end
end