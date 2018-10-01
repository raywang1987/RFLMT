%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Random Forest with Linear Model Tree -- @rf_model
%-------------------------------------------------------------------------
% It aims to learn a model: Y=f([Xp,Xa]) from the dataset ([xp,xs],y) where
% Xp -- 1*Mp predict vector 
% Xs -- 1*Ms spliting vector 
% Y  -- 1*L response vector 
% xp -- N*Mp data matrix
% xs -- N*Ms data matrix
% y  -- N*L data matrix
% 
% leaf linear model: Y=(Xp-Xph)B+Yph
% agragated linear model: Y=Xp*Be+Ype where 
% Be=1/ntree*sum(Bi), Ype=1/ntree*sum(Yphi-Xphi*Bi)
%
% opts:
% N  -- number of rows in x,y
% L  -- number of columns in y
% Mp -- number of columns in xp
% Ms -- number of columns in xs
% Nt -- number of trees
% Msp-- number of try variables in predict vector (Msp <= Mp)
% Mst-- number of try variables in split vector (Mst <= Ms)
% Ns -- minimal data points in a leaf (Ns >= 2*Mp+1)
%-------------------------------------------------------------------------
% Author: Ruigang Wang 
% Email: ray.1987.wang@gmail.com
%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
function [y0,B] = rf_model(x,forest)
opts=forest.opts; M=opts.Mp+opts.Ms; Nt=opts.Nt;
[n,m]=size(x);
if n > 1 || m ~= M
    error('tr_model: dim error, %d*%d ~= 1*%d',n,m,M);
end
y0=zeros(1,opts.L); B=zeros(opts.Mp,opts.L);
for i=1:Nt
    [tB,tmx,tmy]=tr_model(x,forest.trees{i});
    B=B+tB; y0=y0+tmy-tmx*tB;
end
B=B/Nt;
y0=y0/Nt;
end