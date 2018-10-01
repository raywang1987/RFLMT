%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Random Forest with Linear Model Tree -- @rf_eval
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
function ypred = rf_eval(x,forest)
[n,m]=size(x); 
opts=forest.opts;
M=opts.Mp+opts.Ms; Nt=opts.Nt;
if m ~= M
    error('tr_pred: col-dim mismatch, %d* ~= %d', m, M);
end
ypred = zeros(n,opts.L);
for i = 1:Nt
    ytree = tr_pred(x,forest.trees{i},opts);
    ypred = ypred+ytree;
end
ypred = ypred/Nt;
end