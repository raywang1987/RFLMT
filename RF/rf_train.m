%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Random Forest with Linear Model Tree -- @rf_train
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
function forest = rf_train(x,y,opts)

if sum(sum(isnan(x))) > 0 || sum(isnan(y)) > 0
    error('NaN is not permitted');
end

N=opts.N;
Nt=opts.Nt;
trees=cell(Nt,1);

parfor j=1:Nt
    % draw a bootstrap sample for growing a tree
    bsi=ceil(N*rand(N,1));
    xb=x(bsi,:); yb = y(bsi,:);
    % grow the regression tree
    trees{j}=tr_grow(xb,yb,opts);
    fprintf('Tree %4d (%d) is trained.\n',j, Nt);
end

forest.opts=opts;
forest.trees=cell(Nt,1);
for j=1:Nt
    forest.trees{j}=trees{j};
end
end