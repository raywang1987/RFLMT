%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Random Forest with Linear Model Tree -- @tr_model
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
function [B,mx,my] = tr_model(x,tree)
NODE_TERMINAL = -1;
k = 1;
while tree.nodestatus(k) ~= NODE_TERMINAL
    if x(tree.splitVar(k)) <= tree.split(k)
        k = tree.lDaughter(k);
    else
        k = tree.rDaughter(k);
    end
end
B=tree.B{k}; mx=tree.mx{k}; my=tree.my{k};
end