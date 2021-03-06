%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Random Forest with Linear Model Tree -- @tr_pred
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
function y = tr_pred(x,tree,opts)
n=size(x,1);
y=zeros(n,opts.L); Mp=opts.Mp;
NODE_TERMINAL=-1;
for i = 1:n
    k=1;
    while tree.nodestatus(k) ~= NODE_TERMINAL
        if x(i,tree.splitVar(k)) <= tree.split(k)
            k=tree.lDaughter(k);
        else
            k=tree.rDaughter(k);
        end
    end
    y(i,:)=tree.my{k}+(x(i,1:Mp)-tree.mx{k})*tree.B{k};
end
end