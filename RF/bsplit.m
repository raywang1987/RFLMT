%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Random Forest with Linear Model Tree -- @bsplit
%------------------------------------------------------------------------- 
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
%
% node data structure:
%   dex   -- index set
%   cnt   -- counter of data points
%   B     -- linear model on this node
%   sse   -- sum of square error
%   sum_x -- sum of xp vectors
%   sum_y -- sum of y vectors
%-------------------------------------------------------------------------
% Author: Ruigang Wang 
% Email: ray.1987.wang@gmail.com
%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
function [msplit,ubest,lnd,rnd]=bsplit(x,y,opts,parent)

msplit=0; ubest=0; 
L=opts.L;
Mp=opts.Mp;
Ms=opts.Ms;
Mpt=opts.Mpt;
Mst=opts.Mst;
Ns=opts.Ns;

rnd=parent; 
lnd.cnt=0; lnd.B=zeros(Mp,L); lnd.sse=0; 
lnd.sum_x=zeros(1,Mp); lnd.ssum_y=zeros(1,L);

critParent=parent.sse;
if Mst > 0
    mind=[randsample(Mp,Mpt),Mp+randsample(Ms,Mst)];
else
    mind=randsample(Mp,Mpt);
end
critmin=0;
pcnt=parent.cnt;

for i=1:length(mind)
    kv = mind(i);
    sdex = qsort(x,parent.dex,kv);
    if x(sdex(1),kv) >= x(sdex(pcnt),kv)
        continue;
    end
    lB=parent.B; rB=parent.B;
    lsx=zeros(1,Mp); lsy=zeros(1,L);
    rsx=parent.sum_x; rsy=parent.sum_y;
    for j=1:Ns-1
        lsx=lsx+x(sdex(j),1:Mp); lsy=lsy+y(sdex(j),:);
        rsx=rsx-x(sdex(j),1:Mp); rsy=rsy-y(sdex(j),:);
    end
    % start from the left and search to the right
    for j=Ns:pcnt-Ns
        lsx=lsx+x(sdex(j),1:Mp); lsy=lsy+y(sdex(j),:);
        rsx=rsx-x(sdex(j),1:Mp); rsy=rsy-y(sdex(j),:);
        if x(sdex(j),kv) < x(sdex(j+1),kv) 
            % Search through the "gaps" in the x-variable.
            [lB,lsse]=linreg(x(sdex(1:j),1:Mp),lsx/j,y(sdex(1:j),:),lsy/j,lB);
            [rB,rsse]=linreg(x(sdex(j+1:pcnt),1:Mp),rsx/(pcnt-j),...
                             y(sdex(j+1:pcnt),:),rsy/(pcnt-j),rB);
            crit = lsse+rsse-critParent;
            if crit < critmin 
                % find a better split
                critmin=crit; 
                msplit=kv;
                ubest=(x(sdex(j),kv)+x(sdex(j+1),kv))/2;
                lnd.dex=sdex(1:j); rnd.dex=sdex(j+1:pcnt);
                lnd.sse=lsse; rnd.sse=rsse;
                lnd.cnt=j; rnd.cnt=pcnt-j;
                lnd.B=lB; rnd.B=rB;
                lnd.sum_x=lsx; lnd.sum_y=lsy; rnd.sum_x=rsx; rnd.sum_y=rsy;
            end
        end
    end
end
end