%+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% Random Forest with Linear Model Tree -- @tr_grow
%-------------------------------------------------------------------------
% leaf linear model: Y=(Xp-Xph)B+Yph
% agragated linear model: Y=Xp*Be+Ype where 
% Be=1/ntree*sum(Bi), Ype=1/ntree*sum(Yphi-Xphi*Bi)
%
% opts:
% N  -- number of rows in x,y
% L  -- number of columns in y
% Ms -- number of columns in xs
% Mp -- number of columns in xp
% Nt -- number of trees
% Mst-- number of try variables in split vector (Mst <= Ms)
% Msp-- number of try variables in predict vector (Msp <= Mp)
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
function tree = tr_grow(x,y,opts)

N=opts.N;
Mp=opts.Mp;
Ns=opts.Ns;
L=opts.L;

% allocate memory
Nd=2*ceil(N/max(1,Ns))+1;
nodestatus=zeros(Nd,1);
NODE_TERMINAL=-1; NODE_TOSPLIT=-2; NODE_INTERIOR=-3;
lDaughter=zeros(Nd,1);
rDaughter=zeros(Nd,1);
mbest=zeros(Nd,1);
upper=zeros(Nd,1);
node=cell(Nd,1);

% initialize the root node
ncur=1;
node{1}.cnt=N; node{1}.dex=1:N;
node{1}.sum_x=sum(x(:,1:Mp)); node{1}.sum_y=sum(y);
[node{1}.B,node{1}.sse]=linreg(x(:,1:Mp),node{1}.sum_x/N,y,node{1}.sum_y/N,rand(Mp,L));
nodestatus(1)=NODE_TOSPLIT;
    
% start main loop
for k=1:Nd-2
    if k > ncur || ncur >= Nd-2
        break;
    end
    % skip if the node is not to be split
    if nodestatus(k) ~= NODE_TOSPLIT
        continue;
    end
 
    [msplit,ubest,lnd,rnd]=bsplit(x,y,opts,node{k});
    
    if msplit == 0 
        % node is terminal: mark it as such and move on to the next
        nodestatus(k)=NODE_TERMINAL;
        continue;
    end
    % found the best split
    mbest(k)=msplit;
    upper(k)=ubest;
    nodestatus(k)=NODE_INTERIOR;
    
    % left node no.= ncur+1, rightnode no. = ncur+2
    node{ncur+1}=lnd; node{ncur+2}=rnd;
    nodestatus(ncur+1)=NODE_TOSPLIT; nodestatus(ncur+2)=NODE_TOSPLIT;
    
    if node{ncur+1}.cnt <= Ns
        nodestatus(ncur+1)=NODE_TERMINAL;
    end
    if node{ncur+2}.cnt <= Ns
        nodestatus(ncur+2)=NODE_TERMINAL;
    end
    
    % map the daughter nodes
    lDaughter(k)=ncur+1;
    rDaughter(k)=ncur+2;
    % augment the tree by two nodes
    ncur=ncur+2;
    
end

treeSize=Nd;
for k=Nd:-1:1
    if nodestatus(k) == 0
        treeSize=treeSize-1;
    end
    if nodestatus(k) == NODE_TOSPLIT
        nodestatus(k)=NODE_TERMINAL;
    end
end

tree.treeSize=treeSize;
tree.lDaughter=lDaughter(1:treeSize);
tree.rDaughter=rDaughter(1:treeSize);
tree.B=cell(treeSize,1);
tree.mx=cell(treeSize,1);
tree.my=cell(treeSize,1);
tree.sse=zeros(treeSize,1);
for k=1:treeSize
    tree.B{k}=node{k}.B;
    tree.mx{k}=node{k,1}.sum_x/node{k}.cnt;
    tree.my{k}=node{k,1}.sum_y/node{k}.cnt;
    tree.sse(k)=node{k,1}.sse;
end
tree.nodestatus=nodestatus(1:treeSize);
tree.splitVar=mbest(1:treeSize);
tree.split=upper(1:treeSize);
end