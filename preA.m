% construct similarity matrix with probabilistic k-nearest neighbors. It is a parameter free, distance consistent similarity.
function W = preA(X,Xl,F, k, issymmetric,P)
% X: each column is a data point
% k: number of neighbors
% issymmetric: set W = (W+W')/2 if issymmetric=1
% W: similarity matrix

if nargin < 5
    issymmetric = 1;
end;
if nargin < 4
    k = 5;
end;

Nl = size(F,1);
label = unique (F);
c =length(label);

[dim, n] = size(X);
D = L2_distance_1(X, X);
D2 =10* ones(Nl,Nl);

if P>1
   for iii = 1:c
       [id,~] = find (F == label(iii));
        for ii = 1:length(id)
           D2(id(ii),id) = D(id(ii),id);   
           D2(id(ii),id(ii))=0;
        end 
   end
end

D(1:Nl,1:Nl) = D2;
[dumb, idx] = sort(D, 2); 

W = zeros(n);
for i = 1:n
    id = idx(i,2:k+2);
    di = D(i, id);
    W(i,id) = (di(k+1)-di)/(k*di(k+1)-sum(di(1:k))+eps);
end;



if issymmetric == 1
    W = (W+W')/2;
end;




% compute squared Euclidean distance
% ||A-B||^2 = ||A||^2 + ||B||^2 - 2*A'*B
function d = L2_distance_1(a,b)
% a,b: two matrices. each column is a data
% d:   distance matrix of a and b



if (size(a,1) == 1)
  a = [a; zeros(1,size(a,2))]; 
  b = [b; zeros(1,size(b,2))]; 
end

aa=sum(a.*a); bb=sum(b.*b); ab=a'*b; 
d = repmat(aa',[1 size(bb,2)]) + repmat(bb,[size(aa,2) 1]) - 2*ab;

d = real(d);
d = max(d,0);







