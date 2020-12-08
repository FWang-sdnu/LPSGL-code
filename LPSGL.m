function [ W ] = LPSGL( X,Xl,Xu,Yl,A,param,MAX_ITER,islocal)
%2020.
%datasets: X:d*n,include Xl+Xu; Xl:d*l; Xu:d*u; Yl:l*c;
%A:n*n,pre-constructed similarity matrix
%param£ºalpha.lambda.v.c(the reduced dimension=cluster num)
%islocal: only update the similarities of neighbors if islocal=1
%W:d*c,the projection matrix

if nargin < 8
    islocal = 1;  %only update the similarities of neighbors
end;

[d,N] = size(X);
Nl = size(Xl,2);
c = size (Yl,2);
thresh = 1e-11;

%calculate V
Vll = param.v*ones(Nl,1);
Vl = diag(Vll);
Vuu = ones(N-Nl,1);
Vu = diag(Vuu);
V = blkdiag(Vl,Vu);    

%%initialization
S = A;
W = zeros(d,c);

%%optimization
for iter = 1:MAX_ITER
    %calculate Ls
    Ws = (S'+S)/2;
    Ds = diag(sum(Ws));
    Ls = Ds - Ws;
    Luu = Ls((Nl+1):end, (Nl+1):end);
    Lul = Ls((Nl+1):end, 1:Nl);
    
   %update F
    Fu1 = 2*param.lambda*Luu+param.alpha*Vu;
    Fu2 = param.alpha*Vu*Xu'*W-2*param.lambda*Lul*Yl;
    Fu = Fu1\Fu2;
    
   %Adding non-negative constraint 
    for ui = 1 : N-Nl
       for uj = 1:c
           if Fu(ui, uj) <= 0
               Fu(ui, uj) = 0;
           else
               if Fu(ui, uj) >= 1
                  Fu(ui, uj) = 1;
               else
                  Fu(ui, uj) = Fu(ui ,uj);
               end
           end
        end
    end
    F = [Yl;Fu];
   
    %update W
    W = (X*V*X')\X*V*F;

    %update S
        dist = L2_distance_1(F',F');
        for i=1:N
        a0 = A(i,:);  
        if islocal == 1   
            idxa0 = find(a0>0);
        else
            idxa0 = 1:num;
        end;
        ai = a0(idxa0);
        di = dist(i,idxa0);
        ad = ai-0.5*param.lambda*di;
        S(i,idxa0) = EProjSimplex_new(ad);
        end;
        
    %calculate obj
        obj1 = norm(S-A,'fro')^2;
        obj2 = 2*param.lambda*trace(F'*Ls*F);
        obj3 = param.alpha*trace((X'*W-F)'*V*(X'*W-F));
        obj(iter) = obj1 +obj2 + obj3;
        plot(obj);
        if iter>2 && ( obj(iter-1)-obj(iter) )/obj(iter-1) < thresh
             break;
        end
        fprintf('Iter %d\tobj=%f\n',iter,obj(end));
                       
end;   
end




