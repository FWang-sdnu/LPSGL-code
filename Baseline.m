clear all;
clc;

%load('MSRA25_uni.mat');n =size(X,1);
%load('Coil20Data_25_uni');n =size(X,1);
%load('CNAE-9');n =size(X,1);
%load('ORL_32x32');n =size(X,1);
%load('USPSdata_20_uni');n =size(X,1);
%load('dig1-10_uni');Y = Y+1;n =size(X,1);

%% PCA
% options = [];
% options.PCARatio = 0.95;
% [eigvector,eigvalue,meanData,new_data] = PCA(X,options);
% X = new_data;

%% 处理数据集
label = unique (Y);
nlabel = histc (Y,label);
c =length(label);

Result = zeros (1,20);
for iter = 1 :20
%% 1.随机划分数据集
Xl = [];
Yl = [];
Xu = [];
Yu = [];
Xtest = [];
Ytest = [];

rowrank = randperm (size (X,1));
XX = X(rowrank,:);
YY = Y(rowrank,:);


for i = 1:c
    
    p = 1;                                            
    nn = ceil(nlabel(i)/2);

    [id,~] = find(YY==label(i));
    YYY = YY(id,:); 
    XXX = XX(id,:);
    if p == 1
       Xl = [Xl; XXX(1,:)];
       Yl = [Yl;YYY(1,:)];
       Xu = [Xu;XXX(p+1:nn,:)];
       Yu = [Yu;YYY(p+1:nn,:)];
    else
    Xl = [Xl; XXX(1:p,:)];
    Yl = [Yl;YYY(1:p,:)];
    Xu = [Xu;XXX(p+1:nn,:)];
    Yu = [Yu;YYY(p+1:nn,:)];
    end
    
    %%  2.test part 
    Xtest = [Xtest;XXX(nn+1:nlabel(i),:)];
    Ytest = [Ytest;YYY(nn+1:nlabel(i),:)];
            
end
      Xl=NormalizeFea(Xl);
      Xu = NormalizeFea(Xu);
      Xtest= NormalizeFea(Xtest);
      Xtrain = [Xl;Xu];
      Ytrain = [Yl;Yu];  
    

    Nl = size(Xl,1);
    Nu = size(Xu,1);
    NN = Nl+Nu;
    
    YY = zeros(NN,c);
    YYl = zeros(Nl,c);
    YYu = zeros(Nu,c);
    YYtrain = zeros(NN,c);
    YYtest = zeros(n-NN,c);
for k = 1:n
   YY(k,Y(k,1)) = 1;
end
for kk = 1:Nl
    YYl(kk,Yl(kk,1)) = 1;
end
if Nu ==1
    YYu(1,Yu(1,1)) = 1;
else
    for kkk = 1:Nu
       YYu(kkk,Yu(kkk,1)) = 1;
    end
end  
for t =1:NN
    YYtrain(t,Ytrain(t,1)) = 1;
end
for tt = 1: n-NN
    YYtest(tt,Ytest(tt,1)) = 1;
end
    
%% main

%%  1.unlabel
% 
%       rowrank = randperm(size(Xl,1));
%       XX = Xl(rowrank,:); 
%       YY = Yl(rowrank,:);
%       row = randperm (size(Xu,1));
%       xx = Xu(row,:);
%       yy = Yu (row,:);
%       
%       y = KNN(xx', XX', YY', 1);
%       preY =y';            
%       rca = 0;
%       Nu = size(Xu,1); 
%       for u = 1:Nu
%           if preY (u,:) == Yu(u,:);
%          if preY (u,:) == yy(u,:);
%              rca = rca+1;
%          end
%       end
%       result= rca/Nu;  
% 

%%  2.test
% 

      rowrank = randperm(size(Xl,1));
      XX = Xl(rowrank,:); 
      YY = Yl(rowrank,:);

      row = randperm (size(Xtest,1));
      xx = Xtest(row,:);
      yy = Ytest (row,:);
      
      y = KNN(xx', XX', YY', 1);  
      preY =y';            
      rca = 0;
      Nu = size(Xtest,1);
      for u = 1:Nu
         if preY (u,:) == yy(u,:);
             rca = rca+1;
         end
     end
     result= rca/Nu;  
% % 

 Result(1,iter) = result;

end
mean_result = mean(Result);
mean_std = std(Result);













    