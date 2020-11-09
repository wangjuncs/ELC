function [pt]=ELC(X_train, Y_train, k, lamda, beta, tao)

if nargin<6
   tao = 10;
end

if nargin<5
   beta = 1e+8;
end

if nargin<4
   lamda=1e-3;
end

if nargin<3
   k=15;
end

data=X_train;
[n,m]=size(data); 
class=Y_train;
c=size(class,2);

%% normalize data
data=zscore(data);

%% initialize label correlations
dis=EuDist2(class',class');
d=mean(mean(dis));
SC=exp(-dis)/(2*d);


%% estimate matrix T
[Q1,R1]=eig(SC);
Q=zeros(c,c);
R=zeros(c,c);
for i=1:1:c
    Q(:,i)=Q1(:,c-i+1);
    R(i,i)=R1(c-i+1,c-i+1);
end;
T=n*Q*(R^0.5);

%% initialize parameters
tmax=1e+3;
ed=1e-4; 

r=zeros(1,m);
for i=1:1:m
    CC=data(:,i)'*class; 
    CC=CC/n;      
    r(i)=trace((CC'*CC-SC)*(CC'*CC-SC)');
end;
[AX,INX]=sort(r);
A=class'*data; 
Ut1=zeros(m,c);
Vt1=zeros(m,c);
pt1=zeros(m,1);
tao=1/max(abs(eig(A'*A)));


%% select features
t=0;
Wt1=zeros(m,c);
Wt=zeros(m,c); 
Vt=ones(m,1)*ones(1,c)/m;
pt=zeros(m,1); 
for j=1:1:k
    pt(INX(j))=1;
end;

 while t<=tmax      
     if norm(Wt1-Wt,2)<=ed && t>1 
         break;
     end;
     Wt=Wt1;
     theta=norm(Wt+Vt/beta,2)-lamda/beta;
     Ut1=max(theta,0)*(Wt+Vt/beta)/norm(Wt+Vt/beta,2);
     Omagat=diag(pt)*A'*(A*diag(pt)*Wt-T);
     Wt1=(tao/(beta*tao+1))*(beta*Ut1+Vt+(Wt-tao*Omagat)/tao);
     pt1=zeros(m,1);
     rt=zeros(1,m);
     for jj=1:1:m
         x=class'*data(:,jj)*Wt1(jj,:)-T;
         rt(jj)=trace(x*x');
     end;
     [AXr,INXr]=sort(rt);
     for jj=1:1:k
         pt1(INXr(jj))=1;
     end;
     Vt1=Vt-beta*(Wt1-Ut1);
     Vt=Vt1;
     pt=pt1;
     t=t+1;
     tao=1/norm(max(A'*A*Vt),2);  % update \tao
        
 end;   %%% end-while
    pt=find(pt~=0);  

end

