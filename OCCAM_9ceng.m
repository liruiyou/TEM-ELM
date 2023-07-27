%% >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>9-layer model>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
clear;clc;close all;
tic
ti=-6:0.1:-3;T=10.^ti; % observe time 

rou1 = [ 100 10 100 10 100 50 130 100 100]';   
h1  = [ 20 20 20 20 60 20 30 10 inf]'; %%9-layer model

[d_Rou,d_H]=SMS_draw_rou(rou1,h1);
figure(1);
loglog(d_Rou,-d_H,'k');hold on;
%% >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>60 layer model>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
rou0 = 10;
m = log10(...
   [ rou0, rou0, rou0, rou0, rou0, rou0, rou0,rou0,rou0,rou0,... 
     rou0, rou0, rou0, rou0, rou0, rou0, rou0,rou0,rou0,rou0,... 
     rou0, rou0, rou0, rou0, rou0, rou0, rou0,rou0,rou0,rou0,...
     rou0, rou0, rou0, rou0, rou0, rou0, rou0,rou0,rou0,rou0,... 
     rou0, rou0, rou0, rou0, rou0, rou0, rou0,rou0,rou0,rou0,...
     rou0, rou0, rou0, rou0, rou0, rou0, rou0,rou0,rou0,rou0]'); %60 layer model
h0=4;
h = [...
   h0,h0,h0,h0,h0,h0,h0,h0,h0,h0,... 
   h0,h0,h0,h0,h0,h0,h0,h0,h0,h0,... 
   h0,h0,h0,h0,h0,h0,h0,h0,h0,h0,...
   h0,h0,h0,h0,h0,h0,h0,h0,h0,h0,... 
   h0,h0,h0,h0,h0,h0,h0,h0,h0,h0,...
   h0,h0,h0,h0,h0,h0,h0,h0,h0,inf]'; %60 layer model
%%
rou_inv=SMS_m2rou(m,h);

[d_Rou,d_H]=SMS_draw_rou(rou_inv,h);
figure(1)
handle00=loglog(d_Rou,-d_H,'r*');hold on;

% tic;
d_obs = SMS_fwd1D(T,rou1,h1);%%Theoretical forward response
d_obs=(awgn(d_obs,20,'measured'));%Add 5% random noise
% toc;
figure(2);
loglog(T,d_obs,'r-');
hold on;
legend('measured data')
handle001=loglog(T,d_obs,'r-');

Maxiter=10;%%Maximum Number of Iterations
ObjFun_out = zeros(1,Maxiter);
step00=1;
for iter=1:Maxiter
    iter
    fprintf ('%d',iter);
    rou_inv=SMS_m2rou(m,h); 
%     tic;
    Fwd_pre = SMS_fwd1D(T,rou_inv,h);
%     toc;
%     delete(handle001)
    figure(2)
    handle001=loglog(T,Fwd_pre,'b--');hold on;
    legend('measured data','fitting data')
    ObjFun=sum( abs ( abs(d_obs) - abs(Fwd_pre) )./ abs(d_obs) );
    ObjFun_new=ObjFun;
    ObjFun_out(iter)=ObjFun_new;
    if  rem(iter,2) == 0
        figure(1)
        handle00=loglog(d_Rou,-d_H,'r');hold on;
    end
    if iter == 10
       fprintf ('%d',iter);
       figure(4);semilogy(ObjFun_out(1:10),'*-')
    end 
    sigma=diag([0,ones(1,length(m)-1)])+diag(-ones(1,length(m)-1),-1);
    step=step00;
    while001=0;
    
    while  ObjFun<=ObjFun_new;
        while001=while001+1
        if while001>1
            step00=step00/2;%%%%%%%%%%%%%%????????????
            delete(handleMu)
        end
        step=step/2;
        fprintf ('step = %f',step);
        J = SMS_Jacobi1D(m, step, T, h, Fwd_pre);
        
        dn = d_obs - Fwd_pre + J*m;
        
        [mu,handleMu] = SMS_searchU1d(T,sigma,J,d_obs,dn,h);
        m_new = SMS_modelUpdate(10.^mu,sigma,J,dn);
        rou_inv = SMS_m2rou(m_new,h); 
        Fwd_pre = SMS_fwd1D(T,rou_inv,h);
        ObjFun_new = sum(abs( abs(d_obs)-abs(Fwd_pre))./abs(d_obs));
    end
    m=m_new;
    rou_inv = SMS_m2rou(m,h); 
    [d_Rou,d_H]=SMS_draw_rou(rou_inv,h);
end
TimeOCCAM=toc;
%% Result comparison
figure(5);
semilogy(ObjFun_out,'*-');hold on;
legend('Iterative curve')

[d_Rou1,d_H1]=SMS_draw_rou(rou1,h1);
figure(6);
loglog(d_Rou1,-d_H1,'k');hold on;
loglog(d_Rou,-d_H,'r-');hold on;
legend('Fit curve')

% tic;
d_obs1 = SMS_fwd1D(T,rou1,h1);
% toc;
figure(7);
loglog(T,d_obs1,'r-');
legend('measured data')
hold on;
loglog(T,Fwd_pre,'b--');
legend('measured data','fitting data')
