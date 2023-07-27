%% ¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·A set of test data¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·
%% Empty environment variables
clc;clear;close all;
for k=1:10
 k
Strata=9;%%Number of layers
pathname=['...\TrainModel\',num2str(Strata),'LayersTempCase6\'];%%Training model save location

switch Strata  
     case 9
        rho_test=[100 10 100 10 100 50 130 100 100];
        h_test=[20 20 20 20 60 20 30 10];%Nine layer model   
end
%% ¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·Model parameter preset¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·
T_test= [rho_test h_test]';
P_test=(awgn((zhengyan2(T_test(1:Strata),T_test(Strata+1:end))),26,'measured'))';%%Add 5% random noise
%% Test BP
addpath(genpath(pwd));
load([pathname,'BP_net.mat']);
load([pathname,'BP_inputps.mat']);
load([pathname,'BP_outputps.mat']);
tic
inputn_test=mapminmax('apply',P_test,inputps); %Normalize according to inputps format
an_test=sim(net,inputn_test);
BPoutput_test=mapminmax('reverse',an_test,outputps);%Denormalize according to outputps format
TimeBP(k)=toc;
%% Result comparison
RRMSE_testBP(k) = sqrt(sum(sum(((BPoutput_test - T_test)./T_test).^2))./(size(T_test,1)*size(T_test,2)));%Root mean square relative error
APE_testBP(k) = sum(sum(abs((T_test-BPoutput_test)./T_test)))./(size(T_test,1)*size(T_test,2));
N = length(T_test);% coefficient of determination
R2_testBP(k) =(N*sum(BPoutput_test.*T_test)-sum(BPoutput_test).*sum(T_test)).^2/((N*sum((BPoutput_test).^2)-(sum(BPoutput_test)).^2).*(N*sum((T_test).^2)-(sum(T_test)).^2)); 

%% Test ELM
addpath(genpath(pwd));
load([pathname,'ELM_IW.mat']);
load([pathname,'ELM_Bias.mat']);
load([pathname,'ELM_LW.mat']);
load([pathname,'ELM_TF.mat']);
load([pathname,'ELM_TYPE.mat']);
tic
tn_sim = elmpredict(inputn_test,IW,B,LW,TF,TYPE);
T_sim=mapminmax('reverse',tn_sim,outputps);
TimeELM(k)=toc;
%%Result comparison
RRMSE_testELM(k) = sqrt(sum(sum(((T_sim - T_test)./T_test).^2))./(size(T_test,2)*size(T_test,2)));%Mean square relative error
APE_testELM(k) = sum(sum(abs((T_test-T_sim )./T_test)))./(size(T_test,1)*size(T_test,2));
N = length(T_test);% coefficient of determination
R2_testELM(k)=(N*sum(T_sim.*T_test)-sum(T_sim).*sum(T_test)).^2/((N*sum((T_sim).^2)-(sum(T_sim)).^2).*(N*sum((T_test).^2)-(sum(T_test)).^2)); 
%% Result display comments
fprintf('      method      |    Optimal C    |   Testing Acc.   |   Testing Time \n');
fprintf('--------------------------------------------------------------------------------------------\n');
fprintf('      %6s      |     %6s     |      %.5f      |      %.5f     \n','ELM','0',APE_testELM(end),TimeELM(end));
fprintf('      %6s      |     %6s    |      %.5f      |      %.5f    \n','BP', '0' ,APE_testBP(end),TimeBP(end));

TY_testELM(k,:) = T_sim;
TY_testBP(k,:) = BPoutput_test;
end
Mean_RRMSE_ELM = mean(RRMSE_testELM);
Mean_APE_ELM = mean(APE_testELM);
Mean_R2_ELM = mean(R2_testELM);
Mean_TYtest_ELM = mean(TY_testELM,1);
Mean_TimeELM = mean(TimeELM);
Mean_RRMSE_BP = mean(RRMSE_testBP);
Mean_APE_BP = mean(APE_testBP);
Mean_R2_BP = mean(R2_testBP);
Mean_TYtest_BP = mean(TY_testBP,1);
Mean_TimeBP = mean(TimeBP);

%%One-dimensional map of resistivity depth
    rou_Theroy=rho_test';
    h_Theroy=[h_test inf]';
    [d_RouTheroy,d_HTheroy]=SMS_draw_rou(rou_Theroy,h_Theroy);
    rou_InvBP=[Mean_TYtest_BP(1:Strata)]';
    h_InvBP=[Mean_TYtest_BP(Strata+1:end),inf]';
    [d_RouInvBP,d_HInvBP]=SMS_draw_rou(rou_InvBP,h_InvBP);
    rou_InvELM=[Mean_TYtest_ELM(1:Strata)]';
    h_InvELM=[Mean_TYtest_ELM(Strata+1:end),inf]';
    [d_RouInvELM,d_HInvELM]=SMS_draw_rou(rou_InvELM,h_InvELM);
figure(1);
    loglog(d_RouTheroy,-d_HTheroy,'k');hold on;
    loglog(d_RouInvBP,-d_HInvBP,'r--');hold on;
    loglog(d_RouInvELM,-d_HInvELM,'b-.');hold on;
    legend('Theoretical curve','BP','ELM');
    xlabel('Resistivity ¦Ñ(\Omega¡¤m)');
    ylabel('Depth H(m)');
 
P_test=zhengyan2(T_test(1:Strata),T_test(Strata+1:end));   
Fwd_preBP=zhengyan2(Mean_TYtest_BP(1:Strata),Mean_TYtest_BP(Strata+1:end));
Fwd_preELM=zhengyan2(Mean_TYtest_ELM(1:Strata),Mean_TYtest_ELM(Strata+1:end));
dt = -6:0.1:-3;
T = 10.^dt;
figure(2);
    loglog(T,P_test,'-'); hold on;
    loglog(T,Fwd_preBP,'b-.');hold on;
    loglog(T,Fwd_preELM,'r--');hold on;
    legend('measured data','BP','ELM')
    