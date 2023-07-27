%% >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
%% Empty environment variables
clc;clear;close all;
Strata = 9;%%Number of layers
pathbase =['....\TrainData\',num2str(Strata),'LayersTempCase6'];%Dataset file location
T_train=[];
for icpu=1:4
T_train=[T_train;csvread([pathbase,'\rhoh',num2str(icpu),'.csv'])];
end
P_train=[];
for icpu=1:4
P_train=[P_train;csvread([pathbase,'\Hz',num2str(icpu),'.csv'])];
end

input=P_train;
output=T_train;
clear P_train T_train;
P_train=input(1:25000,:);
T_train=output(1:25000,:);  
%% ¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·add noise¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·
P_train=awgn(P_train,26,'measured');
%% ¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·Training and testing data¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·  
input=P_train;
output=T_train;
k=rand(1,size(input,1));
[m,n]=sort(k);
P_train=input(n(1:end-5000),:)';
T_train=output(n(1:end-5000),:)';    
P_test=input(n(end-5000+1:end),:)';
T_test=output(n(end-5000+1:end),:)';
%% ¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·
[inputn,inputps]=mapminmax(P_train);          %Input normalization
[outputn,outputps]=mapminmax(T_train);    %Output normalization
%% ¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·BP¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·
star=tic;
net=newff(inputn,outputn,[5]);
net.trainParam.epochs=10;
net.trainParam.lr=0.1;
net.trainParam.goal=4.0e-12;
net=train(net,inputn,outputn);
inputn_test=mapminmax('apply',P_test,inputps); %Normalize according to inputps format
an_train=sim(net,inputn);
TrainTimeBP=toc(star);
star=tic;
an_test=sim(net,inputn_test);
BPoutput_train=mapminmax('reverse',an_train,outputps);
BPoutput_test=mapminmax('reverse',an_test,outputps);%Denormalize according to outputps format
TestTimeBP=toc(star);
%% Result comparison
RMrse_trainBP = sqrt(sum(sum(((BPoutput_train - T_train)./T_train).^2))./(size(T_train,1)*size(T_train,2)));
APE_trainBP = sum(sum(abs((T_train-BPoutput_train)./T_train)))./(size(T_train,1)*size(T_train,2));
N = length(T_train);% coefficient of determination
R2_trainBP=(N*sum(BPoutput_train.*T_train)-sum(BPoutput_train).*sum(T_train)).^2/((N*sum((BPoutput_train).^2)-(sum(BPoutput_train)).^2).*(N*sum((T_train).^2)-(sum(T_train)).^2)); 
RMrse_testBP = sqrt(sum(sum(((BPoutput_test - T_test)./T_test).^2))./(size(T_test,1)*size(T_test,2)));%Root Mean Square Relative Error 
APE_testBP = sum(sum(abs((T_test-BPoutput_test)./T_test)))./(size(T_test,1)*size(T_test,2));
N = length(T_test);% coefficient of determination
R2_testBP=(N*sum(BPoutput_test.*T_test)-sum(BPoutput_test).*sum(T_test)).^2/((N*sum((BPoutput_test).^2)-(sum(BPoutput_test)).^2).*(N*sum((T_test).^2)-(sum(T_test)).^2)); 
%%Save training model BP
pathname=['...\TrainModel\',num2str(Strata),'LayersTempCase6\'];
if exist(pathname,'dir') == 0
    mkdir(pathname);
end
filename='BP_net';
save([pathname,filename],'net') ;
filename='BP_inputps';
save([pathname,filename],'inputps');
filename='BP_outputps';
save([pathname,filename],'outputps');
%% ¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·ELM¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·¡·
star=tic;
[IW,B,LW,TF,TYPE,Yn_ELM] = elmtrain(inputn,outputn,100);
Y_ELM = mapminmax('reverse',Yn_ELM,outputps);
TrainTimeELM=toc(star);
star=tic;
tn_sim = elmpredict(inputn_test,IW,B,LW,TF,TYPE);
T_sim=mapminmax('reverse',tn_sim,outputps);
TestTimeELM=toc(star);
%% Result comparison
RRMSE_trainElm = sqrt(sum(sum(((Y_ELM - T_train)./T_train).^2))./(size(T_train,1)*size(T_train,2)));%Root Mean Square Relative Error 
APE_trainELM = sum(sum(abs((T_train-Y_ELM)./T_train)))./(size(T_train,1)*size(T_train,2));
N = length(T_train);% coefficient of determination
R2_trainELM=(N*sum(Y_ELM.*T_train)-sum(Y_ELM).*sum(T_train)).^2/((N*sum((Y_ELM).^2)-(sum(Y_ELM)).^2).*(N*sum((T_train).^2)-(sum(T_train)).^2)); 
RRMSE_testELM = sqrt(sum(sum(((T_sim - T_test)./T_test).^2))./(size(T_test,1)*size(T_test,2)));%Root Mean Square Relative Error 
APE_testELM = sum(sum(abs((T_test-T_sim )./T_test)))./(size(T_test,1)*size(T_test,2));
N = length(T_test);% coefficient of determination
R2_testELM=(N*sum(T_sim.*T_test)-sum(T_sim).*sum(T_test)).^2/((N*sum((T_sim).^2)-(sum(T_sim)).^2).*(N*sum((T_test).^2)-(sum(T_test)).^2)); 
%%Save training model ELM
filename='ELM_IW';
save([pathname,filename],'IW') ;
filename='ELM_Bias';
save([pathname,filename],'B') ;
filename='ELM_LW';
save([pathname,filename],'LW') ;
filename='ELM_TF';
save([pathname,filename],'TF') ;
filename='ELM_TYPE';
save([pathname,filename],'TYPE') 

fprintf('      method      |    Optimal C    |  Training Acc.  |    Testing Acc.   |   Training Time \n');
fprintf('--------------------------------------------------------------------------------------------\n');
fprintf('      %6s      |     %.5f     |      %.3f      |      %.5f      |      %.5f      \n','ELM','0',APE_trainELM,APE_testELM,TrainTimeELM);
fprintf('      %6s      |     %6s    |      %.3f      |      %.5f      |      %.5f     \n','BP', '0' ,APE_trainBP,APE_testBP,TrainTimeBP);







