function [Rou,H]=SMS_draw_rou(rou,h)
% rou=rou1;
% h=h1;

H=zeros(1,length(h)*2);
for i=1:length(h)
    H(i)=sum(h(1:i));
end
h=H;
Rou=zeros(1,length(rou));

iR=0;
for i=1:length(rou)
    iR=iR+1;
    Rou(iR)=rou(i);
    if i==1
        H(iR)=1;
    else
        H(iR)=h(i-1);
    end
    iR=iR+1;
    Rou(iR)=rou(i);
    if i==length(rou)
        H(iR)=2*h(i-1);
    else
        H(iR)=h(i);
    end
end