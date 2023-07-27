function [r,handleMu]=SMS_searchU1d(T,sigma,J,d_obs,dn,h)

mu=-24:(5+24)/40:5;%mu=-15:0.5:0;
y=zeros(1,length(mu));
for i=1:length(mu)
    m=SMS_modelUpdate(10.^mu(i),sigma,J,dn); %%Inverted resistivity
    rou_inv=SMS_m2rou(m,h); 
    y(i)= sum(abs( abs(d_obs) - abs(SMS_fwd1D(T,rou_inv,h)) ) ./ abs(d_obs) );
end
ia=find(y==min(y));ia=max(ia);
% ymin=y(ia);

figure(3)
handleMu = semilogy(mu,y,'*-');hold on
legend('mu-yTotal relative error of forward response');

r=mu(ia);