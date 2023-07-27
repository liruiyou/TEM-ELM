function [m_new]=SMS_modelUpdate(mu,sigma,J,dn)
% mu=10^mu(i);

m_new=(mu*(sigma')*sigma+J'*J)\J'*dn;

for i=1:length(m_new)
    if m_new(i)<-2
        m_new(i)=-2;
    elseif  m_new(i)>2.7
        m_new(i)=2.7;
    end
end