function [J]=SMS_Jacobi1D(m_inv, step, T, h_inv, Fwd_pre)
% m_inv=m;  
% h_inv=h;
  
J=zeros(length(Fwd_pre),length(m_inv));

for j=1:length(m_inv)
    m_new = m_inv;
    dm = step*m_new(j);
    m_new(j) = m_new(j) - dm;
    rou_inv = SMS_m2rou(m_new,h_inv); 
    Fwd_new = SMS_fwd1D(T,rou_inv,h_inv);
    J(:,j) = -( abs(Fwd_new) - abs(Fwd_pre) )/dm;
end