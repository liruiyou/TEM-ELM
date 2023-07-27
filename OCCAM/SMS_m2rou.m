function [rou] = SMS_m2rou (m_inv,h_inv)
% rou_up=rou_up;
% m_inv=m;
% h_inv=h;

    len = size (m_inv,2);
    rou = zeros( length(h_inv) ,len);
    rou ( 1 : end , :) =  10.^ m_inv;
end