
function omega  = createcov_ls(para)


sigmaz          = para(31);
sigmag          = para(32);
sigmau          = para(33);
sigmarho        = para(34);
sigmaw          = para(35);
sigmab          = para(36);
sigmasigma      = para(37);
sigmanw         = para(38);
sigmamu         = para(39);
sigmamp         = para(40);


omega = zeros(11,11);

omega(1,1) = sigmaz;
omega(2,2) = sigmag;
omega(3,3) = sigmau;
omega(4,4) = sigmarho;
omega(5,5) = sigmaw;
omega(6,6) = sigmab;
omega(7,7) = sigmasigma;
omega(8,8) = sigmanw;
omega(9,9) = sigmamu;
omega(10,10) = sigmamp;


end
