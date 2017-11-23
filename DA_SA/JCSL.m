% This code is part of the supplementary material to the paper
% 
% @article{FernandoPRL2015,
% author = {Basura Fernando, Tatiana Tommasi, Tinne Tuytelaars},
% title = {Joint cross domain classification and subspace learning for unsupervised adaptation},
% journal = {subitted to Pattern Recognition Letters},
% year = {2015},
% } 
%
% Copyright (c) 2015, Basura Fernando
% All rights reserved.
%
% Redistribution and use in source and binary forms, with or without modification, 
% are permitted provided that the following conditions are met:
%
% 1. Redistributions of source code must retain the above copyright notice, this 
%list of conditions and the following disclaimer.
%
% 2. Redistributions in binary form must reproduce the above copyright notice, 
% this list of conditions and the following disclaimer in the documentation and/or 
% other materials provided with the distribution.
% 
% THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND 
% ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED 
% WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE 
% DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR 
% ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES 
% (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; 
% LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON 
% ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT 
% (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS 
% SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

function [w,U,V,overall_obj] = JCSL(Xs,Xt,Ys,Params,U_o)

ns = size(Xs,1); % number of source samples
[V,~,~] = princomp(Xt); % PCA for the target
d = Params.C; % dimensionality of subspace
V = V(:,1:d); % target subpsace

%---Initialization
U = U_o; % initialize U

niter = 1000;
overall_obj = zeros(1,niter);

L = Params.L; %0.01;
B = Params.B; %0.5;

%--- Initialize w
% w = randn(d,1);
[w, ~,~] = vl_svmtrain( (Xs * U)', Ys', L); % better initialize with vl_feat

batch_size = 25;
learn_rate_0 = 10^-1;
D = d; % dimensionality
learn_rate_w = learn_rate_0;
learn_rate_u = learn_rate_0;
obj_old = w'*w +  L * mean(max(0,1-Ys.*(Xs*U*w))) + B * norm(U-V,'fro');
for iter = 1 : niter
   batches = ceil(ns / batch_size);
   rn = (randperm(ns));
   for bi = 1 : batches
		indx = [(bi-1)*batch_size + 1: min(bi * batch_size,ns)];
		indx = rn(indx);
		shi = Ys(indx).*(Xs(indx,:)*U*w);
		error_indx = indx(find(shi < 1));
		ne = numel(error_indx);		
		if ne > 0
			gradient_w = 2*w + L * mean(  -(Xs(error_indx,:)*U)' .* repmat(Ys(error_indx)',D,1) , 2 );		
			for jj = 1 : ne
				gradient_U = Xs(error_indx(jj),:)' * w' .* Ys(error_indx(jj),:);
			end
			gradient_U =  L * gradient_U/ne + B * 2*(U-V)  ;						
        end		
		w = w - learn_rate_w * gradient_w;	
		U = U - learn_rate_u * gradient_U;				
   end
   obj_new = w'*w +  L * mean(max(0,1-Ys.*(Xs*U*w))) + B * norm(U-V,'fro');    
   overall_obj(iter) = obj_new;   
   if (obj_old - obj_new) < 10^-6    % convergence criteria    
        fprintf('iteration %d obj %1.6f norm(U-V) = %1.6f  mean(max(0,1-Ys.*(Xs*U*w))) = %1.4f S.acuracy = %1.2f  margin = %1.4f\n',iter,obj_new,norm(U-V,'fro'),mean(max(0,1-Ys.*(Xs*U*w))),sum( Ys.*(Xs*U*w) > 0 )/ns*100,w'*w);
        return;
    end
end
fprintf('obj %1.6f norm(U-V) = %1.6f  mean(max(0,1-Ys.*(Xs*U*w))) = %1.4f S.acuracy = %1.2f  margin = %1.4f\n',obj_new,norm(U-V,'fro'),mean(max(0,1-Ys.*(Xs*U*w))),sum( Ys.*(Xs*U*w) > 0 )/ns*100,w'*w);

end




