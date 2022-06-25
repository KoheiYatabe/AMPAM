function [p,os,calcF,Trfactor] = AMPAM(ps,os,y,z,params)

[tau,rho,alpha,pMax,maxIter] = initializeParameters(params);
[F,Ft,S,St,T,Tt,D,Dt,P,Pt] = initializeOperators(ps,os,params);
[params,eta,sigma] = calcOperatorNorm(P,Pt,S,St,params);
[proxSumPsiO,proxSumPsiP,proxSumNorm] = initializeProx(params);

u = eta*St(Pt(Ft(y)));
v = u + sigma*Dt(z);
o = os * pMax / tau; % s stands for scaled
p = ps * pMax;       % s stands for scaled

rfactor = zeros(maxIter,1);
etime = zeros(maxIter,1);

if params.drawFigures
    figure(99)
    subplot(1,3,1), fig1 = imagesc(abs(o));   colormap gray; colorbar; axis image
    subplot(1,3,2), fig2 = imagesc(angle(o)); colormap gray; colorbar; axis image
    subplot(1,3,3), fig3 = imagesc(abs(p));   colormap gray; colorbar; axis image
    sgt = sgtitle(['Iteration: 0/' num2str(maxIter)]);
    drawnow
end

tic
for n = 1:maxIter
    % primal-dual splitting for o-subproblem
    t = y + tau*F(P(S(o-v)));
   yt = t - proxSumPsiO(t);
   vt = v - u + eta*St(Pt(Ft(yt)));
    t = z + tau*D(o-2*vt);
   zt = t - proxSumNorm(t);
    o = o - rho*vt;
    y = y + rho*(yt-y);
    z = z + rho*(zt-z);
    u = u + rho*(vt-v);
    v = u + sigma*Dt(z);
    
    % update operator O
    sampledOs = S(o*tau/pMax);
    O  = @(x) x.*sampledOs;
    Ot = @(x) x.*conj(sampledOs);
    
    % (approximate) gradient descent for p-subproblem
    t = F(O(T(p)));
    t = t - proxSumPsiP(t);
    p = p - alpha * Tt(Ot(Ft(t)));
    
    % update operator P
    ps = p / max(abs(p),[],'all');
    shiftedPs  = T(ps);
    P  = @(x) x.*shiftedPs;
    Pt = @(x) x.*conj(shiftedPs);
    
    if params.drawFigures
        fig1.CData = abs(o*tau/pMax);
        fig2.CData = angle(o);
        fig3.CData = abs(p);
        sgt.String = ['Iteration: ' num2str(n) '/' num2str(maxIter)];
        drawnow
    end
    calcF      = abs(F(O(T(p))));
    rfactor(n) = RFcalc(calcF,params);
    etime(n)   = toc;
    
    fpsv = max(abs(St(Pt(P(S(ones(params.imSize)))))),[],'all');
    fotv = max(abs(Tt(Ot(O(T(ones(params.pbSize)))))),[],'all');
    fprintf('iteration # %d, fps %f, fot %f, rfactor %f, elapsed time %f s. \n', n, fpsv, fotv, rfactor(n), etime(n));
    
end
os = o * tau / pMax;
p  = p / size(params.samplingIdx,3);
calcF = fftshift(fftshift(calcF,1),2);
toc

niter = (1:maxIter).';
Trfactor = table(niter,rfactor,etime,'VariableNames',{'iteration #','RF factor','elapsed time (s)'});

end

function [tau,rho,alpha,pMax,maxIter] = initializeParameters(params)
tau     = params.tau;
rho     = params.rho;
alpha   = params.alpha;
pMax    = params.probeScale;
maxIter = params.maxIter;
end

function [F,Ft,S,St,T,Tt,D,Dt,P,Pt] = initializeOperators(p,o,params)

sizeConst = sqrt(numel(p));
F  = @(x)  fft2(x) / sizeConst;
Ft = @(x) ifft2(x) * sizeConst;

idx = params.samplingIdx;
zeroMat = zeros(size(o));
S  = @(x) x(idx) / sqrt(size(idx,3));
St = @(x) overlapAddProcedure(x,zeroMat,idx) / sqrt(size(idx,3));

subPixelPhase = params.subPixelShiftInRad;
T  = @(x) ifft2(exp(-1i*subPixelPhase).*fft2(x)) / sqrt(size(idx,3));
Tt = @(x) ifft2(sum(exp(1i*subPixelPhase).*fft2(x),3)) / sqrt(size(idx,3));

D  = @(x) discreteGrad(x);
Dt = @(x) discreteGradTranspose(x);

shiftedP  = T(p);
P  = @(x) x.*shiftedP;
Pt = @(x) x.*conj(shiftedP);
end

function [params,eta,sigma] = calcOperatorNorm(P,Pt,S,St,params)

imSize = params.imSize;
normFPS = max(abs(St(Pt(P(S(ones(imSize)))))),[],'all');
normD   = 8; % well-known upperbound

tau   = params.tau;
eta   = 1/(tau*normFPS);
sigma = 1/(tau*normD);

params.eta   = eta;
params.sigma = sigma;
end

function [proxSumPsiO,proxSumPsiP,proxSumNorm] = initializeProx(params)
eta    = params.eta;
sigma  = params.sigma;
gamma  = params.gamma;
lambda = params.lambda;
dataI  = params.intensityData;
excIdx = params.exceptionIdx;

proxParamPsiO = 1/eta;
proxParamPsiP = gamma;
proxParamNorm = lambda/sigma;

proxSumPsiO = @(x) proxGeneralizedKLdiv(x,dataI,excIdx,proxParamPsiO);
proxSumPsiP = @(x) proxGeneralizedKLdiv(x,dataI,excIdx,proxParamPsiP);
proxSumNorm = @(x) proxSumOfNorms(x,proxParamNorm);
end

function z = overlapAddProcedure(x,z,idx)
for n = 1:size(idx,3)
    z(idx(:,:,n)) = z(idx(:,:,n)) + x(:,:,n);
end
end

function y = discreteGrad(x)
dv = diff(x,[],1);
dh = diff(x,[],2);
y = cat(3,[dv;zeros(1,size(x,2))],[dh zeros(size(x,1),1)]);
end

function x = discreteGradTranspose(y)
dv = y(1:end-1,:,1);
dh = y(:,1:end-1,2);
x = [-dv(1,:); -diff(dv,[],1); dv(end,:)] + [-dh(:,1), -diff(dh,[],2), dh(:,end)];
end

function x = proxGeneralizedKLdiv(x,d,excIdx,lambda)
y = abs(x);
amp = (y + sqrt(y.^2 + 8*lambda*(2*lambda+1)*d)) / (4*lambda + 2);
amp(excIdx) = y(excIdx);
x = amp .* sign(x);
end

function x = proxSumOfNorms(x,lambda)
x = max(0,1-lambda./sqrt(sum(abs(x).^2,3))).*x;
end

function rfactor = RFcalc(calcF,params)
dataF  = sqrt(params.intensityData);
excIdx = params.exceptionIdx;
dataF(excIdx) = 0;
calcF(excIdx) = 0;

rfactor = sum(abs(dataF - calcF),'all')/sum(abs(dataF),'all');
end