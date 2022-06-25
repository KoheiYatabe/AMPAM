%% set optimization parameter

outhead = 'rec';

params.maxIter = 100;

params.lambda = 0.5; % total variation weight

params.tau   = 50; % step-size parameter for y,z
params.alpha = 10; % step-size parameter for p
params.gamma = 1;   % parameter adjusting approximation accuracy of cost of p
params.rho   = 1;   % relaxation parameter (rho > 1 for acceleration)

params.imSize = 770;
params.pbSize = 470;

params.drawFigures = true;


%% initialization

load data

[params,probeInit,objectInit,yInit,zInit] = initializeVariables(params,diffData,p,illumPos);


%% main processing

[probeResult,objectResult,calcFResult,Trfactor] = AMPAM(probeInit,objectInit,yInit,zInit,params);


%% visualize result

paramlist = append('_i',num2str(params.maxIter),'_l',num2str(params.lambda),...
                   '_t', num2str(params.tau),'_g',num2str(params.gamma),...
                   '_a',num2str(params.alpha),'_r',num2str(params.rho),...
                   '_red',num2str(numStep));

outTiff(abs(probeResult),    append(outhead,paramlist,'_aprb.tif'))
outTiff(angle(probeResult),  append(outhead,paramlist,'_pprb.tif'))
outTiff(abs(objectResult),   append(outhead,paramlist,'_aobj.tif'))
outTiff(angle(objectResult), append(outhead,paramlist,'_pobj.tif'))
outTiff(calcFResult.^2,      append(outhead,paramlist,'_diff.tif'))
writetable(Trfactor,         append(outhead,paramlist,'_rfac.xlsx'))


%% local functions

function [params,probeInit,objectInit,yInit,zInit] = initializeVariables(params,data,p,positionList)%mod220427yt
% set data
params.intensityData = fftshift(fftshift(abs(double(data)),1),2);
params.exceptionIdx  = fftshift(fftshift(    data < 0     ,1),2);

% initialize p,o,y,z
[probeInit,params] = ProbeInitialize(params,data,p);
objectInit = ones(params.imSize);
yInit = zeros(size(data));
zInit = zeros([size(objectInit) 2]);

% set translation parameters
params = calcTranslationParameters(params,positionList,objectInit,data);
end

function idx = fourierIdx(len)
idx = -floor(len/2):floor(len/2)-mod(len+1,2);
end

function [probeInit,params] = ProbeInitialize(params,data,p)
probeInit = p/max(p,[],'all');
probeInitFFT = sum(sum(probeInit)) / sqrt(numel(probeInit));
scaleConst = max(sqrt(params.intensityData(1,1,:))) / probeInitFFT;
params.probeScale = scaleConst * size(data,3);
end

function params = calcTranslationParameters(params,positionList,o,data)
lenV = size(data,1);
lenH = size(data,2);

phV = 2*pi*ifftshift(fourierIdx(lenV))'/lenV;
phH = 2*pi*ifftshift(fourierIdx(lenH)) /lenH;

posIdx  = floor(positionList);
pv = reshape(posIdx(:,1),1,1,[]);
ph = reshape(posIdx(:,2),1,1,[]);
idx = sub2ind(size(o),repmat(pv+(1:lenV)',[1 lenH]),repmat(ph+(1:lenH),[lenV 1]));
params.samplingIdx = uint32(idx);

posDeci = positionList - posIdx;
dv = reshape(posDeci(:,1),1,1,[]);
dh = reshape(posDeci(:,2),1,1,[]);
params.subPixelShiftInRad = dv.*phV + dh.*phH;
end

function outTiff(x, outname)
tagstruct.Compression = Tiff.Compression.None;
tagstruct.SampleFormat = Tiff.SampleFormat.IEEEFP;
tagstruct.Photometric = Tiff.Photometric.MinIsBlack;
tagstruct.BitsPerSample = 32; 
tagstruct.SamplesPerPixel = 1; 
tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;

tagstruct.ImageLength = size(x, 1);
tagstruct.ImageWidth = size(x, 2);
t = Tiff(outname, 'w');
for iimg = 1:size(x,3)
    t.setTag(tagstruct);
    t.write(single(x(:,:,iimg)));
    t.writeDirectory();
end
t.close();
end

