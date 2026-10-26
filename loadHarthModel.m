function M = loadHarthModel(kind)
% Loads a PRE-TRAINED model (no retraining). kind: 'mat' | 'onnx' | 'onnx_corrected'
% Returns struct M with: predict(Xc) -> categorical, classNames, windowSize,
% stepSize, Fs, useChannels (always 1:6 = back_xyz + thigh_xyz, in g).
% Xc = cell array, each cell a [6 x W] single window.
    if nargin < 1, kind = 'mat'; end
    M.Fs = 50; M.useChannels = 1:6; M.kind = kind;
    switch kind
        case 'mat'
            f = findFile('harth_trained_model.mat');
            S = load(f); net = S.trainedNet;
            M.classNames = string(net.Layers(end).Classes)';  % real class order
            M.windowSize = S.windowSize;                      % 125
            M.stepSize   = round(M.windowSize/2);
            M.predict = @(Xc) classify(net, Xc, 'MiniBatchSize',128);
            fprintf('Loaded MATLAB model: %s (window %d)\n', f, M.windowSize);
        case {'onnx','onnx_corrected'}
            name = 'best_model.onnx';
            if strcmp(kind,'onnx_corrected'), name = 'best_model_corrected.onnx'; end
            f = findFile(name);
            net = importNetworkFromONNX(f);
            MU    = [-0.885417 0.001669 -0.147612 -0.507975 0.059537 0.532519]';
            SIGMA = [ 0.295474 0.225258  0.338422  0.499131 0.308348 0.587967]';
            M.classNames = ["Walking","Running","Shuffling","Stairs (ascending)", ...
                "Stairs (descending)","Standing","Sitting","Lying","Cycling (sit)", ...
                "Cycling (stand)","Cycling (sit, inactive)","Cycling (stand, inactive)"];
            M.windowSize = 128; M.stepSize = 64;
            M.predict = @(Xc) onnxPredict(net, Xc, MU, SIGMA, M.classNames);
            fprintf('Loaded ONNX model: %s (window 128)\n', f);
        otherwise
            error('Unknown kind "%s"', kind);
    end
end

function f = findFile(name)
    cands = {name, fullfile('models',name), fullfile('old_friend_files',name), ...
             fullfile('..',name)};
    for i = 1:numel(cands)
        if isfile(cands{i}), f = cands{i}; return; end
    end
    [a,b] = uigetfile(['*' name(find(name=='.',1,'last'):end)], ['Locate ' name]);
    if isequal(a,0), error('%s not found.', name); end
    f = fullfile(b,a);
end

function pred = onnxPredict(net, Xc, MU, SIGMA, names)
    n = numel(Xc); W = size(Xc{1},2); pred = categorical(strings(n,1), names);
    for s = 1:256:n
        idx = s:min(n,s+255); B = numel(idx);
        A = zeros(6,B,W,'single');                    % [C B T]
        for j = 1:B, A(:,j,:) = reshape((Xc{idx(j)} - MU)./SIGMA, [6 1 W]); end
        try
            Y = predict(net, dlarray(A,'CBT'));
        catch
            Y = predict(net, dlarray(permute(A,[2 3 1]),'BTC'));  % [B T C]
        end
        d = extractdata(Y); lbl = dims(Y);
        if ~isempty(lbl) && contains(lbl,'C') && contains(lbl,'B')
            c = find(lbl=='C'); b = find(lbl=='B');
            d = permute(d,[c b setdiff(1:ndims(d),[c b])]);
            sc = reshape(double(d), size(d,1), []);
        else
            sc = reshape(double(d), numel(names), []);
        end
        [~,k] = max(sc,[],1);
        pred(idx) = names(k);
    end
end
