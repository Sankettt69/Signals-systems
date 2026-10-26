function f = harthFeatures(win, Fs)
% Base-MATLAB only (no toolbox). win: [C x W] in g. Returns 1 x nFeatures.
% ORIGINAL feature set - must match the saved 75.1% thigh model.
    if nargin < 2, Fs = 50; end
    C = size(win,1); W = size(win,2); f = [];
    for c = 1:C
        x = double(win(c,:)); mu = mean(x); sd = std(x) + eps;
        xs = sort(x); q1 = xs(max(1,round(0.25*W))); q3 = xs(max(1,round(0.75*W)));
        P = abs(fft(x - mu)).^2; P = P(2:floor(W/2));
        fr = (1:numel(P))*Fs/W; [pk,ix] = max(P);
        f = [f, mu, sd, min(x), max(x), max(x)-min(x), sqrt(mean(x.^2)), ...
             median(x), q3-q1, mean(((x-mu)/sd).^3), mean(((x-mu)/sd).^4), ...
             mean(abs(diff(x))), fr(ix), sum(P)/W, pk/(sum(P)+eps)]; %#ok<AGROW>
    end
    if C >= 3
        for s = 1:3:C
            R = corrcoef(double(win(s:s+2,:))');
            f = [f, R(1,2), R(1,3), R(2,3)]; %#ok<AGROW>
        end
    end
    f(~isfinite(f)) = 0;
end
