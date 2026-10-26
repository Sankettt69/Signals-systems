%% simulate_imu.m - Advanced part 1: simulated IMU signals + pre-trained classifier
% Uses imuSensor (Navigation / Sensor Fusion toolbox) if installed; otherwise a
% base-MATLAB model of the same physics: f_body = R' * (a_nav - g_nav) + bias + noise.
% Demonstration only: simulated axes/noise differ from the real HARTH sensors.
clear; clc; close all; rng(1);
M = loadHarthModel('mat');                 % or 'onnx'
Fs = M.Fs; W = M.windowSize; g = 9.81; dur = 30; t = (0:1/Fs:dur-1/Fs)'; N = numel(t);
useToolbox = ~isempty(which('imuSensor'));
if useToolbox
    imu = imuSensor('accel-gyro','SampleRate',Fs);
    imu.Accelerometer.NoiseDensity = 0.004*g; imu.Gyroscope.NoiseDensity = 0.0005;
    fprintf('Using imuSensor.\n');
else
    imu = []; fprintf('imuSensor not available -> using base-MATLAB IMU model.\n');
end
% name, thigh freq (Hz), thigh amp (deg), thigh mean pitch (deg), trunk amp, trunk mean pitch
acts = {'Walking',1.0,30,0,3,0; 'Running',2.5,50,0,8,5; 'Standing',0,0,0,0,0; ...
        'Sitting',0,0,85,0,5; 'Lying',0,0,90,0,90};
figure('Name','Simulated thigh accelerometer');
for a = 1:size(acts,1)
    [name,f,A,m,At,mt] = acts{a,:};
    thigh = simSensor(imu,t,Fs,f,A,m,0.25,g);
    back  = simSensor(imu,t,Fs,f,At,mt,0.50,g);
    sig = ([back thigh]/g)';                         % [6 x N] in g
    starts = 1:M.stepSize:(N-W+1);
    Xc = arrayfun(@(s) single(sig(:,s:s+W-1)), starts,'UniformOutput',false)';
    p = M.predict(Xc);
    fprintf('%-9s -> predicted mostly: %s (%.0f%% of windows)\n', name, string(mode(p)), 100*mean(p==mode(p)));
    subplot(size(acts,1),1,a); plot(t, sig(4:6,:)'); ylabel(name); grid on
end
xlabel('Time (s)'); sgtitle('Simulated thigh accelerometer (g)');

function out = simSensor(imu,t,Fs,f,Adeg,mDeg,r,g)
    th  = deg2rad(mDeg + Adeg*sin(2*pi*f*t));        % pitch about y
    w   = gradient(th,1/Fs);
    pos = r*[sin(th), zeros(size(th)), cos(th)];     % pendulum point, NED (z down)
    acc = [gradient(gradient(pos(:,1),1/Fs),1/Fs), gradient(gradient(pos(:,2),1/Fs),1/Fs), ...
           gradient(gradient(pos(:,3),1/Fs),1/Fs)];
    if ~isempty(imu)
        q = quaternion([zeros(size(th)) th zeros(size(th))],'euler','ZYX','frame');
        out = imu(acc, [zeros(size(th)) w zeros(size(th))], q);
    else
        d = acc - [0 0 g];                           % a - g (NED, g points down)
        out = [cos(th).*d(:,1) - sin(th).*d(:,3), ...
               d(:,2), ...
               sin(th).*d(:,1) + cos(th).*d(:,3)];   % R' * (a-g)
        out = out + 0.02*g*ones(1,3) .* [1 -1 1] ... % constant bias
                  + 0.004*g*sqrt(Fs/2)*randn(size(out));  % white noise
    end
end
