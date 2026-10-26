# Project 232 - Human Motion Recognition Using IMUs (HARTH, MATLAB)

## Folder structure
new_imu/
  data/harth/            <- put S006.csv ... S029.csv here (UCI "HARTH" dataset)
  train_harth.m          <- 1) trains the CNN+BiLSTM, saves harth_trained_model.mat
  predict_harth.m        <- 2) tests on a HARTH CSV / phone log
  simulate_imu.m         <- 3) imuSensor simulation + classification (advanced)
  realtime_phone.m       <- 4) live recognition from MATLAB Mobile (advanced)
  models/                <- keep harth_trained_model.mat copies here
  old_friend_files/      <- old ONNX/PyTorch files (not needed any more)

## Steps
1. Download HARTH from UCI (search "HARTH dataset UCI"), unzip CSVs into data/harth.
2. In MATLAB: cd to new_imu, then run train_harth.m (needs R2023b+, Deep Learning Toolbox).
3. Run predict_harth.m, then simulate_imu.m.
4. For live phone demo: set useChannels = 4:6 in train_harth.m, retrain, run realtime_phone.m.
