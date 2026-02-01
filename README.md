# MPC + Residual Torque Learning

This repository implements a torque-based MPC controller
augmented with an offline-trained residual neural network
to compensate model mismatch in MuJoCo.

## Structure
- dataGet: MPC-only data collection
- TrainNN: residual torque neural network training
- ApplyNNtoMPC: MPC + NN inference
- eval: performance evaluation

## Pipeline
1. Run MPC-only simulation to collect residual dataset
2. Train residual NN offline
3. Apply NN during MPC execution
4. Compare MPC vs MPC+NN
