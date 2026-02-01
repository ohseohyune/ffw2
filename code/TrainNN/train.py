# train_residual_nn.py
"""
데이터 로드 → 신경망 학습 → 모델 저장

- 입력: delta_tau_dataset.npz (test5.py에서 생성)

- 학습 과정:

데이터 80/20으로 분할
MSE 손실로 Adam 최적화 (40 epochs)
검증셋에서 성능 평가
모델 저장

- 출력: residual_nn.pt (학습된 모델)

"""

import os
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


# ==============================
# Model (reuse your structure)
# ==============================
class ResidualTorqueNN(nn.Module):
    def __init__(self, delta_tau_max):
        super().__init__()
        self.delta_tau_max = float(delta_tau_max)

        # self.net = nn.Sequential(
        #     nn.Linear(9, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 64),
        #     nn.ReLU(),
        #     nn.Linear(64, 3),
        #     nn.Tanh(),  # output in [-1, 1]
        # )
        self.net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Tanh(),  # output in [-1, 1]
        )

    def forward(self, x):
        # x: (..., 9)
        return self.delta_tau_max * self.net(x)


def set_seed(seed: int = 0):
    """
    실험 결과를 다시 실행해도 거의 동일하게 재현되도록
    모든 ‘무작위성(randomness)’의 출발점을 고정하는 장치
    """
    np.random.seed(seed) # NumPy의 난수 생성기를 고정 
    torch.manual_seed(seed) # PyTorch의 난수 생성기를 고정


def main():
    set_seed(0)

    # ==============================
    # Config : 실험 조건을 정의 
    # ==============================
    DATA_PATH = "/home/seohy/colcon_ws/src/olaf/ffw/code/dataGet/delta_tau_dataset.npz"
    DELTA_TAU_MAX = 50.0
    BATCH_SIZE = 128  # 한 번의 gradient update에 사용할 샘플 수 (Q)
    EPOCHS = 200       # 전체 데이터셋을 몇 번 반복해서 학습할지 
    LR = 1e-3         # 학습률 (learning rate)
    WEIGHT_DECAY = 0.0 # 가중치 감쇠 (L2 정규화)
    GRAD_CLIP_NORM = 1.0  # 그래디언트 클리핑 (None이면 사용 안 함) (Q)  
    TRAIN_RATIO = 0.8 # 훈련셋 비율 (80% 훈련, 20% 검증)

    # JOINT_NAMES = ["r_sh_p", "r_sh_r", "r_elb"]
    JOINT_NAMES = ["r_sh_p"]

    assert os.path.exists(DATA_PATH), f"Dataset not found: {DATA_PATH}"

    # ==============================
    # Load dataset
    # ==============================
    data = np.load(DATA_PATH)
    q = data["q"]              # (N, 3)
    qdot = data["qdot"]        # (N, 3)
    tau_mpc = data["tau_mpc"]  # (N, 3)
    delta_tau = data["delta_tau"]  # (N, 3)

    # Build x = [q(3), qdot(3), tau_mpc(3)] => (N, 9)
    x = np.concatenate([q, qdot, tau_mpc], axis=1).astype(np.float32)

    # Clip y to [-5, +5] as required
    y = np.clip(delta_tau, -DELTA_TAU_MAX, DELTA_TAU_MAX).astype(np.float32)

    # Validate shapes : 학습 중 이상한 shape error를 초기에 잡기 위해
    N = x.shape[0]
    # assert x.shape == (N, 9), f"x shape mismatch: {x.shape}"
    assert x.shape == (N, 3), f"x shape mismatch: {x.shape}"
    # assert y.shape == (N, 3), f"y shape mismatch: {y.shape}"
    assert y.shape == (N, 1), f"y shape mismatch: {y.shape}"



    # ==============================
    # Train/Val split : 데이터셋을 훈련셋과 검증셋으로 분할 👉 데이터 인덱스를 섞은 뒤, 앞쪽은 훈련용 / 뒤쪽은 검증용으로 나눈다
    # ==============================
    """
    시뮬레이션에서 수집한 (상태, 토크) 데이터를
    공정하게 나누고,
    학습이 안정적으로 돌아가도록 PyTorch 학습 파이프라인을 구성하는 단계
    """
    idx = np.arange(N)
    np.random.shuffle(idx)
    n_train = int(TRAIN_RATIO * N) # 전체 중 훈련 데이터 개수    
    train_idx, val_idx = idx[:n_train], idx[n_train:]

    x_train, y_train = x[train_idx], y[train_idx] 
    x_val, y_val = x[val_idx], y[val_idx]

    # Torch tensors (NumPy → PyTorch Tensor 변환)
    x_train_t = torch.from_numpy(x_train)
    y_train_t = torch.from_numpy(y_train)
    x_val_t = torch.from_numpy(x_val)
    y_val_t = torch.from_numpy(y_val)

    # DataLoader 구성 (미니배치 단위로 데이터를 공급)
    train_loader = DataLoader(
        TensorDataset(x_train_t, y_train_t),
        batch_size=BATCH_SIZE,
        shuffle=True, # 매 epoch마다 데이터 순서를 다시 섞음 -> 일반화에 도움(not 편향)
        drop_last=False,
    )

    val_loader = DataLoader(
        TensorDataset(x_val_t, y_val_t),
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )

    # ==============================
    # Model / Optim / Loss
    # ==============================
    device = torch.device("cpu") # CPU에서 학습 (GPU가 있으면 "cuda"로 변경 가능)
    model = ResidualTorqueNN(delta_tau_max=DELTA_TAU_MAX).to(device)
    criterion = nn.MSELoss() # 손실 함수 = 평균 제곱 오차 -> 회귀 문제에 적합 : Δτtrue​−Δτpred
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # ==============================
    # Training loop
    # ==============================

    """
    Epoch 반복
    ├─ Train 단계 (가중치 업데이트 O)
    │    └─ batch 단위로 forward → loss → backward → update
    └─ Validation 단계 (가중치 업데이트 X)
        └─ 성능 평가만

    """
    train_losses, val_losses = [], []

    for epoch in range(1, EPOCHS + 1):
        model.train() # 모델을 훈련 모드로 전환
        running = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device) # 배치 입력을 디바이스로 이동
            yb = yb.to(device) # 배치 타깃을 디바이스로 이동

            pred = model(xb)   # NN이 Δτ_hat 예측
            loss = criterion(pred, yb) # 실제 Δτ_true와 비교 → 손실 계산(MSE)

            optimizer.zero_grad() # 그래디언트 초기화
            loss.backward() # 역전파로 그래디언트 계산 -> loss 기준으로 각 가중치에 대한 loss 변화량 산출 => 이 파라미터를 조금 바꾸면, 오차가 늘어날까 줄어들까 ?
            if GRAD_CLIP_NORM is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step() # Adam이 계산된 그래디언트를 사용해 가중치 업데이트

            running += loss.item() * xb.size(0) # 배치 손실 누적 (배치 크기 곱해서) -> 평균내야해서 

        train_loss = running / len(train_loader.dataset) # epoch 전체 평균 loss
        train_losses.append(train_loss)                  # 기록

        # Validation
        model.eval() # 모델을 평가 모드로 전환 (드롭아웃/배치정규화 등 비활성화) 
        running = 0.0
        with torch.no_grad(): # 평가 단계에서는 그래디언트 계산 안 함 -> 메모리 절약
            # 지금 모델이 처음 보는 데이터에서 얼마나 잘 맞추는가?
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                pred = model(xb)
                loss = criterion(pred, yb)
                running += loss.item() * xb.size(0)

        val_loss = running / len(val_loader.dataset) # Validation 평균 loss
        val_losses.append(val_loss)

        if epoch % 5 == 0 or epoch == 1 or epoch == EPOCHS:
            print(f"[Epoch {epoch:03d}/{EPOCHS}] train={train_loss:.6f}  val={val_loss:.6f}")

    # ==============================
    # Plot loss curves
    # ==============================
    plt.figure(figsize=(7, 4))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("ResidualTorqueNN Offline Training Loss")
    plt.grid(True)
    plt.legend()
    plt.show()

    # ==============================
    # Error metrics on val set
    # ==============================
    model.eval()
    with torch.no_grad():
        pred_val = model(x_val_t.to(device)).cpu().numpy()

    err = y_val - pred_val  # (N_val, 3)
    mean_abs_err = np.mean(np.abs(err), axis=0) # 각 관절별 평균 절대 오차
    max_abs_err = np.max(np.abs(err), axis=0) # 각 관절별 최대 절대 오차

    print("\n[Validation error] joint-wise |Δτ_true - Δτ_hat|")
    for j, name in enumerate(JOINT_NAMES):
        print(f"- {name:6s}: mean={mean_abs_err[j]:.4f} Nm, max={max_abs_err[j]:.4f} Nm")

    # ==============================
    # Save model : 모델 저장 !
    # ==============================
    save_path = "/home/seohy/colcon_ws/src/olaf/ffw/code/TrainNN/residual_nn.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "delta_tau_max": DELTA_TAU_MAX,
            # "input_dim": 9,
            "input_dim": 3,
            "output_dim": 3,
            "joint_names": JOINT_NAMES,
        },
        save_path,
    )
    print(f"\nSaved trained model: {save_path}")

  
    plt.figure(figsize=(10,3))
    for i, name in enumerate(JOINT_NAMES):
        plt.subplot(1,3,i+1)
        plt.hist(y[:,i], bins=100)
        plt.title(name)
        plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
