"""
Evaluation Module: MPC only vs MPC + NN

두 실험의 TrackingLogger 데이터를 받아,
동일한 메트릭으로 구간별 비교를 수행합니다.

사용법:
    evaluator = PerformanceEvaluator(dt=0.005)

    # 각 실험 후 TrackingLogger를 넘김
    result_mpc     = evaluator.evaluate(tracking_logger_mpc,     label="MPC only")
    result_mpc_nn  = evaluator.evaluate(tracking_logger_mpc_nn,  label="MPC + NN")

    # 두 결과를 비교 출력
    evaluator.compare(result_mpc, result_mpc_nn)
"""

import numpy as np
from dataGet.config import TrajectoryConfig


class PerformanceEvaluator:
    """
    구간별 성능 메트릭 계산 및 두 실험 비교

    구간 정의 (TrajectoryConfig 기준):
        전이구간 (transition) : 0            ~ T_raise
        정지구간 (steady)     : T_raise + T_wait ~
    """

    def __init__(self, dt):
        """
        Args:
            dt: 시뮬 타임스텝 (SimulationConfig.SIM_DT)
        """
        self.dt = dt

        cfg = TrajectoryConfig
        self.t_raise = cfg.T_RAISE
        self.t_steady = cfg.T_RAISE + cfg.T_WAIT   # 정지구간 시작점

    # ─────────────────────────────────────────────
    # 메트릭 계산
    # ─────────────────────────────────────────────
    def evaluate(self, tracking_logger, label=""):
        """
        TrackingLogger 데이터로부터 전체 메트릭을 계산합니다.

        Args:
            tracking_logger: TrackingLogger 인스턴스 (이미 데이터 수집 완료)
            label: 실험 이름 (출력용)

        Returns:
            dict: 구간별 메트릭 전체
        """
        arrays = tracking_logger.get_arrays()

        t             = arrays['time']
        q_ref         = arrays['shoulder_ref']
        q_act         = arrays['shoulder_act']
        tau           = arrays['tau_shoulder']
        error         = q_ref - q_act

        # 구간 인덱스
        idx_trans  = t <= self.t_raise                          # 전이구간
        idx_steady = t >= self.t_steady                         # 정지구간

        # 목표값 (정지구간의 참조값 평균 ≈ SHOULDER_TARGET)
        q_target = np.mean(q_ref[idx_steady]) if idx_steady.sum() > 0 else q_ref[-1]
        q_start  = q_ref[0]
        travel   = abs(q_target - q_start)                      # 총 이동 범위

        result = {
            'label': label,
            # ── 전체 구간 ──
            'rmse_all':            self._rmse(error),
            'max_error_all':       np.max(np.abs(error)),
            # ── 전이구간 ──
            'rmse_transition':     self._rmse(error[idx_trans]),
            'max_error_transition':np.max(np.abs(error[idx_trans])) if idx_trans.sum() > 0 else 0.0,
            # ── 정지구간 ──
            'rmse_steady':         self._rmse(error[idx_steady]),
            'steady_state_error':  np.mean(np.abs(error[idx_steady])) if idx_steady.sum() > 0 else 0.0,
            # ── 동적 성능 ──
            'rise_time':           self._rise_time(t, q_act, q_start, q_target),
            'settling_time':       self._settling_time(t, error, q_target, travel),
            'overshoot':           self._overshoot(q_act, q_start, q_target, travel),
            # ── 토크 효율 ──
            'mean_abs_tau':        np.mean(np.abs(tau)),
            'peak_tau':            np.max(np.abs(tau)),
            'control_effort':      np.sum(tau ** 2) * self.dt,  # integral(tau^2 dt)
        }

        return result

    # ─────────────────────────────────────────────
    # 비교 출력
    # ─────────────────────────────────────────────
    def compare(self, result_a, result_b):
        """
        두 실험 결과를 나를 곱셈 테이블로 비교 출력합니다.

        Args:
            result_a: evaluate() 반환값 (예: MPC only)
            result_b: evaluate() 반환값 (예: MPC + NN)
        """
        metrics = [
            # (키, 표시명, 단위, 낮을수록좋음?)
            ('rmse_all',              'RMSE (전체)',           'rad',  True),
            ('rmse_transition',       'RMSE (전이구간)',       'rad',  True),
            ('rmse_steady',           'RMSE (정지구간)',       'rad',  True),
            ('max_error_all',         'Max Error',            'rad',  True),
            ('steady_state_error',    'Steady-State Error',   'rad',  True),
            ('rise_time',             'Rise Time (90%)',       's',    True),
            ('settling_time',         'Settling Time (±2%)',   's',    True),
            ('overshoot',             'Overshoot',            '%',    True),
            ('mean_abs_tau',          'Mean |tau|',           'Nm',   True),
            ('peak_tau',              'Peak |tau|',           'Nm',   True),
            ('control_effort',        'Control Effort',       'Nm²s', True),
        ]

        label_a = result_a.get('label', 'A')
        label_b = result_b.get('label', 'B')

        col_w = 26                          # 메트릭 이름 열 폭
        val_w = 14                          # 값 열 폭

        print("\n" + "=" * (col_w + val_w * 3 + 12))
        print(f"  📊 Performance Comparison: {label_a}  vs  {label_b}")
        print("=" * (col_w + val_w * 3 + 12))

        # 헤더
        print(f"  {'Metric':<{col_w}} {label_a:>{val_w}} {label_b:>{val_w}} {'Winner':>10}")
        print("  " + "-" * (col_w + val_w * 2 + 12))

        # 구간 구분선용
        prev_group = None
        group_map = {
            'rmse_all':             'Tracking',
            'rmse_transition':      'Tracking',
            'rmse_steady':          'Tracking',
            'max_error_all':        'Tracking',
            'steady_state_error':   'Tracking',
            'rise_time':            'Dynamic',
            'settling_time':        'Dynamic',
            'overshoot':            'Dynamic',
            'mean_abs_tau':         'Torque',
            'peak_tau':             'Torque',
            'control_effort':       'Torque',
        }

        for key, name, unit, lower_is_better in metrics:
            # 그룹 구분선
            grp = group_map.get(key, '')
            if grp != prev_group:
                if prev_group is not None:
                    print("  " + "-" * (col_w + val_w * 2 + 12))
                prev_group = grp

            va = result_a.get(key, float('nan'))
            vb = result_b.get(key, float('nan'))

            # 퍼센트 단위는 별도 포맷
            if unit == '%':
                sa = f"{va:>10.2f} %"
                sb = f"{vb:>10.2f} %"
            else:
                sa = f"{va:>10.4f} {unit}"
                sb = f"{vb:>10.4f} {unit}"

            # Winner 판정
            if np.isnan(va) or np.isnan(vb):
                winner = "N/A"
            elif lower_is_better:
                winner = label_a if va < vb else (label_b if vb < va else "tie")
            else:
                winner = label_a if va > vb else (label_b if vb > va else "tie")

            # 강조: 이기는 쪽 볼드 표시 (텍스트로)
            marker_a = " ◀" if winner == label_a else ""
            marker_b = " ◀" if winner == label_b else ""

            print(f"  {name:<{col_w}} {sa}{marker_a:>2} {sb}{marker_b:>2}")

        print("=" * (col_w + val_w * 3 + 12))

    # ─────────────────────────────────────────────
    # 내부 헬퍼
    # ─────────────────────────────────────────────
    @staticmethod
    def _rmse(error):
        if len(error) == 0:
            return 0.0
        return np.sqrt(np.mean(error ** 2))

    @staticmethod
    def _rise_time(t, q_act, q_start, q_target):
        """목표의 90%에 처음 도달하는 시간"""
        threshold = q_start + 0.9 * (q_target - q_start)
        # 부호 방향 고려 (올라가는 경우 vs 내려가는 경우)
        if q_target > q_start:
            crossed = np.where(q_act >= threshold)[0]
        else:
            crossed = np.where(q_act <= threshold)[0]

        if len(crossed) == 0:
            return float('nan')   # 목표 미달성
        return t[crossed[0]]

    @staticmethod
    def _settling_time(t, error, q_target, travel):
        """목표의 ±2% 안에 계속 머무는 시간"""
        if travel == 0:
            return 0.0
        band = 0.02 * travel   # ±2% of total travel

        # 마지막에서 역방향으로 band 밖에 나간 마지막 시점 찾기
        outside = np.where(np.abs(error) > band)[0]
        if len(outside) == 0:
            return t[0]          # 처음부터 band 안
        return t[outside[-1]]

    @staticmethod
    def _overshoot(q_act, q_start, q_target, travel):
        """오버슈트 퍼센트: (최대 오버 거리 / 총 이동거리) * 100"""
        if travel == 0:
            return 0.0

        if q_target > q_start:
            # 올라가는 방향: max가 target을 넘는 경우
            overshoot_dist = max(0.0, np.max(q_act) - q_target)
        else:
            # 내려가는 방향: min이 target을 넘는 경우
            overshoot_dist = max(0.0, q_target - np.min(q_act))

        return (overshoot_dist / travel) * 100.0