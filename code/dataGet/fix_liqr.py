#!/usr/bin/env python3
"""
🔧 iLQR 자동 설치 및 수정 스크립트
==================================

이 스크립트를 실행하면 자동으로 문제를 진단하고 수정합니다.

실행 방법:
    python fix_ilqr.py
"""

import subprocess
import sys
import os

def run_command(cmd, description):
    """명령어 실행 헬퍼"""
    print(f"\n{'='*70}")
    print(f"🔄 {description}")
    print(f"{'='*70}")
    print(f"실행: {cmd}")
    
    try:
        result = subprocess.run(
            cmd, 
            shell=True, 
            check=True,
            capture_output=True,
            text=True
        )
        print(f"✅ 성공!")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 실패: {e}")
        if e.stderr:
            print(f"에러: {e.stderr}")
        return False


def main():
    print("""
╔══════════════════════════════════════════════════════════════╗
║              🔧 iLQR 자동 수정 스크립트                      ║
╠══════════════════════════════════════════════════════════════╣
║                                                              ║
║  이 스크립트는 자동으로:                                      ║
║  1. Python 버전 확인                                         ║
║  2. 필요한 패키지 설치/업데이트                               ║
║  3. iLQR 라이브러리 설치                                     ║
║  4. 설치 확인                                                ║
║                                                              ║
║  주의: 인터넷 연결이 필요합니다!                              ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    # 사용자 확인
    response = input("\n계속하시겠습니까? (y/n): ")
    if response.lower() != 'y':
        print("취소되었습니다.")
        return
    
    print(f"\n시작합니다...\n")
    
    # ========================================================================
    # 1. Python 버전 확인
    # ========================================================================
    print("\n" + "="*70)
    print("1️⃣  Python 버전 확인")
    print("="*70)
    
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version < (3, 7):
        print("❌ Python 3.7 이상이 필요합니다!")
        print("현재 Python을 업그레이드하세요.")
        return
    else:
        print("✅ Python 버전 OK")
    
    # ========================================================================
    # 2. pip 업그레이드
    # ========================================================================
    run_command(
        f"{sys.executable} -m pip install --upgrade pip",
        "pip 업그레이드"
    )
    
    # ========================================================================
    # 3. 기본 패키지 설치/업그레이드
    # ========================================================================
    packages = [
        "numpy",
        "sympy", 
        "numba",
        "scipy",
        "matplotlib"
    ]
    
    for package in packages:
        run_command(
            f"{sys.executable} -m pip install --upgrade {package}",
            f"{package} 설치/업그레이드"
        )
    
    # ========================================================================
    # 4. MuJoCo 설치 (선택사항)
    # ========================================================================
    print("\n" + "="*70)
    print("4️⃣  MuJoCo 설치")
    print("="*70)
    
    try:
        import mujoco
        print(f"✅ MuJoCo 이미 설치됨 (버전: {mujoco.__version__})")
    except ImportError:
        print("MuJoCo 설치 중...")
        run_command(
            f"{sys.executable} -m pip install mujoco",
            "MuJoCo 설치"
        )
    
    # ========================================================================
    # 5. iLQR 설치
    # ========================================================================
    print("\n" + "="*70)
    print("5️⃣  iLQR 설치")
    print("="*70)
    
    # 기존 iLQR 제거
    print("기존 iLQR 제거 중...")
    subprocess.run(
        f"{sys.executable} -m pip uninstall ilqr -y",
        shell=True,
        capture_output=True
    )
    
    # 새로 설치
    success = run_command(
        f"{sys.executable} -m pip install git+https://github.com/Bharath2/iLQR.git",
        "iLQR 설치"
    )
    
    if not success:
        print("\n❌ iLQR 설치 실패!")
        print("\n수동 설치를 시도하세요:")
        print("  pip install git+https://github.com/Bharath2/iLQR.git")
        return
    
    # ========================================================================
    # 6. 설치 확인
    # ========================================================================
    print("\n" + "="*70)
    print("6️⃣  설치 확인")
    print("="*70)
    
    errors = []
    
    # 6-1. ilqr 기본 import
    try:
        import ilqr
        print("✅ ilqr 패키지 import 성공")
    except ImportError as e:
        print(f"❌ ilqr import 실패: {e}")
        errors.append("ilqr")
    
    # 6-2. ilqr 서브모듈
    try:
        from ilqr import iLQR
        from ilqr.containers import Dynamics, Cost
        from ilqr.utils import GetSyms, Bounded
        print("✅ ilqr 서브모듈 import 성공")
    except ImportError as e:
        print(f"❌ ilqr 서브모듈 import 실패: {e}")
        errors.append("ilqr submodules")
    
    # 6-3. 의존성 확인
    deps = ['numpy', 'sympy', 'numba', 'scipy']
    for dep in deps:
        try:
            __import__(dep)
            print(f"✅ {dep}")
        except ImportError as e:
            print(f"❌ {dep}: {e}")
            errors.append(dep)
    
    # ========================================================================
    # 7. 간단한 테스트
    # ========================================================================
    if not errors:
        print("\n" + "="*70)
        print("7️⃣  기능 테스트")
        print("="*70)
        
        try:
            import numpy as np
            import sympy as sp
            from ilqr import iLQR
            from ilqr.containers import Dynamics, Cost
            from ilqr.utils import GetSyms
            
            # 간단한 시스템
            def f(x, u):
                return np.array([x[1], u[0]])
            
            dynamics = Dynamics.Continuous(f, dt=0.1)
            
            x, u = GetSyms(2, 1)
            L = x[0]**2 + 0.1*u[0]**2
            Lf = 10*x[0]**2
            cost = Cost.Symbolic(L, Lf, x, u)
            
            controller = iLQR(dynamics, cost)
            
            print("✅ iLQR 컨트롤러 생성 성공")
            print("⏳ 최적화 테스트 중... (첫 실행은 느림)")
            
            x0 = np.array([1.0, 0.0])
            us_init = np.zeros((10, 1))
            xs, us, cost_trace = controller.fit(x0, us_init)
            
            print(f"✅ 최적화 테스트 성공! (iterations: {len(cost_trace)})")
            
        except Exception as e:
            print(f"❌ 테스트 실패: {e}")
            import traceback
            traceback.print_exc()
            errors.append("test")
    
    # ========================================================================
    # 8. 결과
    # ========================================================================
    print("\n" + "="*70)
    if errors:
        print("❌ 설치 실패")
        print("="*70)
        print(f"\n문제 항목: {', '.join(errors)}")
        print("\n추가 도움말:")
        print("  1. 인터넷 연결 확인")
        print("  2. pip 버전 확인: pip --version")
        print("  3. Python 버전 확인: python --version")
        print("  4. 가상환경 사용 권장")
        print("\n수동 설치:")
        print("  pip install numpy sympy numba scipy matplotlib")
        print("  pip install git+https://github.com/Bharath2/iLQR.git")
    else:
        print("✅ 모든 설치 완료!")
        print("="*70)
        print("\n🎉 성공!")
        print("\n다음 단계:")
        print("  1. main_ilqr_standalone.py 파일을 프로젝트 폴더에 복사")
        print("  2. mpc_controller_ilqr_standalone.py 파일을 프로젝트 폴더에 복사")
        print("  3. python main_ilqr_standalone.py 실행")
        print("\n주의:")
        print("  - 첫 실행은 Numba 컴파일로 5-10초 소요")
        print("  - 이후 실행은 매우 빠름 (5ms)!")
        print("\n✨ Happy Computing! ✨")


if __name__ == "__main__":
    main()