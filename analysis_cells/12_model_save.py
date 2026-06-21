# ══════════════════════════════════════════════════════════════════
#  ★ NEW: 학습 완료 모델 저장 (BL / SF / EF 3 정책)
#   · 기존 학습 결과(all_results)의 trained_model 을 .zip 으로 저장
#   · Stable-Baselines3 PPO.save() → 추후 PPO.load() 로 재사용 가능
#   · Colab 사용 시 files.download() 로 로컬 다운로드까지 지원
# ══════════════════════════════════════════════════════════════════
import os, zipfile

MODEL_DIR = "/content/results/models"
os.makedirs(MODEL_DIR, exist_ok=True)

print("═" * 70)
print("  💾 학습 완료 모델 저장 (PPO, 3-way: BL / SF / EF)")
print("═" * 70)

saved_model_paths = {}
for pol_name, res in all_results.items():
    model_obj = res.get("trained_model", None)
    if model_obj is None:
        print(f"  ⚠️  {pol_name}: trained_model 이 None — 저장 건너뜀")
        continue

    # 파일명: 실험명 + 정책명 + seed
    fname = f"{CONFIG['experiment_name']}_{pol_name}_seed{GLOBAL_SEED}"
    fpath = os.path.join(MODEL_DIR, fname)            # SB3 가 .zip 자동 부착
    model_obj.save(fpath)
    zip_path = fpath + ".zip"
    size_mb  = os.path.getsize(zip_path) / 1e6
    saved_model_paths[pol_name] = zip_path
    print(f"  ✅ {pol_name:<3} 저장 완료 → {os.path.basename(zip_path)}  ({size_mb:.2f} MB)")
    print(f"        Phase1={POLICIES[pol_name]['phase1_mode']:<15} "
          f"Phase2={POLICIES[pol_name]['phase2_mode']:<6} "
          f"FinalReward(Lv4)={res['avg_rewards'][-1]:+.2f}")

# ── 3개 모델을 하나의 zip 으로 묶기 (배포/백업용) ──
bundle_path = os.path.join(MODEL_DIR,
                           f"{CONFIG['experiment_name']}_ALL_models_seed{GLOBAL_SEED}.zip")
if saved_model_paths:
    with zipfile.ZipFile(bundle_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for pol_name, p in saved_model_paths.items():
            zf.write(p, arcname=os.path.basename(p))
    print()
    print(f"  📦 통합 번들 생성 → {os.path.basename(bundle_path)} "
          f"({os.path.getsize(bundle_path)/1e6:.2f} MB)")

print()
print("  ── 재사용 예시 (추론/추가학습) ──")
print("    from stable_baselines3 import PPO")
print(f"    model = PPO.load('{MODEL_DIR}/{CONFIG['experiment_name']}_SF_seed{GLOBAL_SEED}.zip')")
print("    # model.set_env(eval_env)  # 추가 학습 시")

# ── Colab 다운로드 (개별 + 번들) ──
try:
    from google.colab import files
    print()
    print("  ⬇️  Colab 다운로드 시작 (브라우저)…")
    if saved_model_paths:
        files.download(bundle_path)        # 번들 다운로드
    print("  ✅ 다운로드 트리거 완료")
except Exception as e:
    print(f"  ℹ️  (로컬/비Colab 환경) 다운로드 생략 — 저장 경로: {MODEL_DIR}")
