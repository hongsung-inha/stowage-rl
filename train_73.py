#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ══════════════════════════════════════════════════════════════════════
#  train_73.py — 7:3 비율 실험 TRAIN 러너 (단일 정책, VESSL 병렬용)
#
#  사용:
#    POLICY=SF python train_73.py            # 환경변수로 정책 지정
#    python train_73.py --policy SF          # 또는 인자로
#
#  · stowage_core_73.py 의 환경/보상/학습/CBData 를 재사용
#  · 결과: RESULTS_DIR/policies/<POLICY>/ 에 model.zip / result.pkl / done.json
#  · 4개 정책(ES/SF/EF/BL) run 을 병렬로 띄우면 같은 볼륨의 서로 다른 폴더에 저장됨
# ══════════════════════════════════════════════════════════════════════
import os, sys, copy, time, pickle, json, argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", default=None, help="ES/SF/EF/BL (없으면 env POLICY)")
    ap.add_argument("--output_base", default=None, help="출력 루트(=VESSL_OUTPUT_DIR 우선)")
    args = ap.parse_args()

    if args.output_base:
        os.environ["VESSL_OUTPUT_DIR"] = args.output_base

    # core import (env 주입 후 → OUTPUT_BASE/RESULTS_DIR 확정)
    import stowage_core_73 as core

    POLICIES   = core.POLICIES
    CONFIG     = core.CONFIG
    GLOBAL_SEED = core.GLOBAL_SEED
    RESULTS_DIR = core.RESULTS_DIR

    POLICY = (args.policy or os.environ.get("POLICY", "")).strip().upper()
    if POLICY not in POLICIES:
        raise SystemExit(
            f"❌ POLICY 를 ES/SF/EF/BL 중 하나로 지정하세요 (현재: {POLICY!r}).\n"
            "   · VESSL: run 의 Environment Variables 에 POLICY=ES (정책별 4개 run)\n"
            "   · 로컬:  POLICY=ES python train_73.py")

    policy_spec = POLICIES[POLICY]
    POL_DIR = os.path.join(RESULTS_DIR, "policies", POLICY)
    os.makedirs(POL_DIR, exist_ok=True)

    print("=" * 70)
    print(f"🚀 v15 7:3 — 단일 정책 병렬 학습 (POLICY={POLICY})")
    print("=" * 70)
    print(f"  Policy:     {POLICY} — {policy_spec['kor']}")
    print(f"  Mode:       {policy_spec['phase1_mode']} (전 라운드 고정)")
    print(f"  Seed:       {GLOBAL_SEED}")
    print(f"  Episodes:   {sum(CONFIG['episodes_per_level'].values()):,}")
    print(f"  Output dir: {POL_DIR}")
    print()

    cfg = copy.deepcopy(CONFIG)
    cfg["experiment_name"] = f"v15_tier_{POLICY}"
    cfg["lexicographic"]   = policy_spec

    pol_logger = core.ExperimentLogger(cfg["experiment_name"], cfg)

    t0  = time.time()
    res = core._run_training(cfg, pol_logger, GLOBAL_SEED)
    elapsed = time.time() - t0

    res["policy_name"] = POLICY
    res["policy_spec"] = policy_spec
    res["elapsed_sec"] = elapsed

    # 1) PPO 모델 저장
    model_path = os.path.join(POL_DIR, f"model_{POLICY}_seed{GLOBAL_SEED}")
    try:
        res["trained_model"].save(model_path)
        print(f"  💾 model saved → {model_path}.zip")
    except Exception as e:
        print(f"  ⚠️ model save 실패: {e}")

    # 2) 결과 직렬화 (모델 제거 + 콜백 경량화)
    slim = dict(res)
    slim["trained_model"] = None
    slim["model_path"]    = model_path + ".zip"
    slim["callbacks"]     = [core.CBData(cb.timesteps, cb.ep_rewards, cb.ep_lengths)
                             for cb in res["callbacks"]]
    slim["log_dir"]       = pol_logger.log_dir
    with open(os.path.join(POL_DIR, "result.pkl"), "wb") as f:
        pickle.dump(slim, f)
    print(f"  💾 result.pkl saved → {POL_DIR}/result.pkl")

    # 3) 완료 마커
    with open(os.path.join(POL_DIR, "done.json"), "w", encoding="utf-8") as f:
        json.dump({"policy": POLICY,
                   "final_reward_lv4": float(res["avg_rewards"][-1]),
                   "elapsed_sec": elapsed,
                   "episodes": sum(CONFIG["episodes_per_level"].values()),
                   "model_path": model_path + ".zip",
                   "seed": GLOBAL_SEED}, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 70)
    print(f"  ✅ {POLICY} 학습 완료 — final reward (Lv4) = "
          f"{res['avg_rewards'][-1]:+.2f} | {elapsed/60:.1f} min")
    print(f"  → 4개 정책 run 이 모두 끝나면 aggregate_73.py 로 병합/비교")
    print("=" * 70)


if __name__ == "__main__":
    main()
