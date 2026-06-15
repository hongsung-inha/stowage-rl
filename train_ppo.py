#!/usr/bin/env python3
# ══════════════════════════════════════════════════════════════════════
#  train_ppo.py — Container Stowage RL (v15) VESSL 실험 러너
#
#  사용법:
#    python train_ppo.py --config config.yaml
#    python train_ppo.py --config config.yaml --output_base /shared/run1 --seeds 42 123
#
#  · stowage_core.py 의 환경/보상/학습함수/export 를 그대로 재사용
#  · config.yaml 로 seed/정책/에피소드/PPO 하이퍼파라미터 외부화
#  · 산출물(모델·CSV·Excel·TensorBoard) 은 모두 OUTPUT_BASE 하위에 저장
#    (google.colab files.download 없음 — VESSL 볼륨에 영구 저장)
# ══════════════════════════════════════════════════════════════════════
import os, sys, copy, time, json, argparse

def parse_args():
    p = argparse.ArgumentParser(description="Stowage RL PPO trainer (VESSL)")
    p.add_argument("--config", default="config.yaml", help="YAML 설정 파일")
    p.add_argument("--output_base", default=None,
                   help="출력 루트 (지정 시 VESSL_OUTPUT_DIR 보다 우선)")
    p.add_argument("--seeds", type=int, nargs="*", default=None,
                   help="시드 목록 (config 의 seeds 덮어씀)")
    p.add_argument("--policies", nargs="*", default=None,
                   help="학습할 정책 (ES SF EF BL 중)")
    return p.parse_args()


def load_yaml(path):
    import yaml
    if not os.path.exists(path):
        print(f"⚠️ config 파일 없음: {path} — 코드 기본값 사용")
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def main():
    args = parse_args()
    cfg_yaml = load_yaml(args.config)

    # ── OUTPUT_BASE 결정: CLI > config.output_base > VESSL_OUTPUT_DIR > /shared ──
    ob = args.output_base or cfg_yaml.get("output_base") or ""
    if ob:
        os.environ["VESSL_OUTPUT_DIR"] = ob   # core 가 이 env 를 읽음

    # core import (env 주입 후) — 여기서 OUTPUT_BASE/RESULTS_DIR 확정
    import stowage_core as core

    CONFIG   = core.CONFIG
    POLICIES = core.POLICIES

    # ── config.yaml 오버라이드 적용 ──
    seeds = args.seeds or cfg_yaml.get("seeds", [core.GLOBAL_SEED])
    sel_policies = args.policies or cfg_yaml.get("policies", list(POLICIES.keys()))

    if "experiment_name" in cfg_yaml:
        CONFIG["experiment_name"] = cfg_yaml["experiment_name"]
    if "total_rounds" in cfg_yaml:
        CONFIG["total_rounds"] = cfg_yaml["total_rounds"]
    if "episodes_per_level" in cfg_yaml:
        # YAML 키가 문자열이면 int 로 변환
        CONFIG["episodes_per_level"] = {
            int(k): int(v) for k, v in cfg_yaml["episodes_per_level"].items()
        }
    # PPO 하이퍼파라미터: 코드 기본 CONFIG['ppo'] 를 유지하며 일부만 덮어씀
    # (config.yaml 의 ppo 키 이름과 코드 키 이름이 다를 수 있어 안전 매핑)
    if "ppo" in cfg_yaml and isinstance(cfg_yaml["ppo"], dict):
        _keymap = {"learning_rate": "lr"}  # yaml → code 키 별칭
        for yk, yv in cfg_yaml["ppo"].items():
            ck = _keymap.get(yk, yk)
            if ck in CONFIG.get("ppo", {}):
                CONFIG["ppo"][ck] = yv

    # 선택된 정책만 남기기
    POLICIES = {k: v for k, v in POLICIES.items() if k in sel_policies}
    assert POLICIES, f"유효한 정책 없음: {sel_policies}"

    RESULTS_DIR = core.RESULTS_DIR
    CKPT_DIR    = os.path.join(RESULTS_DIR, "checkpoints")
    os.makedirs(CKPT_DIR, exist_ok=True)

    print("=" * 72)
    print("🚀 VESSL PPO 학습 시작")
    print(f"   experiment : {CONFIG['experiment_name']}")
    print(f"   output     : {core.OUTPUT_BASE}")
    print(f"   seeds      : {seeds}")
    print(f"   policies   : {list(POLICIES.keys())}")
    print(f"   rounds     : {CONFIG['total_rounds']}")
    print(f"   episodes/lv: {CONFIG['episodes_per_level']}")
    print("=" * 72)

    grand_t0 = time.time()
    summary = []

    for seed in seeds:
        print("\n" + "▒" * 72)
        print(f"  SEED = {seed}")
        print("▒" * 72)
        for pol_idx, (pol_name, pol_spec) in enumerate(POLICIES.items(), 1):
            print(f"\n  🎯 [{pol_idx}/{len(POLICIES)}] policy={pol_name} | seed={seed}")
            c = copy.deepcopy(CONFIG)
            c["experiment_name"] = f"{CONFIG['experiment_name']}_{pol_name}_seed{seed}"
            c["lexicographic"]   = pol_spec

            logger = core.ExperimentLogger(c["experiment_name"], c)
            t0 = time.time()
            res = core._run_training(c, logger, seed)
            elapsed = time.time() - t0

            # ── 모델 체크포인트 저장 ──
            try:
                mpath = os.path.join(CKPT_DIR, c["experiment_name"])
                res["trained_model"].save(mpath)
                print(f"     💾 model → {mpath}.zip")
            except Exception as e:
                print(f"     ⚠️ model save skipped: {e}")

            final_r = res["avg_rewards"][-1]
            summary.append({
                "seed": seed, "policy": pol_name,
                "final_reward": round(float(final_r), 3),
                "elapsed_min": round(elapsed / 60, 2),
                "csv": logger.csv_path,
            })
            print(f"     ✅ done — final reward(Lv4)={final_r:+.2f} | {elapsed/60:.1f} min")

    # ── 전체 요약 저장 ──
    summ_path = os.path.join(RESULTS_DIR, "run_summary.json")
    with open(summ_path, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": CONFIG["experiment_name"],
            "total_min": round((time.time() - grand_t0) / 60, 2),
            "runs": summary,
        }, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 72)
    print(f"✅ 전체 완료 — {(time.time()-grand_t0)/60:.1f} min")
    print(f"   요약 저장: {summ_path}")
    print(f"   산출물 루트: {RESULTS_DIR}")
    print("=" * 72)


if __name__ == "__main__":
    main()
