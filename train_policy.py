#!/usr/bin/env python3
# ══════════════════════════════════════════════════════════════════
#  train_policy.py — VESSL 병렬 워커 (정책 1개 학습)
#
#  VESSL 배치 잡 6개를 동시에 띄우고, 각 잡은 POLICY_NAME 환경변수로
#  지정된 정책 하나만 학습한다. 결과는 공유 오브젝트 스토리지(/output)에
#  정책별 폴더로 저장되어, 이후 aggregate.py 가 통합 분석한다.
#
#  ── 입력(환경변수) ─────────────────────────────────────────────
#    POLICY_NAME   : 학습할 정책. 다음 중 하나
#                    ES / SF_9_EF_1 / SF_7_EF_3 / BL_SF_5_EF_5 /
#                    EF_7_SF_3 / EF_9_SF_1
#                    (미지정 시 ALL 로 간주하고 6개 순차 학습 — 단일 GPU 폴백)
#    OUTPUT_ROOT   : 결과 루트 (기본 /output)  ← VESSL object storage 마운트
#    SEED          : 시드 (기본 42)
#
#  ── 출력(/output/<policy>/) ────────────────────────────────────
#    model.zip            : SB3 PPO 학습 모델 (재추론·bay plan 용)
#    result.pkl           : 통합 분석에 필요한 직렬화 결과 (모델 제외)
#    meta.json            : 사람이 읽는 요약 (final reward, 시간 등)
#    logger_artifacts/    : ExperimentLogger 가 만든 csv/json/png/xlsx
#    _SUCCESS             : 정상 종료 마커 (aggregate 가 6개 확인)
# ══════════════════════════════════════════════════════════════════
import os, sys, json, time, pickle, shutil, copy

OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "/output")
SEED        = int(os.environ.get("SEED", "42"))
POLICY_NAME = os.environ.get("POLICY_NAME", "ALL").strip()

os.environ["OUTPUT_ROOT"] = OUTPUT_ROOT          # core 의 ExperimentLogger 가 참조
os.makedirs(OUTPUT_ROOT, exist_ok=True)

# core 모듈 (정의 + CONFIG + POLICIES) 로드
import stowage_core as sc


# ── 직렬화 가능한 형태로 result 정리 (SB3 모델·콜백 객체 제거) ──
def _portable_result(res: dict) -> dict:
    """trained_model(SB3)·callback 객체를 제거하고 분석에 필요한 필드만 남긴다.
    callback 은 ep_rewards/timesteps 만 추려 list-of-dict 로 보존한다."""
    out = {
        "policy_name":         res.get("policy_name"),
        "policy_spec":         res.get("policy_spec"),
        "elapsed_sec":         res.get("elapsed_sec"),
        "avg_rewards":         res.get("avg_rewards"),
        "std_rewards":         res.get("std_rewards"),
        "levels":              res.get("levels"),
        "per_round_metrics":   res.get("per_round_metrics"),
        "phase_info":          res.get("phase_info"),
        "per_round_bay_plans": res.get("per_round_bay_plans"),
        "final_metrics":       res.get("final_metrics"),
        # 콜백은 그림(Fig1/2/6)에 쓰는 ep_rewards/timesteps 만 추출
        "callbacks_slim": [
            {"ep_rewards": list(cb.ep_rewards), "timesteps": list(cb.timesteps)}
            for cb in res.get("callbacks", [])
        ],
    }
    return out


def train_one(policy_name: str) -> dict:
    assert policy_name in sc.POLICIES, \
        f"Unknown POLICY_NAME={policy_name!r}. 가능: {list(sc.POLICIES.keys())}"
    policy_spec = sc.POLICIES[policy_name]
    out_dir = os.path.join(OUTPUT_ROOT, policy_name)
    os.makedirs(out_dir, exist_ok=True)

    print("█" * 70)
    print(f"  🎯 VESSL worker — policy={policy_name}: {policy_spec['kor']}")
    print(f"     SEED={SEED} | OUTPUT_ROOT={OUTPUT_ROOT}")
    print("█" * 70)

    # 정책별 독립 config
    cfg = copy.deepcopy(sc.CONFIG)
    cfg["experiment_name"] = f"v17_{policy_name}"
    cfg["lexicographic"]   = policy_spec

    logger = sc.ExperimentLogger(cfg["experiment_name"], cfg)

    t0 = time.time()
    res = sc._run_training(cfg, logger, SEED)
    elapsed = time.time() - t0

    res["policy_name"] = policy_name
    res["policy_spec"] = policy_spec
    res["elapsed_sec"] = elapsed

    # 1) SB3 모델 저장 (재추론/bay plan 용)
    model = res.get("trained_model")
    if model is not None:
        model.save(os.path.join(out_dir, "model.zip"))

    # 2) 직렬화 결과 저장 (모델 제외)
    with open(os.path.join(out_dir, "result.pkl"), "wb") as f:
        pickle.dump(_portable_result(res), f, protocol=pickle.HIGHEST_PROTOCOL)

    # 3) logger 산출물(csv/json/png/xlsx) 복사
    art_dir = os.path.join(out_dir, "logger_artifacts")
    os.makedirs(art_dir, exist_ok=True)
    if os.path.isdir(logger.log_dir):
        for fn in os.listdir(logger.log_dir):
            sp = os.path.join(logger.log_dir, fn)
            if os.path.isfile(sp):
                shutil.copy2(sp, art_dir)
    # log_dir 자체 경로도 기록 (aggregate 가 재구성 시 사용)
    meta = {
        "policy_name":   policy_name,
        "mode":          policy_spec["phase1_mode"],
        "seed":          SEED,
        "final_reward":  float(res["avg_rewards"][-1]) if res.get("avg_rewards") else None,
        "elapsed_min":   round(elapsed / 60, 2),
        "levels":        res.get("levels"),
        "log_dir":       logger.log_dir,
        "has_model":     model is not None,
    }
    with open(os.path.join(out_dir, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    # 4) 성공 마커
    open(os.path.join(out_dir, "_SUCCESS"), "w").write(
        f"{policy_name} done @ {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

    print(f"\n  ✅ {policy_name} 저장 완료 → {out_dir}")
    print(f"     final reward (Lv4) = {meta['final_reward']:+.2f} | {meta['elapsed_min']:.1f} min")
    return meta


def main():
    if POLICY_NAME.upper() == "ALL":
        # 단일 GPU 폴백: 6개 순차 학습 (VESSL 미사용/디버그용)
        print("⚠️  POLICY_NAME=ALL — 6개 정책 순차 학습 (단일 노드 폴백)")
        metas = [train_one(pn) for pn in sc.POLICIES.keys()]
        print("\n=== ALL DONE ===")
        for m in metas:
            print(f"  {m['policy_name']:<14} reward={m['final_reward']:+.2f} "
                  f"time={m['elapsed_min']:.1f}min")
    else:
        train_one(POLICY_NAME)


if __name__ == "__main__":
    main()
