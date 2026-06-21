#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ══════════════════════════════════════════════════════════════════════
#  aggregate_73.py — 7:3 실험 결과 병합 러너 (4개 TRAIN run 종료 후 실행)
#
#  역할:
#    1) (선택) S3_RESULTS_URI 가 설정돼 있으면 S3 에서 policies/ 를 먼저 동기화
#       → Object storage 마운트가 'sync형'이라 시작 시 기존 결과가 안 보일 때의 대비책
#    2) RESULTS_DIR/policies/<정책>/result.pkl 4개를 읽어 all_results 로 병합
#    3) 병합 결과를 aggregate_merged.pkl 로 저장 (그림/Excel 노트북이 이어받기 좋게)
#
#  사용:
#    python aggregate_73.py                      # 마운트된 /output 직접 사용
#    S3_RESULTS_URI=s3://.../results python aggregate_73.py   # S3 동기화 후 병합
#
#  · 그림/Excel/데이터셋 생성(Cell 11~16)은 무겁고 종류가 많아, 병합 산출물을
#    내보낸 뒤 AGGREGATE 노트북에서 이어서 실행하는 것을 권장 (가이드 참고).
# ══════════════════════════════════════════════════════════════════════
import os, sys, pickle, json, subprocess, datetime, argparse


def maybe_sync_from_s3(results_dir: str):
    """S3_RESULTS_URI 가 있으면 policies/ 를 로컬 RESULTS_DIR 로 내려받는다."""
    s3 = os.environ.get("S3_RESULTS_URI", "").rstrip("/")
    if not s3:
        print("ℹ️  S3_RESULTS_URI 미설정 — 마운트된 경로를 그대로 사용")
        return
    dst = os.path.join(results_dir, "policies")
    os.makedirs(dst, exist_ok=True)
    src = f"{s3}/policies/"
    print(f"⬇️  S3 동기화: {src} → {dst}")
    try:
        subprocess.run(["aws", "s3", "cp", src, dst, "--recursive"], check=True)
        print("✅ S3 동기화 완료")
    except FileNotFoundError:
        print("❌ aws CLI 없음 — `pip install awscli` 후 자격증명(AWS_*) 설정 필요")
        raise
    except subprocess.CalledProcessError as e:
        print(f"❌ S3 동기화 실패: {e}")
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_base", default=None)
    ap.add_argument("--require_all", action="store_true",
                    help="4개 정책이 모두 있어야 진행 (없으면 에러)")
    args = ap.parse_args()
    if args.output_base:
        os.environ["VESSL_OUTPUT_DIR"] = args.output_base

    import stowage_core_73 as core
    POLICIES    = core.POLICIES
    RESULTS_DIR = core.RESULTS_DIR
    CBData      = core.CBData          # pickle 복원에 필요 (이름 보장)

    # 0) 필요 시 S3 에서 먼저 가져오기
    maybe_sync_from_s3(RESULTS_DIR)

    POL_BASE = os.path.join(RESULTS_DIR, "policies")
    print("=" * 70)
    print("🔄 v15 7:3 — 4개 정책 결과 로드·병합 (AGGREGATE)")
    print(f"   policies base: {POL_BASE}")
    print("=" * 70)

    # 진단: 실제로 보이는 폴더 출력 (마운트 읽기 여부 확인)
    if os.path.isdir(POL_BASE):
        print("   발견된 정책 폴더:", sorted(os.listdir(POL_BASE)))
    else:
        print(f"   ⚠️ policies 폴더가 없음: {POL_BASE}")

    from stable_baselines3 import PPO

    all_results = {}
    for pol in POLICIES:                       # ES/SF/EF/BL 순서
        pkl = os.path.join(POL_BASE, pol, "result.pkl")
        if not os.path.exists(pkl):
            print(f"  ⚠️  {pol}: result.pkl 없음 — 건너뜀 ({pkl})")
            continue
        with open(pkl, "rb") as f:
            res = pickle.load(f)
        # 모델 복원 (bay plan / reward 분석에 필요)
        mp = res.get("model_path")
        if mp and os.path.exists(mp):
            try:
                res["trained_model"] = PPO.load(mp, device="cpu")
            except Exception as e:
                print(f"  ⚠️  {pol}: 모델 로드 실패 ({e})")
                res["trained_model"] = None
        all_results[pol] = res
        print(f"  ✅ {pol}: loaded | final reward(Lv4) = "
              f"{res['avg_rewards'][-1]:+.2f} | {res.get('elapsed_sec',0)/60:.1f} min")

    if not all_results:
        raise SystemExit(
            "❌ 로드된 정책 결과가 없습니다.\n"
            "   원인 후보: (1) TRAIN run 4개가 아직 안 끝남, "
            "(2) Object storage 마운트가 시작 시 기존 파일을 못 읽음.\n"
            "   → (2)라면 S3_RESULTS_URI 환경변수를 설정해 S3 동기화로 우회하세요.")

    if args.require_all and len(all_results) < len(POLICIES):
        missing = [p for p in POLICIES if p not in all_results]
        raise SystemExit(f"❌ --require_all: 누락 정책 {missing}")

    # 병합 산출물 저장 (그림/Excel 노트북이 이어받도록)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    agg_dir = os.path.join(RESULTS_DIR, f"aggregate_{ts}")
    os.makedirs(agg_dir, exist_ok=True)
    # trained_model 은 pickle 불가 → 제거 후 저장 (경로는 유지)
    slim_all = {}
    for pol, res in all_results.items():
        r = dict(res); r["trained_model"] = None
        slim_all[pol] = r
    with open(os.path.join(agg_dir, "aggregate_merged.pkl"), "wb") as f:
        pickle.dump(slim_all, f)

    # 요약 표 (CSV)
    import csv
    summ = os.path.join(agg_dir, "summary.csv")
    with open(summ, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["policy", "final_reward_lv4", "elapsed_min", "model_path"])
        for pol, res in all_results.items():
            w.writerow([pol, round(float(res["avg_rewards"][-1]), 2),
                        round(res.get("elapsed_sec", 0)/60, 1),
                        res.get("model_path", "")])

    print()
    print(f"  ✅ 병합 완료: {list(all_results.keys())}")
    print(f"  📦 병합 산출물 → {agg_dir}/aggregate_merged.pkl")
    print(f"  📄 요약표      → {summ}")
    print("=" * 70)


if __name__ == "__main__":
    main()
