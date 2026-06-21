#!/usr/bin/env python3
# ══════════════════════════════════════════════════════════════════
#  aggregate.py — VESSL 통합 분석 잡 (6개 정책 결과 병합 → 전체 분석)
#
#  train_policy.py 6개 잡이 /output/<policy>/ 에 저장한 결과를 모두 로드해
#  원본 노트북의 all_results / all_loggers 를 재구성한 뒤,
#  Cell 5~16 (Figure 1~6, 가설검증, 보상성분, Appendix, Excel, 모델저장,
#  POD/Weight Excel, RDB/LPG, RAG, SFT) 14개 분석 셀을 순차 실행한다.
#
#  ── 입력(환경변수) ─────────────────────────────────────────────
#    OUTPUT_ROOT : 결과 루트 (기본 /output). 하위에 정책별 폴더 존재.
#    SEED        : 시드 (기본 42)
#    REQUIRE_ALL : "1"(기본)이면 6개 정책 _SUCCESS 모두 필요. "0"이면
#                  존재하는 정책만으로 부분 분석.
#
#  ── 출력 ───────────────────────────────────────────────────────
#    /output/_aggregate/  : 모든 그림(png)·표·Excel·데이터셋 산출물
#                           (각 분석 셀이 /content/results 에 쓰지만
#                            preamble 이 OUTPUT_ROOT/_aggregate 로 리다이렉트)
# ══════════════════════════════════════════════════════════════════
import os, sys, json, glob, pickle, types, time

OUTPUT_ROOT = os.environ.get("OUTPUT_ROOT", "/output")
SEED        = int(os.environ.get("SEED", "42"))
REQUIRE_ALL = os.environ.get("REQUIRE_ALL", "1") == "1"
AGG_DIR     = os.path.join(OUTPUT_ROOT, "_aggregate")
os.makedirs(AGG_DIR, exist_ok=True)
os.environ["OUTPUT_ROOT"] = OUTPUT_ROOT

# ── matplotlib: 헤드리스(Agg) 백엔드 (VESSL 잡엔 디스플레이 없음) ──
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.show = lambda *a, **k: None        # show() no-op (저장만)

# ── (A) Colab 종속성 shim: google.colab.files.download → no-op ──
#     분석 셀이 'from google.colab import files; files.download(x)' 를 호출하면
#     파일은 이미 디스크에 저장돼 있으므로 다운로드만 건너뛴다.
_colab = types.ModuleType("google.colab")
_files = types.ModuleType("google.colab.files")
def _noop_download(path, *a, **k):
    print(f"     (download skip — saved at: {path})")
_files.download = _noop_download
_colab.files = _files
sys.modules["google"] = sys.modules.get("google", types.ModuleType("google"))
sys.modules["google.colab"] = _colab
sys.modules["google.colab.files"] = _files

# ── (B) /content/results → OUTPUT_ROOT/_aggregate 리다이렉트 ──
#     분석 셀이 '/content/results/...' 경로에 저장하므로, 그 경로가
#     실제로 _aggregate 를 가리키도록 심볼릭 링크를 만든다.
os.makedirs("/content", exist_ok=True) if os.access("/", os.W_OK) else None
try:
    if not os.path.exists("/content/results"):
        os.makedirs("/content", exist_ok=True)
        os.symlink(AGG_DIR, "/content/results")
        print(f"  🔗 /content/results → {AGG_DIR}")
except OSError as e:
    # 심볼릭 불가 환경: 실제 디렉터리로 생성하고 경로 패치는 셀 내부 기본값에 의존
    os.makedirs("/content/results", exist_ok=True)
    print(f"  ⚠️ symlink 불가({e}) — /content/results 디렉터리 생성")

import stowage_core as sc
import numpy as np


# ── (C) 경량 콜백 객체 (분석 셀이 cb.ep_rewards / cb.timesteps 만 사용) ──
class _SlimCallback:
    def __init__(self, ep_rewards, timesteps):
        self.ep_rewards = list(ep_rewards)
        self.timesteps  = list(timesteps)
        self.ep_lengths = []
    def smoothed(self, window: int = 20):
        r = np.array(self.ep_rewards)
        if len(r) < window:
            return r
        k = np.ones(window) / window
        return np.convolve(r, k, mode="valid")


def _discover_policies() -> list:
    found = []
    for pn in sc.POLICIES.keys():
        d = os.path.join(OUTPUT_ROOT, pn)
        if os.path.isfile(os.path.join(d, "result.pkl")):
            ok = os.path.isfile(os.path.join(d, "_SUCCESS"))
            found.append((pn, ok))
    return found


def _load_all_results():
    """6개 정책 result.pkl + model.zip 을 로드해 all_results/all_loggers 재구성."""
    from stable_baselines3 import PPO

    found = _discover_policies()
    present = [pn for pn, _ in found]
    print(f"  발견된 정책: {present}")
    missing = [pn for pn in sc.POLICIES if pn not in present]
    if missing:
        msg = f"누락 정책: {missing}"
        if REQUIRE_ALL:
            raise RuntimeError(
                f"{msg}  (REQUIRE_ALL=1 → 6개 모두 필요). "
                f"각 train_policy 잡이 끝났는지 확인하세요.")
        print(f"  ⚠️ {msg} — 존재하는 {len(present)}개로 부분 분석 진행")

    all_results, all_loggers = {}, {}
    # 정책 순서는 CONFIG 의 POLICIES 정의 순서를 유지
    for pn in sc.POLICIES.keys():
        d = os.path.join(OUTPUT_ROOT, pn)
        rp = os.path.join(d, "result.pkl")
        if not os.path.isfile(rp):
            continue
        with open(rp, "rb") as f:
            slim = pickle.load(f)

        # 콜백 객체 복원
        cbs = [_SlimCallback(c["ep_rewards"], c["timesteps"])
               for c in slim.get("callbacks_slim", [])]

        # SB3 모델 복원 (bay plan 재추론용)
        model = None
        mz = os.path.join(d, "model.zip")
        if os.path.isfile(mz):
            try:
                model = PPO.load(mz, device="cpu")
            except Exception as e:
                print(f"  ⚠️ {pn}: 모델 로드 실패 ({e}) — bay plan 일부 제한")

        res = {
            "algo":               "PPO",
            "policy_name":        slim["policy_name"],
            "policy_spec":        slim["policy_spec"],
            "elapsed_sec":        slim["elapsed_sec"],
            "avg_rewards":        slim["avg_rewards"],
            "std_rewards":        slim["std_rewards"],
            "levels":             slim["levels"],
            "per_round_metrics":  slim["per_round_metrics"],
            "phase_info":         slim["phase_info"],
            "per_round_bay_plans":slim["per_round_bay_plans"],
            "final_metrics":      slim.get("final_metrics"),
            "callbacks":          cbs,
            "trained_model":      model,
        }
        all_results[pn] = res

        # ExperimentLogger 재구성 (분석 셀이 log_dir 만 사용)
        cfg = dict(sc.CONFIG)
        cfg = {**sc.CONFIG, "experiment_name": f"v17_{pn}"}
        lg = sc.ExperimentLogger(f"v17_{pn}", cfg)   # 새 log_dir (OUTPUT_ROOT 하위)
        all_loggers[pn] = lg

    if not all_results:
        raise RuntimeError("로드된 정책 결과가 0개입니다. 학습 잡 출력 경로를 확인하세요.")
    return all_results, all_loggers


def main():
    t0 = time.time()
    print("=" * 70)
    print("  🔗 VESSL 통합 분석 — 6개 정책 결과 병합")
    print(f"     OUTPUT_ROOT={OUTPUT_ROOT} | SEED={SEED} | REQUIRE_ALL={REQUIRE_ALL}")
    print("=" * 70)

    all_results, all_loggers = _load_all_results()

    # ── 다운스트림 호환 별칭 (원본 Cell 4 말미와 동일) ──
    def _pick(name, fb):
        return name if name in all_results else fb
    REF_POLICY  = _pick("BL_SF_5_EF_5", next(iter(all_results)))
    STAB_POLICY = _pick("SF_9_EF_1", REF_POLICY)
    EFF_POLICY  = _pick("EF_9_SF_1", REF_POLICY)
    BAL_POLICY  = _pick("BL_SF_5_EF_5", REF_POLICY)
    ppo_results = all_results[REF_POLICY]
    logger      = all_loggers[REF_POLICY]

    print(f"  별칭: REF={REF_POLICY} STAB={STAB_POLICY} "
          f"EFF={EFF_POLICY} BAL={BAL_POLICY}")
    print(f"  로드된 정책 수: {len(all_results)} / {len(sc.POLICIES)}")

    # ── 분석 셀 실행용 공유 네임스페이스 구성 ──
    #   원본 셀은 모듈 전역(all_results, CONFIG, np, plt, POLICIES 등)을 사용한다.
    ns = {}
    ns.update({k: getattr(sc, k) for k in dir(sc) if not k.startswith("__")})
    ns.update({
        "all_results":  all_results,
        "all_loggers":  all_loggers,
        "REF_POLICY":   REF_POLICY,
        "STAB_POLICY":  STAB_POLICY,
        "EFF_POLICY":   EFF_POLICY,
        "BAL_POLICY":   BAL_POLICY,
        "ppo_results":  ppo_results,
        "logger":       logger,
        "plt":          plt,
        "np":           np,
        "__name__":     "__agg__",
    })

    # ── 분석 셀 순차 실행 (각 셀 try/except 로 격리) ──
    cell_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "analysis_cells")
    manifest = json.load(open(os.path.join(cell_dir, "_manifest.json")))
    print(f"\n  실행할 분석 셀: {len(manifest)}개\n")

    ok, fail = 0, 0
    for fn in manifest:
        path = os.path.join(cell_dir, fn)
        code = open(path, "r", encoding="utf-8").read()
        print(f"  ── ▶ {fn} ──")
        try:
            exec(compile(code, path, "exec"), ns)
            print(f"     ✅ {fn} 완료")
            ok += 1
        except Exception as e:
            import traceback
            print(f"     ⚠️ {fn} 실패: {e}")
            traceback.print_exc()
            fail += 1

    # 산출물을 _aggregate 로 모으기
    import shutil
    # (1) /content/results (심볼릭 안 걸린 경우 대비 복사)
    if not os.path.islink("/content/results") and os.path.isdir("/content/results"):
        for f in glob.glob("/content/results/**/*", recursive=True):
            if os.path.isfile(f):
                rel = os.path.relpath(f, "/content/results")
                dst = os.path.join(AGG_DIR, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(f, dst)
    # (2) aggregate 중 생성된 v17_* 로거 폴더(그림 fig1~6 등)도 모으기
    for d in glob.glob(os.path.join(OUTPUT_ROOT, "v17_*")):
        if os.path.isdir(d) and d != AGG_DIR:
            base = os.path.basename(d)
            for f in glob.glob(os.path.join(d, "**", "*"), recursive=True):
                if os.path.isfile(f):
                    rel = os.path.relpath(f, d)
                    dst = os.path.join(AGG_DIR, "figures", base, rel)
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    shutil.copy2(f, dst)

    print("\n" + "=" * 70)
    print(f"  ✅ 통합 분석 완료 — 성공 {ok} / 실패 {fail} "
          f"(총 {len(manifest)}개 셀)  | {(time.time()-t0)/60:.1f} min")
    print(f"  산출물 위치: {AGG_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
