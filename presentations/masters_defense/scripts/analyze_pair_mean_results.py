"""보관된 Test 결과로 10개 pair 평균과 Learned 선택을 비교한다.

저장된 CSV/NPZ 및 원천 해시만 읽으며 실험·학습·무작위 pair 선택은 실행하지
않는다. 신뢰구간은 episode 전체를 재표집한다. 같은 episode의 10개 pair를
서로 독립인 표본으로 취급하지 않는다. 출력은 pair_mean_results_audit.json뿐이다.
"""

from collections import Counter
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = Path(__file__).resolve().parents[3]
EVIDENCE = ROOT / "evidence"
EVALUATION = Path("outputs/panda_link_selection/fresh_evaluation/"
                  "evaluation-2000-20261020T003931Z")
CASE = "seed2026120069_0014"


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label):
    require(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-10),
            f"{label}: {actual} != {expected}")


def flag(value):
    require(value in {"True", "False", "1", "0"}, f"잘못된 성공 표시: {value}")
    return value in {"True", "1"}


def time_value(value, success):
    if not success:
        return float("nan")
    result = float(value)
    require(math.isfinite(result) and 0 < result <= 120 + 1e-8,
            f"성공 완료시간 범위 오류: {value}")
    return result


def success_comparison(reference, candidate, indices):
    """두 입력 모두 episode별 성공 또는 성공 기대값인 경우의 paired 비교."""
    difference = candidate.astype(float) - reference.astype(float)
    bootstrap = difference[indices].mean(axis=1) * 100
    return {
        "reference_success_rate_percent": float(reference.mean() * 100),
        "candidate_success_rate_percent": float(candidate.mean() * 100),
        "difference_pp": float(difference.mean() * 100),
        "bootstrap_95_ci_pp": np.quantile(bootstrap, [.025, .975]).tolist(),
    }


def time_comparison(reference_success, reference_time, candidate_success,
                    candidate_time):
    """동일 episode 또는 동일 episode–pair 기록의 공동성공에 조건부인 평균."""
    reference_success, candidate_success = np.broadcast_arrays(
        reference_success, candidate_success)
    reference_time, candidate_time = np.broadcast_arrays(reference_time, candidate_time)
    common = reference_success.astype(bool) & candidate_success.astype(bool)
    reference = reference_time[common]
    candidate = candidate_time[common]
    records = int(common.size)
    return {
        "records": records,
        "both_success_records": int(common.sum()),
        "unique_common_success_episodes": int(
            common.any(axis=1).sum() if common.ndim == 2 else common.sum()),
        "reference_only_success_records": int((reference_success & ~candidate_success).sum()),
        "candidate_only_success_records": int((~reference_success & candidate_success).sum()),
        "both_failure_records": int((~reference_success & ~candidate_success).sum()),
        "reference_common_mean_seconds": float(reference.mean()),
        "candidate_common_mean_seconds": float(candidate.mean()),
        "mean_paired_difference_seconds": float((candidate - reference).mean()),
        "relative_reduction_percent": float(100 * (1 - candidate.mean() / reference.mean())),
    }


def main():
    manifest = json.loads((EVIDENCE / "sources.json").read_text())
    source_records = {Path(item["copy"]).name: item for item in manifest}
    checked_sources = []
    for name in ["pair_outcomes.csv", "selector_choices.csv", "recomputed_summary.json",
                 "test_baselines_comparison.csv", "test_baselines_analysis.json",
                 "test_baselines_paired_comparisons.json"]:
        snapshot = EVIDENCE / name
        original = Path(source_records[name]["source"])
        digest = sha256(snapshot)
        require(digest == source_records[name]["sha256"], f"스냅샷 해시 불일치: {name}")
        require(digest == sha256(original), f"원본 해시 불일치: {name}")
        checked_sources.append({"snapshot": str(snapshot.relative_to(ROOT)),
                                "original": str(original), "sha256": digest})

    original_evidence = Path(source_records["pair_outcomes.csv"]["source"]).parent
    provenance_path = original_evidence / "provenance.json"
    provenance = json.loads(provenance_path.read_text())
    legacy = json.loads((EVIDENCE / "test_baselines_paired_comparisons.json").read_text())
    # 원천 NPZ/rollout CSV/split 및 selector 선택 결과와 Test baseline 기록을 확인.
    raw_paths = [EVALUATION / "dataset/dataset.npz", EVALUATION / "dataset/results.csv",
                 EVALUATION / "split.csv", Path("outputs/panda_link_selection/softmax_time/"
                 "softmax-time-21423-20261023T093500Z/heldout_choices.csv")]
    raw_sources = []
    for relative in raw_paths:
        digest = sha256(REPOSITORY / relative)
        require(digest == provenance["source_sha256"][str(relative)],
                f"원천 해시 불일치: {relative}")
        raw_sources.append({"path": str(relative), "sha256": digest})
    for path, digest in legacy["source_sha256"].items():
        require(sha256(Path(path)) == digest, f"Test 비교 원천 해시 불일치: {path}")
        raw_sources.append({"path": path, "sha256": digest})

    summary = json.loads((EVIDENCE / "recomputed_summary.json").read_text())
    test_summary = summary["splits"]["test"]
    pairs = summary["action_pairs"]
    records = [row for row in rows(EVIDENCE / "pair_outcomes.csv") if row["split"] == "test"]
    pair_lookup = {(row["case_id"], int(row["pair_index"])): row for row in records}
    require(len(pair_lookup) == len(records), "episode–pair 중복")
    ids = sorted({row["case_id"] for row in records})
    episode_index = {case_id: index for index, case_id in enumerate(ids)}
    n, k = len(ids), len(pairs)
    require((n, k, len(records)) == (1000, 10, 10000), "Test 모집단 오류")
    require(all((case_id, j) in pair_lookup for case_id in ids for j in range(k)),
            "누락된 episode–pair")
    require(all(json.loads(row["pair"]) == pairs[int(row["pair_index"])]
                for row in records), "pair 인덱스 불일치")
    split_rows = rows(REPOSITORY / EVALUATION / "split.csv")
    require({row["case_id"] for row in split_rows if row["split"] == "test"} == set(ids),
            "원천 Test split과 불일치")
    raw_records = rows(REPOSITORY / EVALUATION / "dataset/results.csv")
    raw_lookup = {(row["case_id"], tuple(json.loads(row["pair"]))): row for row in raw_records}
    require(len(raw_lookup) == len(raw_records), "원천 rollout 중복")
    y = np.zeros((n, k), dtype=bool)
    t = np.full((n, k), np.nan)
    invalid = 0
    with np.load(REPOSITORY / EVALUATION / "dataset/dataset.npz", allow_pickle=False) as data:
        require(data["link_pairs"].tolist() == pairs, "원천 NPZ pair 순서 불일치")
        raw_index = {str(case_id): i for i, case_id in enumerate(data["case_id"])}
        require([str(case_id) for case_id in data["case_id"] if str(case_id) in episode_index]
                == ids, "원천 bootstrap episode 순서 불일치")
        for row in records:
            case_id, j = row["case_id"], int(row["pair_index"])
            i, raw_i = episode_index[case_id], raw_index[case_id]
            raw = raw_lookup[case_id, tuple(pairs[j])]
            valid = bool(data["label_valid"][raw_i, j])
            success = flag(row["success"])
            require(flag(raw["label_valid"]) == valid, "CSV/NPZ 유효성 불일치")
            require(success == (flag(raw["success"]) and valid) ==
                    (valid and int(data["success"][raw_i, j]) == 1), "성공 여부 불일치")
            if not valid:
                require(int(data["success"][raw_i, j]) == -1 and not success,
                        "수치 오류 실패 처리 불일치")
                invalid += 1
            y[i, j] = success
            t[i, j] = time_value(row["completion_time"], success)
            if success:
                close(t[i, j], time_value(raw["completion_time"], True), "원천 완료시간")

    choices = [row for row in rows(EVIDENCE / "selector_choices.csv")
               if row["split"] == "test" and row["family"] == "baseline" and row["seed"] == "11"]
    choice_lookup = {row["case_id"]: row for row in choices}
    require(len(choices) == len(choice_lookup) == n and set(choice_lookup) == set(ids),
            "선택기 Test episode 불일치")
    learned = np.zeros(n, dtype=bool)
    learned_time = np.full(n, np.nan)
    selected_indices = []
    for i, case_id in enumerate(ids):
        row = choice_lookup[case_id]
        j = int(row["selected_pair_index"])
        require(json.loads(row["selected_pair"]) == pairs[j], "선택 pair 인덱스 불일치")
        require(flag(row["success"]) == y[i, j], "선택 결과와 candidate 결과 불일치")
        learned[i] = y[i, j]
        learned_time[i] = time_value(row["completion_time_seconds"], learned[i])
        selected_indices.append(j)
        if learned[i]:
            close(learned_time[i], t[i, j], "선택된 pair 완료시간")

    baseline_rows = rows(EVIDENCE / "test_baselines_comparison.csv")
    baseline = {row["case_id"]: row for row in baseline_rows}
    require(len(baseline_rows) == len(baseline) == n and set(baseline) == set(ids),
            "All-link/Direct episode 불일치")
    all_links = np.array([flag(baseline[case_id]["sequential_120_success"]) for case_id in ids])
    all_links_time = np.array([time_value(baseline[case_id]["sequential_120_completion_time"],
                                        all_links[i]) for i, case_id in enumerate(ids)])
    direct = np.array([flag(baseline[case_id]["direct_120_success"]) for case_id in ids])
    pair_mean = y.mean(axis=1)
    close(pair_mean.mean(), test_summary["uniform_expected_success_rate"], "기존 pair 평균")
    require(y.sum(axis=0).tolist() == test_summary["pair_successes"], "기존 pair별 성공 수")
    require(invalid == test_summary["numerical_errors"] == 26, "Test 수치 오류 수")
    require(int(learned.sum()) == test_summary["models"]["baseline"]["successes"] == 609,
            "기존 Learned 성공 수")
    require(int(all_links.sum()) == legacy["methods"]["sequential"]["successes"] == 552,
            "기존 All-link 성공 수")
    require(int(direct.sum()) == legacy["methods"]["direct"]["successes"] == 455,
            "기존 Direct 성공 수")

    draws, seed = summary["bootstrap_draws"], summary["bootstrap_seed"]
    require(draws == legacy["bootstrap_draws"] == 10000 and seed == legacy["bootstrap_seed"],
            "기존 bootstrap 설정 불일치")
    indices = np.random.default_rng(seed).integers(n, size=(draws, n))
    learned_vs_mean = success_comparison(pair_mean, learned, indices)
    all_links_vs_mean = success_comparison(pair_mean, all_links, indices)
    learned_vs_all_links = success_comparison(all_links, learned, indices)
    mean_vs_all_links_time = time_comparison(all_links[:, None], all_links_time[:, None], y, t)
    learned_vs_mean_time = time_comparison(y, t, learned[:, None], learned_time[:, None])
    learned_vs_all_links_time = time_comparison(all_links, all_links_time, learned, learned_time)
    # 10개 pair 평균이므로 episode별 차이는 정확히 0.1의 정수배이다.
    gain_tenths = learned.astype(int) * k - y.sum(axis=1)
    gain_counts = Counter(int(value) for value in gain_tenths)
    close(gain_tenths.mean() / k * 100, learned_vs_mean["difference_pp"],
          "episode별 차이 분포 평균")
    fixed_index = pairs.index([3, 5])
    learned_vs_fixed = success_comparison(y[:, fixed_index], learned, indices)
    learned_vs_fixed_time = time_comparison(y[:, fixed_index], t[:, fixed_index], learned, learned_time)

    # 독립 집계: numpy 행렬과 별도로 CSV 목록/statistics.mean으로 같은 결과 확인.
    independent_mean = mean(int(row["success"]) for row in records)
    independent_learned = mean(int(row["success"]) for row in choices)
    close(pair_mean.mean(), independent_mean, "독립 CSV 평균 검산")
    close(learned_vs_mean["difference_pp"], 100 * (independent_learned - independent_mean),
          "독립 CSV 성공률 차이 검산")
    independent_common = [row for row in records if flag(row["success"])
                          and flag(choice_lookup[row["case_id"]]["success"])]
    close(learned_vs_mean_time["reference_common_mean_seconds"],
          mean(float(row["completion_time"]) for row in independent_common), "독립 pair 시간 검산")
    close(learned_vs_mean_time["candidate_common_mean_seconds"],
          mean(float(choice_lookup[row["case_id"]]["completion_time_seconds"])
               for row in independent_common), "독립 Learned 시간 검산")
    # 기존 정수 episode 비교의 모든 통계/CI까지 재현하여 재표집 순서와 구현 확인.
    for current_success, current_time, key in [
            (learned_vs_all_links, learned_vs_all_links_time, "neural_vs_sequential"),
            (learned_vs_fixed, learned_vs_fixed_time, "neural_vs_fixed35")]:
        old = legacy["paired_comparisons"][key]
        close(current_success["difference_pp"], old["difference_pp"], key)
        for actual, expected in zip(current_success["bootstrap_95_ci_pp"], old["bootstrap_95_ci_pp"]):
            close(actual, expected, f"기존 CI {key}")
        for new_key, old_key in [("both_success_records", "both_success"),
                                 ("reference_only_success_records", "reference_only"),
                                 ("candidate_only_success_records", "candidate_only"),
                                 ("both_failure_records", "both_failure"),
                                 ("reference_common_mean_seconds", "reference_common_mean"),
                                 ("candidate_common_mean_seconds", "candidate_common_mean"),
                                 ("mean_paired_difference_seconds", "mean_time_difference")]:
            close(current_time[new_key], old[old_key], f"기존 비교 {key}/{new_key}")

    case_i = episode_index[CASE]
    cells = {
        "both_success": learned_vs_mean_time["both_success_records"],
        "pair_mean_only": learned_vs_mean_time["reference_only_success_records"],
        "learned_only": learned_vs_mean_time["candidate_only_success_records"],
        "both_failure": learned_vs_mean_time["both_failure_records"],
    }
    require(sum(cells.values()) == n * k, "four-cell 합 오류")
    case_audit = {
        "case_id": CASE,
        "successful_pair_count": int(y[case_i].sum()),
        "candidate_pair_count": k,
        "successful_pairs": [pair for pair, success in zip(pairs, y[case_i]) if success],
        "pair_mean_success_rate_percent": float(pair_mean[case_i] * 100),
        "learned_selected_pair": pairs[selected_indices[case_i]],
        "learned_success": bool(learned[case_i]),
        "learned_completion_time_seconds": float(learned_time[case_i]),
        "fixed_3_5_success": bool(y[case_i, fixed_index]),
        "interpretation": "10개 pair 중 (5,6) 하나만 성공한 episode. 평균의 단일 궤적은 없으며, "
                          "시각화한 (3,5) 실패 궤적은 9개 실패 candidate 중 한 예시이다.",
    }
    require(case_audit["successful_pairs"] == [[5, 6]] and
            case_audit["learned_selected_pair"] == [5, 6] and
            not case_audit["fixed_3_5_success"], "예시 episode 결과 불일치")

    audit = {
        "description": "Test 1,000 episodes의 10-pair 평균과 Learned 선택 비교",
        "split": "test", "controller_horizon_seconds": 120,
        "no_new_simulation": True, "no_new_rollouts": True, "no_new_training": True,
        "no_random_pair_draws": True,
        "input_sources": checked_sources, "raw_sources": raw_sources,
        "provenance": {"path": str(provenance_path), "sha256": sha256(provenance_path)},
        "analysis_script": {"path": "scripts/analyze_pair_mean_results.py",
                            "sha256": sha256(Path(__file__))},
        "population": {"episodes": n, "candidate_pairs": k, "episode_pair_outcomes": n * k,
                       "action_pairs": pairs, "numerical_errors_included_as_unsuccessful": invalid,
                       "learned_family": "baseline", "learned_seed": 11},
        "definitions": {
            "pair_mean": "m_i=(1/10) sum_k y_ik; 평균 성공률은 (1/N) sum_i m_i.",
            "learned": "l_i=y_i,khat(i), 미리 선택된 family=baseline, seed=11.",
            "difference": "Delta=(1/N) sum_i(l_i-m_i). Learned 재학습/재선택 없음.",
            "baseline_label": "Two-link average",
            "not_a_single_fixed_pair": True,
            "not_a_randomized_trial": "10개 실제 기록의 정확한 평균이며 무작위 pair를 다시 추출하지 않음.",
        },
        "bootstrap": {
            "draws": draws, "seed": seed, "confidence_level": 0.95,
            "quantiles": [.025, .975], "method": "paired episode percentile bootstrap",
            "resampling_unit": "episode (all ten pair outcomes and selected outcome kept together)",
            "episodes_per_draw": n, "episode_order": "case_id ascending; original NPZ Test order와 동일",
            "numpy_rng": "default_rng", "numpy_quantile_method": "linear",
            "conditions": "하나의 이미 학습된 seed=11 모델에 조건부이며 학습 seed 변동은 포함하지 않음.",
        },
        "success": {
            "direct_success_rate_percent": float(direct.mean() * 100),
            "all_links_success_rate_percent": float(all_links.mean() * 100),
            "pair_mean_success_rate_percent": float(pair_mean.mean() * 100),
            "learned_success_rate_percent": float(learned.mean() * 100),
            "pair_successes": int(y.sum()), "pair_outcome_denominator": int(y.size),
            "learned_successes": int(learned.sum()), "learned_episode_denominator": n,
            "learned_vs_pair_mean": learned_vs_mean,
            "all_links_vs_pair_mean": all_links_vs_mean,
            "learned_vs_all_links": learned_vs_all_links,
        },
        "learned_vs_pair_mean_outcomes": {
            "episode_pair_record_counts": cells,
            "record_denominator": n * k,
            "expected_counts_per_1000_episodes": {name: count / k for name, count in cells.items()},
            "percentages": {name: count / (n * k) * 100 for name, count in cells.items()},
            "not_actual_episode_counts": True,
            "interpretation": "각 episode의 모든 pair를 평균한 기대 분할이다. 예: 129.8은 실제 "
                              "129.8개 episode가 아니라 Learned-only 기록 1,298개/10이며 "
                              "four-cell 비교는 %로 제시하는 것이 적절함.",
        },
        "episode_gain_distribution": {
            "definition": "d_i=l_i-m_i; 각 episode를 한 번 세는 실제 정수 episode 분포.",
            "bins_tenths": list(range(-9, 10)),
            "episode_counts": [gain_counts[value] for value in range(-9, 10)],
            "negative_episodes": int((gain_tenths < 0).sum()),
            "zero_episodes": int((gain_tenths == 0).sum()),
            "positive_episodes": int((gain_tenths > 0).sum()),
            "episodes": n,
            "mean_difference": float(gain_tenths.mean() / k),
            "mean_difference_percentage_points": float(gain_tenths.mean() / k * 100),
            "interpretation": "양수는 Learned가 성공하고 일부 다른 pair는 실패한 episode. "
                              "음수는 Learned가 실패했으나 다른 일부 pair가 성공한 episode. "
                              "0은 모든 candidate가 성공하거나 모두 실패한 episode.",
        },
        "paired_time": {
            "learned_vs_pair_mean": learned_vs_mean_time,
            "pair_mean_vs_all_links": mean_vs_all_links_time,
            "learned_vs_all_links": learned_vs_all_links_time,
            "interpretation": {
                "prior": "각 episode와 그 안의 각 pair에 동일 가중치를 둔 뒤 공동성공에 조건부.",
                "learned_vs_pair_mean": "Learned와 해당 candidate가 모두 성공한 episode–pair 기록의 "
                                        "pair 완료시간 및 Learned 완료시간 평균. 동일 Learned 시간이 "
                                        "성공 candidate 개수만큼 반복됨.",
                "matches_slide_26_estimand": True,
                "pair_weights_after_conditioning": "각 pair 공동성공 수에 비례; pair별 조건부 평균의 "
                                                   "단순 산술평균과는 다름.",
                "not_independent_episode_pair_records": True,
                "not_unconditional_runtime": "실패 기록 제외. 모든 시도의 무조건부 runtime 비교가 아님.",
                "no_learned_speed_advantage_over_pair_mean": True,
            },
        },
        "example_episode": case_audit,
        "historical_fixed_3_5_benchmark": {
            "pair": [3, 5], "success_rate_percent": float(y[:, fixed_index].mean() * 100),
            "learned_vs_fixed": learned_vs_fixed,
            "paired_time": learned_vs_fixed_time,
            "role": "Train-best fixed (3,5)는 더 강한 단일 fixed 기준으로 기존 기록 보존. "
                    "평균 대비 개선을 이 fixed pair 대비 유의한 개선으로 확대 해석하지 않음.",
            "existing_statistics_and_ci_reproduced": True,
        },
        "per_pair": [{"pair": pair, "successes": int(y[:, j].sum()),
                      "success_rate_percent": float(y[:, j].mean() * 100)}
                     for j, pair in enumerate(pairs)],
        "verification": {
            "snapshot_and_original_sha256_match": True,
            "raw_npz_csv_split_selector_and_test_baseline_sha256_match": True,
            "test_episode_ids_match_across_all_sources": True,
            "each_episode_has_each_pair_exactly_once": True,
            "raw_npz_and_csv_outcomes_checked": len(records),
            "selected_pair_outcomes_and_successful_times_checked": len(choices),
            "independent_csv_success_and_time_aggregation_matches": True,
            "original_pair_counts_mean_and_numerical_errors_reproduced": True,
            "all_links_learner_and_fixed_learner_counts_times_and_ci_reproduced": True,
            "source_files_unchanged": True,
        },
    }
    # 읽은 스냅샷 및 원천을 마지막에 다시 확인하여 원시 데이터 불변을 보장.
    for item in checked_sources:
        require(sha256(ROOT / item["snapshot"]) == item["sha256"] == sha256(Path(item["original"])),
                "분석 중 스냅샷/원천 변경")
    for item in raw_sources:
        require(sha256(REPOSITORY / item["path"]) == item["sha256"], "분석 중 원천 변경")
    destination = EVIDENCE / "pair_mean_results_audit.json"
    destination.write_text(json.dumps(audit, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps({"audit": str(destination), "success": audit["success"],
                      "time": audit["paired_time"], "example": case_audit},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
