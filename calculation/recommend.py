"""QUBOとアニーリングによる学習項目の推薦。

適性5軸
  ↓
各IT分野との相性
  ↓
低相性分野（min_relevance未満）の寄与を除外
  ↓
所属分野の相性を合計して学習項目価値を計算
  ↓
項目ペアのシナジー（同時に選んだ場合だけ発生する価値）を算出
  ↓
予算制約・前提科目制約・シナジー項をQUBO化
  ↓
アニーリングによって推薦項目を決定
"""

import math

import dimod
import numpy as np
from neal import SimulatedAnnealingSampler

from calculation.encoding import bounded_binary_weights
from calculation.prerequisites import (
    add_prerequisites_to_candidates,
    check_prerequisites,
)
from config.fields import F
from config.items import M, hours, item_fields, item_index, item_names
from config.parameters import (
    DEFAULT_MIN_RELEVANCE,
    DEFAULT_NUM_READS,
    DEFAULT_NUM_SWEEPS,
    DEFAULT_SEED,
)
from config.prerequisites import and_prerequisites, or_prerequisites
from config.synergy import item_synergy


def recommend(
    user,
    T,
    min_relevance=DEFAULT_MIN_RELEVANCE,
    num_reads=DEFAULT_NUM_READS,
    num_sweeps=DEFAULT_NUM_SWEEPS,
    seed=DEFAULT_SEED,
    sampler=None,
):
    """
    QUBOとアニーリングによって学習項目を推薦する。

    Parameters
    ----------
    user : array-like
        適性5軸ベクトル。各値は0～1。

    T : int
        使用可能な総学習時間。

    min_relevance : float
        項目価値へ反映させる分野相性の最低値。

    num_reads : int
        アニーリングの実行回数。

    num_sweeps : int
        neal使用時の1回当たりのスイープ数。

    seed : int
        乱数シード。

    sampler :
        Noneの場合はnealのSimulatedAnnealingSamplerを使用する。
        D-Wave QPU用サンプラーを渡すこともできる。

    Returns
    -------
    z : ndarray
        各学習項目の選択結果。

    field_score : ndarray
        各IT分野との元の相性。

    debug_info : dict
        分野関連度、項目価値、QUBOなどの情報。
    """

    # --------------------------------------------------------
    # 入力検査
    # --------------------------------------------------------

    user = np.asarray(user, dtype=float)

    if user.shape != (5,):
        raise ValueError("userは長さ5のベクトルにしてください。")

    if np.any(user < 0.0) or np.any(user > 1.0):
        raise ValueError("userの各値は0～1にしてください。")

    if not isinstance(T, (int, np.integer)):
        raise ValueError("Tは整数時間で指定してください。")

    if T <= 0:
        raise ValueError("Tは正の値にしてください。")

    if not 0.0 <= min_relevance < 1.0:
        raise ValueError("min_relevanceは0以上1未満にしてください。")

    # ========================================================
    # 1. 各IT分野との相性
    # ========================================================

    # 5軸には個別の重みを付けず、
    # すべて等重みで平均二乗距離を計算する。
    #
    # 完全一致：field_score = 1
    # 不一致が大きいほど0に近づく。

    mean_squared_difference = np.mean((F - user) ** 2, axis=1)

    field_score = np.clip(1.0 - mean_squared_difference, 0.0, 1.0)

    # ========================================================
    # 2. 低相性分野の寄与を0にする
    # ========================================================

    # 閾値以上の相性は変形せずそのまま使う。
    #
    # min_relevance未満：項目価値への寄与は0
    # min_relevance以上：元の相性をそのまま使用

    field_relevance = np.where(field_score >= min_relevance, field_score, 0.0)

    # すべての分野関連度が0の場合は推薦不能
    if np.all(field_relevance == 0.0):
        raise RuntimeError("min_relevance以上の相性を持つIT分野がありません。")

    # ========================================================
    # 3. 学習項目の価値
    # ========================================================

    # 項目が所属するすべての分野について、
    # field_relevanceをそのまま合計する。
    #
    # 複数の関連分野で利用できる項目ほど高い価値を持つ。
    #
    # 例：
    #   バックエンド関連度 = 0.90
    #   データ/ML関連度    = 0.95
    #
    #   SQL/データベース = 0.90 + 0.95 = 1.85

    item_value = np.array(
        [field_relevance[item_fields[j]].sum() for j in range(M)],
        dtype=float,
    )

    # 分野関連度がまったくない項目は候補外。
    # ただし、候補項目に必要な前提科目は候補へ追加される。

    candidate_mask = add_prerequisites_to_candidates(item_value > 0.0)

    candidate_indices = np.flatnonzero(candidate_mask).tolist()

    if len(candidate_indices) == 0:
        raise RuntimeError("推薦候補となる学習項目がありません。")

    # ========================================================
    # 4. 時間を最大公約数で圧縮
    # ========================================================

    # 候補項目と予算の最大公約数を使う。
    #
    # 例：
    #   40時間 → 4単位
    #   60時間 → 6単位
    #   240時間 → 24単位

    time_unit = int(T)

    for j in candidate_indices:
        time_unit = math.gcd(time_unit, int(hours[j]))

    time_unit = max(time_unit, 1)

    cost_units = hours // time_unit
    budget_units = T // time_unit

    # ========================================================
    # 5. QUBO/BQMの作成
    # ========================================================

    bqm = dimod.BinaryQuadraticModel.empty(dimod.BINARY)

    # 項目の価値を最大化する。
    # QUBOはエネルギー最小化なので、価値を負の線形係数として追加する。

    for j in candidate_indices:
        bqm.add_variable(f"item_{j}", -float(item_value[j]))

    # ========================================================
    # 5b. シナジー項の追加
    # ========================================================

    # 項目iと項目jを両方選んだ場合にのみ発生する価値を
    # QUBOの二次項として反映する。
    #
    # これは単独項目の価値（線形項）とは独立に働く効果であり、
    # 「AとBを同時に選ぶ」という組み合わせそのものに
    # 価値（またはペナルティ）を与える。
    #
    # エネルギー最小化問題なので、
    # 価値（プラス方向のシナジー）は負の係数として加える。

    synergy_upper_bound = 0.0

    for a in range(len(candidate_indices)):
        i = candidate_indices[a]

        for b in range(a + 1, len(candidate_indices)):
            j = candidate_indices[b]

            synergy_value = item_synergy[i, j]

            # シナジーが定義されていないペアは何もしない
            if synergy_value == 0.0:
                continue

            bqm.add_quadratic(f"item_{i}", f"item_{j}", -float(synergy_value))

            # 正のシナジーだけを「報酬の上限」に積み上げる。
            # 負のシナジー（非効率ペナルティ）は
            # 報酬を増やす方向には働かないため無視する。
            if synergy_value > 0.0:
                synergy_upper_bound += synergy_value

    # ========================================================
    # 6. ペナルティ係数
    # ========================================================

    # 項目とシナジーをすべて得た場合に得られる報酬の上限。
    #
    # 前提科目として追加された価値0の項目は報酬に影響しない。
    #
    # シナジー項が加わった分、報酬の上限も引き上げておかないと、
    # 制約を破ってでもシナジーを稼いだ方が得になってしまう。

    reward_upper_bound = (
        sum(max(0.0, float(item_value[j])) for j in candidate_indices)
        + synergy_upper_bound
    )

    # 最小の制約違反ペナルティが、得られる最大報酬より大きくなるようにする。
    constraint_penalty = reward_upper_bound + 1.0

    # ========================================================
    # 7. 予算制約
    # ========================================================

    # 予算不等式：
    #   sum(cost_i * z_i) <= budget
    #
    # スラック変数sを使って等式へ変換：
    #   sum(cost_i * z_i) + s = budget
    #
    # QUBOへ追加するペナルティ：
    #   A * (sum(cost_i * z_i) + s - budget)^2

    slack_weights = bounded_binary_weights(budget_units)

    budget_terms = {}

    # 学習項目
    for j in candidate_indices:
        budget_terms[f"item_{j}"] = int(cost_units[j])

    # スラック変数
    for k, weight in enumerate(slack_weights):
        slack_variable = f"slack_{k}"

        bqm.add_variable(slack_variable, 0.0)

        budget_terms[slack_variable] = int(weight)

    budget_variables = list(budget_terms.keys())

    # 二乗ペナルティの線形項
    for variable in budget_variables:
        coefficient = budget_terms[variable]

        linear_bias = constraint_penalty * (
            coefficient ** 2 - 2 * budget_units * coefficient
        )

        bqm.add_linear(variable, linear_bias)

    # 二乗ペナルティの交差項
    for p in range(len(budget_variables)):
        variable_p = budget_variables[p]
        coefficient_p = budget_terms[variable_p]

        for q in range(p + 1, len(budget_variables)):
            variable_q = budget_variables[q]
            coefficient_q = budget_terms[variable_q]

            quadratic_bias = 2 * constraint_penalty * coefficient_p * coefficient_q

            bqm.add_quadratic(variable_p, variable_q, quadratic_bias)

    # 二乗展開の定数項
    bqm.offset += constraint_penalty * budget_units ** 2

    # ========================================================
    # 8. AND前提科目制約
    # ========================================================

    # 子項目を選ぶなら親項目も必要：
    #   child <= parent
    #
    # 違反するのは child = 1, parent = 0 の場合だけ。
    #
    # QUBOペナルティ：
    #   A * child * (1 - parent)
    #
    # 展開すると：
    #   A * child - A * child * parent

    for child_name, parent_names in and_prerequisites.items():
        child = item_index[child_name]

        # 子項目が候補外なら制約も不要
        if not candidate_mask[child]:
            continue

        child_variable = f"item_{child}"

        for parent_name in parent_names:
            parent = item_index[parent_name]

            if not candidate_mask[parent]:
                raise RuntimeError(
                    f"前提科目「{parent_name}」が候補に追加されていません。"
                )

            parent_variable = f"item_{parent}"

            bqm.add_linear(child_variable, constraint_penalty)

            bqm.add_quadratic(child_variable, parent_variable, -constraint_penalty)

    # ========================================================
    # 8b. OR前提科目制約（言語必須ルール）
    # ========================================================

    # 子項目を選ぶなら、選択肢のうち少なくとも1つが必要：
    #   child <= parent_1 + parent_2 + ... + parent_n
    #
    # AND前提（8）とは異なり、右辺が定数ではなく
    # 複数の変数の合計になるため、予算制約（7）と同じ
    # 「スラック変数で不等式を等式に変換する」手法を流用する。
    #
    # 等式へ変換：
    #   parent_1 + ... + parent_n - child - s = 0
    #
    # sは0～n（選択肢の数）の範囲を取るスラック変数。
    #
    # 満たされているとき（child <= sum(parents)）：
    #   左辺は0以上n以下の値になるので、
    #   sをちょうどその値に選べば式は0になる。
    #
    # 満たされていないとき（child=1, parents全部0）：
    #   左辺は-1になるが、sは0以上しか取れないため、
    #   どうやっても式を0にできず、必ずペナルティが発生する。
    #
    # QUBOへ追加するペナルティ：
    #   A * (sum(parents) - child - s)^2

    for child_name, parent_names in or_prerequisites.items():
        child = item_index[child_name]

        # 子項目が候補外なら制約も不要
        if not candidate_mask[child]:
            continue

        child_variable = f"item_{child}"

        parent_variables = []

        for parent_name in parent_names:
            parent = item_index[parent_name]

            if not candidate_mask[parent]:
                raise RuntimeError(
                    f"前提科目「{parent_name}」が候補に追加されていません。"
                )

            parent_variables.append(f"item_{parent}")

        # このOR制約専用のスラック変数を用意する
        # （budget用のslack_kと名前が衝突しないよう、
        #   子項目のインデックスを名前に含める）

        or_slack_weights = bounded_binary_weights(len(parent_variables))

        # 係数：親項目は+1、子項目は-1、スラックは-weight
        or_terms = {}

        for parent_variable in parent_variables:
            or_terms[parent_variable] = or_terms.get(parent_variable, 0) + 1

        or_terms[child_variable] = or_terms.get(child_variable, 0) - 1

        for k, weight in enumerate(or_slack_weights):
            or_slack_variable = f"or_slack_{child}_{k}"

            bqm.add_variable(or_slack_variable, 0.0)

            or_terms[or_slack_variable] = (
                or_terms.get(or_slack_variable, 0) - weight
            )

        or_variables = list(or_terms.keys())

        # 二乗ペナルティの線形項（目標値は0）
        for variable in or_variables:
            coefficient = or_terms[variable]

            bqm.add_linear(variable, constraint_penalty * coefficient ** 2)

        # 二乗ペナルティの交差項
        for p in range(len(or_variables)):
            variable_p = or_variables[p]
            coefficient_p = or_terms[variable_p]

            for q in range(p + 1, len(or_variables)):
                variable_q = or_variables[q]
                coefficient_q = or_terms[variable_q]

                quadratic_bias = (
                    2 * constraint_penalty * coefficient_p * coefficient_q
                )

                bqm.add_quadratic(variable_p, variable_q, quadratic_bias)

        # 目標値が0なので、二乗展開の定数項は発生しない

    # ========================================================
    # 9. アニーリング
    # ========================================================

    if sampler is None:
        sampler = SimulatedAnnealingSampler()

    sample_kwargs = {"num_reads": num_reads}

    # nealのシミュレーテッド・アニーリングで使用できる引数
    if isinstance(sampler, SimulatedAnnealingSampler):
        sample_kwargs.update({
            "num_sweeps": num_sweeps,
            "seed": seed,
        })

    sampleset = sampler.sample(bqm, **sample_kwargs)

    # ========================================================
    # 10. 実行可能解の抽出
    # ========================================================

    best_feasible_z = None
    best_feasible_energy = np.inf

    feasible_count = 0
    budget_violation_count = 0
    slack_violation_count = 0
    prerequisite_violation_count = 0

    for datum in sampleset.data(fields=["sample", "energy"], sorted_by="energy"):
        sample = datum.sample

        z_candidate = np.zeros(M, dtype=int)

        for j in candidate_indices:
            z_candidate[j] = int(sample.get(f"item_{j}", 0))

        # 選択された学習時間
        selected_units = int(cost_units @ z_candidate)

        # サンプル内のスラック値
        sampled_slack_units = sum(
            weight * int(sample.get(f"slack_{k}", 0))
            for k, weight in enumerate(slack_weights)
        )

        # 予算以下か
        budget_ok = selected_units <= budget_units

        # 予算等式が成立しているか
        slack_ok = selected_units + sampled_slack_units == budget_units

        # 前提科目制約
        prerequisites_ok, _ = check_prerequisites(z_candidate)

        if not budget_ok:
            budget_violation_count += 1

        if not slack_ok:
            slack_violation_count += 1

        if not prerequisites_ok:
            prerequisite_violation_count += 1

        if budget_ok and slack_ok and prerequisites_ok:
            feasible_count += 1

            if datum.energy < best_feasible_energy:
                best_feasible_z = z_candidate.copy()
                best_feasible_energy = float(datum.energy)

    if best_feasible_z is None:
        raise RuntimeError(
            "予算制約と前提科目制約を満たすサンプルが得られませんでした。\n"
            "num_readsまたはnum_sweepsを増やしてください。"
        )

    z = best_feasible_z

    # ========================================================
    # 11. 最終検査
    # ========================================================

    total_hours = int(hours @ z)

    prerequisites_ok, violations = check_prerequisites(z)

    if total_hours > T:
        raise RuntimeError(f"予算制約違反：{total_hours}h > {T}h")

    if not prerequisites_ok:
        raise RuntimeError(f"前提科目制約違反：{violations}")

    # ========================================================
    # 12. デバッグ情報
    # ========================================================

    # 選択された項目の中で、実際に発生したシナジーを集計しておく
    # （結果表示で「なぜこの組み合わせが選ばれたか」を説明するため）
    active_synergies = []

    for a in range(len(candidate_indices)):
        i = candidate_indices[a]

        if z[i] != 1:
            continue

        for b in range(a + 1, len(candidate_indices)):
            j = candidate_indices[b]

            if z[j] != 1:
                continue

            synergy_value = item_synergy[i, j]

            if synergy_value != 0.0:
                active_synergies.append(
                    (item_names[i], item_names[j], float(synergy_value))
                )

    debug_info = {
        "field_score": field_score,
        "field_relevance": field_relevance,
        "item_value": item_value,
        "candidate_mask": candidate_mask,
        "candidate_indices": candidate_indices,
        "time_unit": time_unit,
        "cost_units": cost_units,
        "budget_units": budget_units,
        "slack_weights": slack_weights,
        "reward_upper_bound": reward_upper_bound,
        "constraint_penalty": constraint_penalty,
        "synergy_upper_bound": synergy_upper_bound,
        "active_synergies": active_synergies,
        "energy": best_feasible_energy,
        "feasible_count": feasible_count,
        "budget_violation_count": budget_violation_count,
        "slack_violation_count": slack_violation_count,
        "prerequisite_violation_count": prerequisite_violation_count,
        "bqm": bqm,
        "sampleset": sampleset,
    }

    return z, field_score, debug_info
