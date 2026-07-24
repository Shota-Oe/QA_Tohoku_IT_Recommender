"""QUBOとアニーリングによる学習項目の推薦。

適性5軸
  ↓
各IT分野との相性
  ↓
低相性分野（min_relevance未満）の寄与を除外
  ↓
所属分野の相性を合計して学習項目価値を計算
  ↓
学習済み科目を候補から除外し、実効前提・実効価値へ畳み込む
  ↓
候補集合の生成＋前提科目の前処理
（予算内で前提を満たせない項目の除外・前提科目の候補追加）
  ↓
項目ペアのシナジー（同時に選んだ場合だけ発生する価値）を算出
  ↓
予算制約・シナジー項・条件付き価値をQUBO化
  ↓
アニーリング
  ↓
各サンプルを修復デコード（前提未達の子項目を除去／不足親を補完）し、
真の目的関数値（価値＋シナジー）が最大の実行可能解を採用
"""

import math

import dimod
import numpy as np
from neal import SimulatedAnnealingSampler

from calculation.encoding import bounded_binary_weights
from calculation.prerequisites import (
    add_prerequisites_to_candidates,
    check_prerequisites,
    complete_prerequisites,
    effective_prerequisites,
    find_learned_prerequisite_gaps,
    min_closure_hours,
    repair_prerequisites,
    restrict_candidates_by_budget,
)
from calculation.sampler import summarize_sampleset
from config.fields import F
from config.items import M, hours, item_fields, item_index, item_names
from config.parameters import (
    DEFAULT_MIN_RELEVANCE,
    DEFAULT_NUM_READS,
    DEFAULT_NUM_SWEEPS,
    DEFAULT_PENALTY_MARGIN,
    DEFAULT_SEED,
)
from config.synergy import item_synergy


def coefficient_ratio(bqm):
    """QUBO係数のダイナミックレンジ（最大絶対値 ÷ 最小絶対値）。

    実機QPUは係数を「最大絶対値が1になる」ように自動スケールしてから
    磁束として印加するため、この比が大きいほど小さい係数が
    ハードウェアの制御誤差（ICE）に埋もれる。
    予算制約ペナルティは重み最大32のスラック変数と掛け合わさって
    数千の係数になる一方、項目価値は1前後なので、この比は数千に達する。
    QPUで解の品質が落ちる場合、まずここを疑う指標になる
    （nealは倍精度で計算するため影響を受けない）。
    """
    magnitudes = [
        abs(value)
        for value in list(bqm.linear.values()) + list(bqm.quadratic.values())
        if value != 0.0
    ]

    if not magnitudes:
        return 1.0

    return max(magnitudes) / min(magnitudes)


def recommend(
    user,
    T,
    learned=(),
    min_relevance=DEFAULT_MIN_RELEVANCE,
    num_reads=DEFAULT_NUM_READS,
    num_sweeps=DEFAULT_NUM_SWEEPS,
    seed=DEFAULT_SEED,
    penalty_margin=DEFAULT_PENALTY_MARGIN,
    sampler=None,
    sampler_kwargs=None,
):
    """
    QUBOとアニーリングによって学習項目を推薦する。

    Parameters
    ----------
    user : array-like
        適性5軸ベクトル。各値は0～1。

    T : int
        使用可能な総学習時間。

    learned : iterable of str
        すでに学習を終えた項目名。候補から外し、
        実効前提（親を満たし済みとみなす）と
        実効価値（シナジーの畳み込み）へ反映する。
        学習済みの時間はTを消費せず、価値も目的関数に計上しない。

    min_relevance : float
        項目価値へ反映させる分野相性の最低値。

    num_reads : int
        アニーリングの実行回数。

    num_sweeps : int
        neal使用時の1回当たりのスイープ数。

    seed : int
        乱数シード。

    penalty_margin : float
        予算制約ペナルティ係数の安全マージン。
        A = max_j(V_j + Σ正シナジー) × (1 + penalty_margin)。

    sampler :
        Noneの場合はnealのSimulatedAnnealingSamplerを使用する。
        D-Wave QPU用サンプラーを渡すこともできる
        （calculation.sampler.build_sampler が生成する）。

    sampler_kwargs : dict or None
        サンプラー固有の追加引数（QPUのannealing_time、chain_strengthなど）。
        num_reads・num_sweeps・seedより優先される。

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

    # 学習済み科目。以降はハッシュ可能な集合として扱う
    # （前提クロージャのキャッシュキーになるため）。
    learned_names = tuple(learned)

    unknown_learned = [
        name for name in learned_names if name not in item_index
    ]

    if unknown_learned:
        raise ValueError(f"学習済みに未知の項目名があります: {unknown_learned}")

    learned = frozenset(learned_names)

    learned_indices = [item_index[name] for name in learned_names]

    # 学習済み集合そのものの前提不整合（「Unityは学習済みだがC#は未習」など）。
    # 自己申告を尊重してエラーにも自動補完にもせず、警告として呼び出し側へ返す。
    learned_gaps = find_learned_prerequisite_gaps(learned_names)

    # 実効前提 P'(c) = P(c) - learned。
    # 以降の前提に関する処理はすべてこの辞書を見る。
    effective_prereq = effective_prerequisites(learned)

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
    #
    # 学習済みの項目も候補外にする（これが「学習済みを推薦しない」の実体）。
    # 候補入りの判定に使うのはこの元の価値item_valueであり、
    # 後述の実効価値effective_valueではない。
    # 学習済みシナジーのボーナスだけを理由に、
    # 相性0.60未満の分野の項目が候補へ紛れ込むのを防ぐためである。
    #
    # さらに、前提クロージャ込みの最小学習時間が予算Tを超える項目は、
    # 「前提を満たしつつ予算内」の解に決して現れないため候補から外す
    # （前提科目制約の前処理その1：候補集合の制限）。
    # 学習済みの親はクロージャから外れるので、
    # 学習済みがあるとこの枝刈りを通る項目はむしろ増えることがある。

    candidate_mask = item_value > 0.0

    candidate_mask[learned_indices] = False

    candidate_mask, pruned_items = restrict_candidates_by_budget(
        candidate_mask, T, learned
    )

    # 候補項目に必要な前提科目を候補へ追加する
    # （前提科目制約の前処理その2：候補展開）。

    # 候補展開の後に枝刈りをやり直す必要はない。
    # 追加される親の前提クロージャは子のクロージャの部分集合なので、
    # 子が枝刈りを通っていれば親の最小クロージャ時間も必ずT以下になる
    # （OR前提があった頃は、選択肢として追加した項目が
    #   単独では予算に収まらない場合があり、2回目の枝刈りが要った）。

    candidate_mask = add_prerequisites_to_candidates(candidate_mask, learned)

    candidate_indices = np.flatnonzero(candidate_mask).tolist()

    if len(candidate_indices) == 0:
        # 「予算が足りない」のか「学習済みで尽きた」のかを切り分けられるように、
        # 予算Tと、前提込みで最も短く学べる項目の時間を添える。
        remaining = [
            name
            for name, j in item_index.items()
            if item_value[j] > 0.0 and name not in learned
        ]

        if not remaining:
            raise RuntimeError(
                "推薦候補となる学習項目がありません。"
                f"相性{min_relevance:.2f}以上の分野に属する項目は、すべて学習済みです。"
            )

        cheapest = min(min_closure_hours(name, learned) for name in remaining)

        raise RuntimeError(
            "推薦候補となる学習項目がありません。"
            f"予算{T}hに対し、前提込みで最も短く学べる項目でも{cheapest}h必要です。"
        )

    # ========================================================
    # 3b. 学習済みシナジーの畳み込み（実効価値）
    # ========================================================

    # 学習済みの項目は「z=1に固定してQUBOから消した変数」とみなす。
    # 二次項 S_ij z_i z_j に z_i=1 を代入すると S_ij z_j、
    # つまり項目jの線形ボーナスへ落ちる。これを項目価値へ畳み込む。
    #
    #   B_j  = Σ_{i∈learned} S_ij     （負にもなりうる）
    #   V'_j = V_j + B_j
    #
    # 正負とも畳み込む。負のシナジー（Unity×Unreal Engineなど）は
    # 「同時に学ぶと非効率」であると同時に「重複スキルの冗長性」でもあり、
    # 既習との関係でも成立するとみなす。
    #
    # 以降、QUBOの係数と目的関数値にはこの実効価値を使う。
    # 候補入りの判定だけは元のitem_valueで済ませてある（3節）。

    if learned_indices:
        learned_synergy = item_synergy[learned_indices].sum(axis=0)
    else:
        learned_synergy = np.zeros(M, dtype=float)

    effective_value = item_value + learned_synergy

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
    #
    # ただし前提科目を持つ項目の価値は、
    # 「前提が揃って初めて発生する条件付き価値」として
    # 8節で二次項として追加するため、ここでは線形項を与えない。
    #
    # 前提を持つかどうかは実効前提P'で判定する。
    # 親がすべて学習済みになった項目は「前提のない項目」に戻り、
    # 価値は条件付きではなく線形項として入る。

    for j in candidate_indices:
        if effective_prereq.get(item_names[j]):
            bqm.add_variable(f"item_{j}", 0.0)
        else:
            bqm.add_variable(f"item_{j}", -float(effective_value[j]))

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
    # 6. ペナルティ係数（予算制約専用）
    # ========================================================

    # QUBO に残るペナルティ項は予算制約（7節）ただ一つ
    # （前提科目は前処理＋修復デコードで担保し、ペナルティ項を持たない）。
    # その係数 A は「違反を防ぐのに必要な最小値」から決める。
    #
    # 予算を1単位超えた解からは、他の選択項目の前提になっていない項目
    # （前提DAGの極大元。選択が空でなければ必ず存在する）を1つ除去すれば、
    # 子項目の条件付き価値を巻き添えにせず実行可能化できる。
    # このとき失う報酬の主要項は「単一項目の限界寄与」である：
    #
    #   限界寄与_j = V_j + Σ_i max(0, S_ij)
    #             （項目jの価値 ＋ jが持つ正のシナジーの合計）
    #
    # 予算を1単位超えるごとにペナルティは A 増えるので、
    #   A > max_j 限界寄与
    # なら1単位違反の解はその除去先（実行可能）に負ける。
    #
    # 旧実装の「得られる最大報酬の合計＋1」は「1単位の超過で全報酬が
    # 消える」ほど過大で、目的差（0.15〜2.4）に対し1〜2桁大きい
    # ペナルティになっていた（issue #8）。単一項目の限界寄与を基準に
    # すると、この桁差が解消する。
    #
    # OR前提を廃止したことで、この基準は厳密な上界になった。
    # 8節の条件付き価値はAND前提のみとなり、親が揃ったときの報酬は
    # ちょうど V_j（満額）で頭打ちになるためである
    # （OR前提があった頃は、親をk個選んだ子の報酬が
    #   V_j·k/min(m,2) まで膨らみ限界寄与を上回りうるため、
    #   「A > 限界寄与 なら常に実行可能解が勝つ」と主張できなかった）。
    #
    # 前提科目として追加された価値0の項目は限界寄与も0で影響しない。
    #
    # なお桁差の解消は解の品質を改善しなかった（Aを1/12にしても達成率の差は
    # シード間ばらつきに埋もれる）。桁差は品質のボトルネックではなかった、
    # というのが tools/exact_bb.py との比較で得た実測の結論である。
    #
    # report 表示用に、報酬の総量上限も併せて求めておく
    # （係数の決定には使わない）。

    positive_synergy_per_item = {j: 0.0 for j in candidate_indices}

    for a in range(len(candidate_indices)):
        i = candidate_indices[a]

        for b in range(a + 1, len(candidate_indices)):
            j = candidate_indices[b]

            synergy_value = item_synergy[i, j]

            if synergy_value > 0.0:
                positive_synergy_per_item[i] += float(synergy_value)
                positive_synergy_per_item[j] += float(synergy_value)

    # 価値の側は実効価値V'を使う。学習済みシナジーB_jは
    # 「項目jを1つ追加したときに解放される報酬」に含まれるので、
    # V_jのままだとAが過小になり、1単位違反の解が実行可能解に勝ちうる。
    # 内側の和は候補内のペアのみ（学習済みとの分はV'側に入っており、二重計上しない）。

    marginal_contribution = max(
        max(0.0, float(effective_value[j])) + positive_synergy_per_item[j]
        for j in candidate_indices
    )

    # 厳密不等号を保つため、限界寄与に微小マージンを掛ける。
    constraint_penalty = marginal_contribution * (1.0 + penalty_margin)

    reward_upper_bound = (
        sum(max(0.0, float(effective_value[j])) for j in candidate_indices)
        + synergy_upper_bound
    )

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
    # 8. 前提科目の条件付き価値
    # ========================================================

    # 前提科目はハード制約であり、ペナルティ項では担保しない
    # （担保は前処理＝候補集合の制限と、10節の修復デコードが行う）。
    # そのため、旧実装にあった前提科目のペナルティ項と
    # OR用スラック変数（or_slack_*）はQUBOに存在しない。
    #
    # その代わり、前提科目を持つ子項目の価値V_childを
    # 「前提が揃って初めて発生する条件付き価値」の二次項として表現し、
    # 前提を無視した選択が得にならない方向へアニーリングを誘導する。
    # 係数はペナルティ係数Aではなく価値Vそのものなので、
    # エネルギー地形が巨大な制約項に支配されることはない。
    #
    # AND前提（親p_1..p_nがすべて必要）：
    #   -V * child * (p_1 + ... + p_n - (n-1))
    #
    #   全親を選択 → -V（満額）／親が1つ欠ける → 0（価値なし）
    #   ／さらに欠ける → 正（予算を消費するだけで損）
    #
    # 前提科目はAND前提のみなので、この表現は近似ではなく
    # 「前提が揃ったとき、かつそのときだけ満額」を厳密に表す
    # （OR前提があった頃は選択肢数に比例する近似項が別に必要だった）。

    for j in candidate_indices:
        child_name = item_names[j]

        # 前提のない項目（親がすべて学習済みになった項目を含む）の価値は
        # 5節で線形項として追加済み
        if not effective_prereq.get(child_name):
            continue

        # 学習済みシナジーB_cも条件付き価値の側に含める（V'ごと二次項にする）。
        # B_cだけを無条件の線形項にすると、前提を満たさないまま子を立てても
        # ボーナスだけ得られる形になり、探索の誘導が濁るためである。
        child_value = float(effective_value[j])

        # 前提科目として追加された価値0の項目は誘導項も不要
        if child_value == 0.0:
            continue

        child_variable = f"item_{j}"

        # 前提科目は候補展開で必ず候補に入っている
        parent_indices = []

        for parent_name in effective_prereq[child_name]:
            parent = item_index[parent_name]

            if not candidate_mask[parent]:
                raise RuntimeError(
                    f"前提科目「{parent_name}」が候補に追加されていません。"
                )

            parent_indices.append(parent)

        for parent in parent_indices:
            bqm.add_quadratic(child_variable, f"item_{parent}", -child_value)

        bqm.add_linear(child_variable, child_value * (len(parent_indices) - 1))

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

    # サンプラー固有の引数（QPUのannealing_timeなど）を上書きで反映する
    if sampler_kwargs:
        sample_kwargs.update(sampler_kwargs)

    sampleset = sampler.sample(bqm, **sample_kwargs)

    # ========================================================
    # 10. 修復デコードと実行可能解の抽出
    # ========================================================

    # 各サンプルの選択から、前提を満たさない子項目を取り除く
    # （前提科目制約の担保その3：修復デコード）。
    #
    # 修復後の選択は定義上すべて前提科目制約を満たすため、
    # 前提違反を理由にサンプルが捨てられることはない。
    # 修復は項目を取り除くだけで学習時間が増えることもなく、
    # 残る検査は予算のみになる。
    #
    # 順位付けはQUBOエネルギーではなく、修復後の選択の
    # 真の目的関数値（項目価値＋シナジー）で行う。
    # エネルギーは修復前のビット列に対応する値であり、
    # 修復で項目が除かれたサンプルでは実際の価値とずれるため。

    best_z = None
    best_score = -np.inf
    best_hours = 0
    best_energy = np.inf

    feasible_count = 0
    budget_violation_count = 0
    prerequisite_repair_count = 0

    # エネルギー分布の可視化用。
    # 実行可能解と制約違反解を分けて記録しておく。
    feasible_energies = []
    infeasible_energies = []

    for datum in sampleset.data(fields=["sample", "energy"], sorted_by="energy"):
        sample = datum.sample

        z_raw = np.zeros(M, dtype=int)

        for j in candidate_indices:
            z_raw[j] = int(sample.get(f"item_{j}", 0))

        z_repaired, removed_names = repair_prerequisites(z_raw, learned)

        if removed_names:
            prerequisite_repair_count += 1

        # 予算検査は修復後の選択に対して行う
        selected_hours = int(hours @ z_repaired)

        if selected_hours > T:
            budget_violation_count += 1
            infeasible_energies.append(float(datum.energy))
            continue

        feasible_count += 1
        feasible_energies.append(float(datum.energy))

        decoded_selections = [(z_repaired, selected_hours)]

        # 前提が欠けていたサンプルは、子項目を諦める下方修復だけでなく、
        # 不足親を追加する上方補完も試し、予算内に収まれば候補に加える
        # （高価値の子項目を捨てずに済む解を拾うため）
        if removed_names:
            z_completed, _ = complete_prerequisites(
                z_raw, candidate_mask, learned
            )

            if z_completed is not None:
                completed_hours = int(hours @ z_completed)

                if completed_hours <= T:
                    decoded_selections.append((z_completed, completed_hours))

        for z_decoded, decoded_hours in decoded_selections:
            # 修復後の真の目的関数値（大きいほど良い）。
            # 学習済みシナジーを畳み込んだ実効価値で測る。
            # item_synergyは対角0の対称行列なので、
            # 二次形式の半分がペア合計になる。
            score = float(effective_value @ z_decoded) + 0.5 * float(
                z_decoded @ item_synergy @ z_decoded
            )

            # 同スコアなら学習時間の短い解を採用する
            # （価値0の項目だけを追加した冗長な解を除くため）
            if score > best_score or (
                score == best_score and decoded_hours < best_hours
            ):
                best_z = z_decoded
                best_score = score
                best_hours = decoded_hours
                best_energy = float(datum.energy)

    if best_z is None:
        raise RuntimeError(
            "予算制約を満たすサンプルが得られませんでした。\n"
            "num_readsまたはnum_sweepsを増やしてください。"
        )

    z = best_z

    # ========================================================
    # 11. 最終検査
    # ========================================================

    total_hours = int(hours @ z)

    prerequisites_ok, violations = check_prerequisites(z, learned)

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

    # 学習済みと選択項目のあいだで発生したシナジー。
    # 実効価値へ畳み込んで表に出なくなった分を、結果表示で説明するために残す。
    learned_synergies = []

    for name in learned_names:
        i = item_index[name]

        for j in candidate_indices:
            if z[j] != 1:
                continue

            synergy_value = item_synergy[i, j]

            if synergy_value != 0.0:
                learned_synergies.append(
                    (name, item_names[j], float(synergy_value))
                )

    debug_info = {
        "field_score": field_score,
        "field_relevance": field_relevance,
        "item_value": item_value,
        "effective_value": effective_value,
        "learned": learned_names,
        "learned_indices": learned_indices,
        "learned_hours": int(sum(hours[j] for j in learned_indices)),
        "learned_gaps": learned_gaps,
        "learned_synergy": learned_synergy,
        "learned_synergies": learned_synergies,
        "effective_prerequisites": effective_prereq,
        "candidate_mask": candidate_mask,
        "candidate_indices": candidate_indices,
        "pruned_items": pruned_items,
        "time_unit": time_unit,
        "cost_units": cost_units,
        "budget_units": budget_units,
        "slack_weights": slack_weights,
        "reward_upper_bound": reward_upper_bound,
        "marginal_contribution": marginal_contribution,
        "constraint_penalty": constraint_penalty,
        "synergy_upper_bound": synergy_upper_bound,
        "active_synergies": active_synergies,
        "energy": best_energy,
        "best_score": best_score,
        "feasible_count": feasible_count,
        "feasible_energies": feasible_energies,
        "infeasible_energies": infeasible_energies,
        "budget_violation_count": budget_violation_count,
        "prerequisite_repair_count": prerequisite_repair_count,
        "bqm": bqm,
        "sampleset": sampleset,
        "coefficient_ratio": coefficient_ratio(bqm),
        "solver_info": summarize_sampleset(sampleset),
    }

    return z, field_score, debug_info
