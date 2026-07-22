"""前提科目制約に関する計算処理。

前提科目はハード制約であり、QUBOの目的関数（ペナルティ項）では担保しない。
本モジュールが以下の3段階で構造的に担保する。

1. 候補集合の制限（restrict_candidates_by_budget）：
   前提クロージャ込みの最小学習時間が予算を超える項目を候補から外す。

2. 候補展開（add_prerequisites_to_candidates）：
   候補項目が必要とする前提科目を候補へ追加する。

3. 修復デコード（repair_prerequisites / complete_prerequisites）：
   アニーリングのサンプルを「前提未達の子項目を取り除く（下方修復）」
   「不足している前提科目を追加する（上方補完）」の両方向で
   前提を満たす選択へ変換し、それだけを解の候補にする。

check_prerequisitesは最終検査として残す。
"""

from functools import lru_cache

from config.items import hours, item_index
from config.prerequisites import and_prerequisites, or_prerequisites


def check_prerequisites(z):
    """
    選択結果zが前提科目制約を満たしているか確認する。

    Returns
    -------
    ok : bool
        すべて満たしていればTrue。

    violations : list
        違反している (制約種別, 子項目, 前提項目) の一覧。
    """
    violations = []

    # AND前提：リストの全項目が必要
    for child_name, parent_names in and_prerequisites.items():
        child = item_index[child_name]

        if z[child] == 0:
            continue

        for parent_name in parent_names:
            parent = item_index[parent_name]

            if z[parent] == 0:
                violations.append(("AND", child_name, parent_name))

    # OR前提：リストのいずれか1項目が選ばれていればよい
    for child_name, parent_names in or_prerequisites.items():
        child = item_index[child_name]

        if z[child] == 0:
            continue

        parent_indices = [item_index[parent_name] for parent_name in parent_names]

        # 選択肢の中に1つでも選ばれていればOK
        any_parent_selected = any(z[parent] == 1 for parent in parent_indices)

        if not any_parent_selected:
            violations.append(("OR", child_name, tuple(parent_names)))

    return len(violations) == 0, violations


def add_prerequisites_to_candidates(candidate_mask):
    """
    価値が正の項目を起点として、
    必要な前提科目（AND・OR両方）を
    再帰的に候補へ追加する。

    OR前提については、どの選択肢が選ばれるかは
    アニーリング側が決めるため、選択肢を"すべて"候補に
    追加しておく（候補に入っていない項目はQUBOの変数として
    存在しないので、選ばれる余地すらなくなってしまう）。

    前提科目へ追加の価値は与えない。
    """
    candidate_mask = candidate_mask.copy()

    changed = True

    while changed:
        changed = False

        # AND前提：全項目を候補に追加する
        for child_name, parent_names in and_prerequisites.items():
            child = item_index[child_name]

            # 子項目が候補でなければ、前提科目を見る必要はない
            if not candidate_mask[child]:
                continue

            for parent_name in parent_names:
                parent = item_index[parent_name]

                # 前提科目がまだ候補に入っていなければ追加する
                if not candidate_mask[parent]:
                    candidate_mask[parent] = True
                    changed = True  # 追加が発生したので、もう一周チェックする

        # OR前提：選択肢をすべて候補に追加する
        # （実際に選ばれるのはこの中の1つ以上でよい）
        for child_name, parent_names in or_prerequisites.items():
            child = item_index[child_name]

            if not candidate_mask[child]:
                continue

            for parent_name in parent_names:
                parent = item_index[parent_name]

                if not candidate_mask[parent]:
                    candidate_mask[parent] = True
                    changed = True

    return candidate_mask


@lru_cache(maxsize=None)
def _min_closure_hours(selected):
    """
    選択済み集合selected（項目名のfrozenset）を含み、
    前提科目制約をすべて満たす項目集合のうち、
    合計学習時間が最小のものの時間を返す。

    AND前提は不足親の追加が強制なのでそのまま再帰し、
    OR前提は選択肢ごとに分岐して最小値を取る。
    共有された親は集合なので二重計上されない。
    """
    # AND前提：不足している親をすべて追加する（強制）
    for child_name, parent_names in and_prerequisites.items():
        if child_name not in selected:
            continue

        missing = frozenset(
            parent_name
            for parent_name in parent_names
            if parent_name not in selected
        )

        if missing:
            return _min_closure_hours(selected | missing)

    # OR前提：どの選択肢を選ぶかで分岐し、最小の合計時間を取る
    for child_name, parent_names in or_prerequisites.items():
        if child_name not in selected:
            continue

        if any(parent_name in selected for parent_name in parent_names):
            continue

        return min(
            _min_closure_hours(selected | frozenset([parent_name]))
            for parent_name in parent_names
        )

    # すべての前提が満たされた
    return int(sum(hours[item_index[name]] for name in selected))


def min_closure_hours(item_name):
    """
    項目を1つ選ぶために最低限必要な合計学習時間
    （前提クロージャ込み）を返す。
    """
    return _min_closure_hours(frozenset([item_name]))


def restrict_candidates_by_budget(candidate_mask, T):
    """
    前提クロージャ込みの最小学習時間が予算Tを超える項目を
    候補から外す。

    そのような項目は、どんな選び方をしても
    「前提を満たしつつ予算内」の解に現れることができないため、
    除外しても最適性を損なわない（安全な枝刈り）。

    Returns
    -------
    candidate_mask : ndarray
        除外後の候補マスク。

    pruned_names : list
        除外した項目名の一覧。
    """
    candidate_mask = candidate_mask.copy()

    pruned_names = []

    for item_name, j in item_index.items():
        if candidate_mask[j] and min_closure_hours(item_name) > T:
            candidate_mask[j] = False
            pruned_names.append(item_name)

    return candidate_mask, pruned_names


def repair_prerequisites(z):
    """
    前提科目を満たさない子項目を、すべて満たされるまで
    取り除く（下方修復）。

    AND・OR前提はどちらも親の選択に対して単調
    （親が増えて制約が破れることはない）なので、
    取り除く順序によらず「元の選択に含まれる
    最大の実行可能部分集合」に収束する。

    項目を取り除くだけなので、合計学習時間は元の選択以下になり、
    予算制約を新たに破ることはない。

    Returns
    -------
    z : ndarray
        修復後の選択（元の配列は変更しない）。

    removed_names : list
        取り除いた項目名の一覧。
    """
    z = z.copy()

    removed_names = []

    changed = True

    while changed:
        changed = False

        # AND前提：親が1つでも欠けている子項目を取り除く
        for child_name, parent_names in and_prerequisites.items():
            child = item_index[child_name]

            if z[child] == 0:
                continue

            if any(z[item_index[p]] == 0 for p in parent_names):
                z[child] = 0
                removed_names.append(child_name)
                changed = True  # 連鎖的な違反を再チェックする

        # OR前提：選択肢が1つも選ばれていない子項目を取り除く
        for child_name, parent_names in or_prerequisites.items():
            child = item_index[child_name]

            if z[child] == 0:
                continue

            if all(z[item_index[p]] == 0 for p in parent_names):
                z[child] = 0
                removed_names.append(child_name)
                changed = True

    return z, removed_names


def complete_prerequisites(z, item_value, candidate_mask):
    """
    前提科目が不足している子項目を諦める代わりに、
    不足している前提科目を追加して前提を満たす（上方補完）。

    AND前提は不足親をすべて追加する。
    OR前提は候補内の選択肢のうち価値が最大
    （同価値なら学習時間が最短）の親を1つ追加する。
    追加した親にさらに前提があれば連鎖的に補完する。

    項目を追加するので合計学習時間は増える。
    予算に収まるかの検査は呼び出し側で行うこと。

    Returns
    -------
    z : ndarray or None
        補完後の選択（元の配列は変更しない）。
        候補内の項目だけでは前提を満たせない場合はNone。

    added_names : list
        追加した項目名の一覧。
    """
    z = z.copy()

    added_names = []

    changed = True

    while changed:
        changed = False

        # AND前提：不足親をすべて追加する
        for child_name, parent_names in and_prerequisites.items():
            child = item_index[child_name]

            if z[child] == 0:
                continue

            for parent_name in parent_names:
                parent = item_index[parent_name]

                if z[parent] == 1:
                    continue

                if not candidate_mask[parent]:
                    return None, []

                z[parent] = 1
                added_names.append(parent_name)
                changed = True  # 追加した親の前提も再チェックする

        # OR前提：最も価値の高い選択肢を1つ追加する
        for child_name, parent_names in or_prerequisites.items():
            child = item_index[child_name]

            if z[child] == 0:
                continue

            if any(z[item_index[p]] == 1 for p in parent_names):
                continue

            allowed_names = [
                p for p in parent_names if candidate_mask[item_index[p]]
            ]

            if len(allowed_names) == 0:
                return None, []

            best_parent_name = min(
                allowed_names,
                key=lambda p: (-item_value[item_index[p]], hours[item_index[p]]),
            )

            z[item_index[best_parent_name]] = 1
            added_names.append(best_parent_name)
            changed = True

    return z, added_names
