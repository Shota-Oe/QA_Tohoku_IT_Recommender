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

前提科目はAND（全親必須）のみである。OR前提の廃止により、
子項目の前提クロージャは選択の余地がない一意の集合になり、
本モジュールの各処理から分岐（どの選択肢を選ぶか）が消えている。

学習済み科目（learned）を渡すと、本モジュールの各処理は
「実効前提」P'(c) = P(c) - learned を見る。
学習済みの親はすでに満たされているとみなす、ということである。
config/prerequisites.py の and_prerequisites 自体は書き換えない
（係数データは不変に保ち、畳んだ辞書は実行時に作る）。
"""

from functools import lru_cache

from config.items import hours, item_index
from config.prerequisites import and_prerequisites


@lru_cache(maxsize=None)
def effective_prerequisites(learned=frozenset()):
    """学習済みの親を満たし済みとみなして畳んだ前提辞書 P' を返す。

        P'(c) = P(c) - learned      （c は learned に属さない項目）

    親がすべて学習済みになった子は空リストを持ち、
    「前提のない項目」として扱われる（QUBOでは線形項へ戻る）。
    学習済みの項目自身は、子としても親としても現れない。

    キャッシュした辞書を共有して返すので、呼び出し側で変更しないこと。
    """
    if not learned:
        return and_prerequisites

    return {
        child_name: [
            parent_name
            for parent_name in parent_names
            if parent_name not in learned
        ]
        for child_name, parent_names in and_prerequisites.items()
        if child_name not in learned
    }


def find_learned_prerequisite_gaps(learned):
    """学習済み集合そのものが前提を満たしていない箇所を返す。

    「Unityは学習済みだがC#は未習」のような自己申告の不整合である。
    仕様上これはエラーにも自動補完にもせず、自己申告を尊重して
    警告を出すだけに留める（docs/requirements.md 第10.3節）。
    未習の親は候補に残り、価値があれば推薦されうる。

    Returns
    -------
    gaps : list
        (学習済みの子項目, 未習の前提項目) の一覧。
    """
    learned_set = set(learned)

    return [
        (child_name, parent_name)
        for child_name in learned
        for parent_name in and_prerequisites.get(child_name, [])
        if parent_name not in learned_set
    ]


def check_prerequisites(z, learned=frozenset()):
    """
    選択結果zが前提科目制約を満たしているか確認する。

    Returns
    -------
    ok : bool
        すべて満たしていればTrue。

    violations : list
        違反している (子項目, 前提項目) の一覧。
    """
    violations = []

    # AND前提：リストの全項目が必要
    for child_name, parent_names in effective_prerequisites(learned).items():
        child = item_index[child_name]

        if z[child] == 0:
            continue

        for parent_name in parent_names:
            parent = item_index[parent_name]

            if z[parent] == 0:
                violations.append((child_name, parent_name))

    return len(violations) == 0, violations


def add_prerequisites_to_candidates(candidate_mask, learned=frozenset()):
    """
    候補項目が必要とする前提科目を、再帰的に候補へ追加する。

    前提科目へ追加の価値は与えない。
    学習済みの親は追加されない（実効前提から外れているため）。
    """
    candidate_mask = candidate_mask.copy()

    changed = True

    while changed:
        changed = False

        for child_name, parent_names in effective_prerequisites(learned).items():
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

    return candidate_mask


@lru_cache(maxsize=None)
def prerequisite_closure(item_name, learned=frozenset()):
    """
    項目を1つ選ぶときに一緒に選ばざるを得ない項目の集合
    （その項目自身を含む前提クロージャ）を返す。

    AND前提だけなので、どの親を選ぶかの分岐はなく、
    クロージャは項目ごとに一意に定まる。
    共有された親は集合なので二重に現れない。

    学習済みの親はすでに満たされているのでクロージャに含めない。
    そのぶんクロージャは小さくなり、必要な学習時間も減る。
    """
    closure = {item_name}

    for parent_name in and_prerequisites.get(item_name, []):
        # 学習済みの親は改めて学ぶ必要がない
        if parent_name in learned:
            continue

        closure |= prerequisite_closure(parent_name, learned)

    return frozenset(closure)


def min_closure_hours(item_name, learned=frozenset()):
    """
    項目を1つ選ぶために最低限必要な合計学習時間
    （前提クロージャ込み）を返す。
    """
    return int(
        sum(
            hours[item_index[name]]
            for name in prerequisite_closure(item_name, learned)
        )
    )


def restrict_candidates_by_budget(candidate_mask, T, learned=frozenset()):
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
        if candidate_mask[j] and min_closure_hours(item_name, learned) > T:
            candidate_mask[j] = False
            pruned_names.append(item_name)

    return candidate_mask, pruned_names


def repair_prerequisites(z, learned=frozenset()):
    """
    前提科目を満たさない子項目を、すべて満たされるまで
    取り除く（下方修復）。

    AND前提は親の選択に対して単調
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

        # 親が1つでも欠けている子項目を取り除く
        for child_name, parent_names in effective_prerequisites(learned).items():
            child = item_index[child_name]

            if z[child] == 0:
                continue

            if any(z[item_index[p]] == 0 for p in parent_names):
                z[child] = 0
                removed_names.append(child_name)
                changed = True  # 連鎖的な違反を再チェックする

    return z, removed_names


def complete_prerequisites(z, candidate_mask, learned=frozenset()):
    """
    前提科目が不足している子項目を諦める代わりに、
    不足している前提科目を追加して前提を満たす（上方補完）。

    不足している親をすべて追加する。
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

        for child_name, parent_names in effective_prerequisites(learned).items():
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

    return z, added_names
