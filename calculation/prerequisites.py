"""前提科目制約に関する計算処理。"""

from config.items import item_index
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
