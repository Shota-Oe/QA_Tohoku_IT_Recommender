"""推薦結果の整理とテキスト表示。"""

from config.fields import field_index, field_names
from config.items import M, hours, item_fields, item_index, item_names
from config.parameters import DEFAULT_MIN_RELEVANCE, axis_names
from config.prerequisites import and_prerequisites


def sort_fields_by_score(field_score, field_relevance):
    """相性の高い順に (分野名, 相性, 関連度) を並べる。"""
    return sorted(
        zip(field_names, field_score, field_relevance),
        key=lambda x: -x[1],
    )


def summarize_selection(z, debug_info):
    """選択された項目を表示用の辞書リストへ整理する。"""
    selected = []

    learned = set(debug_info["learned"])

    # 前提の表示は実効前提を見る
    # （学習済みで満たされた親は「必要とした項目」に数えない）。
    effective_prereq = debug_info["effective_prerequisites"]

    for j in range(M):
        if z[j] != 1:
            continue

        related_fields = [field_names[field] for field in item_fields[j]]

        effective_fields = [
            field_names[field]
            for field in item_fields[j]
            if debug_info["field_relevance"][field] > 0.0
        ]

        # この項目を必要とする上位項目
        required_by = [
            child_name
            for child_name, parent_names in effective_prereq.items()
            if item_names[j] in parent_names
            and z[item_index[child_name]] == 1
        ]

        # この項目の前提のうち、学習済みで満たされているもの
        learned_parents = [
            parent_name
            for parent_name in and_prerequisites.get(item_names[j], [])
            if parent_name in learned
        ]

        # この項目に関わっているシナジー（選ばれた相手のみ）
        synergies = [
            (partner, value)
            for name_a, name_b, value in debug_info["active_synergies"]
            for partner in [name_b if name_a == item_names[j] else name_a]
            if item_names[j] in (name_a, name_b)
        ]

        # 学習済みとのシナジー（実効価値へ畳み込まれている分）
        learned_synergies = [
            (learned_name, value)
            for learned_name, item_name, value in debug_info["learned_synergies"]
            if item_name == item_names[j]
        ]

        selected.append({
            "index": j,
            "name": item_names[j],
            "hours": int(hours[j]),
            "value": float(debug_info["effective_value"][j]),
            "field_value": float(debug_info["item_value"][j]),
            "related_fields": related_fields,
            "effective_fields": effective_fields,
            "required_by": required_by,
            "learned_parents": learned_parents,
            "synergies": synergies,
            "learned_synergies": learned_synergies,
        })

    # 項目価値の高い順、同価値なら学習時間の短い順
    selected.sort(
        key=lambda item: (-item["value"], item["hours"], item["name"])
    )

    return selected


def print_learned(debug_info):
    """学習済み科目（前処理で候補から外した項目）を表示する。

    学習済みの時間は予算Tを消費しない（過去に消費済みの時間）ので、
    合計時間は「今回の学習計画から外れた分」の目安として出す。
    """
    learned_names = debug_info["learned"]

    if not learned_names:
        return

    print(
        f"\n【学習済み（前処理で候補から除外）】{len(learned_names)}件"
        f"・計{debug_info['learned_hours']}h"
    )

    for name in learned_names:
        print(f"  ✔ {name}（{hours[item_index[name]]}h）")

    # 学習済み集合そのものが前提を満たしていない場合の警告。
    # 自己申告を尊重し、未習の前提項目は候補に残したまま続行している。
    for child_name, parent_name in debug_info["learned_gaps"]:
        print(
            f"  ⚠ 「{child_name}」は学習済みですが、"
            f"前提の「{parent_name}」は未習です"
            "（自己申告のまま扱い、前提項目は候補に残しています）"
        )


def print_interests(interests, field_score, debug_info):
    """アンケートで申告された興味のある分野を表示する。

    興味は推薦計算には使っていない（分野は適性5軸からの相性だけで決まる）。
    自己申告と相性が食い違う回答が実際にあるため、相性と閾値の判定を併記して
    「申告した分野が推薦に効いているのか」を読み取れるようにする。
    """
    if not interests:
        return

    print(f"\n【興味のある分野】{len(interests)}件（自己申告・推薦計算には未使用）")

    for name in interests:
        field = field_index[name]

        if debug_info["field_relevance"][field] > 0.0:
            status = "推薦計算に使用"
        else:
            status = f"相性{DEFAULT_MIN_RELEVANCE:.2f}未満のため除外"

        print(f"  ・{name}（適性からの相性 {field_score[field]:.3f}：{status}）")


def print_solver_info(debug_info, backend_description):
    """アニーリング実行基盤の情報を表示する。

    実機QPUのときは、マイナー埋め込みの規模・鎖切れ率・QPU時間という
    「実機でしか出てこない指標」を併せて出す。
    """
    solver_info = debug_info.get("solver_info") or {}

    print(f"\n【アニーリング実行基盤】{backend_description}")

    print(
        f"【QUBO規模】{debug_info['bqm'].num_variables}変数／"
        f"二次項{debug_info['bqm'].num_interactions}"
        f"｜係数のダイナミックレンジ {debug_info['coefficient_ratio']:.0f}倍"
    )

    if not solver_info:
        return

    if "physical_qubits" in solver_info:
        print(
            f"【マイナー埋め込み】物理量子ビット{solver_info['physical_qubits']}個"
            f"（鎖長 平均{solver_info['mean_chain_length']:.1f}／"
            f"最大{solver_info['max_chain_length']}）"
        )

    if "chain_break_mean" in solver_info:
        print(
            f"【鎖切れ率】平均{solver_info['chain_break_mean']:.3f}／"
            f"最大{solver_info['chain_break_max']:.3f}"
            f"｜鎖切れなしのサンプル {solver_info['chain_break_free_ratio']:.1%}"
        )

    if "qpu_access_time" in solver_info:
        print(
            f"【QPUアクセス時間】{solver_info['qpu_access_time'] / 1000:.1f}ms"
            f"（うちアニール {solver_info['qpu_anneal_time_per_sample']:.0f}µs/サンプル）"
        )

    if "problem_id" in solver_info:
        print(f"【問題ID】{solver_info['problem_id']}")


def print_report(
    user,
    hours_per_week,
    weeks,
    T,
    z,
    field_score,
    debug_info,
    interests=(),
    backend_description="neal（シミュレーテッド・アニーリング／ローカル）",
):
    """推薦結果を標準出力へ表示する。"""
    selected = summarize_selection(z, debug_info)

    sorted_fields = sort_fields_by_score(
        field_score,
        debug_info["field_relevance"],
    )

    total_hours = int(hours @ z)
    remaining_hours = T - total_hours

    print("=" * 76)

    print("【入力された適性】")

    for axis_name, value in zip(axis_names, user):
        print(f"  {axis_name}: {value:.1f}")

    print_interests(interests, field_score, debug_info)

    print(f"\n【学習時間】週{hours_per_week}h × {weeks}週 = {T}h")

    print_learned(debug_info)

    print("\n【各IT分野との相性】")

    for field_name, score, relevance in sorted_fields:
        if relevance > 0.0:
            status = f"推薦計算に使用：{relevance:.3f}"
        else:
            status = f"相性{DEFAULT_MIN_RELEVANCE:.2f}未満のため除外"

        print(f"  {field_name}: 相性 {score:.3f}｜{status}")

    print("\n【推薦する学習項目】")

    if len(selected) == 0:
        print("  推薦項目はありません。")

    else:
        for item in selected:
            print(f"\n  ✅ {item['name']}（{item['hours']}h）")

            if item["learned_synergies"]:
                learned_bonus = sum(
                    value for _, value in item["learned_synergies"]
                )
                print(
                    f"     項目価値: {item['value']:.3f}"
                    f"（分野価値 {item['field_value']:.3f}"
                    f" ＋ 既習シナジー {learned_bonus:+.3f}）"
                )
            else:
                print(f"     項目価値: {item['value']:.3f}")

            if item["effective_fields"]:
                print(
                    "     価値に反映された分野: "
                    + ", ".join(item["effective_fields"])
                )
            else:
                print("     価値に反映された分野: なし（前提科目として選択）")

            if item["required_by"]:
                print(
                    "     前提として必要とした項目: "
                    + ", ".join(item["required_by"])
                )

            if item["learned_parents"]:
                print(
                    "     学習済みで満たした前提: "
                    + ", ".join(item["learned_parents"])
                )

            if item["synergies"]:
                synergy_text = ", ".join(
                    f"{partner}({value:+.2f})"
                    for partner, value in item["synergies"]
                )
                print("     シナジー: " + synergy_text)

            if item["learned_synergies"]:
                learned_synergy_text = ", ".join(
                    f"{partner}({value:+.2f})"
                    for partner, value in item["learned_synergies"]
                )
                print("     既習シナジー: " + learned_synergy_text)

    print(f"\n【合計学習時間】{total_hours}h / {T}h")

    print(f"【残り時間】{remaining_hours}h")

    print(f"【時間圧縮単位】1単位 = {debug_info['time_unit']}h")

    print(
        f"【報酬上限】{debug_info['reward_upper_bound']:.3f}"
        f"（うちシナジー上限：{debug_info['synergy_upper_bound']:.3f}）"
    )

    print(
        f"【予算ペナルティ係数】{debug_info['constraint_penalty']:.3f}"
        f"（基準＝単一項目の限界寄与 {debug_info['marginal_contribution']:.3f}）"
    )

    if debug_info["pruned_items"]:
        print(
            "【予算内で前提を満たせないため候補から除外】"
            + ", ".join(debug_info["pruned_items"])
        )

    print(
        f"【実行可能サンプル数】{debug_info['feasible_count']}"
        f" / 取得サンプル数{len(debug_info['sampleset'])}"
    )

    print(
        f"【前提科目を修復したサンプル数】"
        f"{debug_info['prerequisite_repair_count']}"
        f"（修復後の候補はすべて前提充足）"
    )

    print(f"【採用解の目的関数値（価値＋シナジー）】{debug_info['best_score']:.3f}")

    print(f"【採用解のQUBOエネルギー】{debug_info['energy']:.3f}")

    print_solver_info(debug_info, backend_description)

    print("=" * 76)
