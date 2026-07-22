"""推薦結果の整理とテキスト表示。"""

from config.fields import field_names
from config.items import M, hours, item_fields, item_index, item_names
from config.parameters import axis_names
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
            for child_name, parent_names in and_prerequisites.items()
            if item_names[j] in parent_names
            and z[item_index[child_name]] == 1
        ]

        # この項目に関わっているシナジー（選ばれた相手のみ）
        synergies = [
            (partner, value)
            for name_a, name_b, value in debug_info["active_synergies"]
            for partner in [name_b if name_a == item_names[j] else name_a]
            if item_names[j] in (name_a, name_b)
        ]

        selected.append({
            "index": j,
            "name": item_names[j],
            "hours": int(hours[j]),
            "value": float(debug_info["item_value"][j]),
            "related_fields": related_fields,
            "effective_fields": effective_fields,
            "required_by": required_by,
            "synergies": synergies,
        })

    # 項目価値の高い順、同価値なら学習時間の短い順
    selected.sort(
        key=lambda item: (-item["value"], item["hours"], item["name"])
    )

    return selected


def print_report(user, hours_per_week, weeks, T, z, field_score, debug_info):
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

    print(f"\n【学習時間】週{hours_per_week}h × {weeks}週 = {T}h")

    print("\n【各IT分野との相性】")

    for field_name, score, relevance in sorted_fields:
        if relevance > 0.0:
            status = f"推薦計算に使用：{relevance:.3f}"
        else:
            status = "相性0.60未満のため除外"

        print(f"  {field_name}: 相性 {score:.3f}｜{status}")

    print("\n【推薦する学習項目】")

    if len(selected) == 0:
        print("  推薦項目はありません。")

    else:
        for item in selected:
            print(f"\n  ✅ {item['name']}（{item['hours']}h）")

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

            if item["synergies"]:
                synergy_text = ", ".join(
                    f"{partner}({value:+.2f})"
                    for partner, value in item["synergies"]
                )
                print("     シナジー: " + synergy_text)

    print(f"\n【合計学習時間】{total_hours}h / {T}h")

    print(f"【残り時間】{remaining_hours}h")

    print(f"【時間圧縮単位】1単位 = {debug_info['time_unit']}h")

    print(
        f"【報酬上限】{debug_info['reward_upper_bound']:.3f}"
        f"（うちシナジー上限：{debug_info['synergy_upper_bound']:.3f}）"
    )

    print(f"【制約ペナルティ係数】{debug_info['constraint_penalty']:.3f}")

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

    print("=" * 76)
