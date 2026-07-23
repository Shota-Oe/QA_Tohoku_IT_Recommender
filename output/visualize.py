"""推薦結果の可視化。"""

import japanize_matplotlib  # noqa: F401  日本語フォントの設定
import matplotlib.pyplot as plt
import numpy as np

from config.items import hours, item_names
from config.parameters import DEFAULT_MIN_RELEVANCE
from output.report import sort_fields_by_score


def plot_results(z, T, field_score, debug_info):
    """分野相性・項目価値・エネルギー分布の3枚のグラフを表示する。"""
    sorted_fields = sort_fields_by_score(
        field_score,
        debug_info["field_relevance"],
    )

    total_hours = int(hours @ z)

    _, axes = plt.subplots(1, 3, figsize=(22, 11))

    # --------------------------------------------------------
    # 1. IT分野の相性
    # --------------------------------------------------------

    field_plot_names = [field for field, _, _ in sorted_fields]
    field_plot_scores = [score for _, score, _ in sorted_fields]

    field_plot_colors = [
        "steelblue" if relevance > 0.0 else "lightgray"
        for _, _, relevance in sorted_fields
    ]

    axes[0].barh(field_plot_names, field_plot_scores, color=field_plot_colors)

    axes[0].axvline(
        DEFAULT_MIN_RELEVANCE,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"最低相性 {DEFAULT_MIN_RELEVANCE:.2f}",
    )

    axes[0].set_title("適性と各IT分野の相性\n青＝項目価値に使用、灰＝除外")
    axes[0].set_xlabel("相性スコア")
    axes[0].set_xlim(0.0, 1.05)
    axes[0].invert_yaxis()
    axes[0].legend()

    # --------------------------------------------------------
    # 2. 学習項目の価値
    # --------------------------------------------------------

    # 学習済みの項目は候補に入っていないので、そもそも棒として現れない。
    # 棒の長さは実効価値（分野価値＋既習シナジー）で、レポートの
    # 「項目価値」と同じ値になる。

    candidate_indices = debug_info["candidate_indices"]

    effective_value = debug_info["effective_value"]

    item_order = sorted(
        candidate_indices,
        key=lambda j: (-effective_value[j], hours[j]),
    )

    plot_item_names = [item_names[j] for j in item_order]
    plot_item_values = [effective_value[j] for j in item_order]

    plot_item_colors = [
        "salmon" if z[j] == 1 else "lightgray"
        for j in item_order
    ]

    axes[1].barh(plot_item_names, plot_item_values, color=plot_item_colors)

    learned_note = ""

    if debug_info["learned"]:
        learned_note = f"、学習済み{len(debug_info['learned'])}件は候補外"

    axes[1].set_title(
        f"学習項目の価値\n赤＝選定、使用{total_hours}h／予算{T}h{learned_note}"
    )
    axes[1].set_xlabel("所属分野の有効相性の合計（＋既習シナジー）")
    axes[1].invert_yaxis()

    # --------------------------------------------------------
    # 3. アニーリング結果のエネルギー分布
    # --------------------------------------------------------

    # 得られた全サンプルのエネルギーを、実行可能解
    # （修復デコード後に予算内）と予算違反解に分けて重ねる。
    #
    # 前提科目は修復デコードが構造的に担保するため、
    # 前提違反でサンプルが捨てられることはない。
    # 採用解は修復後の真の目的関数値（価値＋シナジー）で選ぶため、
    # 必ずしもエネルギー最小のサンプルとは一致しない。

    feasible_energies = debug_info["feasible_energies"]
    infeasible_energies = debug_info["infeasible_energies"]

    all_energies = feasible_energies + infeasible_energies

    # 2つのヒストグラムを重ねるので、ビンの境界は共通にする
    bin_edges = np.histogram_bin_edges(all_energies, bins=60)

    axes[2].hist(
        infeasible_energies,
        bins=bin_edges,
        color="lightgray",
        label=f"予算違反 {len(infeasible_energies)}件",
    )

    axes[2].hist(
        feasible_energies,
        bins=bin_edges,
        color="steelblue",
        label=f"実行可能 {len(feasible_energies)}件",
    )

    axes[2].axvline(
        debug_info["energy"],
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"採用解 {debug_info['energy']:.2f}",
    )

    axes[2].set_title(
        f"アニーリング結果のエネルギー分布\n"
        f"{len(all_energies)}サンプル中 {len(feasible_energies)}件が実行可能"
    )
    axes[2].set_xlabel("QUBOエネルギー")
    axes[2].set_ylabel("サンプル数")

    # 採用解の線が左端に張り付くため、左右に少し余白を持たせる
    axes[2].margins(x=0.02)

    axes[2].legend()

    plt.tight_layout()
    plt.show()
