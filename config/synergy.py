"""項目間シナジーの定義。

item_synergy[i][j] は、項目iと項目jを"同時に"選んだときに
追加で発生する価値（正）または非効率のペナルティ（負）を表す。

item_value（単独の項目価値）には現れない、
"組み合わせて選んだ場合だけ"発生する効果をここに記述する。

値は仮の目安であり、根拠となるデータ（求人票の共起頻度など）が
得られたら差し替えることを想定している。
"""

import numpy as np

from config.items import M, item_index

synergy_pairs = {
    # --- 積み上げ型（前提科目に近いが、必須ではなく相乗効果） ---
    ("HTML/CSS", "デザインツール(Figma)"): 0.3,
    ("JavaScript/TS", "フロントFW(React等)"): 0.2,
    ("Python基礎", "統計・数学"): 0.4,
    ("Docker", "クラウド(AWS等)"): 0.3,
    ("SQL/データベース", "統計・数学"): 0.2,
    ("アルゴリズム", "統計・数学"): 0.2,

    # --- 分野をまたいだ組み合わせで市場価値が上がるペア ---
    ("UI/UXデザイン", "フロントFW(React等)"): 0.2,
    ("API設計", "Flutter/RN"): 0.2,
    ("テスト自動化", "Git"): 0.15,

    # --- 同時に学ぶと非効率になりやすいペア（負のシナジー） ---
    ("Unity", "Unreal Engine"): -0.5,
    ("Swift/Kotlin", "Flutter/RN"): -0.2,
}


def build_synergy_matrix():
    """
    synergy_pairsからM×Mの対称行列を作る。
    定義されていないペアの値は0とする。
    """
    synergy_matrix = np.zeros((M, M), dtype=float)

    for (name_a, name_b), value in synergy_pairs.items():
        if name_a not in item_index:
            raise ValueError(f"シナジー定義に存在しない項目: {name_a}")

        if name_b not in item_index:
            raise ValueError(f"シナジー定義に存在しない項目: {name_b}")

        i = item_index[name_a]
        j = item_index[name_b]

        # 対称行列として両方向に書き込む
        synergy_matrix[i, j] += value
        synergy_matrix[j, i] += value

    return synergy_matrix


item_synergy = build_synergy_matrix()
