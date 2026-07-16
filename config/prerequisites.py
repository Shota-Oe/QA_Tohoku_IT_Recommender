"""前提科目の定義。

and_prerequisites：
    子項目を選択する場合、リスト内の前提科目を「すべて」選択する必要がある。

    child <= parent

    例：
        Reactを選ぶ → JavaScript/TSも必要
        機械学習を選ぶ → Python基礎と統計・数学がともに必要

or_prerequisites：
    子項目を選択する場合、リスト内の前提科目のうち
    「少なくとも1つ」を選択する必要がある。

    child <= sum(parents)

    例：
        モバイルUIを選ぶ → Swift/KotlinかFlutter/RNのどちらかが必要
        Gitを選ぶ → フロントエンド・バックエンド・モバイル・QAのいずれかの
                    言語を1つ以上選んでいる必要がある
                    （Gitは複数分野にまたがる汎用ツールのため、
                      特定の1分野の言語だけに縛らない）
"""

from config.items import item_index

and_prerequisites = {
    "JavaScript/TS": ["HTML/CSS"],
    "フロントFW(React等)": ["JavaScript/TS"],
    "API設計": ["バックエンド言語"],
    "機械学習": ["Python基礎", "統計・数学"],
    "データ可視化": ["Python基礎"],
    "アプリストア公開": ["アプリ設計"],
    "Unity": ["C#"],
    "Unreal Engine": ["C++"],
    "ゲーム数学・物理": ["アルゴリズム"],
    "クラウド(AWS等)": ["Linux基礎", "ネットワーク基礎"],
    "Docker": ["Linux基礎"],

    # ------------------------------------------------------
    # 言語必須ルール（AND）：
    # 選択肢が1つしかない分野は、
    # 既存のAND前提科目としてそのまま追加する。
    # ------------------------------------------------------
    "Firebase/Supabase": ["バックエンド言語"],
    "統計・数学": ["Python基礎"],
    "テスト技法(JSTQB)": ["Python基礎"],
    "テスト自動化": ["Python基礎"],
}

or_prerequisites = {
    "SQL/データベース": ["バックエンド言語", "Python基礎"],
    "アルゴリズム": ["バックエンド言語", "Python基礎", "C#", "C++"],
    "モバイルUI": ["Swift/Kotlin", "Flutter/RN"],
    "アプリ設計": ["Swift/Kotlin", "Flutter/RN"],
    "ゲーム数学・物理": ["C#", "C++"],
    "3Dグラフィックス": ["C#", "C++"],
    "Git": [
        "JavaScript/TS",
        "バックエンド言語",
        "Swift/Kotlin",
        "Flutter/RN",
        "Python基礎",
    ],
    "開発環境/ターミナル": [
        "JavaScript/TS",
        "バックエンド言語",
        "Swift/Kotlin",
        "Flutter/RN",
    ],
}


def validate_prerequisite_data():
    """
    前提科目（AND・OR両方）に記載された項目が
    items_data内に存在するか確認する。
    """
    errors = []

    for label, prerequisite_dict in (
        ("AND", and_prerequisites),
        ("OR", or_prerequisites),
    ):
        for child_name, parent_names in prerequisite_dict.items():
            if child_name not in item_index:
                errors.append(f"[{label}] 子項目が存在しません: {child_name}")

            for parent_name in parent_names:
                if parent_name not in item_index:
                    errors.append(
                        f"[{label}] 前提項目が存在しません: {parent_name}"
                    )

            # OR前提は「選択肢がゼロ」だと
            # 絶対に満たせない制約になってしまうため検査する
            if label == "OR" and len(parent_names) == 0:
                errors.append(f"[OR] 選択肢が0個です: {child_name}")

    if errors:
        raise ValueError("\n".join(errors))
