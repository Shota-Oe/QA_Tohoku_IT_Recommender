"""前提科目の定義。

and_prerequisites：
    子項目を選択する場合、リスト内の前提科目を「すべて」選択する必要がある。

    child <= parent

    例：
        Reactを選ぶ → JavaScript/TSも必要
        機械学習を選ぶ → Python基礎と統計・数学がともに必要

前提科目はAND（全親必須）の一種類だけであり、
「選択肢のうちいずれか1つ」を要求するOR前提は廃止した。

OR前提は「Gitを選ぶなら何か言語を1つ以上やっていること」のような
弱い依存を表していたが、
    - 必須性が低く、本来はシナジー（同時に選ぶと得）で表すべき関係だった
    - 条件付き価値の分母 min(m, 2) が近似でしかなく、探索の誘導が偏っていた
    - 予算枝刈り・修復デコードの分岐の大半がOR側の処理だった
ため、制約からは外している。廃止したOR関係は以下。

    SQL/データベース   ← バックエンド言語 / Python基礎
    アルゴリズム       ← バックエンド言語 / Python基礎 / C# / C++
    モバイルUI         ← Swift/Kotlin / Flutter/RN
    アプリ設計         ← Swift/Kotlin / Flutter/RN
    ゲーム数学・物理   ← C# / C++          （AND前提「アルゴリズム」は残る）
    3Dグラフィックス   ← C# / C++          （AND前提「ゲーム数学・物理」を後から追加）
    Git                ← JavaScript/TS / バックエンド言語 / Swift/Kotlin
                          / Flutter/RN / Python基礎
                                             （AND前提「開発環境/ターミナル」を後から追加）
    開発環境/ターミナル ← JavaScript/TS / バックエンド言語 / Swift/Kotlin
                          / Flutter/RN

3DグラフィックスとGitは、廃止したOR前提とは別の「親が一意に定まる依存」を
AND前提として持たせている。「言語のうちいずれか」はANDにできないが、
「ゲーム数学・物理」「開発環境/ターミナル」なら選択の余地がないためである。
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
    "テスト自動化": ["Python基礎", "テスト技法(JSTQB)"],

    # ------------------------------------------------------
    # 必須性の高い依存の追加：
    # 親が一意に定まる（＝ANDとして書ける）ものだけを入れる。
    # 「言語のうちいずれか」のように親が選択制の依存は
    # ANDにすると誤った制約になるため、ここには含めない。
    # ------------------------------------------------------
    "WordPress": ["HTML/CSS"],
    "SEO/サイト運営": ["HTML/CSS"],
    "画像編集": ["デザインツール(PS/AI)"],
    "3Dグラフィックス": ["ゲーム数学・物理"],
    "Git": ["開発環境/ターミナル"],
}


def validate_prerequisite_data():
    """
    前提科目に記載された項目がitems_data内に存在するか、
    また前提関係に循環がないかを確認する。
    """
    errors = []

    for child_name, parent_names in and_prerequisites.items():
        if child_name not in item_index:
            errors.append(f"[AND] 子項目が存在しません: {child_name}")

        for parent_name in parent_names:
            if parent_name not in item_index:
                errors.append(f"[AND] 前提項目が存在しません: {parent_name}")

    # 循環があると前提クロージャの計算が停止しない。
    # 項目名が未定義の場合はここまでにエラーが出ているので、
    # 存在する項目だけを辿って検査する。
    visiting = set()
    visited = set()

    def find_cycle(name):
        if name in visited:
            return

        if name in visiting:
            errors.append(f"[AND] 前提関係が循環しています: {name}")
            return

        visiting.add(name)

        for parent_name in and_prerequisites.get(name, []):
            if parent_name in item_index:
                find_cycle(parent_name)

        visiting.discard(name)
        visited.add(name)

    for child_name in and_prerequisites:
        if child_name in item_index:
            find_cycle(child_name)

    if errors:
        raise ValueError("\n".join(errors))
