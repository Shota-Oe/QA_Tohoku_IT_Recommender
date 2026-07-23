"""項目間シナジーの定義。

item_synergy[i][j] は、項目iと項目jを"同時に"選んだときに
追加で発生する価値（正）または非効率のペナルティ（負）を表す。

item_value（単独の項目価値）には現れない、
"組み合わせて選んだ場合だけ"発生する効果をここに記述する。

値は仮の目安であり、根拠となるデータ（求人票の共起頻度など）が
得られたら差し替えることを想定している（issue #12）。

前提科目との関係
----------------
前提クロージャに含まれるペアはここに書いてはならない。
子cの前提に親pがあるなら z_c = 1 ⇒ z_p = 1 が制約で保証されるので、
二次項 S_cp·z_c·z_p は z_c と同時にしか立たず、
「組み合わせ効果」ではなく子の価値への定数上乗せに退化する。
これは線形項で書くべきものであり、二次項の意味を薄めるだけである。
"""

import numpy as np

from config.items import M, item_index
from config.prerequisites import and_prerequisites

synergy_pairs = {
    # ========================================================
    # 積み上げ型（前提科目に近いが、必須ではなく相乗効果）
    # ========================================================

    # --- Webサイト管理 ---
    ("WordPress", "ドメイン/サーバー設定"): 0.30,
    ("WordPress", "SEO/サイト運営"): 0.25,

    # --- グラフィックデザイン ---
    ("デザイン基礎", "デザインツール(PS/AI)"): 0.50,

    # --- UI/UX・フロントエンド ---
    ("HTML/CSS", "デザインツール(Figma)"): 0.50,

    # --- バックエンド ---
    ("バックエンド向け言語", "SQL/データベース"): 0.70,
    ("API設計", "SQL/データベース"): 0.70,

    # --- データ/ML ---
    ("SQL/データベース", "統計・数学"): 0.50,
    ("アルゴリズム", "統計・数学"): 0.70,
    ("機械学習", "データ可視化"): 0.50,
    ("統計・数学", "データ可視化"): 0.30,
    ("SQL/データベース", "データ可視化"): 0.40,
    ("Python基礎", "SQL/データベース"): 0.30,

    # --- モバイルアプリ ---
    ("Swift/Kotlin", "モバイルUI"): 0.50,
    ("Flutter/RN", "モバイルUI"): 0.50,
    ("アプリストア公開", "モバイルUI"): 0.30,

    # --- ゲーム開発 ---
    ("C#", "アルゴリズム"): 0.25,
    ("C++", "アルゴリズム"): 0.40,
    ("Unity", "3Dグラフィックス"): 0.20,
    ("Unreal Engine", "3Dグラフィックス"): 0.20,
    ("ゲーム数学・物理", "Unity"): 0.40,
    ("ゲーム数学・物理", "Unreal Engine"): 0.40,

    # --- インフラ/クラウド ---
    ("Docker", "クラウド(AWS等)"): 0.30,
    ("Linux基礎", "開発環境/ターミナル"): 0.25,
    ("Linux基礎", "ネットワーク基礎"): 0.40,

    # ========================================================
    # 分野をまたいだ組み合わせで市場価値が上がるペア
    #
    # 単独の項目価値は所属分野の相性だけで決まるため、
    # 「分野をまたいで初めて価値が出る」関係は線形項では表せない。
    # 二次項が最も効くのはこの層である。
    # ========================================================

    # --- デザイン → 実装 ---
    ("UI/UXデザイン", "フロントFW(React等)"): 0.40,
    ("デザイン基礎", "UI/UXデザイン"): 0.6,
    ("デザイン基礎", "デザインツール(Figma)"): 0.60,
    ("画像編集", "HTML/CSS"): 0.10,
    ("デザインツール(PS/AI)", "3Dグラフィックス"): 0.40,
    ("3Dグラフィックス", "デザイン基礎"): 0.40,
    ("モバイルUI", "UI/UXデザイン"): 0.60,
    ("モバイルUI", "デザインツール(Figma)"): 0.60,

    # --- フロントエンド ↔ バックエンド ---
    ("API設計", "フロントFW(React等)"): 0.25,
    ("JavaScript/TS", "Firebase/Supabase"): 0.20,

    # --- モバイルアプリ ↔ バックエンド ---
    ("API設計", "Flutter/RN"): 0.10,
    ("Firebase/Supabase", "Flutter/RN"): 0.30,
    ("Firebase/Supabase", "モバイルUI"): 0.10,

    # --- 開発 ↔ インフラ/運用 ---
    ("バックエンド向け言語", "Docker"): 0.40,
    ("Python基礎", "Docker"): 0.20,
    ("機械学習", "クラウド(AWS等)"): 0.40,
    ("ドメイン/サーバー設定", "ネットワーク基礎"): 0.50,
    ("開発環境/ターミナル", "Docker"): 0.25,

    # --- QA・テスト ---
    ("テスト自動化", "Git"): 0.10,
    ("Docker", "テスト自動化"): 0.60,
    ("JavaScript/TS", "テスト自動化"): 0.50,
    ("テスト技法(JSTQB)", "API設計"): 0.20,
    ("テスト技法(JSTQB)", "Git"): 0.20,
    ("Git", "Docker"): 0.30,

    # --- 開発プロセス・公開 ---
    ("アプリストア公開", "SEO/サイト運営"): 0.15,
    ("SEO/サイト運営", "データ可視化"): 0.15,
    ("ゲーム数学・物理", "統計・数学"): 0.40,
    ("画像編集", "WordPress"): 0.25,
    ("画像編集", "SEO/サイト運営"): 0.20,

    # ========================================================
    # 同時に学ぶと非効率になりやすいペア（負のシナジー）
    #
    # 代替・競合関係にある技術を両方選んでも市場価値は足し算にならず、
    # 学習時間だけを二重に消費する。
    # 項目を独立に評価する線形モデルでは原理的に表現できない層であり、
    # 二次項を持つQUBOを使う直接の動機になる。
    #
    # ここに書くのは「予算が潤沢でも両方やるのは損」という関係だけに限る。
    # 「時間が足りないから両方は選べない」は予算制約が既に表しているので、
    # それを負の項でも表すと二重計上になる
    # （例：JavaScript/TS × Swift/Kotlin のような単なる言語の並行学習）。
    #
    # なお6節のペナルティ係数 A は正のシナジーしか参照しないため、
    # 負の項を増やしてもAと係数のダイナミックレンジは変化しない。
    # ========================================================

    # --- ゲームエンジン・言語の二者択一 ---
    ("Unity", "Unreal Engine"): -0.70,
    ("C#", "C++"): -0.30,
    ("Unity", "C++"): -0.20,
    ("Unreal Engine", "C#"): -0.20,

    # --- ゲームエンジンとアプリ開発の路線違い ---
    ("Unity", "Flutter/RN"): -0.15,
    ("Unity", "Swift/Kotlin"): -0.15,

    # --- モバイル開発の二者択一 ---
    ("Swift/Kotlin", "Flutter/RN"): -0.20,

    # --- 言語の二重取得 ---
    # 「バックエンド向け言語」は総称項目なので、
    # 既にサーバ側を書ける言語を学ぶ人には重複投資になる。
    ("JavaScript/TS", "バックエンド向け言語"): -0.25,
    ("C#", "バックエンド向け言語"): -0.25,
    ("C++", "バックエンド向け言語"): -0.15,

    # --- マネージドサービスと自前運用の二者択一 ---
    ("Firebase/Supabase", "クラウド(AWS等)"): -0.30,
    ("Firebase/Supabase", "Docker"): -0.25,
    ("クラウド(AWS等)", "ドメイン/サーバー設定"): -0.20,
    ("Docker", "ドメイン/サーバー設定"): -0.15,

    # --- デザインツールの習熟先の重複 ---
    ("デザインツール(Figma)", "デザインツール(PS/AI)"): -0.4,
    ("デザインツール(PS/AI)", "UI/UXデザイン"): -0.25,
    ("デザインツール(PS/AI)", "モバイルUI"): -0.20,

    # --- CMS運用と自前実装の路線違い ---
    ("WordPress", "フロントFW(React等)"): -0.20,
    ("WordPress", "バックエンド向け言語"): -0.15,
}


def prerequisite_closure(name, seen=None):
    """nameの前提クロージャ（自身は含まない）を返す。"""
    if seen is None:
        seen = set()

    for parent_name in and_prerequisites.get(name, []):
        if parent_name not in seen:
            seen.add(parent_name)
            prerequisite_closure(parent_name, seen)

    return seen


def validate_synergy_data():
    """シナジー定義の検査。

    - 項目名がitems_dataに存在すること
    - 同じペアが順序違いで二重定義されていないこと
      （対称行列へ2回加算され、意図の倍の値になる）
    - 前提クロージャに含まれる退化ペアでないこと
      （モジュール冒頭の「前提科目との関係」を参照）
    """
    errors = []
    seen = set()

    for name_a, name_b in synergy_pairs:
        if name_a not in item_index:
            errors.append(f"シナジー定義に存在しない項目: {name_a}")

        if name_b not in item_index:
            errors.append(f"シナジー定義に存在しない項目: {name_b}")

        # 以降の検査は両方の項目が実在することを前提とする
        if name_a not in item_index or name_b not in item_index:
            continue

        if name_a == name_b:
            errors.append(f"同一項目のペアは定義できません: {name_a}")
            continue

        key = frozenset((name_a, name_b))

        if key in seen:
            errors.append(f"ペアが二重に定義されています: {name_a} × {name_b}")

        seen.add(key)

        if name_a in prerequisite_closure(name_b):
            errors.append(
                f"退化ペア（{name_b} の前提に {name_a}）: {name_a} × {name_b}"
                "／前提が保証する関係は二次項ではなく項目価値で表してください。"
            )
        elif name_b in prerequisite_closure(name_a):
            errors.append(
                f"退化ペア（{name_a} の前提に {name_b}）: {name_a} × {name_b}"
                "／前提が保証する関係は二次項ではなく項目価値で表してください。"
            )

    if errors:
        raise ValueError("\n".join(errors))


def build_synergy_matrix():
    """
    synergy_pairsからM×Mの対称行列を作る。
    定義されていないペアの値は0とする。
    """
    validate_synergy_data()

    synergy_matrix = np.zeros((M, M), dtype=float)

    for (name_a, name_b), value in synergy_pairs.items():
        i = item_index[name_a]
        j = item_index[name_b]

        # 対称行列として両方向に書き込む
        synergy_matrix[i, j] += value
        synergy_matrix[j, i] += value

    return synergy_matrix


item_synergy = build_synergy_matrix()
