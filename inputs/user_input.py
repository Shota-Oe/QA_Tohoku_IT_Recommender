"""ユーザー入力（適性5軸ベクトル・学習時間・学習済み科目・興味のある分野）。

適性値と学習時間は ``inputs/data/user_input.csv`` から読み込む。
条件を変えたいときは Python を編集せず CSV を編集すればよい。

CSV は ``key,value`` 形式：

    key,value
    クリエイティブ,1.0
    ロジカル,1.0
    作る↔運用,1.0
    数学,0.5
    計画性・正確さ,1.0
    hours_per_week,30
    weeks,24
    learned,"HTML/CSS, Python基礎"
    interests,"フロントエンド, ゲーム開発"

適性値は 0.0～1.0 の数値、または HIGH / MID / LOW で指定できる。
「#」で始まる行と空行は無視される。

``learned`` はすでに学習を終えた科目で、省略できる（キーなし・空値＝学習済みなし）。
カンマ区切りで並べ、CSVの仕様上ダブルクォートで囲む。
ここに挙げた項目は前処理で候補から外れる（README「学習済み科目の畳み込み」）。

``interests`` はアンケートで申告された興味のある分野で、同じく省略できる。
値は項目名ではなく分野名。**推薦計算には一切使わず、レポートに表示するだけ**である
（分野は適性5軸からの相性だけで決まる。README「興味のある分野」）。

環境変数 ``USER_INPUT_CSV`` で読み込む CSV のパスを差し替えられる。
"""

import csv
import os

import numpy as np

from config.fields import field_index
from config.items import item_index
from config.parameters import HIGH, LOW, MID, axis_names

FILE_NAME = "user_input_r5.csv"

# デフォルトの CSV パス（inputs/data/user_input.csv）
DEFAULT_CSV_PATH = os.path.join(
    os.path.dirname(__file__), "data", FILE_NAME
)

# 適性値のラベル → 数値
LABEL_TO_VALUE = {"HIGH": HIGH, "MID": MID, "LOW": LOW}

# 学習時間の必須キー
TIME_KEYS = ["hours_per_week", "weeks"]

# 学習済み科目のキー（省略可）
LEARNED_KEY = "learned"

# 興味のある分野のキー（省略可）
INTERESTS_KEY = "interests"


def _split_names(raw):
    """カンマ区切りの値を、記載順のままの名前リストへ変換する。

    前後の空白は落とし、空要素（末尾のカンマなど）は無視し、重複は1つに畳む。
    """
    names = []

    for part in raw.split(","):
        name = part.strip()

        if not name:
            continue

        if name in names:
            continue

        names.append(name)

    return names


def _parse_learned(raw, csv_path):
    """学習済み科目（カンマ区切り）を項目名のタプルへ変換する。

    表記ゆれを黙って捨てると「除外したつもりが除外されていない」事故になるため、
    config/items.py に無い項目名はエラーにする。
    """
    names = _split_names(raw)

    unknown = [name for name in names if name not in item_index]

    if unknown:
        raise ValueError(
            f"学習済み科目に未知の項目名があります: {unknown}（{csv_path}）\n"
            "config/items.py の項目名と一致させてください。"
        )

    return tuple(names)


def _parse_interests(raw, csv_path):
    """興味のある分野（カンマ区切り）を分野名のタプルへ変換する。

    表示専用の入力だが、綴りの違う分野名を黙って通すと
    「申告したのにレポートに出ない」ことになるため、learned と同じく
    config/fields.py に無い分野名はエラーにする。
    アンケート回答の表記ゆれ（例：「モバイルアプリ開発」）は
    CSV へ書き起こす時点で分野名へ直す。
    """
    names = _split_names(raw)

    unknown = [name for name in names if name not in field_index]

    if unknown:
        raise ValueError(
            f"興味のある分野に未知の分野名があります: {unknown}（{csv_path}）\n"
            "config/fields.py の分野名と一致させてください。"
        )

    return tuple(names)


def _parse_aptitude(raw):
    """適性値をパースする。数値または HIGH / MID / LOW を受け付ける。"""
    label = raw.strip().upper()

    if label in LABEL_TO_VALUE:
        return LABEL_TO_VALUE[label]

    try:
        return float(raw)
    except ValueError:
        raise ValueError(
            f"適性値は数値または HIGH / MID / LOW で指定してください: {raw!r}"
        )


def _read_key_values(csv_path):
    """CSV を key,value の辞書として読む。コメント行・空行・ヘッダーは無視する。"""
    values = {}

    with open(csv_path, encoding="utf-8") as f:
        for lineno, row in enumerate(csv.reader(f), start=1):
            # 空行を無視
            if not row:
                continue

            first = row[0].strip()

            # 空セル行・コメント行（先頭セルが # で始まる）を無視
            if first == "" or first.startswith("#"):
                continue

            # ヘッダー行（key,value）を無視
            if first == "key" and len(row) >= 2 and row[1].strip() == "value":
                continue

            if len(row) < 2:
                raise ValueError(
                    f"{csv_path} の {lineno} 行目は key,value 形式ではありません: {row}"
                )

            key = first

            if key in values:
                raise ValueError(f"CSV にキーが重複しています: {key}")

            values[key] = row[1].strip()

    return values


def load_user_input(csv_path=None):
    """CSV から適性ベクトルと学習時間を読み込む。

    Parameters
    ----------
    csv_path : str, optional
        読み込む CSV のパス。None の場合は環境変数 ``USER_INPUT_CSV``、
        なければ :data:`DEFAULT_CSV_PATH` を使う。

    Returns
    -------
    user : ndarray
        適性5軸ベクトル（``axis_names`` の順）。各値は 0.0～1.0。
    hours_per_week : int
        1週間あたりの学習時間。
    weeks : int
        学習週数。
    T : int
        総学習時間（hours_per_week × weeks）。
    learned : tuple of str
        学習済み科目の項目名（CSVに書かれた順）。指定がなければ空タプル。
    interests : tuple of str
        興味のある分野の分野名（CSVに書かれた順）。指定がなければ空タプル。
        表示専用で、推薦計算には使わない。
    """
    if csv_path is None:
        csv_path = os.environ.get("USER_INPUT_CSV", DEFAULT_CSV_PATH)

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"ユーザー入力 CSV が見つかりません: {csv_path}")

    values = _read_key_values(csv_path)

    # 必須キーの確認
    required = list(axis_names) + TIME_KEYS

    missing = [key for key in required if key not in values]
    if missing:
        raise ValueError(f"CSV に必要なキーがありません: {missing}（{csv_path}）")

    # learned と interests は省略可能なキー
    optional = [LEARNED_KEY, INTERESTS_KEY]

    unknown = [
        key for key in values if key not in required and key not in optional
    ]
    if unknown:
        raise ValueError(f"CSV に未知のキーがあります: {unknown}（{csv_path}）")

    # 適性ベクトル（axis_names の順に並べる）
    user = np.array(
        [_parse_aptitude(values[axis]) for axis in axis_names],
        dtype=float,
    )

    if np.any(user < 0.0) or np.any(user > 1.0):
        raise ValueError("適性値は 0.0～1.0 にしてください。")

    # 学習時間
    try:
        hours_per_week = int(values["hours_per_week"])
        weeks = int(values["weeks"])
    except ValueError:
        raise ValueError("hours_per_week と weeks は整数で指定してください。")

    if hours_per_week <= 0 or weeks <= 0:
        raise ValueError("hours_per_week と weeks は正の整数にしてください。")

    T = hours_per_week * weeks

    # 学習済み科目（省略可）
    learned = _parse_learned(values.get(LEARNED_KEY, ""), csv_path)

    # 興味のある分野（省略可・表示専用）
    interests = _parse_interests(values.get(INTERESTS_KEY, ""), csv_path)

    return user, hours_per_week, weeks, T, learned, interests


# モジュール読み込み時にデフォルト CSV から値を読み込む。
# main.py は user, hours_per_week, weeks, T, learned, interests をそのまま利用する。
user, hours_per_week, weeks, T, learned, interests = load_user_input()
