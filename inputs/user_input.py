"""ユーザー入力（適性5軸ベクトルと学習時間）。

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

適性値は 0.0～1.0 の数値、または HIGH / MID / LOW で指定できる。
「#」で始まる行と空行は無視される。

環境変数 ``USER_INPUT_CSV`` で読み込む CSV のパスを差し替えられる。
"""

import csv
import os

import numpy as np

from config.parameters import HIGH, LOW, MID, axis_names

# デフォルトの CSV パス（inputs/data/user_input.csv）
DEFAULT_CSV_PATH = os.path.join(
    os.path.dirname(__file__), "data", "user_input.csv"
)

# 適性値のラベル → 数値
LABEL_TO_VALUE = {"HIGH": HIGH, "MID": MID, "LOW": LOW}

# 学習時間の必須キー
TIME_KEYS = ["hours_per_week", "weeks"]


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

    unknown = [key for key in values if key not in required]
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

    return user, hours_per_week, weeks, T


# モジュール読み込み時にデフォルト CSV から値を読み込む。
# main.py は user, hours_per_week, weeks, T をそのまま利用する。
user, hours_per_week, weeks, T = load_user_input()
