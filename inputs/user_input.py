"""ユーザー入力。

適性5軸ベクトルと使用可能な学習時間をここで設定する。
フロントエンドはないので、このファイルを直接編集する。
"""

import numpy as np

from config.parameters import HIGH, LOW, MID  # noqa: F401  適性の設定に使う

# クリエイティブ:◎, ロジカル:◎, 作る運用:作る, 数学:○, 計画性正確さ:◎

user = np.array([
    HIGH,  # クリエイティブ
    HIGH,  # ロジカル
    HIGH,  # 作る↔運用
    MID,   # 数学
    HIGH,  # 計画性・正確さ
])

# 使用可能な総学習時間
hours_per_week = 30
weeks = 24

T = hours_per_week * weeks
