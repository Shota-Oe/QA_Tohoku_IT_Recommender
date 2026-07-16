"""IT分野の5軸プロファイル。

各分野を [クリエイティブ, ロジカル, 作る↔運用, 数学, 計画性・正確さ] の
5軸ベクトルで表す。
"""

import numpy as np

from config.parameters import HIGH, LOW, MID

field_profile = {
    "UI/UXデザイン":        [HIGH, MID, 1.0, LOW, MID],
    "グラフィックデザイン":  [HIGH, LOW, 1.0, LOW, LOW],
    "Webサイト管理":         [MID, LOW, 0.0, LOW, HIGH],
    "フロントエンド":        [MID, MID, 1.0, LOW, MID],
    "バックエンド":          [LOW, HIGH, 1.0, MID, HIGH],
    "モバイルアプリ":        [MID, MID, 1.0, LOW, MID],
    "ゲーム開発":            [HIGH, MID, 1.0, HIGH, LOW],
    "データ/ML":             [LOW, HIGH, 0.5, HIGH, MID],
    "インフラ/クラウド":     [LOW, MID, 0.0, MID, HIGH],
    "QA・テスト":            [LOW, MID, 0.0, LOW, HIGH],
}

field_names = list(field_profile.keys())

# 分野×5軸の行列
F = np.array([field_profile[field] for field in field_names], dtype=float)

field_index = {field: i for i, field in enumerate(field_names)}
