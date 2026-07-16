"""IT学習ナビゲーター（シナジー行列版）エントリーポイント。

実行方法：
    python main.py

構成：
    config/       係数・データ定義（適性軸、分野、学習項目、シナジー、前提科目）
    inputs/       ユーザー入力（適性ベクトルと学習時間）
    calculation/  QUBO構築・アニーリングなどの計算ロジック
    output/       テキスト表示とグラフ描画
"""

import sys

from calculation.recommend import recommend
from config.prerequisites import validate_prerequisite_data
from inputs.user_input import T, hours_per_week, user, weeks
from output.report import print_report
from output.visualize import plot_results


def main():
    # Windowsのコンソール（cp932）でも「↔」「✅」などを表示できるようにする
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")

    # 前提科目データの検査
    validate_prerequisite_data()

    # 推薦の実行
    z, field_score, debug_info = recommend(user=user, T=T)

    # 結果表示
    print_report(
        user=user,
        hours_per_week=hours_per_week,
        weeks=weeks,
        T=T,
        z=z,
        field_score=field_score,
        debug_info=debug_info,
    )

    # 可視化
    plot_results(
        z=z,
        T=T,
        field_score=field_score,
        debug_info=debug_info,
    )


if __name__ == "__main__":
    main()
