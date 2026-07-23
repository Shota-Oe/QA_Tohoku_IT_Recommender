"""IT学習ナビゲーター（シナジー行列版）エントリーポイント。

実行方法：
    python main.py                  ローカルのシミュレーテッド・アニーリング（neal）
    python main.py --backend qpu    D-Wave の実機 QPU（Leap 経由）

QPU を使うには、プロジェクト直下の .env に
DWAVE_API_TOKEN=DEV-xxxx を設定しておく。

構成：
    config/       係数・データ定義（適性軸、分野、学習項目、シナジー、前提科目）
    inputs/       ユーザー入力（適性ベクトルと学習時間）
    calculation/  QUBO構築・アニーリングなどの計算ロジック
    output/       テキスト表示とグラフ描画
"""

import argparse
import sys

from calculation.recommend import recommend
from calculation.sampler import BACKENDS, build_sampler, default_num_reads
from config.parameters import DEFAULT_ANNEALING_TIME, DEFAULT_QPU_SOLVER
from config.prerequisites import validate_prerequisite_data
from inputs.user_input import T, hours_per_week, user, weeks
from output.report import print_report
from output.visualize import plot_results


def use_utf8_stdout():
    """Windowsのコンソール（cp932）でも「↔」「✅」などを表示できるようにする。

    argparseのヘルプも日本語を含むため、引数解釈より先に呼ぶ。
    """
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")


def parse_arguments():
    """コマンドライン引数を解釈する。"""
    parser = argparse.ArgumentParser(
        description="適性と学習時間からIT学習項目を推薦する。"
    )

    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        default="neal",
        help="アニーリングの実行基盤（既定：neal）",
    )

    parser.add_argument(
        "--num-reads",
        type=int,
        default=None,
        help="アニーリングの実行回数（既定：neal 5000／qpu 1000）",
    )

    parser.add_argument(
        "--solver",
        default=DEFAULT_QPU_SOLVER,
        help="QPUの機種名（例：Advantage_system6。既定：Leapの自動選択）",
    )

    parser.add_argument(
        "--annealing-time",
        type=float,
        default=DEFAULT_ANNEALING_TIME,
        help=f"QPUの1回当たりのアニール時間（マイクロ秒。既定：{DEFAULT_ANNEALING_TIME}）",
    )

    parser.add_argument(
        "--chain-strength",
        type=float,
        default=None,
        help="QPUのマイナー埋め込みの鎖の強さ（既定：自動決定）",
    )

    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="グラフを表示せずテキスト結果だけを出力する",
    )

    return parser.parse_args()


def main():
    use_utf8_stdout()

    args = parse_arguments()

    # 前提科目データの検査
    validate_prerequisite_data()

    # サンプラーの準備（--backend qpu なら実機QPUへ接続する）
    sampler, sampler_kwargs, backend_description = build_sampler(
        backend=args.backend,
        solver=args.solver,
        annealing_time=args.annealing_time,
        chain_strength=args.chain_strength,
    )

    num_reads = args.num_reads

    if num_reads is None:
        num_reads = default_num_reads(args.backend)

    print(f"アニーリング実行基盤：{backend_description}（num_reads={num_reads}）")

    # 推薦の実行
    z, field_score, debug_info = recommend(
        user=user,
        T=T,
        num_reads=num_reads,
        sampler=sampler,
        sampler_kwargs=sampler_kwargs,
    )

    # 結果表示
    print_report(
        user=user,
        hours_per_week=hours_per_week,
        weeks=weeks,
        T=T,
        z=z,
        field_score=field_score,
        debug_info=debug_info,
        backend_description=backend_description,
    )

    # 可視化
    if not args.no_plot:
        plot_results(
            z=z,
            T=T,
            field_score=field_score,
            debug_info=debug_info,
        )


if __name__ == "__main__":
    main()
