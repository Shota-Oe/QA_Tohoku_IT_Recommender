"""複数のユーザー入力を一括でアニーリングし、結果をファイルへ保存するバッチランナー。

``inputs/data/user_input_N.csv`` を順に読み込み、各件について
``output/result/N/`` に次を書き出す。

    report.txt   … print_report の全文（推薦・QPU情報・目的関数値など）
    graph.png    … 分野相性・項目価値の2枚グラフ
    error.txt    … その件が例外で落ちたときだけ（トレースバック）

さらに全件を横断する比較表を ``output/result/summary.md`` /
``output/result/summary.csv`` に書き出す。

本体（main.py）とは独立したスタンドアロンスクリプト。既定はローカルの
neal（課金なし）で、実機QPUで回すときだけ ``--backend qpu`` を明示する。

実行方法（リポジトリルートから）:
    python tools/run_all_inputs.py                       # neal で 1〜11
    python tools/run_all_inputs.py --backend qpu         # 実機QPU で 1〜11
    python tools/run_all_inputs.py --backend neal --inputs 1   # neal で1件だけ
"""

import argparse
import csv
import io
import os
import re
import sys
import traceback
from contextlib import redirect_stdout
from pathlib import Path

# スタンドアロン実行用：リポジトリルートを import パスに追加する
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 日本語まじりの進捗表示が Windows コンソール（cp932）で化けないように。
for _stream in (sys.stdout, sys.stderr):
    if _stream.encoding and _stream.encoding.lower() != "utf-8":
        _stream.reconfigure(encoding="utf-8")

# グラフはウィンドウ表示せずファイルへ保存する。
# output.visualize が matplotlib.pyplot を import する前に Agg を選ぶ必要がある。
import matplotlib  # noqa: E402

matplotlib.use("Agg")

# inputs.user_input は import 時に既定CSV（user_input.csv）を1回読み込む
# モジュール副作用がある。既定CSVが無い環境でも import を通すため、実在する
# 入力CSVを USER_INPUT_CSV に指定しておく。本ランナーの各件ロードは明示パスで
# 行うので、ここで指す既定値は実際の処理には使われない。
_default_csv = ROOT / "inputs" / "data" / "user_input.csv"
if "USER_INPUT_CSV" not in os.environ and not _default_csv.exists():
    for _n in range(1, 12):
        _candidate = ROOT / "inputs" / "data" / f"user_input_{_n}.csv"
        if _candidate.exists():
            os.environ["USER_INPUT_CSV"] = str(_candidate)
            break

import numpy as np  # noqa: E402

from calculation.sampler import build_sampler, default_num_reads  # noqa: E402
from calculation.recommend import recommend  # noqa: E402
from config.items import hours, item_names  # noqa: E402
from config.prerequisites import validate_prerequisite_data  # noqa: E402
from inputs.user_input import load_user_input  # noqa: E402
from output.report import print_report  # noqa: E402
from output.visualize import plot_results  # noqa: E402

DATA_DIR = ROOT / "inputs" / "data"
RESULT_DIR = ROOT / "output" / "result"

# 出力ファイルは実験データとして公開する前提のため、個人を特定できる情報
# （アンケート回答者のメールアドレス・回答日時、実行環境の絶対パス）が
# 書き出されないようにする。
EMAIL_PATTERN = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
TIMESTAMP_PATTERN = re.compile(
    r"\d{4}[/-]\d{1,2}[/-]\d{1,2}(\s+\d{1,2}:\d{2}(:\d{2})?)?"
)

# 絶対パスに含まれる OS ユーザー名を隠すための置換先
ROOT_PLACEHOLDER = "<repo>"

# summary の列（キー → 見出し）。順序を保つため list of (key, header)。
SUMMARY_COLUMNS = [
    ("input", "入力"),
    ("source", "回答者"),
    ("status", "状態"),
    ("budget_hours", "予算T(h)"),
    ("hours_per_week", "週h"),
    ("weeks", "週数"),
    ("num_items", "推薦数"),
    ("used_hours", "使用(h)"),
    ("objective", "目的関数値"),
    ("energy", "QUBOエネルギー"),
    ("feasible_count", "実行可能数"),
    ("qpu_access_time_us", "QPU時間(µs)"),
    ("chain_break_mean", "鎖切れ平均"),
    ("problem_id", "problem_id"),
    ("recommended_items", "推薦項目"),
]


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="user_input_N.csv を一括でアニーリングし結果を保存する。"
    )
    parser.add_argument(
        "--backend",
        choices=("neal", "qpu"),
        default="neal",
        help="アニーリングの実行基盤（既定：neal。実機は qpu）",
    )
    parser.add_argument(
        "--inputs",
        type=int,
        nargs="+",
        default=list(range(1, 12)),
        help="対象の入力番号（既定：1〜11）",
    )
    parser.add_argument(
        "--num-reads",
        type=int,
        default=None,
        help="アニーリングの実行回数（既定：backend ごとの既定値）",
    )
    parser.add_argument(
        "--annealing-time",
        type=float,
        default=None,
        help="QPUの1回当たりのアニール時間（マイクロ秒。既定：config の既定値）",
    )
    return parser.parse_args()


def scrub_root(text):
    """文字列中のリポジトリ絶対パスを ``<repo>`` に置き換える。

    トレースバックやエラーメッセージをそのまま書き出すと
    ``C:\\Users\\<ユーザー名>\\...`` の形で実行環境のユーザー名が残るため。
    """
    return str(text).replace(str(ROOT), ROOT_PLACEHOLDER)


def strip_identifiers(text):
    """回答者を特定できる情報（メールアドレス・日時）を取り除く。

    残った区切り文字（``/``）や空白だけになった場合は空文字を返す。
    """
    text = EMAIL_PATTERN.sub("", text)
    text = TIMESTAMP_PATTERN.sub("", text)

    return text.strip(" \t/-").strip()


def read_source_comment(csv_path):
    """CSV 冒頭の ``#`` コメントから回答者を表す文字列を取り出す。

    ``# 元回答: <日時> / <メール>`` 行の中身を返すが、メールアドレスと日時は
    :func:`strip_identifiers` で落とす（summary は公開する前提のため）。
    そもそも自由記述のコメントは回答者名などが紛れ込む経路になるので、
    ``元回答:`` 以外のコメント行は読まず、該当行が無ければ空文字を返す。
    """
    with open(csv_path, encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()

            if stripped.startswith("#"):
                text = stripped.lstrip("#").strip()

                if text.startswith("元回答:"):
                    return strip_identifiers(text[len("元回答:") :])

                continue

            if stripped:
                # コメント行が続く間だけ走査し、中身が来たら打ち切る
                break

    return ""


def run_one(n, sampler, sampler_kwargs, num_reads, backend_description):
    """入力 n を1件処理し、summary 用の1行（dict）を返す。"""
    csv_path = DATA_DIR / f"user_input_{n}.csv"
    outdir = RESULT_DIR / str(n)
    os.makedirs(outdir, exist_ok=True)

    source = read_source_comment(csv_path) if csv_path.exists() else ""

    row = {key: "" for key, _ in SUMMARY_COLUMNS}
    row["input"] = n
    row["source"] = source

    if not csv_path.exists():
        row["status"] = "missing"
        (outdir / "error.txt").write_text(
            f"入力CSVが見つかりません: {scrub_root(csv_path)}\n", encoding="utf-8"
        )
        print(f"[{n}] CSVが見つかりません: {csv_path}")
        return row

    try:
        user, hours_per_week, weeks, T, learned, interests = load_user_input(
            str(csv_path)
        )

        z, field_score, debug_info = recommend(
            user=user,
            T=T,
            learned=learned,
            num_reads=num_reads,
            sampler=sampler,
            sampler_kwargs=sampler_kwargs,
        )

        # report.txt（print_report の標準出力を丸ごと捕捉）
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            print(
                f"アニーリング実行基盤：{backend_description}"
                f"（num_reads={num_reads}）"
            )
            print_report(
                user=user,
                hours_per_week=hours_per_week,
                weeks=weeks,
                T=T,
                z=z,
                field_score=field_score,
                debug_info=debug_info,
                interests=interests,
                backend_description=backend_description,
            )
        (outdir / "report.txt").write_text(buffer.getvalue(), encoding="utf-8")

        # graph.png
        plot_results(
            z=z,
            T=T,
            field_score=field_score,
            debug_info=debug_info,
            save_path=str(outdir / "graph.png"),
        )

        # summary 用の値
        selected_names = [item_names[j] for j in range(len(z)) if z[j] == 1]
        solver_info = debug_info.get("solver_info", {})

        row["status"] = "ok"
        row["budget_hours"] = T
        row["hours_per_week"] = hours_per_week
        row["weeks"] = weeks
        row["num_items"] = int(np.count_nonzero(z))
        row["used_hours"] = int(hours @ z)
        row["objective"] = round(float(debug_info["best_score"]), 3)
        row["energy"] = round(float(debug_info["energy"]), 3)
        row["feasible_count"] = int(debug_info["feasible_count"])
        row["recommended_items"] = "; ".join(selected_names)

        if "qpu_access_time" in solver_info:
            row["qpu_access_time_us"] = round(
                float(solver_info["qpu_access_time"]), 1
            )
        if "chain_break_mean" in solver_info:
            row["chain_break_mean"] = round(
                float(solver_info["chain_break_mean"]), 4
            )
        if "problem_id" in solver_info:
            row["problem_id"] = solver_info["problem_id"]

        print(
            f"[{n}] OK  推薦{row['num_items']}件 / 使用{row['used_hours']}h "
            f"/ 予算{T}h / 目的値{row['objective']}"
        )

    except Exception:
        row["status"] = "error"
        tb = scrub_root(traceback.format_exc())
        (outdir / "error.txt").write_text(tb, encoding="utf-8")
        # 前回成功時の graph.png が残ると誤解を招くので消しておく
        stale_graph = outdir / "graph.png"
        if stale_graph.exists():
            stale_graph.unlink()
        print(f"[{n}] ERROR: {tb.strip().splitlines()[-1]}")

    return row


def write_summary(rows, backend_description):
    """summary.md と summary.csv を書き出す。"""
    os.makedirs(RESULT_DIR, exist_ok=True)
    headers = [header for _, header in SUMMARY_COLUMNS]
    keys = [key for key, _ in SUMMARY_COLUMNS]

    # summary.csv
    with open(
        RESULT_DIR / "summary.csv", "w", encoding="utf-8-sig", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        for row in rows:
            writer.writerow([row.get(key, "") for key in keys])

    # summary.md
    lines = []
    lines.append("# 一括アニーリング結果サマリ")
    lines.append("")
    lines.append(f"- 実行基盤：{backend_description}")
    lines.append(f"- 件数：{len(rows)}")
    ok = sum(1 for r in rows if r.get("status") == "ok")
    lines.append(f"- 成功：{ok} / {len(rows)}")
    lines.append("")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in rows:
        cells = [str(row.get(key, "")) for key in keys]
        # markdown の表を壊さないようパイプをエスケープ
        cells = [c.replace("|", "\\|") for c in cells]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    (RESULT_DIR / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    args = parse_arguments()

    validate_prerequisite_data()

    # サンプラーは1回だけ生成して全件で使い回す（QPUなら認証・接続も1回）。
    build_kwargs = {"backend": args.backend}
    if args.annealing_time is not None:
        build_kwargs["annealing_time"] = args.annealing_time

    sampler, sampler_kwargs, backend_description = build_sampler(**build_kwargs)

    num_reads = args.num_reads
    if num_reads is None:
        num_reads = default_num_reads(args.backend)

    print(
        f"アニーリング実行基盤：{backend_description}"
        f"（num_reads={num_reads}） 対象入力：{args.inputs}"
    )
    print("=" * 60)

    rows = []
    for n in args.inputs:
        rows.append(
            run_one(n, sampler, sampler_kwargs, num_reads, backend_description)
        )

    write_summary(rows, backend_description)

    print("=" * 60)
    ok = sum(1 for r in rows if r.get("status") == "ok")
    print(f"完了：{ok}/{len(rows)} 件成功。結果は {RESULT_DIR} 以下。")


if __name__ == "__main__":
    main()
