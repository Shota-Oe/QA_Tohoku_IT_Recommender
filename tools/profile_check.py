"""分野プロファイルの健全性チェック（検証専用ツール）。

推薦パイプライン本体（calculation/ のQUBO構築・アニーリング）とは
独立したスタンドアロンスクリプトであり、本体側のコードは
このファイルを一切参照しない（依存は tools → config/inputs の一方向のみ）。

config/fields.py の field_profile が満たすべき性質を、
5軸の回答をすべて列挙した仮想ユーザー集合に対して測る。

  1. 不変条件    ： 各軸が0.00と1.00を持つか／同一プロファイルの分野が無いか
  2. 構造バイアス： 分野ごとの平均相性のばらつき（小さいほど公平）
  3. 閾値の感度  ： min_relevanceごとの採用分野数と候補0件パターン数
  4. 実回答      ： inputs/data/user_input*.csv の分野相性と採用分野

アンケートの適性は◎○×の3択なので、ありうる回答は 3^5 = 243 パターンに限られる。
この全数に対して統計を取るため、結果はサンプリング誤差を含まない。

実行方法（リポジトリルートから）:
    python tools/profile_check.py
"""

import itertools
import sys
from pathlib import Path

# スタンドアロン実行用：リポジトリルートをimportパスに追加する
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

from config.fields import F, field_max_distance, field_names
from config.parameters import DEFAULT_MIN_RELEVANCE, axis_names
from inputs.user_input import load_user_input

DATA_DIR = ROOT / "inputs" / "data"

# アンケートの◎○×に対応する適性値
ANSWER_VALUES = [0.0, 0.5, 1.0]

THRESHOLDS = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]


def field_scores(users):
    """ユーザー行列（n×5）に対する分野相性（n×分野数）を返す。

    calculation/recommend.py と同じ式。本体をimportしていないのは、
    このツールが分野プロファイル単体の性質を見るためのものだからである。
    """
    squared = ((users[:, None, :] - F[None, :, :]) ** 2).mean(axis=2)

    return np.clip(1.0 - squared / field_max_distance, 0.0, 1.0)


def raw_field_scores(users):
    """正規化前の相性（比較用）。"""
    squared = ((users[:, None, :] - F[None, :, :]) ** 2).mean(axis=2)

    return np.clip(1.0 - squared, 0.0, 1.0)


def print_invariants():
    print("=" * 78)
    print("1. 不変条件")
    print("=" * 78)

    for a, axis in enumerate(axis_names):
        low = [field_names[f] for f in np.flatnonzero(F[:, a] == 0.0)]
        high = [field_names[f] for f in np.flatnonzero(F[:, a] == 1.0)]
        mark = "OK" if low and high else "NG"
        print(f"  {mark} {axis:<16} 0.00={low or 'なし'} / 1.00={high or 'なし'}")

    print("     ※ 0.00と1.00の分野が両方あって初めて、その軸の◎と×が区別に効く")

    pairwise = ((F[:, None, :] - F[None, :, :]) ** 2).mean(axis=2)
    np.fill_diagonal(pairwise, np.inf)
    i, j = np.unravel_index(pairwise.argmin(), pairwise.shape)
    mark = "OK" if pairwise.min() > 0.0 else "NG"
    print(
        f"  {mark} 最も近い分野の組: {field_names[i]} と {field_names[j]}"
        f"（平均二乗距離 {pairwise.min():.4f}）"
    )


def print_bias(grid, scores):
    print()
    print("=" * 78)
    print("2. 構造バイアス（243パターンの仮想ユーザー全体に対する統計）")
    print("=" * 78)

    raw = raw_field_scores(grid)
    top1 = np.bincount(scores.argmax(axis=1), minlength=len(field_names))

    header = f"  {'分野':<12}{'平均':>8}{'標準偏差':>10}{'最小':>8}{'最大':>8}{'1位回数':>9}"
    print(header)

    for f, name in enumerate(field_names):
        print(
            f"  {name:<12}{scores[:, f].mean():>8.3f}{scores[:, f].std():>10.3f}"
            f"{scores[:, f].min():>8.3f}{scores[:, f].max():>8.3f}{top1[f]:>9d}"
        )

    print()
    print(
        f"  分野平均のばらつき（最大−最小）: 正規化後 {np.ptp(scores.mean(axis=0)):.3f}"
        f" ／ 正規化前 {np.ptp(raw.mean(axis=0)):.3f}"
    )
    print(
        "  ※ 正規化前は 平均相性 = 0.8333 − mean((F−0.5)^2) が恒等的に成り立ち、"
        "プロファイルが\n     中央に寄った分野ほど、内容に関係なく誰にでも高得点を返す"
    )


def print_threshold_sensitivity(scores):
    print()
    print("=" * 78)
    print("3. 閾値 min_relevance の感度")
    print("=" * 78)
    print(f"  {'閾値':>6}{'平均採用分野数':>16}{'候補0件のパターン':>20}")

    for threshold in THRESHOLDS:
        counts = (scores >= threshold).sum(axis=1)
        mark = " ←現在の設定" if threshold == DEFAULT_MIN_RELEVANCE else ""
        empty = f"{(counts == 0).sum()} / {len(counts)}"
        print(f"  {threshold:>6.2f}{counts.mean():>16.2f}{empty:>20}{mark}")

    print("  ※ 候補0件でも最上位1分野へのフォールバックが働くのでエラーにはならない")


def print_real_answers():
    print()
    print("=" * 78)
    print("4. 実回答（inputs/data/user_input*.csv）")
    print("=" * 78)

    for path in sorted(DATA_DIR.glob("user_input*.csv")):
        try:
            user, _, _, _, _, interests = load_user_input(str(path))
        except ValueError:
            # アンケートの生データなど、入力形式ではないCSVは飛ばす
            continue

        scores = field_scores(np.array([user]))[0]
        order = np.argsort(-scores)
        adopted = [f for f in order if scores[f] >= DEFAULT_MIN_RELEVANCE]
        dropped = [f for f in order if scores[f] < DEFAULT_MIN_RELEVANCE]

        print()
        print(f"  {path.name}  適性={[float(v) for v in user]}")

        if adopted:
            names = "、".join(f"{field_names[f]} {scores[f]:.2f}" for f in adopted)
            print(f"    採用({len(adopted)}分野): {names}")
        else:
            print(f"    採用: なし → 最上位 {field_names[order[0]]} へフォールバック")

        print(
            "    除外: "
            + "、".join(f"{field_names[f]} {scores[f]:.2f}" for f in dropped)
        )

        if interests:
            hit = [name for name in interests if field_names.index(name) in adopted]
            print(
                f"    申告した興味: {'、'.join(interests)}"
                f" → 採用されたのは {'、'.join(hit) if hit else 'なし'}"
            )


def main():
    grid = np.array(list(itertools.product(ANSWER_VALUES, repeat=len(axis_names))))
    scores = field_scores(grid)

    print_invariants()
    print_bias(grid, scores)
    print_threshold_sensitivity(scores)
    print_real_answers()


if __name__ == "__main__":
    main()
