"""分枝限定法による厳密最適解ソルバー（検証専用ツール）。

推薦パイプライン本体（calculation/ のQUBO構築・アニーリング）とは
独立したスタンドアロンスクリプトであり、本体側のコードは
このファイルを一切参照しない（依存は tools → calculation の一方向のみ）。

アニーリング解の品質評価のために、本体と同じ候補集合・項目価値・
シナジー・前提条件のもとで厳密最適解を分枝限定法で求め、
アニーリング解との達成率を比較する。

実行方法（リポジトリルートから）:
    python tools/exact_bb.py
"""

import sys
import time
from pathlib import Path

# スタンドアロン実行用：リポジトリルートをimportパスに追加する
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np

from calculation.prerequisites import check_prerequisites, effective_prerequisites
from calculation.recommend import recommend
from config.items import M, hours, item_index, item_names
from config.synergy import synergy_pairs
from inputs.user_input import T, learned, user

z_anneal, _, dbg = recommend(user=user, T=T, learned=learned)

# 学習済みがある場合、本体は実効前提P'と実効価値V'（既習シナジー込み）で解く。
# 同じ土俵で比較するため、厳密解も同じ前提・同じ価値で求める。
V = dbg["effective_value"]
and_prerequisites = effective_prerequisites(frozenset(learned))
cand = sorted(dbg["candidate_indices"])
n = len(cand)
pos = {j: k for k, j in enumerate(cand)}

v = np.array([V[j] for j in cand])
h = np.array([hours[j] for j in cand], dtype=int)

# シナジー（候補内のペアのみ、局所インデックス）
pairs = []
for (a, b), val in synergy_pairs.items():
    i, j = item_index[a], item_index[b]
    if i in pos and j in pos:
        pairs.append((pos[i], pos[j], val))

# 前提条件（局所インデックス）。候補外の親は存在しない前提で構築。
and_c = []
for ch, ps in and_prerequisites.items():
    ci = item_index[ch]
    if ci in pos:
        and_c.append((pos[ci], [pos[item_index[p]] for p in ps]))

# 添字kより後ろで得られる価値の上界（価値 + 正シナジー）。
# ペア(i,j) i<j の利得は「後から選ぶ側」jの時点でaccに加算されるため、
# 上界には max(i,j) >= k のペアをすべて含めなければならない。
suffix = np.zeros(n + 1)
for k in range(n - 1, -1, -1):
    extra = sum(max(0.0, s) for i, j, s in pairs if max(i, j) == k)
    suffix[k] = suffix[k + 1] + max(0.0, v[k]) + extra

best = [-np.inf, None]


def feasible(sel):
    for c, ps in and_c:
        if sel[c] and not all(sel[p] for p in ps):
            return False
    return True


def dfs(k, sel, used, acc):
    if acc + suffix[k] <= best[0]:      # 限定操作
        return
    if k == n:
        if feasible(sel) and acc > best[0]:
            best[0], best[1] = acc, sel.copy()
        return
    # 選ばない
    sel[k] = 0
    dfs(k + 1, sel, used, acc)
    # 選ぶ
    if used + h[k] <= T:
        sel[k] = 1
        gain = v[k] + sum(s for i, j, s in pairs
                          if (i == k and sel[j]) or (j == k and sel[i]))
        dfs(k + 1, sel, used + h[k], acc + gain)
    sel[k] = 0


t0 = time.time()
dfs(0, [0] * n, 0, 0.0)
elapsed = time.time() - t0

opt = set(cand[k] for k in range(n) if best[1][k])
anneal = set(np.flatnonzero(z_anneal).tolist())


def report(label, sel):
    tot = sum(V[j] for j in sel)
    syn = sum(s for i, j, s in pairs if cand[i] in sel and cand[j] in sel)
    print(f"{label}: 目的={tot + syn:.3f}（価値{tot:.3f} シナジー{syn:+.3f}）"
          f" {sum(hours[j] for j in sel)}h/{T}h {len(sel)}項目")


report("厳密最適解(分枝限定)", opt)
report("アニーリング解      ", anneal)

# プロジェクト本体の検査関数で最適解の実行可能性を独立に確認する
z_opt = np.array([1 if i in opt else 0 for i in range(M)])
ok, viol = check_prerequisites(z_opt, frozenset(learned))
print(f"\n[検証] 最適解の前提充足={ok} {viol if not ok else ''} / "
      f"予算 {int(hours @ z_opt)}h <= {T}h : {int(hours @ z_opt) <= T}")

# アニーリング解の目的値には dbg["best_score"]（修復後の価値＋シナジー）を使う。
# QUBOエネルギーは修復前のビット列に対応し、予算ペナルティ項や
# 条件付き価値の近似を含むため、真の目的関数値とは一致しない。
anneal_obj = dbg["best_score"]
print(f"[検証] 最適 {best[0]:.3f} / アニーリング {anneal_obj:.3f} = "
      f"達成率 {100 * anneal_obj / best[0]:.1f}%（乖離 {100 * (1 - anneal_obj / best[0]):.1f}%）")
print(f"\n探索時間 {elapsed:.1f}s / 探索空間 2^{n}")
print("最適解のみ:", sorted(item_names[j] for j in opt - anneal))
print("アニーリングのみ:", sorted(item_names[j] for j in anneal - opt))
print("\n最適解:", sorted(item_names[j] for j in opt))
