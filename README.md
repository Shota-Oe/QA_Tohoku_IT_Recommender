# QA_Tohoku_IT_Recommender
This is the repo of the team-1 of the "Practical Quantum Solution Creation" class.

## ディレクトリ構成

```
config/       係数・データ定義（適性5軸、分野プロファイル、学習項目、シナジー、前提科目）
inputs/       ユーザー入力（適性ベクトルと学習時間。data/user_input.csv を編集する）
calculation/  QUBO構築・アニーリングなどの計算ロジック
output/       テキスト表示とグラフ描画
tools/        検証用スタンドアロンスクリプト（厳密解ソルバーなど。本体からは参照されない）
main.py       エントリーポイント
```

## 実行方法

```
pip install -r requirements.txt
python main.py
```

### D-Wave 実機 QPU で解く

`--backend qpu` を付けると、アニーリングを Leap 経由で D-Wave の実機 QPU に投げる。

```
python main.py --backend qpu
```

事前にプロジェクト直下の `.env` へ API トークンを置く（`.env` は `.gitignore` 済み）。

```
DWAVE_API_TOKEN=DEV-xxxxxxxxxxxxxxxx
```

| オプション | 既定値 | 説明 |
| --- | --- | --- |
| `--backend` | `neal` | `neal`（ローカルSA）／`qpu`（実機） |
| `--num-reads` | neal 5000／qpu 1000 | アニーリングの実行回数 |
| `--solver` | Leap の自動選択 | QPU の機種名（例 `Advantage_system6`） |
| `--annealing-time` | `20` | 1回のアニール時間（マイクロ秒） |
| `--chain-strength` | 自動決定 | マイナー埋め込みの鎖の強さ |
| `--no-plot` | — | グラフを出さずテキスト結果だけ表示 |

QPU 実行時は、埋め込み規模・鎖切れ率・QPU アクセス時間・問題 ID がレポート末尾に出る。
QPU はマシン時間が課金対象（Leap 無料枠は月1分）で、`--num-reads 1000` の1回で
約 0.13 秒を消費する。

適性や学習時間を変えたいときは `inputs/data/user_input.csv` を、
係数（分野・項目・シナジー・前提科目・アニーリング設定）を変えたいときは
`config/` 以下の各ファイルを編集する。

### `inputs/data/user_input.csv` の形式

`key,value` 形式で、適性5軸と学習時間を記述する。

```
key,value
クリエイティブ,1.0
ロジカル,1.0
作る↔運用,1.0
数学,0.5
計画性・正確さ,1.0
hours_per_week,30
weeks,24
learned,"HTML/CSS, Python基礎"
```

- 適性値は `0.0`～`1.0` の数値、または `HIGH` / `MID` / `LOW` で指定できる
  （`0.0`=低い / `0.5`=中程度 / `1.0`=高い。「作る↔運用」は `1.0`=作る側 / `0.0`=運用側）。
- `#` で始まる行と空行は無視される。
- 環境変数 `USER_INPUT_CSV` で読み込む CSV のパスを差し替えられる。

### 学習済み科目（`learned`）

すでに学習を終えた科目を `learned` 行に書くと、その項目は推薦されなくなる。

- 項目名は `config/items.py` の名前と一致させる（一致しない名前はエラーになる）。
  カンマ区切りで並べ、CSV の仕様上ダブルクォートで囲む。
- **省略できる**（行が無い／値が空＝学習済みなし）。
- 学習済みの時間は予算 `T` を消費しない（過去に消費した時間として扱う）。
- 学習済みの科目は**前提科目としては満たされたものとみなす**。
  たとえば `HTML/CSS` が学習済みなら、`JavaScript/TS` を
  `HTML/CSS` を学び直さずに選べるようになる。
- 学習済みの科目とのシナジーは、相手の項目価値へ加算される
  （`Python基礎` が学習済みなら `統計・数学` の価値が `+0.40` される）。

詳細は `docs/requirements.md` 第10節。
