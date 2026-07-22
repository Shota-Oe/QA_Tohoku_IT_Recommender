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
```

- 適性値は `0.0`～`1.0` の数値、または `HIGH` / `MID` / `LOW` で指定できる
  （`0.0`=低い / `0.5`=中程度 / `1.0`=高い。「作る↔運用」は `1.0`=作る側 / `0.0`=運用側）。
- `#` で始まる行と空行は無視される。
- 環境変数 `USER_INPUT_CSV` で読み込む CSV のパスを差し替えられる。
