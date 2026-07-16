# QA_Tohoku_IT_Recommender
This is the repo of the team-1 of the "Practical Quantum Solution Creation" class.

## ディレクトリ構成

```
config/       係数・データ定義（適性5軸、分野プロファイル、学習項目、シナジー、前提科目）
inputs/       ユーザー入力（適性ベクトルと学習時間。ファイルを直接編集する）
calculation/  QUBO構築・アニーリングなどの計算ロジック
output/       テキスト表示とグラフ描画
main.py       エントリーポイント
```

## 実行方法

```
pip install -r requirements.txt
python main.py
```

適性や学習時間を変えたいときは `inputs/user_input.py` を、
係数（分野・項目・シナジー・前提科目・アニーリング設定）を変えたいときは
`config/` 以下の各ファイルを編集する。
