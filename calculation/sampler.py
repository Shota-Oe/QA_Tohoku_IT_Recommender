"""アニーリングを実行するサンプラーの生成。

実行基盤（バックエンド）は2種類ある。

  neal : ローカルのシミュレーテッド・アニーリング（既定）
  qpu  : D-Wave の実機 QPU（Leap クラウド経由。APIトークンが必要）

QPUを使うには、プロジェクト直下の ``.env`` に

    DWAVE_API_TOKEN=DEV-xxxxxxxxxxxxxxxx

の形式でトークンを置く（シェルの環境変数に直接設定してもよい）。

QPUは論理変数をそのまま解けるわけではないので、
`EmbeddingComposite` がQUBOのグラフをハードウェアのグラフへ
マイナー埋め込み（1つの論理変数を複数の物理量子ビットの鎖で表現）する。
鎖が途中で切れた（chain break）サンプルは多数決で補正されるため、
鎖切れ率は解の信頼度を測る指標になる（`summarize_sampleset` が集計する）。
"""

import os

from neal import SimulatedAnnealingSampler

from config.parameters import (
    DEFAULT_ANNEALING_TIME,
    DEFAULT_NUM_READS,
    DEFAULT_QPU_NUM_READS,
    DEFAULT_QPU_SOLVER,
)

# 選択できるバックエンド
BACKENDS = ("neal", "qpu")

# APIトークンを入れる環境変数の名前
TOKEN_ENV_NAME = "DWAVE_API_TOKEN"

# プロジェクト直下の .env
ENV_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"
)


def default_num_reads(backend):
    """バックエンドごとの既定の実行回数。

    QPUはマシン時間が課金対象（無料枠は月1分）なので、
    nealの5000回よりも控えめな回数を既定にする。
    """
    if backend == "qpu":
        return DEFAULT_QPU_NUM_READS

    return DEFAULT_NUM_READS


def load_dwave_token():
    """D-WaveのAPIトークンを取得する。

    環境変数が既に設定されていればそれを使い、
    なければプロジェクト直下の ``.env`` から読み込む。
    """
    token = os.environ.get(TOKEN_ENV_NAME)

    if token:
        return token

    try:
        from dotenv import load_dotenv
    except ImportError:
        raise RuntimeError(
            "python-dotenv がインストールされていません。\n"
            "pip install -r requirements.txt を実行してください。"
        )

    load_dotenv(ENV_PATH)

    token = os.environ.get(TOKEN_ENV_NAME)

    if not token:
        raise RuntimeError(
            f"D-WaveのAPIトークンが見つかりません。\n"
            f"{ENV_PATH} に {TOKEN_ENV_NAME}=DEV-xxxx を設定してください。"
        )

    return token


def build_sampler(
    backend="neal",
    solver=DEFAULT_QPU_SOLVER,
    annealing_time=DEFAULT_ANNEALING_TIME,
    chain_strength=None,
):
    """バックエンド名からサンプラーと `sample()` へ渡す追加引数を作る。

    Parameters
    ----------
    backend : str
        "neal" または "qpu"。

    solver : str or None
        QPUの機種名（例："Advantage_system6"）。
        Noneの場合はLeapが既定のQPUを選ぶ。

    annealing_time : float
        1回のアニーリングにかける時間（マイクロ秒）。QPUのみ有効。

    chain_strength : float or None
        マイナー埋め込みの鎖を束ねる強さ。
        Noneの場合は `uniform_torque_compensation` による自動決定に任せる。

    Returns
    -------
    sampler :
        `recommend(..., sampler=...)` へ渡すサンプラー。

    sampler_kwargs : dict
        `recommend(..., sampler_kwargs=...)` へ渡す追加引数。

    description : str
        表示用のバックエンド名。
    """
    if backend not in BACKENDS:
        raise ValueError(
            f"backendは{BACKENDS}のいずれかにしてください: {backend!r}"
        )

    if backend == "neal":
        # 既定のローカル実行。追加引数はrecommend側が組み立てる。
        return None, {}, "neal（シミュレーテッド・アニーリング／ローカル）"

    try:
        from dwave.system import DWaveSampler, EmbeddingComposite
    except ImportError:
        raise RuntimeError(
            "dwave-system がインストールされていません。\n"
            "pip install -r requirements.txt を実行してください。"
        )

    from dwave.cloud.exceptions import SolverAuthenticationError

    token = load_dwave_token()

    solver_spec = solver if solver else {"qpu": True}

    try:
        qpu = DWaveSampler(token=token, solver=solver_spec)
    except SolverAuthenticationError:
        raise RuntimeError(
            f"D-Waveの認証に失敗しました。\n"
            f"{ENV_PATH} の {TOKEN_ENV_NAME} が正しいか、"
            f"Leapアカウントが有効か確認してください。"
        )

    sampler = EmbeddingComposite(qpu)

    sampler_kwargs = {
        "annealing_time": annealing_time,
        # 既定の "histogram" は同一サンプルを1件へまとめてしまい、
        # nealの結果とサンプル数を比べられなくなるため生の読み出しを使う
        "answer_mode": "raw",
        # 埋め込み（論理変数→物理量子ビットの対応）を結果に含めさせる。
        # 既定はFalseで、鎖長を報告できなくなる。
        "return_embedding": True,
        "label": "QA_Tohoku_IT_Recommender",
    }

    if chain_strength is not None:
        sampler_kwargs["chain_strength"] = chain_strength

    description = f"D-Wave QPU 実機（{qpu.solver.name}）"

    return sampler, sampler_kwargs, description


def summarize_sampleset(sampleset):
    """サンプル集合から実行基盤に関する情報を取り出す。

    QPU実行のときだけ得られる情報（鎖切れ率・QPU時間・埋め込みの鎖長）を
    集計する。nealのときは空の辞書を返す。
    """
    info = {}

    # 鎖切れ率：EmbeddingCompositeが各サンプルへ付与する
    if "chain_break_fraction" in sampleset.record.dtype.names:
        chain_break_fractions = sampleset.record.chain_break_fraction

        info["chain_break_mean"] = float(chain_break_fractions.mean())
        info["chain_break_max"] = float(chain_break_fractions.max())
        info["chain_break_free_ratio"] = float(
            (chain_break_fractions == 0.0).mean()
        )

    # 埋め込み：1論理変数あたり何個の物理量子ビットを使ったか
    embedding = sampleset.info.get("embedding_context", {}).get("embedding")

    if embedding:
        chain_lengths = [len(chain) for chain in embedding.values()]

        info["physical_qubits"] = int(sum(chain_lengths))
        info["max_chain_length"] = int(max(chain_lengths))
        info["mean_chain_length"] = float(
            sum(chain_lengths) / len(chain_lengths)
        )

    # QPUの実時間（マイクロ秒）。
    # nealもtimingを返すがキーの内容が異なるため、QPU固有のキーで判定する。
    timing = sampleset.info.get("timing") or {}

    if "qpu_access_time" in timing:
        info["qpu_access_time"] = float(timing["qpu_access_time"])
        info["qpu_anneal_time_per_sample"] = float(
            timing.get("qpu_anneal_time_per_sample", 0.0)
        )

    if "problem_id" in sampleset.info:
        info["problem_id"] = sampleset.info["problem_id"]

    return info
