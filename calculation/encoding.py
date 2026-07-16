"""スラック変数用の符号化。"""


def bounded_binary_weights(maximum):
    """
    0～maximumのすべての整数を表現できる
    バイナリ変数の重みを返す。

    例：
    maximum = 24

    戻り値：
    [1, 2, 4, 8, 9]

    これらの組み合わせによって、
    0～24を表現できる。
    """
    if maximum < 0:
        raise ValueError("maximumは0以上にしてください。")

    if maximum == 0:
        return []

    weights = []
    represented_maximum = 0
    next_power = 1

    while represented_maximum < maximum:
        weight = min(next_power, maximum - represented_maximum)

        weights.append(weight)

        represented_maximum += weight
        next_power *= 2

    return weights
