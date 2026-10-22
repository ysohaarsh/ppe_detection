import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "pokerHandDetector"))

from PokerHandFunction import findPokerHand  # noqa: E402


@pytest.mark.parametrize(
    "hand, expected",
    [
        (["KH", "AH", "QH", "JH", "10H"], "Royal Flush"),
        (["QC", "JC", "10C", "9C", "8C"], "Straight Flush"),
        (["5C", "5S", "5H", "5D", "QH"], "Four of a Kind"),
        (["2H", "2D", "2S", "10H", "10C"], "Full House"),
        (["2D", "KD", "7D", "6D", "5D"], "Flush"),
        (["JC", "10H", "9C", "8C", "7D"], "Straight"),
        (["10H", "10C", "10D", "2D", "5S"], "Three of a Kind"),
        (["KD", "KH", "5C", "5S", "6D"], "Two Pair"),
        (["2D", "2S", "9C", "KD", "10C"], "Pair"),
        (["KD", "5H", "2D", "10C", "JH"], "High Card"),
    ],
)
def test_find_poker_hand(hand, expected):
    assert findPokerHand(hand) == expected


def test_input_order_does_not_matter():
    assert findPokerHand(["10H", "KH", "JH", "AH", "QH"]) == "Royal Flush"


def test_ten_is_parsed_as_two_character_rank():
    assert findPokerHand(["10S", "10D", "4C", "7H", "KS"]) == "Pair"
