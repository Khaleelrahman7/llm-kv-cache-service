"""Frozen character tokenizer used by the serving fixture."""

from __future__ import annotations

import hashlib


class Tokenizer:
    PAD = 0
    BOS = 1
    EOS = 2
    UNK = 3
    _ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789 .,!?;:-_'/"

    def __init__(self) -> None:
        self.id_to_piece = ["<pad>", "<bos>", "<eos>", "<unk>", *self._ALPHABET]
        self.piece_to_id = {piece: index for index, piece in enumerate(self.id_to_piece)}

    @property
    def fingerprint(self) -> str:
        table = "\u0000".join(self.id_to_piece).encode("utf-8")
        return "frozen-char-v1:" + hashlib.sha256(table).hexdigest()

    @property
    def vocab_size(self) -> int:
        return len(self.id_to_piece)

    def encode(self, text: str) -> list[int]:
        return [self.BOS, *(self.piece_to_id.get(ch.lower(), self.UNK) for ch in text)]

    def decode(self, token_ids: list[int]) -> str:
        pieces: list[str] = []
        for token_id in token_ids:
            if token_id == self.UNK:
                pieces.append("�")
            elif 4 <= token_id < self.vocab_size:
                pieces.append(self.id_to_piece[token_id])
        return "".join(pieces)
