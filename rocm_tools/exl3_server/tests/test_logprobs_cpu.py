import json
import math
from types import SimpleNamespace

import pytest
import torch

from rocm_tools.exl3_server.logprobs import Logprobs, _BYTE_ALPHABET
from rocm_tools.exl3_server.protocol import IncrementalAssistantParser
from rocm_tools.exl3_server.lifecycle import GenerationFailure


class Tokenizer:
    def __init__(self, pieces):
        self.pieces = pieces
        inverse = {byte: character for character, byte in _BYTE_ALPHABET.items()}
        class Decoder:
            def __getstate__(self): return b'{"type":"ByteLevel"}'
        self.tokenizer = SimpleNamespace(decoder=Decoder(), id_to_token=lambda i: ''.join(inverse[b] for b in pieces[i]))
        self.extended_id_to_piece = {}


def event(pieces, probabilities=None):
    return {'token_ids': torch.tensor([list(range(len(pieces)))]),
            'token_probs': torch.tensor([probabilities or [.5] * len(pieces)]),
            'top_k_tokens': torch.tensor([[[i] for i in range(len(pieces))]]),
            'top_k_probs': torch.tensor([[[.5] for _ in pieces]])}


@pytest.mark.parametrize('initial,prefix,pieces,expected', [
    (False, '', [b'hello'], b'hello'),
    (False, '', [b' hello '], b' hello '),
    (True, '', [b'thought', b'</think>', b'\n', b'answer'], b'answer'),
    (False, '', [b'<think>', b'thought', b'</think>', b'answer'], b'answer'),
    (False, '{', [b'"x":1}'], b'"x":1}'),
    (False, '', [b'\xe3', b'\x81\x82'], 'あ'.encode()),
    (False, '', [b'checking', b'<tool_call><function=weather><parameter=city>"Tokyo"</parameter></function></tool_call>'], b'checking'),
])
def test_logprobs_align_visible_text_and_preserve_utf8_bytes(initial, prefix, pieces, expected):
    tokenizer = Tokenizer(pieces)
    trace = Logprobs(tokenizer)
    trace.add(event(pieces))
    parser = IncrementalAssistantParser(initial_reasoning=initial)
    parser.feed(prefix + b''.join(pieces).decode())
    parsed = parser.finish()
    result = trace.finish(parser, parsed, prefix)
    assert bytes(b for token in result['content'] for b in token['bytes']) == expected
    assert all(math.isclose(token['logprob'], math.log(.5)) for token in result['content'])
    assert len(trace.records) == len(pieces)
    assert all(len(token['top_logprobs']) == 1 for token in trace.records)


@pytest.mark.parametrize('probability', [float('nan'), -1., 2.])
def test_invalid_probabilities_fail_instead_of_returning_false_confidence(probability):
    trace = Logprobs(Tokenizer([b'x']))
    with pytest.raises(GenerationFailure, match='logprobs_unavailable'):
        trace.add(event([b'x'], [probability]))


def test_zero_probability_and_missing_native_trace():
    trace = Logprobs(Tokenizer([b'x']))
    trace.add(event([b'x'], [0.]))
    assert trace.records[0]['logprob'] == -9999.
    with pytest.raises(GenerationFailure):
        trace.add({'text': 'x'})
