"""Native sampled probabilities, projected onto visible assistant text."""
import json
import math
import re

from . import protocol
from .lifecycle import GenerationFailure


def _byte_alphabet():
    values = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    characters = list(values)
    extra = 0
    for byte in range(256):
        if byte not in values:
            values.append(byte)
            characters.append(256 + extra)
            extra += 1
    return {chr(character): byte for byte, character in zip(values, characters)}


_BYTE_ALPHABET = _byte_alphabet()


def _log(probability):
    if not math.isfinite(probability) or not 0 <= probability <= 1:
        raise GenerationFailure("logprobs_unavailable")
    return math.log(probability) if probability else -9999.0


def _list(tensor):
    return tensor.detach().cpu().tolist()


def content_ranges(parser, parsed, prefix):
    """Use the same tag/call parser and retain original byte positions."""
    source = parser._text
    indices = list(range(len(source)))
    if parser.initial_reasoning and parser._initial_reasoning_done:
        indices = indices[parser._initial_reasoning_cursor:]
    text = ''.join(source[i] for i in indices)
    think = list(protocol._THINK_RE.finditer(text))
    first_tool = next((start for start, _ in protocol._iter_tool_call_starts(text)
                       if not any(m.start() <= start < m.end() for m in think)), len(text))
    removed = set()
    for match in think:
        if match.end() <= first_tool:
            removed.update(range(match.start(), match.end()))
    indices = [i for position, i in enumerate(indices) if position not in removed]
    text = ''.join(source[i] for i in indices)
    _, spans = protocol._parse_qwen_xml_calls(text)
    removed = set()
    for start, end in spans:
        removed.update(range(start, end))
    indices = [i for position, i in enumerate(indices) if position not in removed]
    text = ''.join(source[i] for i in indices)
    indices = indices[len(text) - len(text.lstrip()):len(text.rstrip())]
    if ''.join(source[i] for i in indices) != (parsed.get('content') or ''):
        raise GenerationFailure("logprobs_alignment_failed")
    offsets = [0]
    for character in source:
        offsets.append(offsets[-1] + len(character.encode('utf-8')))
    prefix_bytes = len(prefix.encode('utf-8'))
    ranges = []
    for i in indices:
        if i < len(prefix):
            continue  # Server-injected generation prefixes are not sampled tokens.
        start, end = offsets[i] - prefix_bytes, offsets[i + 1] - prefix_bytes
        if ranges and ranges[-1][1] == start:
            ranges[-1] = (ranges[-1][0], end)
        else:
            ranges.append((start, end))
    return ranges


class Logprobs:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.records = []
        self.offsets = []
        self.raw = bytearray()
        decoder = getattr(getattr(tokenizer, 'tokenizer', None), 'decoder', None)
        try:
            state = decoder.__getstate__()
            state = json.loads(state) if isinstance(state, (str, bytes)) else {}
        except (AttributeError, ValueError, TypeError):
            state = {}
        self.byte_level = state.get('type') == 'ByteLevel'

    def token_bytes(self, token_id):
        added = getattr(self.tokenizer, 'extended_id_to_piece', {})
        if token_id in added:
            return added[token_id].encode('utf-8')
        backend = getattr(self.tokenizer, 'tokenizer', None)
        raw = backend.id_to_token(token_id) if backend is not None else None
        if raw and self.byte_level:
            try:
                return bytes(_BYTE_ALPHABET[c] for c in raw)
            except KeyError as exc:
                raise GenerationFailure("logprobs_unavailable") from exc
        if raw and re.fullmatch(r'<0x[0-9A-Fa-f]{2}>', raw):
            return bytes([int(raw[3:5], 16)])
        return self.tokenizer.get_id_to_piece_list(True)[token_id].encode('utf-8')

    def token(self, token_id, probability):
        data = self.token_bytes(token_id)
        return {'token': data.decode('utf-8', errors='replace'), 'logprob': _log(float(probability)), 'bytes': list(data)}

    def add(self, result):
        ids, probabilities = result.get('token_ids'), result.get('token_probs')
        if ids is None:
            if result.get('text'):
                raise GenerationFailure("logprobs_unavailable")
            return
        if probabilities is None:
            raise GenerationFailure("logprobs_unavailable")
        ids, probabilities = _list(ids)[0], _list(probabilities)[0]
        candidates = _list(result['top_k_tokens'])[0] if result.get('top_k_tokens') is not None else [[] for _ in ids]
        candidate_probs = _list(result['top_k_probs'])[0] if result.get('top_k_probs') is not None else [[] for _ in ids]
        if not len(ids) == len(probabilities) == len(candidates) == len(candidate_probs):
            raise GenerationFailure("logprobs_unavailable")
        for token_id, probability, alternatives, scores in zip(ids, probabilities, candidates, candidate_probs):
            record = self.token(token_id, probability)
            record['top_logprobs'] = [self.token(i, p) for i, p in zip(alternatives, scores)]
            self.offsets.append((len(self.raw), len(self.raw) + len(record['bytes'])))
            self.raw.extend(record['bytes'])
            self.records.append({'token_id': token_id, **record})

    def finish(self, parser, parsed, prefix=''):
        if bytes(self.raw) != parser._text[len(prefix):].encode('utf-8'):
            raise GenerationFailure("logprobs_alignment_failed")
        ranges = content_ranges(parser, parsed, prefix)
        selected = []
        for record, (start, end) in zip(self.records, self.offsets):
            visible = [False] * (end - start)
            for a, b in ranges:
                for i in range(max(start, a), min(end, b)):
                    visible[i - start] = True
            excluded = bytes(byte for byte, shown in zip(record['bytes'], visible) if not shown)
            if any(visible) and not excluded.strip():
                selected.append({k: v for k, v in record.items() if k != 'token_id'})
        return {'content': selected, 'refusal': None}
