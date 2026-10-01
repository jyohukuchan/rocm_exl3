import base64
import io
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from rocm_tools.exl3_server.protocol import ProtocolError, messages_for_template
from rocm_tools.exl3_server.vision import VisionRuntime, image_bytes, decode_image, remote_image


def image_part(color="red", detail="auto", mode="RGB"):
    output = io.BytesIO()
    Image.new(mode, (32, 32), color).save(output, format="PNG")
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(output.getvalue()).decode(), "detail": detail}}


def test_image_limits_corrupt_payloads_and_unknown_encodings():
    data, detail = image_bytes(image_part(), 1024, False)
    assert detail == "auto"
    assert decode_image(data, 1024).size == (32, 32)
    transparent, _ = image_bytes(image_part((255, 0, 0, 0), mode="RGBA"), 1024, False)
    assert decode_image(transparent, 1024).getpixel((0, 0)) == (255, 255, 255)
    with pytest.raises(ProtocolError, match="byte limit"):
        image_bytes(image_part(), 1, False)
    with pytest.raises(ProtocolError, match="pixel limit"):
        decode_image(data, 16)
    for url in ("data:image/png;base64,invalid$", "data:image/svg+xml;base64,AAAA"):
        with pytest.raises(ProtocolError):
            image_bytes({"image_url": url}, 1024, False)
    with pytest.raises(ProtocolError):
        decode_image(b"not an image", 1024)
    with pytest.raises(ProtocolError, match="disabled"):
        image_bytes({"image_url": "https://example.com/image.png"}, 1024, False)


@pytest.mark.parametrize("ip", ["127.0.0.1", "::1", "192.168.1.1", "169.254.169.254", "::ffff:127.0.0.1"])
def test_remote_images_reject_internal_addresses_without_connecting(monkeypatch, ip):
    import socket
    monkeypatch.setattr(socket, "getaddrinfo", lambda *args, **kwargs: [(socket.AF_INET, socket.SOCK_STREAM, 0, '', (ip, 80))])
    monkeypatch.setattr(socket, "create_connection", lambda *args, **kwargs: pytest.fail("Private address must not be connected"))
    with pytest.raises(ProtocolError, match="public addresses"):
        remote_image("http://example.test/image.png", 1024)


def test_remote_connection_uses_the_validated_ip_without_second_dns_lookup(monkeypatch):
    import socket
    queries, connections = [], []
    def resolve(*args, **kwargs):
        queries.append(args)
        return [(socket.AF_INET, socket.SOCK_STREAM, 0, '', ('1.1.1.1', 80))]
    class Socket:
        def sendall(self, value): pass
        def makefile(self, mode): return io.BytesIO(b'HTTP/1.1 200 OK\r\nContent-Length: 4\r\n\r\nabcd')
        def close(self): pass
    def connect(address, timeout):
        connections.append(address)
        return Socket()
    monkeypatch.setattr(socket, "getaddrinfo", resolve)
    monkeypatch.setattr(socket, "create_connection", connect)
    assert remote_image("http://example.test/image.png", 1024) == b"abcd"
    assert len(queries) == 1 and connections == [('1.1.1.1', 80)]


class VisionModel:
    def __init__(self):
        self.config = SimpleNamespace(vision_pp=SimpleNamespace(min_pixels=65536, max_pixels=16777216))
        self.calls = []
        self.unloaded = False

    def get_image_embeddings(self, tokenizer, image):
        import torch
        from exllamav3.tokenizer.mm_embedding import MMEmbedding
        self.calls.append((image.getpixel((0, 0)), self.config.vision_pp.max_pixels, image.mode))
        return MMEmbedding(embeddings=torch.zeros((3, 4), dtype=torch.float16),
                           token_string=torch.tensor([[248053, -1, -1, -1, 248054]]),
                           grid_thw=(1, 2, 6), mrope_merge_size=2)

    def unload(self):
        self.unloaded = True


@pytest.mark.asyncio
async def test_embedding_cache_keeps_aliases_stable_and_is_bounded():
    model = VisionModel()
    runtime = VisionRuntime(model, cache_bytes=24)
    messages = [{"role": "user", "content": [image_part()]}]
    first = (await runtime.prepare(messages, None))[0]
    second = (await runtime.prepare(messages, None))[0]
    assert first is second and len(model.calls) == 1
    assert model.calls[0] == ((255, 0, 0), 262144, "RGB")
    assert model.config.vision_pp.max_pixels == 16777216
    low = (await runtime.prepare([{"role": "user", "content": [image_part("blue", "low")]}], None))[0]
    assert low.first_index != first.first_index
    assert model.calls[-1][1] == 65536
    assert runtime.cache_size == 24 and len(runtime.cache) == 1
    new_first = (await runtime.prepare(messages, None))[0]
    assert new_first.first_index != first.first_index
    runtime.close()
    assert runtime.cache_size == 0 and model.unloaded


@pytest.mark.asyncio
async def test_limits_and_roles_reject_before_embedding_inference():
    model = VisionModel()
    runtime = VisionRuntime(model, max_images=1)
    for messages in ([{"role": "user", "content": [image_part(), image_part()]}],
                     [{"role": "system", "content": [image_part()]}]):
        with pytest.raises(ProtocolError):
            await runtime.prepare(messages, None)
    assert not model.calls


@pytest.mark.asyncio
async def test_nonfinite_features_never_enter_the_cache_and_restore_processor_limits():
    import torch
    from rocm_tools.exl3_server.lifecycle import GenerationFailure
    model = VisionModel()
    original = model.get_image_embeddings
    def invalid(*args, **kwargs):
        embedding = original(*args, **kwargs)
        embedding.embeddings.fill_(float('nan'))
        return embedding
    model.get_image_embeddings = invalid
    runtime = VisionRuntime(model)
    with pytest.raises(GenerationFailure, match="image_inference_failed"):
        await runtime.prepare([{"role": "user", "content": [image_part()]}], None)
    assert runtime.cache_size == 0 and not runtime.cache
    assert model.config.vision_pp.max_pixels == 16777216


def test_real_qwen_template_splices_one_vision_delimiter_pair():
    path = Path(os.environ.get("EXL3_SCHEMA_TEST_TOKENIZER", ""))
    if not path.is_file():
        pytest.skip("Actual model tokenizer is unavailable")
    from exllamav3 import Config, Tokenizer
    from exllamav3.tokenizer.mm_embedding import MMEmbedding
    import torch
    config = Config.from_directory(str(path.parent))
    tokenizer = Tokenizer(config)
    embedding = MMEmbedding(embeddings=torch.zeros((3, 2560)),
                           token_string=torch.tensor([[config.vision_start_token_id, -1, -1, -1, config.vision_end_token_id]]))
    messages = messages_for_template([{"role": "user", "content": [image_part(), {"type": "text", "text": "describe"}]}], allow_images=True)
    assert messages[0]["content"][0] == {"type": "image"}
    ids = tokenizer.hf_chat_template(messages, embeddings=[embedding], enable_thinking=False)
    assert int((ids == config.vision_start_token_id).sum()) == 1
    assert int((ids == config.vision_end_token_id).sum()) == 1
    assert int(((ids >= embedding.first_index) & (ids < embedding.last_index)).sum()) == 3
