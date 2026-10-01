"""Bounded image preprocessing and reusable native EXL3 image embeddings."""
import asyncio
from contextlib import nullcontext
import base64
from collections import OrderedDict
import hashlib
import http.client
import io
import ipaddress
import socket
import ssl
from urllib.parse import urljoin, urlsplit

from .protocol import ProtocolError, normalize_chat_messages
from .lifecycle import GenerationFailure


def remote_image(url, max_bytes, timeout=8):
    """Fetch public HTTP(S) images, pinning the checked IP on each redirect."""
    for _ in range(5):
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            raise ProtocolError("Image URL must be a public HTTP(S) URL without credentials")
        try:
            port = parsed.port or (443 if parsed.scheme == "https" else 80)
            addresses = socket.getaddrinfo(parsed.hostname, port, type=socket.SOCK_STREAM)
            ips = {a[4][0] for a in addresses}
            if not ips or any(not ipaddress.ip_address(ip).is_global for ip in ips):
                raise ProtocolError("Image URL must resolve only to public addresses")
            address = addresses[0][4][0]
            host = parsed.hostname
            host_header = f"[{host}]" if ":" in host else host
            if parsed.port:
                host_header += f":{port}"

            class Connection(http.client.HTTPConnection):
                def connect(self):
                    # Do not resolve again after validation (DNS rebinding).
                    self.sock = socket.create_connection((address, port), timeout)
                    if parsed.scheme == "https":
                        self.sock = ssl.create_default_context().wrap_socket(self.sock, server_hostname=host)

            connection = Connection(host, port, timeout=timeout)
            try:
                path = parsed.path or "/"
                if parsed.query:
                    path += "?" + parsed.query
                connection.request("GET", path, headers={"Host": host_header, "Accept": "image/*"})
                response = connection.getresponse()
                if response.status in {301, 302, 303, 307, 308}:
                    location = response.getheader("Location")
                    if not location:
                        raise ProtocolError("Image redirect has no target")
                    url = urljoin(url, location)
                    continue
                if response.status != 200:
                    raise ProtocolError("Image download failed")
                length = response.getheader("Content-Length")
                if length and int(length) > max_bytes:
                    raise ProtocolError("Image exceeds the configured byte limit")
                data = response.read(max_bytes + 1)
                if len(data) > max_bytes:
                    raise ProtocolError("Image exceeds the configured byte limit")
                return data
            finally:
                connection.close()
        except ProtocolError:
            raise
        except (OSError, ValueError, http.client.HTTPException) as exc:
            raise ProtocolError("Image download failed") from exc
    raise ProtocolError("Too many image redirects")


def image_bytes(part, max_bytes, allow_remote):
    value = part.get("image_url")
    if isinstance(value, str):
        url, detail = value, "auto"
    elif isinstance(value, dict):
        url, detail = value.get("url"), value.get("detail", "auto")
    else:
        raise ProtocolError("image_url requires a URL string or an object with url")
    if not isinstance(url, str) or not isinstance(detail, str) or detail not in {"auto", "low", "high"}:
        raise ProtocolError("Invalid image URL or detail (auto, low or high)")
    if url.startswith("data:"):
        header, separator, payload = url.partition(",")
        if not separator or header.lower() not in {
            "data:image/png;base64", "data:image/jpeg;base64", "data:image/webp;base64", "data:image/gif;base64"}:
            raise ProtocolError("Images must be base64 PNG, JPEG, WebP or GIF data URLs")
        if len(payload) > 4 * ((max_bytes + 2) // 3):
            raise ProtocolError("Image exceeds the configured byte limit")
        try:
            data = base64.b64decode(payload, validate=True)
        except ValueError as exc:
            raise ProtocolError("Invalid base64 image") from exc
    elif allow_remote:
        data = remote_image(url, max_bytes)
    else:
        raise ProtocolError("Remote image URLs are disabled; use a base64 data URL or --vision-remote-urls")
    if len(data) > max_bytes:
        raise ProtocolError("Image exceeds the configured byte limit")
    return data, detail


def decode_image(data, max_input_pixels):
    from PIL import Image, ImageOps, UnidentifiedImageError
    try:
        with Image.open(io.BytesIO(data)) as image:
            if image.format not in {"PNG", "JPEG", "WEBP", "GIF"}:
                raise ProtocolError("Unsupported image encoding")
            if image.width * image.height > max_input_pixels:
                raise ProtocolError("Image exceeds the configured input pixel limit")
            if getattr(image, "n_frames", 1) != 1:
                raise ProtocolError("Animated images are unsupported; supply one frame")
            from exllamav3.architecture.mm_processing.common import convert_to_rgb
            return convert_to_rgb(ImageOps.exif_transpose(image))
    except ProtocolError:
        raise
    except (UnidentifiedImageError, OSError, ValueError, Image.DecompressionBombError) as exc:
        raise ProtocolError("Invalid or oversized image") from exc


class VisionRuntime:
    def __init__(self, model, *, device=None, max_pixels=262144, max_input_pixels=16777216,
                 max_bytes=20 * 1024**2, max_images=16, cache_bytes=128 * 1024**2, allow_remote=False):
        self.model = model
        self.device = device
        self.max_pixels = max_pixels
        self.max_input_pixels = max_input_pixels
        self.max_bytes = max_bytes
        self.max_images = max_images
        self.cache_limit = cache_bytes
        self.allow_remote = allow_remote
        self.cache = OrderedDict()
        self.cache_size = 0
        self.lock = asyncio.Lock()

    async def prepare(self, messages, tokenizer):
        normalize_chat_messages(messages, allow_images=True)
        images = [part for message in messages for part in (message.get("content") or [])
                  if isinstance(part, dict) and part.get("type") == "image_url"]
        if len(images) > self.max_images:
            raise ProtocolError("Too many images in the conversation")
        result = []
        async with self.lock:
            for part in images:
                data, detail = await asyncio.to_thread(image_bytes, part, self.max_bytes, self.allow_remote)
                pixels = min(self.max_pixels, 65536) if detail == "low" else self.max_pixels
                key = (hashlib.sha256(data).digest(), pixels)
                cached = self.cache.pop(key, None)
                if cached is None:
                    image = await asyncio.to_thread(decode_image, data, self.max_input_pixels)
                    pp = self.model.config.vision_pp
                    old_max = pp.max_pixels
                    pp.max_pixels = max(pp.min_pixels, pixels)
                    try:
                        # Serialize the GPU preprocessing with inference's event
                        # loop; do not overlap two callers mutating vision_pp.
                        import torch
                        with torch.cuda.device(self.device) if self.device is not None else nullcontext():
                            embedding = self.model.get_image_embeddings(tokenizer=tokenizer, image=image)
                        embedding.embeddings = embedding.embeddings.cpu()
                        if embedding.deepstack_embeddings:
                            embedding.deepstack_embeddings = [t.cpu() for t in embedding.deepstack_embeddings]
                        if not all(bool(torch.isfinite(t).all()) for t in
                                   [embedding.embeddings] + (embedding.deepstack_embeddings or [])):
                            raise GenerationFailure("image_inference_failed")
                    except (ValueError, AssertionError) as exc:
                        raise ProtocolError("Image cannot be processed by this model") from exc
                    except RuntimeError as exc:
                        raise GenerationFailure("image_inference_failed") from exc
                    finally:
                        pp.max_pixels = old_max
                        image.close()
                    tensors = [embedding.embeddings] + (embedding.deepstack_embeddings or [])
                    size = sum(t.numel() * t.element_size() for t in tensors)
                    cached = (embedding, size)
                    if size <= self.cache_limit:
                        while self.cache and self.cache_size + size > self.cache_limit:
                            _, (_, removed) = self.cache.popitem(last=False)
                            self.cache_size -= removed
                        self.cache_size += size
                    else:
                        result.append(embedding)
                        continue
                self.cache[key] = cached
                result.append(cached[0])
        return result

    def report(self):
        return {"enabled": True, "device": self.device, "max_pixels": self.max_pixels,
                "max_images": self.max_images, "remote_urls": self.allow_remote,
                "cache_bytes": self.cache_size, "cache_limit_bytes": self.cache_limit}

    def close(self):
        self.cache.clear()
        self.cache_size = 0
        self.model.unload()


def load_vision(args):
    from exllamav3 import Config, Model
    config = Config.from_directory(args.model_dir)
    if "vision" not in config.model_classes or config.image_token_id is None:
        raise ValueError("--vision requires image processor metadata and vision weights")
    if not all(hasattr(getattr(config, "vision_pp", None), field) for field in ("min_pixels", "max_pixels")):
        raise ValueError("--vision requires a compatible min_pixels/max_pixels image processor")
    if args.vision_device < 0 or args.vision_max_pixels < config.vision_pp.min_pixels \
            or args.vision_max_input_pixels < args.vision_max_pixels \
            or args.vision_max_images < 1 or args.vision_cache_mb < 0:
        raise ValueError("Invalid Vision device, pixel, image or cache limits")
    model = Model.from_config(config, component="vision")
    try:
        model.load(device=f"cuda:{args.vision_device}", progressbar=True)
        return VisionRuntime(model, device=args.vision_device, max_pixels=args.vision_max_pixels,
                             max_input_pixels=args.vision_max_input_pixels, max_images=args.vision_max_images,
                             cache_bytes=args.vision_cache_mb * 1024**2, allow_remote=args.vision_remote_urls)
    except BaseException:
        model.unload()
        raise
