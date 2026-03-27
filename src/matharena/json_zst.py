import json
from pathlib import Path

import zstandard as zstd

OUTPUT_JSON_SUFFIX = ".json.zst"


def load_json_zst(path):
    path = Path(path)
    with path.open("rb") as f:
        data = zstd.ZstdDecompressor().decompress(f.read())
    return json.loads(data.decode("utf-8"))


def dump_json_zst(data, path, *, indent=None, ensure_ascii=True, sort_keys=False):
    path = Path(path)
    payload = json.dumps(data, indent=indent, ensure_ascii=ensure_ascii, sort_keys=sort_keys).encode("utf-8")
    compressed = zstd.ZstdCompressor(level=3).compress(payload)
    with path.open("wb") as f:
        f.write(compressed)


def output_json_stem(path):
    return Path(path).name.removesuffix(OUTPUT_JSON_SUFFIX)
