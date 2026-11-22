"""Retrieve only the checksum-pinned shards required by a BridgeData V2 index."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from .reconstruct_bridge import digest, validate_index


DATASET = "youliangtan/bridge_dataset"
REVISION = "bf83119398c0ac0debe88ecfd9fbacef5a57dbca"
SOURCES = {
    "official": "https://rail.eecs.berkeley.edu/datasets/bridge_release/data/tfds/bridge_dataset/1.0.0/",
    "hf": f"https://huggingface.co/datasets/{DATASET}/resolve/{REVISION}/1.0.0/",
}


def verify_local(path, item):
    if (path.is_symlink() or not path.is_file() or path.stat().st_size != item["bytes"]
            or digest(path) != item["sha256"]):
        raise ValueError(f"Local shard checksum mismatch: {path.name}")


def fetch_one(item, destination, source):
    path = destination / item["name"]
    if path.exists() or path.is_symlink():
        verify_local(path, item)
        return "present_verified"
    request = Request(SOURCES[source] + path.name, headers={"Accept-Encoding": "identity"})
    count = 0
    checksum = hashlib.sha256()
    with tempfile.NamedTemporaryFile(mode="wb", prefix=f".{path.name}.", suffix=".part",
                                     dir=destination, delete=False) as stream:
        temporary = Path(stream.name)
        try:
            with urlopen(request, timeout=45) as response:
                if response.status != 200 or urlsplit(response.geturl()).scheme != "https":
                    raise ValueError(f"Expected a full HTTPS response for {path.name}")
                while chunk := response.read(2**20):
                    count += len(chunk)
                    if count > item["bytes"]:
                        raise ValueError(f"Download exceeded the expected byte count: {path.name}")
                    checksum.update(chunk)
                    stream.write(chunk)
            if count != item["bytes"] or checksum.hexdigest() != item["sha256"]:
                raise ValueError(f"Downloaded shard checksum mismatch: {path.name}")
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
            try:
                os.link(temporary, path)
            except FileExistsError:
                verify_local(path, item)
                return "present_verified"
            return "downloaded_verified"
        finally:
            stream.close()
            temporary.unlink()


def sealed_json(path, expected):
    contents = path.read_bytes()
    if hashlib.sha256(contents).hexdigest() != expected:
        raise ValueError(f"Input SHA256 mismatch: {path.name}")
    return json.loads(contents)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-index", required=True, type=Path)
    parser.add_argument("--source-index-sha256", required=True)
    parser.add_argument("--shard-checksums", required=True, type=Path)
    parser.add_argument("--shard-checksums-sha256", required=True)
    parser.add_argument("--shard-dir", required=True, type=Path)
    parser.add_argument("--source", choices=SOURCES, default="official")
    parser.add_argument("--shard", action="append", help="Repeat to fetch a subset of shards in the index")
    args = parser.parse_args(argv)
    index = sealed_json(args.source_index, args.source_index_sha256)
    manifest = sealed_json(args.shard_checksums, args.shard_checksums_sha256)
    if (manifest.get("dataset"), manifest.get("revision")) != (DATASET, REVISION):
        raise ValueError("Expected the pinned BridgeData V2 shard manifest")
    shards = validate_index(index, manifest)
    required = {row["shard"] for row in index["sources"]}
    names = sorted(set(args.shard)) if args.shard else sorted(required)
    if not set(names) <= required:
        raise ValueError("Requested shard is outside the source index")
    destination = args.shard_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    results = {name: fetch_one(shards[name], destination, args.source) for name in names}
    print(json.dumps({"source": args.source, "verified_shards": len(results), "results": results}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
