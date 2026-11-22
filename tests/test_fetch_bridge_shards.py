from hashlib import sha256
from io import BytesIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from deepvision_vla.diagnostics.fetch_bridge_shards import fetch_one


class Response(BytesIO):
    status = 200

    def geturl(self):
        return "https://example.org/shard"


class FetchBridgeShardTest(unittest.TestCase):
    def test_only_verified_bytes_are_installed_and_existing_files_are_reused(self):
        value = b"source bytes"
        item = {"name": "bridge_dataset-train.tfrecord-00001-of-01024",
                "bytes": len(value), "sha256": sha256(value).hexdigest()}
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root)
            with patch("deepvision_vla.diagnostics.fetch_bridge_shards.urlopen", return_value=Response(b"wrong bytes!")):
                with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                    fetch_one(item, folder, "official")
                self.assertEqual(list(folder.iterdir()), [])
            with patch("deepvision_vla.diagnostics.fetch_bridge_shards.urlopen", return_value=Response(value)) as urlopen:
                self.assertEqual(fetch_one(item, folder, "official"), "downloaded_verified")
                self.assertEqual(fetch_one(item, folder, "official"), "present_verified")
                self.assertEqual((folder / item["name"]).read_bytes(), value)
                self.assertEqual(urlopen.call_count, 1)


if __name__ == "__main__":
    unittest.main()
