"""Strict compatibility for the pinned author's Pi0 Bridge configuration.

The published JSON contains two evaluation controls that the accompanying
policy dataclass does not declare. The pinned LeRobot source declares those
controls on EvalPipelineConfig/TrainPipelineConfig instead. They are retained
as explicit runtime metadata; every policy field still uses the real author's
typed parser and must survive an exact encode/decode round trip.
"""
import hashlib
import inspect
import json
from pathlib import Path


CONFIG_GIT_BLOB = "0664d11b484c5391f05567e4f2f9cd94498bdd11"
AUTHOR_CONFIG_GIT_BLOB = "5dc6d0cfcbc26b8d3200f845d67efe001f2707ff"


def _git_blob(data):
    return hashlib.sha1(("blob " + str(len(data)) + "\0").encode() + data).hexdigest()


def split_bridge_runtime_fields(raw):
    """Preserve exactly the pinned CPU/non-AMP runtime controls explicitly."""
    if not isinstance(raw, dict):
        raise ValueError("Expected a policy configuration object")
    if raw.get("device") != "cpu" or type(raw.get("use_amp")) is not bool or raw["use_amp"]:
        raise ValueError("Expected the published device=cpu/use_amp=false runtime controls")
    if raw.get("type") != "pi":
        raise ValueError("Expected the author's registered policy type pi")
    policy = dict(raw)
    runtime = {name: policy.pop(name) for name in ("device", "use_amp")}
    return policy, runtime


def parse_bridge_pi0_config(checkpoint, config_class):
    """Return actual author config plus retained runtime compatibility audit.

    Does not construct a neural network, load weights, mutate the source JSON,
    register a replacement config, ignore arbitrary unknown fields, or change
    the weight loader's strictness.
    """
    import draccus
    from lerobot.configs.policies import PreTrainedConfig

    checkpoint = Path(checkpoint).resolve()
    source = checkpoint / "configuration_pi0.py"
    raw_bytes = (checkpoint / "config.json").read_bytes()
    if _git_blob(raw_bytes) != CONFIG_GIT_BLOB:
        raise ValueError("Configuration differs from the pinned Bridge checkpoint")
    if _git_blob(source.read_bytes()) != AUTHOR_CONFIG_GIT_BLOB:
        raise ValueError("Author configuration source differs from its pin")
    if Path(inspect.getfile(config_class)).resolve() != source:
        raise ValueError("Use the actual checkpoint author configuration class")
    policy_data, runtime = split_bridge_runtime_fields(json.loads(raw_bytes))
    # The saved `type` is consumed by the genuine registry base, exactly as
    # its standard from_pretrained parser does. Decoding a concrete subclass
    # would incorrectly treat the discriminator itself as an unknown field.
    config = draccus.decode(PreTrainedConfig, policy_data)
    if type(config) is not config_class:
        raise ValueError("Parser selected another policy configuration class")
    encoded = json.loads(json.dumps(draccus.encode(config)))
    if encoded != policy_data:
        raise ValueError("Policy configuration fields did not round-trip exactly")
    audit = {
        "operator": "explicit_policy_config_with_separate_runtime_controls",
        "runtime_controls": runtime,
        "runtime_source": "pinned LeRobot EvalPipelineConfig/TrainPipelineConfig",
        "policy_class": config_class.__module__ + "." + config_class.__qualname__,
        "policy_config_git_blob": CONFIG_GIT_BLOB,
        "architecture_normalization_fields_roundtrip_equal": True,
        "unrecognized_fields_ignored": [],
    }
    return config, audit
