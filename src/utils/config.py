import copy
import yaml


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _set_nested(d: dict, dotted_key: str, value):
    keys = dotted_key.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur[k]
    leaf = keys[-1]
    old = cur.get(leaf)
    # 원래 타입을 최대한 유지해서 파싱 (yaml.safe_load로 bool/int/float/str 자동 판별)
    parsed = yaml.safe_load(value)
    cur[leaf] = parsed
    return old, parsed


def apply_overrides(cfg: dict, overrides: list) -> dict:
    """
    overrides: ["experiment.split_seed=1", "data.max_modality_dropout=0.3", ...]
    """
    cfg = copy.deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--set 값은 key=value 형태여야 합니다: {item}")
        key, value = item.split("=", 1)
        old, new = _set_nested(cfg, key.strip(), value.strip())
        print(f"[config override] {key} : {old} -> {new}")
    return cfg
