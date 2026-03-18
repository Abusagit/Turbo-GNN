def pytest_make_parametrize_id(config, val, argname):
    """Generate short, readable IDs for parametrized test values."""
    import torch

    # torch.dtype → short name
    dtype_names = {
        torch.float32: "fp32",
        torch.float16: "fp16",
        torch.bfloat16: "bf16",
        torch.int32: "i32",
        torch.int64: "i64",
        torch.uint32: "u32",
    }
    if isinstance(val, torch.dtype):
        return dtype_names.get(val, str(val).split(".")[-1])

    # bool → argname / no_argname
    if isinstance(val, bool):
        return f"{argname}" if val else f"no_{argname}"

    # int/float → prefix with argname
    if isinstance(val, (int, float)):
        return f"{argname}={val}"

    # str → use as-is (already readable)
    if isinstance(val, str):
        return val

    # Everything else: let pytest use its default
    return None
