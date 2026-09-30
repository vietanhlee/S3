"""
01_data_article.modules.curation
================================
Module quản lý sinh siêu dữ liệu, phân vùng chống rò rỉ mẫu vật, và kiểm định trùng lặp quang học.
"""


def generate_assets(*args, **kwargs):
    from .asset_generator import generate_assets as _fn
    return _fn(*args, **kwargs)


def run_leakage_audit(*args, **kwargs):
    from .leakage_auditor import run_leakage_audit as _fn
    return _fn(*args, **kwargs)


def generate_disjoint_split(*args, **kwargs):
    from .disjoint_splitter import generate_disjoint_split as _fn
    return _fn(*args, **kwargs)


__all__ = ["generate_assets", "run_leakage_audit", "generate_disjoint_split"]
