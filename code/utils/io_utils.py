# -*- coding: utf-8 -*-
"""I/O helpers: JSON / CSV read-write and package-relative path resolution."""
import csv
import json
import os

PKG_ROOT = os.environ.get(
    "CFLG_ROOT",
    os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "..", "..")))


def root(*parts):
    """Path inside the package root."""
    return os.path.join(PKG_ROOT, *parts)


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_json(path, obj, indent=2):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)
    return path


def read_csv(path, encoding="utf-8-sig"):
    with open(path, newline="", encoding=encoding) as f:
        return list(csv.DictReader(f))


def write_csv(path, header, rows, encoding="utf-8-sig"):
    ensure_dir(os.path.dirname(path))
    with open(path, "w", newline="", encoding=encoding) as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)
    return path
