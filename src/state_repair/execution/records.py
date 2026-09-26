"""Lossless sealed-log compression, preserving the checkpoint's raw byte hash."""
from pathlib import Path
import gzip
import hashlib
import json
import shutil
from .durable import atomic_write


def open_steps(output: Path):
    plain=output/'steps.jsonl'
    return plain.open('rb') if plain.exists() else gzip.open(output/'steps.jsonl.gz','rb')


def step_rows(output: Path):
    with open_steps(output) as handle:
        for line in handle:
            yield json.loads(line)


def step_bytes(output: Path) -> bytes:
    with open_steps(output) as handle:
        return handle.read()


def raw_step_hash(output: Path) -> str:
    digest=hashlib.sha256()
    with open_steps(output) as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):
            digest.update(block)
    return digest.hexdigest()


def stored_steps(output: Path) -> Path:
    return output/('steps.jsonl' if (output/'steps.jsonl').exists() else 'steps.jsonl.gz')


def compress_steps(output: Path) -> None:
    plain=output/'steps.jsonl'
    expected=raw_step_hash(output)
    def write(handle):
        with plain.open('rb') as source, gzip.GzipFile(fileobj=handle,mode='wb',compresslevel=1,mtime=0) as target:
            shutil.copyfileobj(source,target)
    atomic_write(output/'steps.jsonl.gz',write)
    digest=hashlib.sha256()
    with gzip.open(output/'steps.jsonl.gz','rb') as handle:
        for block in iter(lambda:handle.read(1024*1024),b''):
            digest.update(block)
    if digest.hexdigest()!=expected:
        raise ValueError('compressed log does not preserve raw bytes')
    plain.unlink()
