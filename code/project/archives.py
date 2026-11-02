"""Verified, on-demand extraction of the immutable experiment records."""
from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tarfile
import uuid

from workspace import ROOT, EVIDENCE, bounded_map, io_resources, read_json, sha256, write_json


def safe_member(name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or '..' in path.parts or '\\' in name or str(path) != name:
        raise ValueError('Unsafe archive member: ' + repr(name))
    return path


def extract_checked(archive, target, expected):
    """Extract regular files only, verifying every member before it becomes usable."""
    target = Path(target)
    seen = set()
    with tarfile.open(archive, 'r:gz') as stream:
        for item in stream:
            path = safe_member(item.name)
            if not item.isfile() or item.name in seen or item.name not in expected:
                raise ValueError('Unexpected archive member: ' + item.name)
            row = expected[item.name]
            if item.size != row['bytes']:
                raise ValueError('Archive size mismatch: ' + item.name)
            destination = target.joinpath(*path.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            digest = hashlib.sha256()
            with stream.extractfile(item) as source, destination.open('xb') as output:
                for block in iter(lambda: source.read(1024 * 1024), b''):
                    output.write(block)
                    digest.update(block)
            if digest.hexdigest() != row['sha256']:
                raise ValueError('Archive content mismatch: ' + item.name)
            seen.add(item.name)
    if seen != set(expected):
        raise ValueError('Archive is missing declared members')


def catalog():
    manifest = read_json(ROOT / 'data/manifest.json')
    path = ROOT / manifest['members']['path']
    if sha256(path) != manifest['members']['sha256']:
        raise ValueError('Changed data member manifest')
    with gzip.open(path, 'rt') as stream:
        members = [json.loads(line) for line in stream]
    if len({r['path'] for r in members}) != len(members):
        raise ValueError('Duplicate data member')
    return manifest, members


def verify_files(rows, root=EVIDENCE):
    plan = io_resources()

    def verify(row):
        path = root.joinpath(*safe_member(row['path']).parts)
        if path.is_symlink() or not path.is_file() or path.stat().st_size != row['bytes'] or sha256(path) != row['sha256']:
            return row['path']
    with ThreadPoolExecutor(plan['workers']) as pool:
        errors = [name for name in bounded_map(pool, verify, rows) if name]
    if errors:
        raise ValueError('Changed extracted records: ' + repr(errors[:10]))


def package_file(package, root=None):
    """Validate a local package and explain how to obtain missing external data."""
    archive = (Path(root) if root is not None else ROOT) / package['path']
    if archive.is_symlink():
        raise ValueError('Evidence package must be a regular file: ' + package['path'])
    if not archive.is_file():
        message = 'Missing evidence package: ' + package['path']
        if package.get('download_url'):
            message += '\nDownload from: ' + package['download_url']
            message += '\nSave as: ' + str(archive)
            message += '\nExpected bytes: ' + str(package['bytes']) + '; SHA-256: ' + package['sha256']
        raise FileNotFoundError(message)
    if archive.stat().st_size != package['bytes'] or sha256(archive) != package['sha256']:
        raise ValueError('Changed evidence package: ' + package['name'])
    return archive


def prepare(names=None):
    manifest, members = catalog()
    names = set(names or (p['name'] for p in manifest['packages']))
    if names - {p['name'] for p in manifest['packages']}:
        raise ValueError('Unknown experiment package')
    selected = [(p, package_file(p)) for p in manifest['packages'] if p['name'] in names]
    EVIDENCE.mkdir(parents=True, exist_ok=True)
    for package, archive in selected:
        rows = [r for r in members if r['package'] == package['name']]
        if len(rows) != package['files'] or sum(r['bytes'] for r in rows) != package['unpacked_bytes']:
            raise ValueError('Package manifest count mismatch')
        prefix = safe_member(package['prefix'])
        if any(not PurePosixPath(r['path']).is_relative_to(prefix) for r in rows):
            raise ValueError('Data member leaves its package prefix')
        destination = EVIDENCE.joinpath(*prefix.parts)
        receipt = EVIDENCE / ('.' + package['name'] + '.json')
        if destination.exists():
            if not receipt.is_file() or read_json(receipt).get('archive_sha256') != package['sha256']:
                raise ValueError('Unrecognized extraction; preserve it before preparing a new copy: ' + str(destination))
            verify_files(rows)
            continue
        temporary = EVIDENCE / ('.extract-' + package['name'] + '-' + uuid.uuid4().hex[:8])
        temporary.mkdir()
        try:
            extract_checked(archive, temporary, {r['path']: r for r in rows})
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary.joinpath(*prefix.parts).rename(destination)
            write_json(receipt, {'archive_sha256':package['sha256'], 'files':len(rows)})
            shutil.rmtree(temporary)
        except BaseException as error:
            write_json(temporary / 'failure.json', {'status':'FAIL', 'error':repr(error)})
            raise
    return EVIDENCE
