#!/usr/bin/env python3
"""Verify the curated files, replay saved data, and run the KBS correctness tests."""
import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import gzip
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import unittest

from workspace import ROOT, bounded_map, io_resources, new_run, now, project_path, read_json, sha256, write_json


def is_local_artifact(path):
    """Recognize local settings, caches and rebuildable TeX files."""
    if any(part.startswith('.') or part in {'__pycache__', 'venv', 'htmlcov'} for part in path.parts):
        return True
    if path.name == 'Thumbs.db' or path.name.endswith(('~', '.swp', '.swo', '.pyc', '.pyo', '.pyd')):
        return True
    if path.parts[0] == 'latex':
        if path.name == 'build.sh':
            return True
        if len(path.parts) > 2 and (path.parts[1] in {'build', 'out'} or path.parts[1].startswith('_minted-')):
            return True
        return path.name.endswith((
            '.aux', '.bbl', '.bcf', '.blg', '.fdb_latexmk', '.fls', '.log', '.out',
            '.run.xml', '.synctex', '.synctex.gz', '.spl', '.toc', '.lof', '.lot',
            '.loa', '.lol', '.nav', '.snm', '.vrb', '.idx', '.ilg', '.ind',
            '.acn', '.acr', '.alg', '.glg', '.glo', '.gls', '.dvi', '.xdv',
        ))
    return False


def manifest_check(output):
    manifest_path = ROOT / 'code/project/manifest.json'
    manifest = read_json(manifest_path)
    rows = manifest['files']
    expected = {row['path'] for row in rows}
    actual = {str(p.relative_to(ROOT)) for folder in ['code','data','docs','latex']
              for p in (ROOT / folder).rglob('*') if p.is_file() and p != ROOT/'latex/main.pdf'}
    actual |= {p.name for p in ROOT.iterdir() if p.is_file()}
    actual.discard(str(manifest_path.relative_to(ROOT)))
    actual = {name for name in actual if name in expected or not is_local_artifact(Path(name))}
    external = {p['path'] for p in read_json(ROOT / 'data/manifest.json')['packages'] if p.get('download_url')}
    missing_external = (expected - actual) & external
    if actual - expected or (expected - actual) - external:
        raise ValueError({'unlisted':sorted(actual-expected), 'missing':sorted((expected-actual)-external)})
    if any(project_path(name).is_symlink() for name in missing_external):
        raise ValueError('External evidence package must not be a symlink')
    available = [row for row in rows if row['path'] not in missing_external]
    plan = io_resources()
    write_json(output / 'resources.json', plan)
    def verify(row):
        path = project_path(row['path'])
        if path.is_symlink() or path.stat().st_size != row['bytes'] or sha256(path) != row['sha256']:
            return row['path']
    with ThreadPoolExecutor(plan['workers']) as pool:
        errors = [p for p in bounded_map(pool, verify, available) if p]
    if errors:
        raise ValueError('Changed packaged files: ' + repr(errors[:20]))
    bindings = read_json(ROOT / 'code/source_bindings.json')['files']
    for row in bindings:
        if sha256(project_path(row['path'])) != row['packaged_sha256']:
            raise ValueError('Changed frozen dependency: ' + row['path'])
    for row in rows:
        if not row['path'].startswith('code/') or not row['path'].endswith('.py'):
            continue
        path = project_path(row['path'])
        ast.parse(path.read_text(), filename=str(path))
    return {'files':len(available),'bytes':sum(row['bytes'] for row in available),
            'source_files':len(bindings),'hashes':'PASS',
            'external_packages_missing':sorted(missing_external)}


def tests(output):
    sys.path.insert(0, str(ROOT / 'code/tests'))
    suite = unittest.defaultTestLoader.discover(str(ROOT / 'code/tests'))
    with (output / 'tests.log').open('w') as log:
        result = unittest.TextTestRunner(stream=log, verbosity=2).run(suite)
    summary = dict(run=result.testsRun, failed=len(result.failures), errors=len(result.errors), skipped=len(result.skipped))
    write_json(output / 'tests.json', summary)
    if not result.wasSuccessful():
        raise ValueError('Correctness tests failed; see tests.log')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hashes-only', action='store_true')
    parser.add_argument('--data-only', action='store_true', help='Replay saved records without the test suite')
    parser.add_argument('--tests-only', action='store_true')
    args = parser.parse_args()
    output = new_run('checks')
    write_json(output / 'resources.json', io_resources())
    dependencies = {}
    for package in ['numpy','matplotlib','scipy','pandas','jsonschema']:
        try:
            dependencies[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            dependencies[package] = None
    state = {'status':'RUNNING', 'started_at':now(), 'directory':str(output.relative_to(ROOT)),
             'python':sys.version, 'platform':platform.platform(), 'new_model_requests':0,
             'dependencies':dependencies}
    print(output, flush=True)
    write_json(output / 'status.json', state)
    try:
        missing = [name for name, version in dependencies.items() if version is None]
        if missing and not args.hashes_only:
            raise RuntimeError('Install code/requirements-analysis.txt in the selected environment; missing: ' + ', '.join(missing))
        if not args.data_only and not args.tests_only:
            state['manifest'] = manifest_check(output)
            print('File manifest PASS', flush=True)
            if state['manifest']['external_packages_missing']:
                print('External packages not downloaded: ' + ', '.join(state['manifest']['external_packages_missing'])
                      + '; see data/README.md', flush=True)
        if not args.hashes_only and not args.tests_only:
            from verify_data import run
            state['data'] = run(output)
        if not args.hashes_only and not args.data_only:
            state['tests'] = tests(output)
        state.update(status='PASS', finished_at=now())
    except BaseException as error:
        state.update(status='FAIL', finished_at=now(), error=repr(error))
        raise
    finally:
        write_json(output / 'status.json', state)
        write_json(ROOT / 'work/latest_check.json', state)
        print(json.dumps({'status':state['status'], 'directory':str(output.relative_to(ROOT))}), flush=True)


if __name__ == '__main__':
    main()
