#!/usr/bin/env python3
"""Build the KBS2026B manuscript in isolation; retain sources, logs, PDF and receipts."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[2]
LATEX = ROOT / 'latex'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', action='store_true', help='Also compile and package a flat Editorial Manager source archive')
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    work = ROOT / 'work/builds' / stamp
    work.mkdir(parents=True, exist_ok=False)
    source = work / 'source'
    source.mkdir()
    files = [p for p in LATEX.rglob('*') if p.is_file() and p.suffix in
             {'.tex', '.bib', '.sty', '.cls', '.bst', '.pdf', '.png', '.jpg', '.eps'} and p != LATEX / 'main.pdf']
    for p in files:
        dest = source / p.relative_to(LATEX)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
    state = {'status': 'RUNNING', 'at': datetime.now(timezone.utc).isoformat(),
             'source_hashes': {str(p.relative_to(LATEX)): sha(p) for p in files},
             'new_model_requests': 0}
    status = work / 'status.json'

    def save():
        status.write_text(json.dumps(state, indent=2) + '\n')

    def compile_dir(folder, log_name):
        with (work / log_name).open('w') as log:
            subprocess.run(['latexmk', '-pdf', '-interaction=nonstopmode', '-halt-on-error',
                            '-file-line-error', '-recorder', 'main.tex'], cwd=folder,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        log = (folder / 'main.log').read_text(errors='replace')
        bad = [s for s in ['There were undefined references', 'There were undefined citations',
                           'multiply defined', 'LaTeX Error:', 'Overfull \\hbox', 'Overfull \\vbox'] if s in log]
        if bad:
            raise RuntimeError('Compilation quality check: ' + ', '.join(bad))
        pdfinfo = subprocess.check_output(['pdfinfo', str(folder / 'main.pdf')], text=True)
        pages = int(re.search(r'^Pages:\s+(\d+)', pdfinfo, re.M)[1])
        fonts = subprocess.check_output(['pdffonts', str(folder / 'main.pdf')], text=True)
        if 'Type 3' in fonts:
            raise RuntimeError('Type 3 fonts found')
        (work / (log_name + '.pdfinfo.txt')).write_text(pdfinfo)
        (work / (log_name + '.fonts.txt')).write_text(fonts)
        # All manuscript-specific inputs must come from this isolated folder.
        outside = []
        for line in (folder / 'main.fls').read_text().splitlines():
            if not line.startswith('INPUT '):
                continue
            p = Path(line[6:])
            p = (folder / p).resolve() if not p.is_absolute() else p.resolve()
            forbidden = Path(os.environ.get('KBS_FORBID_ROOT', str(ROOT))).resolve()
            if p.is_relative_to(forbidden) and not p.is_relative_to(folder):
                outside.append(str(p))
        if outside:
            raise RuntimeError('External workspace dependencies: ' + str(sorted(set(outside))))
        return pages

    save()
    try:
        state['pages'] = compile_dir(source, 'build.log')
        state['pdf_sha256'] = sha(source / 'main.pdf')
        release = work / 'release'
        release.mkdir(parents=True, exist_ok=False)
        shutil.copy2(source / 'main.pdf', release / 'KBS2026B_Manuscript.pdf')
        with zipfile.ZipFile(release / 'KBS2026B_sources.zip', 'w', compression=zipfile.ZIP_DEFLATED) as z:
            for p in files:
                relative = p.relative_to(LATEX)
                z.write(source / relative, str(relative))
            z.write(source / 'main.bbl', 'main.bbl')
        if args.package:
            flat = work / 'flat_source'
            flat.mkdir()
            names = [str(p.relative_to(LATEX)) for p in files]
            if len({Path(n).name for n in names}) != len(names):
                raise RuntimeError('Flattening would cause a basename collision')
            mapping = {n: Path(n).name for n in names}
            for relative, name in mapping.items():
                p = source / relative
                if p.suffix == '.tex':
                    text = p.read_text()
                    for old, new in sorted(mapping.items(), key=lambda x: -len(x[0])):
                        text = text.replace('{' + old + '}', '{' + new + '}')
                        if old.endswith('.tex'):
                            text = text.replace('{' + old[:-4] + '}', '{' + new[:-4] + '}')
                    (flat / name).write_text(text)
                else:
                    shutil.copy2(p, flat / name)
            flat_pages = compile_dir(flat, 'flat_build.log')
            if flat_pages != state['pages']:
                raise RuntimeError('Flat source page count differs')
            for folder in [source, flat]:
                subprocess.run(['pdftotext', '-layout', str(folder / 'main.pdf'), str(folder / 'paper.txt')], check=True)
            if (source / 'paper.txt').read_bytes() != (flat / 'paper.txt').read_bytes():
                raise RuntimeError('Flat source PDF text differs')
            with zipfile.ZipFile(release / 'KBS2026B_EM_flat.zip', 'w', compression=zipfile.ZIP_DEFLATED) as z:
                for name in mapping.values():
                    z.write(flat / name, name)
                z.write(flat / 'main.bbl', 'main.bbl')
            state['flat_package'] = {'status': 'PASS', 'pages': flat_pages, 'pdf_text_equal': True}
        # Publish the local convenience PDF only after all requested checks pass.
        shutil.copy2(source / 'main.pdf', LATEX / 'main.pdf')
        state.update(status='PASS', finished_at=datetime.now(timezone.utc).isoformat(),
                     release=str(release.relative_to(ROOT)))
        save()
        (release / 'build_receipt.json').write_text(json.dumps(state, indent=2) + '\n')
        (ROOT / 'work/latest_build.json').write_text(json.dumps({'receipt': str(status.relative_to(ROOT)), **state}, indent=2) + '\n')
        print(json.dumps({'status': 'PASS', 'pages': state['pages'], 'pdf': 'latex/main.pdf',
                          'release': str(release.relative_to(ROOT)), 'log': str((work / 'build.log').relative_to(ROOT))}))
    except BaseException as error:
        state.update(status='FAILED', error=repr(error))
        save()
        print('Build failed; sources and logs retained at ' + str(work))
        raise


if __name__ == '__main__':
    main()
