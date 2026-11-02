"""Integrity and filesystem boundaries of the evidence-package extractor."""
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest.mock import patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'project'))
from archives import extract_checked, package_file
import check as project_check


class ArchiveIntegrity(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.archive = self.base / 'records.tar.gz'
        self.output = self.base / 'output'
        self.output.mkdir()

    def tearDown(self):
        self.temporary.cleanup()

    def make(self, rows):
        with tarfile.open(self.archive, 'w:gz') as stream:
            for name, content in rows:
                info = tarfile.TarInfo(name)
                info.size = len(content)
                stream.addfile(info, io.BytesIO(content))

    def expected(self, name='main/value.json', content=b'{"n":1}'):
        return {name:{'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()}}

    def test_roundtrip_and_no_overwrite(self):
        self.make([('main/value.json',b'{"n":1}')])
        extract_checked(self.archive,self.output,self.expected())
        self.assertEqual((self.output/'main/value.json').read_bytes(),b'{"n":1}')
        with self.assertRaises(FileExistsError):
            extract_checked(self.archive,self.output,self.expected())

    def test_parent_traversal_rejected(self):
        self.make([('../outside',b'x')])
        with self.assertRaisesRegex(ValueError,'Unsafe'):
            extract_checked(self.archive,self.output,self.expected('../outside',b'x'))
        self.assertFalse((self.base/'outside').exists())

    def test_symlink_rejected(self):
        with tarfile.open(self.archive,'w:gz') as stream:
            info=tarfile.TarInfo('main/value.json');info.type=tarfile.SYMTYPE;info.linkname='/tmp/outside'
            stream.addfile(info)
        with self.assertRaisesRegex(ValueError,'Unexpected'):
            extract_checked(self.archive,self.output,self.expected())

    def test_duplicate_member_rejected(self):
        self.make([('main/value.json',b'{"n":1}')]*2)
        with self.assertRaisesRegex(ValueError,'Unexpected'):
            extract_checked(self.archive,self.output,self.expected())

    def test_missing_record_rejected(self):
        self.make([])
        with self.assertRaisesRegex(ValueError,'missing'):
            extract_checked(self.archive,self.output,self.expected())

    def test_changed_same_size_record_rejected(self):
        self.make([('main/value.json',b'{"n":2}')])
        with self.assertRaisesRegex(ValueError,'content mismatch'):
            extract_checked(self.archive,self.output,self.expected())


class ExternalPackages(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.base = Path(self.temporary.name)
        self.package = {'name':'cpu', 'path':'data/archives/cpu.tar.gz', 'bytes':3,
                        'sha256':hashlib.sha256(b'abc').hexdigest(),
                        'download_url':'https://example.invalid/shared-cpu'}

    def tearDown(self):
        self.temporary.cleanup()

    def test_missing_package_reports_download_and_destination(self):
        with self.assertRaises(FileNotFoundError) as failure:
            package_file(self.package, self.base)
        self.assertIn(self.package['download_url'], str(failure.exception))
        self.assertIn(str(self.base / self.package['path']), str(failure.exception))
        self.assertFalse((self.base / 'data').exists())

    def test_downloaded_package_is_checked_without_modification(self):
        path = self.base / self.package['path']
        path.parent.mkdir(parents=True)
        path.write_bytes(b'abc')
        self.assertEqual(package_file(self.package, self.base), path)
        for content in (b'bad', b'<html>sign in</html>'):
            with self.subTest(content=content):
                path.write_bytes(content)
                with self.assertRaisesRegex(ValueError, 'Changed evidence package'):
                    package_file(self.package, self.base)
                self.assertEqual(path.read_bytes(), content)

    def test_manifest_allows_missing_external_but_rejects_wrong_download(self):
        rows = []
        for relative, value in [('data/manifest.json', {'packages':[self.package]}),
                                ('code/source_bindings.json', {'files':[]})]:
            path = self.base / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            content = json.dumps(value).encode()
            path.write_bytes(content)
            rows.append({'path':relative, 'bytes':len(content), 'sha256':hashlib.sha256(content).hexdigest()})
        rows.append({k:self.package[k] for k in ['path','bytes','sha256']})
        manifest = self.base / 'code/project/manifest.json'
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps({'files':rows}))
        output = self.base / 'work'; output.mkdir()
        with patch.object(project_check, 'ROOT', self.base), \
             patch.object(project_check, 'project_path', lambda relative:self.base / relative):
            result = project_check.manifest_check(output)
            self.assertEqual(result['external_packages_missing'], [self.package['path']])
            self.assertEqual(result['files'], 2)
            path = self.base / self.package['path']
            path.parent.mkdir(parents=True)
            path.write_bytes(b'bad')
            with self.assertRaisesRegex(ValueError, 'Changed packaged files'):
                project_check.manifest_check(output)
            path.write_bytes(b'abc')
            self.assertEqual(project_check.manifest_check(output)['external_packages_missing'], [])
