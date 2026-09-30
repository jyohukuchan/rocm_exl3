"""Publication helper regressions: do not remove files/endpoints owned by others."""
import errno
import os
from pathlib import Path
import socket
import tempfile
import unittest
from unittest import mock

from rocm_tools.rdna2 import power_server as power, summarize_tp


class PublicationReviewTests(unittest.TestCase):
    def test_report_temp_collision_preserves_existing_file(self):
        with tempfile.TemporaryDirectory() as td:
            report = power._ReportFile(Path(td) / 'report.json')
            report.create({'original': ['auto', 'auto']})
            other = Path(f'{report.path}.tmp{os.getpid()}')
            other.write_text('unrelated existing file')
            with self.assertRaises(power.PowerServerError):
                report.save({'events': []})
            self.assertEqual(other.read_text(), 'unrelated existing file')

    def test_bind_race_does_not_unlink_another_server(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            paths = [root / 'gpu0', root / 'gpu1']
            for path in paths:
                path.write_text('auto\n')
            endpoint = root / 'power.sock'
            foreign = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            fake = mock.Mock()
            def collide(address):
                foreign.bind(address)
                raise OSError(errno.EADDRINUSE, 'another server won the bind race')
            fake.bind.side_effect = collide
            try:
                with mock.patch.object(power.socket, 'socket', return_value=fake):
                    with self.assertRaises(power.PowerServerError):
                        power.serve(paths, endpoint, root / 'report.json')
                self.assertTrue(endpoint.is_socket())
                self.assertEqual([p.read_text().strip() for p in paths], ['auto', 'auto'])
            finally:
                foreign.close()

    def test_existing_public_directory_is_rejected_without_chmod(self):
        with tempfile.TemporaryDirectory() as td:
            directory = Path(td) / 'shared'
            directory.mkdir()
            directory.chmod(0o777)
            with self.assertRaises(power.PowerServerError):
                power.ensure_private_dir(directory, None)
            self.assertEqual(directory.stat().st_mode & 0o777, 0o777)
            power.ensure_private_dir(Path(td), None)

    def test_pci_device_and_function_bounds(self):
        for invalid in ['0000:43:20.0', '0000:43:00.8']:
            with self.subTest(bdf=invalid), self.assertRaises(power.PowerServerError):
                power.validate_bdfs([invalid, '0000:03:00.0'])
        self.assertEqual(power.validate_bdfs(['ffff:ff:1f.7', '0000:03:00.0']),
                         ['ffff:ff:1f.7', '0000:03:00.0'])

    def test_invalid_group_is_a_report_error(self):
        with self.assertRaises(summarize_tp.ReportError):
            summarize_tp.summarize({'complete': True, 'groups': [None], 'runs': []})
