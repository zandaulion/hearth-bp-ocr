# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 zandaulion
import importlib.util
from pathlib import Path
import shlex
import unittest

spec = importlib.util.spec_from_file_location('phones', Path(__file__).resolve().parents[1] / 'scripts/phones.py')
phones = importlib.util.module_from_spec(spec); spec.loader.exec_module(phones)
LIST = '''List of devices attached
usbserial device usb:1-2 product:a52 model:SM_A525F device:a52q
adb-changing-name._adb-tls-connect._tcp device product:q4q model:SM_F936B device:q4q
'''


class PhonesTests(unittest.TestCase):
    def test_aliases_select_distinct_devices(self):
        devices = phones.parse_devices(LIST)
        self.assertEqual(phones.select_device(devices, 'a52')['serial'], 'usbserial')
        self.assertEqual(phones.select_device(devices, 'fold4')['serial'], 'adb-changing-name._adb-tls-connect._tcp')

    def test_reconnected_wireless_name_is_rediscovered(self):
        devices = phones.parse_devices(LIST.replace('adb-changing-name', 'adb-new-service-name'))
        self.assertEqual(phones.select_device(devices, 'fold4')['serial'], 'adb-new-service-name._adb-tls-connect._tcp')

    def test_missing_device_does_not_select_other_phone(self):
        with self.assertRaises(ValueError): phones.select_device(phones.parse_devices(LIST.split('adb-')[0]), 'fold4')

    def test_duplicate_model_requires_explicit_serial(self):
        devices = phones.parse_devices(LIST + 'second device model:SM_F936B\n')
        with self.assertRaises(ValueError): phones.select_device(devices, 'fold4')
        self.assertEqual(phones.select_device(devices, 'second')['serial'], 'second')

    def test_offline_is_not_usable(self):
        devices = phones.parse_devices('usbserial offline model:SM_A525F\n')
        with self.assertRaises(ValueError): phones.select_device(devices, 'a52')

    def test_nested_android_shell_preserves_literal_url(self):
        url = "http://127.0.0.1:8797/?one=hello world&two=$(not-a-command)&three='quoted'"
        command = phones.adb_words('exact-serial', ['am', 'start', '-d', url], shell=True)
        remote_args = shlex.split(shlex.join(command))
        self.assertEqual(remote_args[:4], ['adb', '-s', 'exact-serial', 'shell'])
        self.assertEqual(shlex.split(remote_args[4]), ['am', 'start', '-d', url])


if __name__ == '__main__': unittest.main()
