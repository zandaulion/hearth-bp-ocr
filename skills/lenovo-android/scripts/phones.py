# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (C) 2026 zandaulion
"""Named Android devices over trusted SSH; stdlib only, no local ADB required."""
import argparse
import json
from pathlib import Path
import re
import shlex
import socket
import subprocess
import sys
import xml.etree.ElementTree as ET

PHONES = {
    'a52': {'model': 'SM_A525F', 'remote_web': 18875, 'remote_debug': 19223, 'local_debug': 19333},
    'fold4': {'model': 'SM_F936B', 'remote_web': 18876, 'remote_debug': 19224, 'local_debug': 19334},
    'poco': {'model': '24117RK2CG', 'remote_web': 18877, 'remote_debug': 19225, 'local_debug': 19335},
}


def ssh_base(host):
    return ['ssh', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes',
            '-o', 'ConnectTimeout=10', host]


def remote(host, words, timeout=30):
    result = subprocess.run(ssh_base(host) + [shlex.join(list(map(str, words)))],
                            capture_output=True, timeout=timeout)
    if result.returncode:
        raise RuntimeError(result.stderr.decode(errors='replace').strip() or
                           result.stdout.decode(errors='replace').strip() or
                           f'Remote command exited {result.returncode}')
    return result.stdout


def parse_devices(output):
    result = []
    for line in output.splitlines():
        parts = line.split()
        if len(parts) < 2 or parts[0] == 'List' or line.startswith('*'):
            continue
        details = dict(p.split(':', 1) for p in parts[2:] if ':' in p)
        aliases = [name for name, spec in PHONES.items() if spec['model'] == details.get('model')]
        result.append({'serial': parts[0], 'state': parts[1], **details,
                       'alias': aliases[0] if len(aliases) == 1 else None,
                       'connection': 'usb' if 'usb' in details else 'wireless' if '_adb-tls-' in parts[0] or ':' in parts[0] else 'unknown'})
    return result


def select_device(devices, selector):
    matches = [d for d in devices if d['alias'] == selector or d['serial'] == selector]
    if len(matches) != 1:
        raise ValueError(f'Expected one {selector!r}, found {len(matches)}. Run list; select an exact current serial if ambiguous. No other phone was selected.')
    if matches[0]['state'] != 'device':
        raise ValueError(f"{selector} is {matches[0]['state']}; unlock/authorize or reconnect it on Lenovo, then run list again.")
    return matches[0]


def adb_words(serial, words, shell=False):
    # adb shell rejoins arguments: preserve Android quoting as ONE shell string.
    return ['adb', '-s', serial, 'shell', shlex.join(list(map(str, words)))] if shell else ['adb', '-s', serial, *map(str, words)]


def bounds(node):
    values = list(map(int, re.findall(r'\d+', node.get('bounds', ''))))
    return values if len(values) == 4 and values[2] > values[0] and values[3] > values[1] else None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--host', default='lenovo')
    parser.add_argument('--phone', help='a52, fold4, poco, or an exact current ADB serial')
    parser.add_argument('--dry-run', action='store_true', help='Discover/select, but do not perform the requested mutation')
    commands = parser.add_subparsers(dest='action', required=True)
    for name in ('list', 'info', 'snapshot', 'foreground-chrome'):
        commands.add_parser(name)
    p = commands.add_parser('open'); p.add_argument('url')
    p = commands.add_parser('tap-text'); p.add_argument('text')
    p = commands.add_parser('tap'); p.add_argument('x', type=int); p.add_argument('y', type=int)
    p = commands.add_parser('swipe')
    for name in ('x1', 'y1', 'x2', 'y2', 'duration'):
        p.add_argument(name, type=int)
    p = commands.add_parser('key'); p.add_argument('code', type=int)
    p = commands.add_parser('screenshot'); p.add_argument('output', type=Path)
    p = commands.add_parser('web')
    p.add_argument('--desktop-port', type=int, required=True)
    p.add_argument('--phone-port', type=int, default=8797)
    p.add_argument('--lenovo-port', type=int)
    p.add_argument('--debug-browser', action='store_true')
    p.add_argument('--local-debug-port', type=int)
    p.add_argument('--remote-debug-port', type=int)
    args = parser.parse_args()
    devices = parse_devices(remote(args.host, ['adb', 'devices', '-l']).decode())
    if args.action == 'list':
        print(json.dumps(devices, indent=2)); return
    if not args.phone:
        parser.error('--phone is required for every device operation')
    device = select_device(devices, args.phone)
    serial = device['serial']

    def adb(*words, shell=False):
        command = adb_words(serial, words, shell)
        if args.dry_run:
            print(json.dumps({'device': device, 'remote_command': shlex.join(command)})); return b''
        return remote(args.host, command)

    def tree():
        remote(args.host, adb_words(serial, ['uiautomator', 'dump', '/data/local/tmp/lenovo-android-ui.xml'], True))
        return ET.fromstring(remote(args.host, adb_words(serial, ['cat', '/data/local/tmp/lenovo-android-ui.xml'], True)))

    if args.action == 'info':
        if args.dry_run:
            print(json.dumps({'device': device, 'action': 'info', 'read_only': True})); return
        properties = ('ro.product.model', 'ro.build.version.release', 'ro.build.version.sdk')
        report = {'device': device, 'properties': {key: adb('getprop', key, shell=True).decode().strip() for key in properties},
                  'display': adb('wm', 'size', shell=True).decode().strip(),
                  'density': adb('wm', 'density', shell=True).decode().strip()}
        print(json.dumps(report, indent=2)); return
    if args.action == 'snapshot':
        if args.dry_run:
            print(json.dumps({'device': device, 'action': 'snapshot'})); return
        nodes = [{'text': n.get('text'), 'description': n.get('content-desc'), 'bounds': n.get('bounds'),
                  'resource': n.get('resource-id'), 'clickable': n.get('clickable'), 'enabled': n.get('enabled')}
                 for n in tree().iter('node') if bounds(n) and (n.get('text') or n.get('content-desc'))]
        print(json.dumps({'device': device, 'nodes': nodes}, indent=2)); return
    if args.action == 'tap-text':
        if args.dry_run:
            print(json.dumps({'device': device, 'action': 'tap-text', 'text': args.text})); return
        matches = [n for n in tree().iter('node') if n.get('text') == args.text and n.get('enabled') == 'true' and bounds(n)]
        clickable = [n for n in matches if n.get('clickable') == 'true']
        matches = clickable or matches
        if len(matches) != 1:
            raise ValueError(f'Expected one visible text match, got {len(matches)}; take a fresh snapshot before using coordinates.')
        x1, y1, x2, y2 = bounds(matches[0]); adb('input', 'tap', (x1+x2)//2, (y1+y2)//2, shell=True)
    elif args.action == 'open':
        from urllib.parse import urlsplit
        if urlsplit(args.url).scheme not in ('http', 'https'):
            raise ValueError('open accepts http/https URLs')
        output = adb('am', 'start', '-W', '-a', 'android.intent.action.VIEW', '-d', args.url, 'com.android.chrome', shell=True)
        if not args.dry_run: print(output.decode().strip())
    elif args.action == 'foreground-chrome':
        adb('am', 'start', '-a', 'android.intent.action.MAIN', '-c', 'android.intent.category.LAUNCHER', '-p', 'com.android.chrome', shell=True)
    elif args.action == 'tap':
        if min(args.x, args.y) < 0: raise ValueError('Coordinates must be nonnegative')
        adb('input', 'tap', args.x, args.y, shell=True)
    elif args.action == 'swipe':
        if min(args.x1, args.y1, args.x2, args.y2, args.duration) < 0: raise ValueError('Coordinates/duration must be nonnegative')
        adb('input', 'swipe', args.x1, args.y1, args.x2, args.y2, args.duration, shell=True)
    elif args.action == 'key':
        if args.code < 0: raise ValueError('Key code must be nonnegative')
        adb('input', 'keyevent', args.code, shell=True)
    elif args.action == 'screenshot':
        target = args.output.resolve()
        if not target.is_relative_to(Path.cwd().resolve()):
            raise ValueError('Save screenshots inside the current workspace')
        if args.dry_run:
            print(json.dumps({'device': device, 'output': str(target)})); return
        pixels = adb('exec-out', 'screencap', '-p')
        target.parent.mkdir(parents=True, exist_ok=True); target.write_bytes(pixels)
        print(json.dumps({'device': device, 'output': str(target)})); return
    elif args.action == 'web':
        spec = PHONES.get(device['alias'], {})
        remote_web = args.lenovo_port or spec.get('remote_web')
        remote_debug = args.remote_debug_port or spec.get('remote_debug')
        local_debug = args.local_debug_port or spec.get('local_debug')
        ports = [args.desktop_port, args.phone_port, remote_web]
        if args.debug_browser: ports += [remote_debug, local_debug]
        if any(p is None or not 1024 <= p <= 65535 for p in ports):
            raise ValueError('Use valid ports; unknown models need explicit --lenovo-port and optional debug ports')
        mappings = [('reverse', ['tcp:'+str(args.phone_port), 'tcp:'+str(remote_web)])]
        if args.debug_browser: mappings.append(('forward', ['tcp:'+str(remote_debug), 'localabstract:chrome_devtools_remote']))
        for direction, pair in mappings:
            listing = remote(args.host, adb_words(serial, [direction, '--list'])).decode().splitlines()
            for line in listing:
                parts = line.split()
                if len(parts) >= 3 and parts[-2] == pair[0] and (parts[-1] != pair[1] or (direction == 'forward' and parts[0] != serial)):
                    raise ValueError(f'{direction} {pair[0]} already belongs to another mapping; choose another port, do not overwrite it')
        tunnel = ['ssh', '-o', 'BatchMode=yes', '-o', 'StrictHostKeyChecking=yes', '-o', 'ExitOnForwardFailure=yes',
                  '-o', 'ServerAliveInterval=15', '-N', '-T', '-R', f'127.0.0.1:{remote_web}:127.0.0.1:{args.desktop_port}']
        if args.debug_browser: tunnel += ['-L', f'127.0.0.1:{local_debug}:127.0.0.1:{remote_debug}']
        tunnel.append(args.host)
        print(json.dumps({'device': device, 'phone_url': f'http://127.0.0.1:{args.phone_port}/',
                          'cdp_endpoint': f'http://127.0.0.1:{local_debug}' if args.debug_browser else None,
                          'ssh_argv': tunnel, 'mappings': mappings, 'lifetime': 'foreground; keep the exec session running'}, indent=2), flush=True)
        if args.dry_run: return
        with socket.create_connection(('127.0.0.1', args.desktop_port), timeout=3): pass
        created = []
        try:
            for direction, pair in mappings:
                listing = remote(args.host, adb_words(serial, [direction, '--list'])).decode()
                existed = any(line.split()[-2:] == pair for line in listing.splitlines())
                adb(direction, *pair)
                if not existed: created.append((direction, pair))
            try:
                code = subprocess.call(tunnel)
                if code: raise RuntimeError(f'SSH tunnel exited {code}; do not open the page until forwarding succeeds')
            except KeyboardInterrupt:
                pass
        finally:
            for direction, pair in reversed(created):
                try:
                    listing = remote(args.host, adb_words(serial, [direction, '--list'])).decode()
                    if any(line.split()[-2:] == pair for line in listing.splitlines()):
                        remote(args.host, adb_words(serial, [direction, '--remove', pair[0]]))
                except (RuntimeError, subprocess.TimeoutExpired): pass
        return
    if not args.dry_run: print(json.dumps({'device': device, 'action': args.action, 'completed': True}))


if __name__ == '__main__':
    try: main()
    except (ValueError, RuntimeError, OSError, subprocess.TimeoutExpired) as error:
        print(json.dumps({'error': str(error)}), file=sys.stderr); sys.exit(1)
