#!/usr/bin/env python3
import subprocess


def run(cmd):
    print('[run]', ' '.join(cmd))
    subprocess.check_call(cmd)


def main():
    # Safe non-interactive cleanup of common space hogs
    run(['docker', 'image', 'prune', '-f'])
    run(['docker', 'container', 'prune', '-f'])
    run(['docker', 'builder', 'prune', '-f'])
    # Show space after cleanup
    run(['docker', 'system', 'df'])


if __name__ == '__main__':
    main()


