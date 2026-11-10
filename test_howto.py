from howto import danger_warnings

# command → should it show a warning?
cases = {
    "find /tmp -type d -empty -exec rmdir {} \\;": True,
    "rm -rf build": True,
    "find . -name '*.log' -delete": True,
    "echo '* * * * * date' | crontab": True,
    "sudo apt update": True,
    "ls -la": False,
    "find . -name '*.png'": False,
    "df -h": False,
}

for cmd, should_warn in cases.items():
    warned = bool(danger_warnings(cmd))
    print("OK  " if warned == should_warn else "FAIL", cmd)
