"""The kernel layer of the sandbox. Two kinds of test, for two kinds of claim.

**The filter's logic**, everywhere: a small interpreter runs the exact BPF program
`containment.program()` produces against synthetic syscalls and checks each verdict.
That needs no kernel, so it runs on macOS too, and it is what catches a wrong jump
offset or a missing syscall number.

**The kernel's enforcement**, on Linux only: a child process applies the real
containment and then runs *unrestricted* Python — as though every in-process layer
had already been escaped — and reports what the kernel let it do. This is the claim
the sandbox actually rests on, so it is tested without the layers that are not
claimed to be sound.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from research_agent.tools import containment

LINUX = sys.platform.startswith("linux")
ALLOW = containment.SECCOMP_RET_ALLOW
DENY = containment.SECCOMP_RET_ERRNO | containment.EPERM
KILL = containment.SECCOMP_RET_KILL_PROCESS


def verdict(arch: str, nr: int, args: tuple[int, ...] = (), audit_arch: int | None = None) -> int:
    """Interpret the filter for one syscall, the way the kernel would."""
    data = {
        0: nr & 0xFFFFFFFF,
        4: containment.AUDIT_ARCH[arch] if audit_arch is None else audit_arch,
    }
    for index, value in enumerate(args):
        data[16 + 8 * index] = value & 0xFFFFFFFF

    program = containment.program(arch)
    accumulator = 0
    pc = 0
    while pc < len(program):
        code, jt, jf, k = program[pc]
        if code == 0x20:  # ld word at absolute offset
            accumulator = data.get(k, 0)
            pc += 1
        elif code in (0x15, 0x35, 0x45):  # jeq / jge / jset
            taken = {
                0x15: accumulator == k,
                0x35: accumulator >= k,
                0x45: bool(accumulator & k),
            }[code]
            pc += 1 + (jt if taken else jf)
        elif code == 0x06:  # ret
            return k
        else:
            raise AssertionError(f"unexpected opcode {code:#x}")
    raise AssertionError("fell off the end of the program")


# --- the filter's logic, on every platform ------------------------------------------


@pytest.mark.parametrize("arch", sorted(containment.AUDIT_ARCH))
def test_every_forbidden_syscall_is_refused(arch: str) -> None:
    for name, number in containment.DENIED[arch].items():
        assert verdict(arch, number) == DENY, f"{name} ({number}) must be refused on {arch}"


@pytest.mark.parametrize(
    ("arch", "harmless"),
    [
        ("x86_64", {"read": 0, "write": 1, "close": 3, "mmap": 9, "brk": 12, "getpid": 39}),
        ("aarch64", {"read": 63, "write": 64, "close": 57, "mmap": 222, "brk": 214, "getpid": 172}),
    ],
)
def test_ordinary_syscalls_are_allowed(arch: str, harmless: dict[str, int]) -> None:
    for name, number in harmless.items():
        assert verdict(arch, number) == ALLOW, f"{name} must stay usable on {arch}"


@pytest.mark.parametrize("arch", sorted(containment.AUDIT_ARCH))
def test_files_open_for_reading_but_not_for_writing(arch: str) -> None:
    for number, flags_arg in containment.OPENERS[arch].items():

        def opened(flags: int, number: int = number, flags_arg: int = flags_arg) -> int:
            args = [0, 0, 0]
            args[flags_arg] = flags
            return verdict(arch, number, tuple(args))

        assert opened(0) == ALLOW, "O_RDONLY must be allowed"
        assert opened(0o2000000) == ALLOW, "O_CLOEXEC on its own is still a read"
        for flag in (0o1, 0o2, 0o100, 0o1000, 0o2000):
            assert opened(flag) == DENY, f"flag {flag:#o} asks to write and must be refused"


def test_the_x32_abi_cannot_be_used_to_get_around_the_filter() -> None:
    # x32 syscalls share x86_64's audit arch with this bit set on the number, so a
    # filter matching only plain numbers would wave `socket | 0x40000000` through.
    socket = containment.DENIED["x86_64"]["socket"]
    assert verdict("x86_64", socket | 0x40000000) == KILL


@pytest.mark.parametrize("arch", sorted(containment.AUDIT_ARCH))
def test_a_syscall_from_another_architecture_is_killed(arch: str) -> None:
    other = next(value for key, value in containment.AUDIT_ARCH.items() if key != arch)
    assert verdict(arch, 0, audit_arch=other) == KILL


def test_an_unknown_architecture_has_no_filter_rather_than_a_wrong_one() -> None:
    with pytest.raises(ValueError):
        containment.program("mips")


def test_off_linux_containment_reports_itself_unavailable() -> None:
    if LINUX:
        pytest.skip("this is the non-Linux path")
    assert containment.contain([]) == {
        "seccomp": False,
        "landlock": False,
        "platform": sys.platform,
    }


# --- the kernel's enforcement, on Linux --------------------------------------------

PROBE = """
import json, os, socket, subprocess, sys
sys.path.insert(0, {src!r})
from research_agent.tools import containment

secret_file = {secret_file!r}
victim = {victim!r}
readable = {readable!r}
report = containment.contain({read_paths!r})
results = {{}}

def attempt(name, action):
    try:
        action()
        results[name] = "allowed"
    except OSError as exc:
        results[name] = "denied"

attempt("open a socket", lambda: socket.socket())
attempt("run a program", lambda: subprocess.run(["/bin/true"], check=False))
attempt("create a file", lambda: open(os.path.join({scratch!r}, "new"), "w"))
attempt("delete a file", lambda: os.unlink(victim))
attempt("make a directory", lambda: os.mkdir(os.path.join({scratch!r}, "dir")))
attempt("signal the parent", lambda: os.kill(os.getppid(), 0))
attempt("read the parent's environment", lambda: open("/proc/%d/environ" % os.getppid()).read())
attempt("read a file outside the allowed paths", lambda: open(secret_file).read())
attempt("read an allowed file", lambda: open(readable).read())
print(json.dumps({{"report": report, "results": results}}))
"""


@pytest.mark.skipif(not LINUX, reason="seccomp and Landlock are Linux kernel features")
def test_a_complete_escape_is_still_contained_by_the_kernel(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    victim = scratch / "victim"
    victim.write_text("leave me")
    outside = tmp_path / "outside"
    outside.mkdir()
    secret_file = outside / "secret"
    secret_file.write_text("sk-not-for-you")
    readable = scratch / "readable"
    readable.write_text("fine")

    # The parent is protected the way code_exec protects the app.
    containment.protect_parent()
    src = str(Path(containment.__file__).parents[2])
    import sysconfig

    paths = sysconfig.get_paths()
    probe = PROBE.format(
        src=src,
        secret_file=str(secret_file),
        victim=str(victim),
        readable=str(readable),
        scratch=str(scratch),
        read_paths=[paths["stdlib"], paths["platstdlib"], str(scratch)],
    )
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(probe)],
        capture_output=True,
        text=True,
        timeout=60,
        env={"PATH": "/usr/bin:/bin", "SECRET_IN_PARENT_ONLY": "x"},
    )
    assert completed.returncode == 0, completed.stderr
    outcome = json.loads(completed.stdout.strip().splitlines()[-1])
    report, results = outcome["report"], outcome["results"]

    assert report["seccomp"] is True
    for forbidden in (
        "open a socket",
        "run a program",
        "create a file",
        "delete a file",
        "make a directory",
        "signal the parent",
        "read the parent's environment",
    ):
        assert results[forbidden] == "denied", f"{forbidden}: {results}"
    assert results["read an allowed file"] == "allowed", "containment must not break reading"
    assert victim.read_text() == "leave me"
    if report["landlock"]:
        assert results["read a file outside the allowed paths"] == "denied"


@pytest.mark.skipif(not LINUX, reason="seccomp is a Linux kernel feature")
def test_code_execution_reports_the_layers_that_applied() -> None:
    from research_agent.tools.code_exec import execute

    result = execute("print(sum(range(10)))")
    assert result.ok and result.content == "45"
    assert result.meta["isolation"]["seccomp"] is True


def test_containment_failure_means_the_code_does_not_run(monkeypatch: pytest.MonkeyPatch) -> None:
    # Fail closed: a host that cannot contain model-written code does not run it.
    from research_agent.tools import code_exec

    class Refused:
        returncode = code_exec.CONTAINMENT_FAILED_EXIT

        def __init__(self, *args: object, **kwargs: object) -> None:
            pass

        def communicate(self, timeout: float | None = None) -> tuple[str, str]:
            return "", "containment could not be established: no seccomp"

    monkeypatch.setattr(code_exec.subprocess, "Popen", Refused)
    result = code_exec.execute("print(1)")
    assert not result.ok and "unavailable" in (result.error or "")


def test_a_secret_that_reaches_stdout_is_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    import base64

    from research_agent.tools.code_exec import _redact

    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-abcdefghijklmnop")
    leaked = "sk-proj-abcdefghijklmnop"
    for form in (leaked, base64.b64encode(leaked.encode()).decode(), leaked[::-1]):
        assert "[redacted]" in _redact(f"here: {form}"), form
    assert _redact("nothing secret") == "nothing secret"
    assert os.environ["OPENAI_API_KEY"] == leaked
