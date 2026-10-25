"""Kernel-enforced limits on what model-written code can do — without privileges.

`code_exec` runs Python the model wrote, and the model reads web pages anyone can
write, so the code it runs is ultimately chosen by strangers. The in-process
restrictions in `code_exec` (an import allowlist, restricted builtins, module
proxies) cannot make that safe: CPython offers too many ways back to the full
interpreter — `typing.get_type_hints` alone evaluates strings and bypasses every
static check — and in-process Python sandboxes have no record of staying sound.

What can be made sound is what the process is allowed to ask the kernel for. Both
mechanisms below are available to an unprivileged process once it has set
`PR_SET_NO_NEW_PRIVS`, which is why the old docstring's claim that "real isolation
requires CAP_SYS_ADMIN" was wrong:

- **seccomp-bpf** refuses, by syscall number: opening any socket, running any
  program, signalling or tracing another process, and creating, writing, renaming
  or deleting any file. A complete Python-level escape still cannot phone home,
  spawn a shell, kill the app, or leave anything behind.
- **Landlock** confines reads to the standard library and the scratch directory.
  That is what keeps every `/proc/<pid>/environ` — and so every API key — out of
  reach, including those of processes other than the immediate parent.

This module is copied into the child's scratch directory and imported by the
bootstrap before the model's code runs, so it is stdlib-only and import-safe on
every platform. On anything but Linux both layers report themselves unavailable
and `code_exec` records that in its result; on Linux a missing seccomp is fatal and
the code does not run.
"""

import ctypes
import os
import platform
import sys
from typing import Any

PR_SET_DUMPABLE = 4
PR_SET_SECCOMP = 22
PR_SET_NO_NEW_PRIVS = 38
SECCOMP_MODE_FILTER = 2

SECCOMP_RET_KILL_PROCESS = 0x80000000
SECCOMP_RET_ERRNO = 0x00050000
SECCOMP_RET_ALLOW = 0x7FFF0000
EPERM = 1

# Classic BPF, the only form seccomp accepts.
_LD_W_ABS = 0x20  # BPF_LD | BPF_W | BPF_ABS
_JEQ_K = 0x15  # BPF_JMP | BPF_JEQ | BPF_K
_JGE_K = 0x35  # BPF_JMP | BPF_JGE | BPF_K
_JSET_K = 0x45  # BPF_JMP | BPF_JSET | BPF_K
_RET_K = 0x06  # BPF_RET | BPF_K

# Offsets into struct seccomp_data { int nr; u32 arch; u64 ip; u64 args[6]; }.
# Arguments are 64-bit; on little-endian hosts the low word sits at the offset.
_NR, _ARCH = 0, 4


def _arg(index: int) -> int:
    return 16 + 8 * index


AUDIT_ARCH = {"x86_64": 0xC000003E, "aarch64": 0xC00000B7}

# On x86_64 the x32 ABI reuses the same audit arch with this bit set on the syscall
# number. A filter that only matches plain numbers can be bypassed through it, so
# anything carrying the bit is killed outright.
_X32_SYSCALL_BIT = 0x40000000

# What contained code may never do, by syscall number. Forking is deliberately not
# here: glibc forks through clone(2), whose flags decide thread versus process and
# cannot be told apart cheaply. RLIMIT_NPROC bounds it, every child inherits this
# filter, and none of them can exec.
DENIED: dict[str, dict[str, int]] = {
    "x86_64": {
        "socket": 41,
        "connect": 42,
        "accept": 43,
        "sendto": 44,
        "sendmsg": 46,
        "bind": 49,
        "listen": 50,
        "socketpair": 53,
        "accept4": 288,
        "execve": 59,
        "execveat": 322,
        "kill": 62,
        "tkill": 200,
        "tgkill": 234,
        "pidfd_send_signal": 424,
        "ptrace": 101,
        "process_vm_readv": 310,
        "process_vm_writev": 311,
        "pidfd_getfd": 438,
        "truncate": 76,
        "rename": 82,
        "mkdir": 83,
        "rmdir": 84,
        "creat": 85,
        "link": 86,
        "unlink": 87,
        "symlink": 88,
        "chmod": 90,
        "fchmod": 91,
        "chown": 92,
        "fchown": 93,
        "lchown": 94,
        "mknod": 133,
        "mkdirat": 258,
        "mknodat": 259,
        "fchownat": 260,
        "unlinkat": 263,
        "renameat": 264,
        "linkat": 265,
        "symlinkat": 266,
        "fchmodat": 268,
        "renameat2": 316,
        "fchmodat2": 452,
        "io_uring_setup": 425,
        "io_uring_enter": 426,
        "io_uring_register": 427,
        "openat2": 437,
    },
    "aarch64": {
        "socket": 198,
        "socketpair": 199,
        "bind": 200,
        "listen": 201,
        "accept": 202,
        "connect": 203,
        "sendto": 206,
        "sendmsg": 211,
        "accept4": 242,
        "execve": 221,
        "execveat": 281,
        "kill": 129,
        "tkill": 130,
        "tgkill": 131,
        "pidfd_send_signal": 424,
        "ptrace": 117,
        "process_vm_readv": 270,
        "process_vm_writev": 271,
        "pidfd_getfd": 438,
        "mknodat": 33,
        "mkdirat": 34,
        "unlinkat": 35,
        "symlinkat": 36,
        "linkat": 37,
        "renameat": 38,
        "truncate": 45,
        "fchmod": 52,
        "fchmodat": 53,
        "fchownat": 54,
        "fchown": 55,
        "renameat2": 276,
        "fchmodat2": 452,
        "io_uring_setup": 425,
        "io_uring_enter": 426,
        "io_uring_register": 427,
        "openat2": 437,
    },
}

# Syscalls that open files, mapped to the argument holding their flags. They are
# allowed for reading and refused if any of these flags asks to write or create.
OPENERS: dict[str, dict[int, int]] = {
    "x86_64": {2: 1, 257: 2},  # open(path, flags, ...), openat(dirfd, path, flags, ...)
    "aarch64": {56: 2},  # openat only; aarch64 has no open(2)
}
WRITE_FLAGS = 0o1 | 0o2 | 0o100 | 0o1000 | 0o2000  # O_WRONLY|O_RDWR|O_CREAT|O_TRUNC|O_APPEND


def program(arch: str) -> list[tuple[int, int, int, int]]:
    """The filter for `arch`, as (code, jt, jf, k) instructions. Pure; unit-testable."""
    if arch not in AUDIT_ARCH:
        raise ValueError(f"no syscall table for {arch!r}")

    code: list[tuple[Any, ...]] = [
        (_LD_W_ABS, None, None, _ARCH),
        (_JEQ_K, None, "kill", AUDIT_ARCH[arch]),
        (_LD_W_ABS, None, None, _NR),
    ]
    if arch == "x86_64":
        code.append((_JGE_K, "kill", None, _X32_SYSCALL_BIT))
    for number in sorted(set(DENIED[arch].values())):
        code.append((_JEQ_K, "deny", None, number))
    for number in sorted(OPENERS[arch]):
        code.append((_JEQ_K, f"open_{number}", None, number))
    code.append((_RET_K, None, None, SECCOMP_RET_ALLOW))
    for number, flags_arg in sorted(OPENERS[arch].items()):
        code.append(("label", f"open_{number}"))
        code.append((_LD_W_ABS, None, None, _arg(flags_arg)))
        code.append((_JSET_K, "deny", None, WRITE_FLAGS))
        code.append((_RET_K, None, None, SECCOMP_RET_ALLOW))
    code.append(("label", "deny"))
    code.append((_RET_K, None, None, SECCOMP_RET_ERRNO | EPERM))
    code.append(("label", "kill"))
    code.append((_RET_K, None, None, SECCOMP_RET_KILL_PROCESS))
    return _assemble(code)


def _assemble(code: list[tuple[Any, ...]]) -> list[tuple[int, int, int, int]]:
    """Resolve symbolic jump targets into the relative offsets BPF expects."""
    labels: dict[Any, int] = {}
    position = 0
    for item in code:
        if item[0] == "label":
            labels[item[1]] = position
        else:
            position += 1

    out: list[tuple[int, int, int, int]] = []
    position = 0
    for item in code:
        if item[0] == "label":
            continue
        op, true_target, false_target, k = item

        def offset(target: object, here: int = position) -> int:
            if target is None:
                return 0
            distance = labels[target] - here - 1
            if not 0 <= distance <= 255:
                raise ValueError(f"jump to {target!r} out of range ({distance})")
            return distance

        out.append((int(op), offset(true_target), offset(false_target), int(k)))
        position += 1
    return out


class _SockFilter(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_ushort),
        ("jt", ctypes.c_ubyte),
        ("jf", ctypes.c_ubyte),
        ("k", ctypes.c_uint),
    ]


class _SockFprog(ctypes.Structure):
    _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.POINTER(_SockFilter))]


def _libc() -> ctypes.CDLL:
    return ctypes.CDLL(None, use_errno=True)


def _prctl(option: int, *args: int | ctypes.c_void_p) -> None:
    """prctl(2) with every argument passed as an unsigned long or a pointer.

    prctl is variadic, so ctypes would otherwise pass Python ints as 32-bit ints.
    The argument order is the call's own: PR_SET_SECCOMP takes the mode second and
    the filter third.
    """
    values: list[ctypes.c_ulong | ctypes.c_void_p] = [
        arg if isinstance(arg, ctypes.c_void_p) else ctypes.c_ulong(arg) for arg in args
    ]
    values += [ctypes.c_ulong(0)] * (4 - len(values))
    if _libc().prctl(ctypes.c_int(option), *values) != 0:
        errno = ctypes.get_errno()
        raise OSError(errno, f"prctl({option}) failed: {os.strerror(errno)}")


def protect_parent() -> bool:
    """Make this process's /proc entries — its environ above all — unreadable to its
    own children. Idempotent. Returns whether it applied (Linux only)."""
    if not sys.platform.startswith("linux"):
        return False
    try:
        _prctl(PR_SET_DUMPABLE, 0)
    except OSError:
        return False
    return True


# --- Landlock ----------------------------------------------------------------------

_LANDLOCK_CREATE_RULESET, _LANDLOCK_ADD_RULE, _LANDLOCK_RESTRICT_SELF = 444, 445, 446
_LANDLOCK_CREATE_RULESET_VERSION = 1 << 0
_LANDLOCK_RULE_PATH_BENEATH = 1
_FS_READ_FILE, _FS_READ_DIR = 1 << 2, 1 << 3
# Every filesystem right ABI v1 knows about, so the ruleset handles — and therefore
# denies by default — all of them, and grants back only reading.
_FS_ALL_V1 = (1 << 13) - 1


class _RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _PathBeneath(ctypes.Structure):
    _pack_ = 1
    _fields_ = [("allowed_access", ctypes.c_uint64), ("parent_fd", ctypes.c_int32)]


def _landlock(read_paths: list[str]) -> bool:
    """Confine reads to `read_paths`. False if the kernel has no Landlock."""
    libc = _libc()
    syscall = libc.syscall
    syscall.restype = ctypes.c_long

    version = syscall(
        ctypes.c_long(_LANDLOCK_CREATE_RULESET),
        None,
        ctypes.c_size_t(0),
        ctypes.c_uint32(_LANDLOCK_CREATE_RULESET_VERSION),
    )
    if version < 1:
        return False

    attr = _RulesetAttr(_FS_ALL_V1)
    ruleset = syscall(
        ctypes.c_long(_LANDLOCK_CREATE_RULESET),
        ctypes.byref(attr),
        ctypes.c_size_t(ctypes.sizeof(attr)),
        ctypes.c_uint32(0),
    )
    if ruleset < 0:
        return False
    try:
        for path in read_paths:
            try:
                # O_PATH is Linux-only, and so is everything that reaches this line.
                fd = os.open(path, getattr(os, "O_PATH", 0) | os.O_CLOEXEC)
            except OSError:
                continue
            try:
                rule = _PathBeneath(_FS_READ_FILE | _FS_READ_DIR, fd)
                if os.path.isfile(path):
                    rule.allowed_access = _FS_READ_FILE
                syscall(
                    ctypes.c_long(_LANDLOCK_ADD_RULE),
                    ctypes.c_int(ruleset),
                    ctypes.c_int(_LANDLOCK_RULE_PATH_BENEATH),
                    ctypes.byref(rule),
                    ctypes.c_uint32(0),
                )
            finally:
                os.close(fd)
        if (
            syscall(
                ctypes.c_long(_LANDLOCK_RESTRICT_SELF), ctypes.c_int(ruleset), ctypes.c_uint32(0)
            )
            != 0
        ):
            return False
    finally:
        os.close(ruleset)
    return True


def _seccomp(arch: str) -> None:
    instructions = program(arch)
    array = (_SockFilter * len(instructions))(*(_SockFilter(*i) for i in instructions))
    prog = _SockFprog(len(instructions), ctypes.cast(array, ctypes.POINTER(_SockFilter)))
    _prctl(PR_SET_SECCOMP, SECCOMP_MODE_FILTER, ctypes.cast(ctypes.pointer(prog), ctypes.c_void_p))


def contain(read_paths: list[str], require_landlock: bool = True) -> dict[str, object]:
    """Apply every layer this platform supports, strongest first. Call once, in the
    child, before untrusted code runs.

    Raises on Linux if seccomp cannot be installed, and — unless told otherwise — if
    Landlock cannot either. Seccomp alone closes the network, so a secret cannot be
    sent anywhere; but without Landlock the child can still *read* another process's
    environment and print a key back through stdout in an encoding redaction does
    not anticipate. On a public endpoint that is not a gap worth accepting, so the
    default is to refuse to run rather than run half-contained.
    """
    if not sys.platform.startswith("linux"):
        return {"seccomp": False, "landlock": False, "platform": sys.platform}

    arch = platform.machine()
    _prctl(PR_SET_NO_NEW_PRIVS, 1)
    try:
        landlocked = _landlock(read_paths)
    except OSError:
        landlocked = False
    if require_landlock and not landlocked:
        raise OSError("Landlock is unavailable on this kernel, and it is required")
    _seccomp(arch)  # raises; deliberately not caught
    return {"seccomp": True, "landlock": landlocked, "platform": f"linux/{arch}"}
