from __future__ import annotations

import ctypes
import os
from ctypes import wintypes


CRED_TYPE_GENERIC = 1
CRED_PERSIST_LOCAL_MACHINE = 2


class CredentialError(RuntimeError):
    pass


def target_name(provider: str) -> str:
    return f"Mimir:{provider.strip().lower()}"


if os.name == "nt":
    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)

    class CREDENTIALW(ctypes.Structure):
        _fields_ = [
            ("Flags", wintypes.DWORD),
            ("Type", wintypes.DWORD),
            ("TargetName", wintypes.LPWSTR),
            ("Comment", wintypes.LPWSTR),
            ("LastWritten", wintypes.FILETIME),
            ("CredentialBlobSize", wintypes.DWORD),
            ("CredentialBlob", ctypes.c_void_p),
            ("Persist", wintypes.DWORD),
            ("AttributeCount", wintypes.DWORD),
            ("Attributes", ctypes.c_void_p),
            ("TargetAlias", wintypes.LPWSTR),
            ("UserName", wintypes.LPWSTR),
        ]

    PCREDENTIALW = ctypes.POINTER(CREDENTIALW)

    advapi32.CredWriteW.argtypes = [ctypes.POINTER(CREDENTIALW), wintypes.DWORD]
    advapi32.CredWriteW.restype = wintypes.BOOL
    advapi32.CredReadW.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(PCREDENTIALW),
    ]
    advapi32.CredReadW.restype = wintypes.BOOL
    advapi32.CredDeleteW.argtypes = [wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD]
    advapi32.CredDeleteW.restype = wintypes.BOOL
    advapi32.CredFree.argtypes = [ctypes.c_void_p]
    advapi32.CredFree.restype = None


def write_secret(provider: str, secret: str) -> None:
    if os.name != "nt":
        raise CredentialError("Windows Credential Manager is only available on Windows")
    secret_bytes = secret.encode("utf-8")
    blob = ctypes.create_string_buffer(secret_bytes)
    credential = CREDENTIALW()
    credential.Type = CRED_TYPE_GENERIC
    credential.TargetName = target_name(provider)
    credential.CredentialBlobSize = len(secret_bytes)
    credential.CredentialBlob = ctypes.cast(blob, ctypes.c_void_p)
    credential.Persist = CRED_PERSIST_LOCAL_MACHINE
    credential.UserName = provider.strip().lower()
    if not advapi32.CredWriteW(ctypes.byref(credential), 0):
        raise CredentialError(f"CredWrite failed: {ctypes.get_last_error()}")


def read_secret(provider: str) -> str | None:
    if os.name != "nt":
        return None
    credential_ptr = PCREDENTIALW()
    ok = advapi32.CredReadW(target_name(provider), CRED_TYPE_GENERIC, 0, ctypes.byref(credential_ptr))
    if not ok:
        return None
    try:
        credential = credential_ptr.contents
        if not credential.CredentialBlob or credential.CredentialBlobSize == 0:
            return ""
        raw = ctypes.string_at(credential.CredentialBlob, credential.CredentialBlobSize)
        return raw.decode("utf-8")
    finally:
        advapi32.CredFree(credential_ptr)


def delete_secret(provider: str) -> None:
    if os.name != "nt":
        return
    advapi32.CredDeleteW(target_name(provider), CRED_TYPE_GENERIC, 0)
