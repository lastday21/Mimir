from __future__ import annotations

import ctypes
import os
import threading
import time
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ctypes import wintypes

from .capture import AudioCaptureConfig, AudioCaptureError


PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
STILL_ACTIVE = 259
COINIT_MULTITHREADED = 0
RPC_E_CHANGED_MODE = 0x80010106
VT_BLOB = 65
AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK = 1
PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE = 0
AUDCLNT_SHAREMODE_SHARED = 0
AUDCLNT_STREAMFLAGS_LOOPBACK = 0x00020000
AUDCLNT_BUFFERFLAGS_SILENT = 0x00000002
WAVE_FORMAT_EXTENSIBLE = 0xFFFE
MF_VERSION = 0x00020070
MFSTARTUP_LITE = 1
PROCESS_LOOPBACK_DEVICE = "VAD\\Process_Loopback"


class _Guid(ctypes.Structure):
    _fields_ = [("bytes", ctypes.c_ubyte * 16)]

    @classmethod
    def parse(cls, value: str) -> "_Guid":
        result = cls()
        result.bytes[:] = uuid.UUID(value).bytes_le
        return result


IID_IUNKNOWN = _Guid.parse("00000000-0000-0000-C000-000000000046")
IID_IAGILE_OBJECT = _Guid.parse("94EA2B94-E9CC-49E0-C0FF-EE64CA8F5B90")
IID_ACTIVATION_HANDLER = _Guid.parse("41D949AB-9862-444A-80F6-C261334DA5EB")
IID_AUDIO_CLIENT = _Guid.parse("1CB9AD4C-DBFA-4C32-B178-C2F568A703B2")
IID_AUDIO_CAPTURE_CLIENT = _Guid.parse("C8ADBD64-E71E-48A0-A4DE-185C395CD317")
KSDATAFORMAT_SUBTYPE_IEEE_FLOAT = _Guid.parse("00000003-0000-0010-8000-00AA00389B71")


class _ProcessLoopbackParams(ctypes.Structure):
    _fields_ = [
        ("process_id", wintypes.DWORD),
        ("mode", ctypes.c_int),
    ]


class _AudioClientActivationParams(ctypes.Structure):
    _fields_ = [
        ("activation_type", ctypes.c_int),
        ("process_loopback", _ProcessLoopbackParams),
    ]


class _Blob(ctypes.Structure):
    _fields_ = [
        ("size", wintypes.ULONG),
        ("data", ctypes.POINTER(ctypes.c_ubyte)),
    ]


class _PropVariantValue(ctypes.Union):
    _fields_ = [
        ("blob", _Blob),
        ("pointer", ctypes.c_void_p),
    ]


class _PropVariant(ctypes.Structure):
    _anonymous_ = ("value",)
    _fields_ = [
        ("vt", wintypes.USHORT),
        ("reserved1", wintypes.USHORT),
        ("reserved2", wintypes.USHORT),
        ("reserved3", wintypes.USHORT),
        ("value", _PropVariantValue),
    ]


class _WaveFormatExtensible(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("format_tag", wintypes.WORD),
        ("channels", wintypes.WORD),
        ("samples_per_second", wintypes.DWORD),
        ("average_bytes_per_second", wintypes.DWORD),
        ("block_align", wintypes.WORD),
        ("bits_per_sample", wintypes.WORD),
        ("extra_size", wintypes.WORD),
        ("valid_bits_per_sample", wintypes.WORD),
        ("channel_mask", wintypes.DWORD),
        ("sub_format", _Guid),
    ]


_QueryInterfaceCallback = ctypes.WINFUNCTYPE(
    ctypes.c_long,
    ctypes.c_void_p,
    ctypes.POINTER(_Guid),
    ctypes.POINTER(ctypes.c_void_p),
)
_AddRefCallback = ctypes.WINFUNCTYPE(wintypes.ULONG, ctypes.c_void_p)
_ReleaseCallback = ctypes.WINFUNCTYPE(wintypes.ULONG, ctypes.c_void_p)
_ActivateCompletedCallback = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.c_void_p)


class _ActivationHandlerVtable(ctypes.Structure):
    _fields_ = [
        ("query_interface", _QueryInterfaceCallback),
        ("add_ref", _AddRefCallback),
        ("release", _ReleaseCallback),
        ("activate_completed", _ActivateCompletedCallback),
    ]


class _ActivationHandlerStruct(ctypes.Structure):
    _fields_ = [("vtable", ctypes.POINTER(_ActivationHandlerVtable))]


class _ActivationHandler:
    def __init__(self) -> None:
        self.completed = threading.Event()
        self.call_result = -1
        self.activation_result = -1
        self.audio_interface = ctypes.c_void_p()
        self.reference_count = 1

        self._query_callback = _QueryInterfaceCallback(self._query_interface)
        self._add_ref_callback = _AddRefCallback(self._add_ref)
        self._release_callback = _ReleaseCallback(self._release)
        self._completed_callback = _ActivateCompletedCallback(self._activate_completed)
        self._vtable = _ActivationHandlerVtable(
            self._query_callback,
            self._add_ref_callback,
            self._release_callback,
            self._completed_callback,
        )
        self.struct = _ActivationHandlerStruct(ctypes.pointer(self._vtable))

    def _query_interface(
        self,
        this: int,
        requested: ctypes.POINTER(_Guid),
        result: ctypes.POINTER(ctypes.c_void_p),
    ) -> int:
        requested_bytes = ctypes.string_at(requested, 16)
        supported = {
            bytes(IID_IUNKNOWN.bytes),
            bytes(IID_IAGILE_OBJECT.bytes),
            bytes(IID_ACTIVATION_HANDLER.bytes),
        }
        if requested_bytes not in supported:
            result[0] = None
            return _signed_hresult(0x80004002)
        result[0] = this
        self.reference_count += 1
        return 0

    def _add_ref(self, _this: int) -> int:
        self.reference_count += 1
        return self.reference_count

    def _release(self, _this: int) -> int:
        self.reference_count = max(1, self.reference_count - 1)
        return self.reference_count

    def _activate_completed(self, _this: int, operation: int) -> int:
        activation_result = ctypes.c_long()
        audio_interface = ctypes.c_void_p()
        try:
            get_result = _com_method(
                operation,
                3,
                ctypes.c_long,
                ctypes.POINTER(ctypes.c_long),
                ctypes.POINTER(ctypes.c_void_p),
            )
            self.call_result = int(
                get_result(
                    operation,
                    ctypes.byref(activation_result),
                    ctypes.byref(audio_interface),
                )
            )
            self.activation_result = int(activation_result.value)
            self.audio_interface = audio_interface
        finally:
            self.completed.set()
        return 0


@dataclass(frozen=True)
class AudioApplication:
    process_id: int
    executable: str
    title: str

    def to_dict(self) -> dict[str, object]:
        return {
            "processId": self.process_id,
            "executable": self.executable,
            "title": self.title,
        }


class ProcessLoopbackPcmSource:
    def __init__(self, process_id: int, config: AudioCaptureConfig | None = None) -> None:
        self.process_id = int(process_id)
        self.config = config or AudioCaptureConfig()

    def chunks(self, stop_event: threading.Event) -> Iterator[bytes]:
        if os.name != "nt":
            raise AudioCaptureError("Захват звука выбранного приложения поддерживается только в Windows")
        if self.process_id <= 0 or not process_exists(self.process_id):
            raise AudioCaptureError("Выбранное приложение созвона не запущено")
        if self.config.sample_rate_hertz != 16_000:
            raise AudioCaptureError("Захват приложения поддерживает частоту 16000 Гц")

        converter = _FloatStereoConverter(
            target_rate=self.config.sample_rate_hertz,
            chunk_duration_ms=self.config.chunk_duration_ms,
        )
        silence_interval = self.config.chunk_duration_ms / 1_000
        source_silence = bytes(round(48_000 * silence_interval) * 8)
        next_silence_at = time.monotonic() + silence_interval
        with _ProcessLoopbackSession(self.process_id) as session:
            last_process_check = time.monotonic()
            while not stop_event.is_set():
                received = False
                for packet in session.read_packets():
                    received = True
                    for chunk in converter.feed(packet):
                        yield chunk
                    next_silence_at = time.monotonic() + silence_interval

                now = time.monotonic()
                if not received and now >= next_silence_at:
                    yield from converter.feed(source_silence)
                    next_silence_at = now + silence_interval
                if now - last_process_check >= 1.0:
                    if not process_exists(self.process_id):
                        raise AudioCaptureError("Выбранное приложение созвона было закрыто")
                    last_process_check = now
                if not received:
                    stop_event.wait(0.01)


class _FloatStereoConverter:
    def __init__(self, target_rate: int, chunk_duration_ms: int) -> None:
        self.target_rate = target_rate
        self.chunk_bytes = max(2, round(target_rate * chunk_duration_ms / 1000) * 2)
        self.pending_pcm = bytearray()
        self._filter_taps: Any | None = None
        self._filter_state: Any | None = None
        self._source_frame_index = 0

    def feed(self, raw: bytes) -> Iterator[bytes]:
        if not raw:
            return
        try:
            import numpy as np
        except ImportError as error:
            raise AudioCaptureError("Для обработки звука приложения требуется numpy") from error

        samples = np.frombuffer(raw, dtype="<f4")
        complete_samples = len(samples) - len(samples) % 2
        if complete_samples <= 0:
            return
        mono = samples[:complete_samples].reshape(-1, 2).mean(axis=1)
        ratio = 48_000 // self.target_rate
        if ratio <= 0 or ratio * self.target_rate != 48_000:
            raise AudioCaptureError("Частота звука приложения должна быть делителем 48000 Гц")
        if self._filter_taps is None:
            self._filter_taps = _downsample_filter(np, ratio)
        if self._filter_state is None:
            self._filter_state = np.full(len(self._filter_taps) - 1, mono[0], dtype=np.float32)

        combined = np.concatenate((self._filter_state, mono))
        filtered = np.convolve(combined, self._filter_taps, mode="valid")
        first_output = (-self._source_frame_index) % ratio
        downsampled = filtered[first_output::ratio]
        self._source_frame_index += len(mono)
        self._filter_state = combined[-(len(self._filter_taps) - 1) :].copy()
        pcm = np.rint(np.clip(downsampled, -1.0, 1.0) * 32767).astype("<i2").tobytes()
        self.pending_pcm.extend(pcm)
        while len(self.pending_pcm) >= self.chunk_bytes:
            chunk = bytes(self.pending_pcm[: self.chunk_bytes])
            del self.pending_pcm[: self.chunk_bytes]
            yield chunk


def _downsample_filter(np: Any, ratio: int, taps_count: int = 63) -> Any:
    taps_count = max(15, int(taps_count) | 1)
    cutoff = 0.45 / ratio
    positions = np.arange(taps_count, dtype=np.float64) - (taps_count - 1) / 2
    taps = 2 * cutoff * np.sinc(2 * cutoff * positions)
    taps *= np.hamming(taps_count)
    taps /= taps.sum()
    return taps.astype(np.float32)


class _ProcessLoopbackSession:
    def __init__(self, process_id: int) -> None:
        self.process_id = process_id
        self.audio_client = ctypes.c_void_p()
        self.capture_client = ctypes.c_void_p()
        self.com_initialized = False
        self.media_foundation_started = False

    def __enter__(self) -> "_ProcessLoopbackSession":
        ole32 = ctypes.WinDLL("ole32", use_last_error=True)
        ole32.CoInitializeEx.argtypes = [ctypes.c_void_p, wintypes.DWORD]
        ole32.CoInitializeEx.restype = ctypes.c_long
        result = int(ole32.CoInitializeEx(None, COINIT_MULTITHREADED))
        code = result & 0xFFFFFFFF
        if result >= 0:
            self.com_initialized = True
        elif code != RPC_E_CHANGED_MODE:
            raise AudioCaptureError(f"Не удалось подготовить Windows Audio: {hex(code)}")

        try:
            mfplat = ctypes.WinDLL("mfplat", use_last_error=True)
            mfplat.MFStartup.argtypes = [wintypes.ULONG, wintypes.DWORD]
            mfplat.MFStartup.restype = ctypes.c_long
            _check_hresult(
                mfplat.MFStartup(MF_VERSION, MFSTARTUP_LITE),
                "Не удалось запустить обработку звука Windows",
            )
            self.media_foundation_started = True

            self.audio_client = _activate_audio_client(self.process_id)
            wave_format = _process_wave_format()
            initialize = _com_method(
                self.audio_client,
                3,
                ctypes.c_long,
                ctypes.c_int,
                wintypes.DWORD,
                ctypes.c_longlong,
                ctypes.c_longlong,
                ctypes.c_void_p,
                ctypes.c_void_p,
            )
            _check_hresult(
                initialize(
                    self.audio_client,
                    AUDCLNT_SHAREMODE_SHARED,
                    AUDCLNT_STREAMFLAGS_LOOPBACK,
                    0,
                    0,
                    ctypes.byref(wave_format),
                    None,
                ),
                "Не удалось открыть звук выбранного приложения",
            )

            get_service = _com_method(
                self.audio_client,
                14,
                ctypes.c_long,
                ctypes.POINTER(_Guid),
                ctypes.POINTER(ctypes.c_void_p),
            )
            _check_hresult(
                get_service(
                    self.audio_client,
                    ctypes.byref(IID_AUDIO_CAPTURE_CLIENT),
                    ctypes.byref(self.capture_client),
                ),
                "Не удалось получить поток выбранного приложения",
            )
            start = _com_method(self.audio_client, 10, ctypes.c_long)
            _check_hresult(start(self.audio_client), "Не удалось начать захват выбранного приложения")
            return self
        except Exception:
            self.close()
            raise

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        self.close()

    def read_packets(self) -> Iterator[bytes]:
        if not self.capture_client:
            return
        next_packet_size = _com_method(
            self.capture_client,
            5,
            ctypes.c_long,
            ctypes.POINTER(wintypes.UINT),
        )
        get_buffer = _com_method(
            self.capture_client,
            3,
            ctypes.c_long,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(wintypes.UINT),
            ctypes.POINTER(wintypes.DWORD),
            ctypes.POINTER(ctypes.c_ulonglong),
            ctypes.POINTER(ctypes.c_ulonglong),
        )
        release_buffer = _com_method(
            self.capture_client,
            4,
            ctypes.c_long,
            wintypes.UINT,
        )

        packet_frames = wintypes.UINT()
        _check_hresult(
            next_packet_size(self.capture_client, ctypes.byref(packet_frames)),
            "Не удалось прочитать звук выбранного приложения",
        )
        while packet_frames.value:
            data = ctypes.c_void_p()
            frames = wintypes.UINT()
            flags = wintypes.DWORD()
            _check_hresult(
                get_buffer(
                    self.capture_client,
                    ctypes.byref(data),
                    ctypes.byref(frames),
                    ctypes.byref(flags),
                    None,
                    None,
                ),
                "Не удалось получить звуковой фрагмент приложения",
            )
            try:
                byte_count = int(frames.value) * 8
                if flags.value & AUDCLNT_BUFFERFLAGS_SILENT or not data.value:
                    yield bytes(byte_count)
                elif byte_count:
                    yield ctypes.string_at(data, byte_count)
            finally:
                _check_hresult(
                    release_buffer(self.capture_client, frames.value),
                    "Не удалось освободить звуковой фрагмент приложения",
                )
            _check_hresult(
                next_packet_size(self.capture_client, ctypes.byref(packet_frames)),
                "Не удалось продолжить чтение звука приложения",
            )

    def close(self) -> None:
        if self.audio_client:
            stop = _com_method(self.audio_client, 11, ctypes.c_long)
            stop(self.audio_client)
        if self.capture_client:
            _release_com(self.capture_client)
            self.capture_client = ctypes.c_void_p()
        if self.audio_client:
            _release_com(self.audio_client)
            self.audio_client = ctypes.c_void_p()
        if self.media_foundation_started:
            ctypes.WinDLL("mfplat").MFShutdown()
            self.media_foundation_started = False
        if self.com_initialized:
            ctypes.WinDLL("ole32").CoUninitialize()
            self.com_initialized = False


def list_audio_applications() -> list[dict[str, object]]:
    if os.name != "nt":
        return []
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _configure_kernel32(kernel32)
    applications: dict[int, AudioApplication] = {}
    ignored = {
        "applicationframehost.exe",
        "dwm.exe",
        "explorer.exe",
        "searchhost.exe",
        "shellexperiencehost.exe",
        "startmenuexperiencehost.exe",
    }

    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user32.EnumWindows.argtypes = [callback_type, wintypes.LPARAM]
    user32.EnumWindows.restype = wintypes.BOOL
    user32.IsWindowVisible.argtypes = [wintypes.HWND]
    user32.IsWindowVisible.restype = wintypes.BOOL
    user32.GetWindowTextLengthW.argtypes = [wintypes.HWND]
    user32.GetWindowTextLengthW.restype = ctypes.c_int
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.GetWindowTextW.restype = ctypes.c_int
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user32.GetWindowThreadProcessId.restype = wintypes.DWORD

    def visit_window(window: int, _parameter: int) -> bool:
        if not user32.IsWindowVisible(window):
            return True
        length = user32.GetWindowTextLengthW(window)
        if length <= 0:
            return True
        title_buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(window, title_buffer, len(title_buffer))
        title = title_buffer.value.strip()
        if not title:
            return True

        process_id = wintypes.DWORD()
        user32.GetWindowThreadProcessId(window, ctypes.byref(process_id))
        pid = int(process_id.value)
        if pid <= 0 or pid == os.getpid() or pid in applications:
            return True
        executable_path = _process_path(kernel32, pid)
        if not executable_path:
            return True
        executable = Path(executable_path).name
        if executable.lower() in ignored:
            return True
        applications[pid] = AudioApplication(pid, executable, title)
        return True

    callback = callback_type(visit_window)
    user32.EnumWindows(callback, 0)
    return [
        application.to_dict()
        for application in sorted(
            applications.values(),
            key=lambda item: (item.executable.lower(), item.title.lower()),
        )
    ]


def process_exists(process_id: int) -> bool:
    if os.name != "nt" or process_id <= 0:
        return False
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _configure_kernel32(kernel32)
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, process_id)
    if not handle:
        return False
    try:
        exit_code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return False
        return exit_code.value == STILL_ACTIVE
    finally:
        kernel32.CloseHandle(handle)


def _process_path(kernel32: Any, process_id: int) -> str:
    handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, process_id)
    if not handle:
        return ""
    try:
        size = wintypes.DWORD(32768)
        buffer = ctypes.create_unicode_buffer(size.value)
        if not kernel32.QueryFullProcessImageNameW(handle, 0, buffer, ctypes.byref(size)):
            return ""
        return buffer.value
    finally:
        kernel32.CloseHandle(handle)


def _configure_kernel32(kernel32: Any) -> None:
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.QueryFullProcessImageNameW.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.LPWSTR,
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel32.QueryFullProcessImageNameW.restype = wintypes.BOOL
    kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL


def _activate_audio_client(process_id: int) -> ctypes.c_void_p:
    mmdevapi = ctypes.WinDLL("mmdevapi", use_last_error=True)
    mmdevapi.ActivateAudioInterfaceAsync.argtypes = [
        wintypes.LPCWSTR,
        ctypes.POINTER(_Guid),
        ctypes.POINTER(_PropVariant),
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    mmdevapi.ActivateAudioInterfaceAsync.restype = ctypes.c_long

    activation_params = _AudioClientActivationParams(
        AUDIOCLIENT_ACTIVATION_TYPE_PROCESS_LOOPBACK,
        _ProcessLoopbackParams(
            process_id,
            PROCESS_LOOPBACK_MODE_INCLUDE_TARGET_PROCESS_TREE,
        ),
    )
    prop_variant = _PropVariant()
    prop_variant.vt = VT_BLOB
    prop_variant.blob = _Blob(
        ctypes.sizeof(activation_params),
        ctypes.cast(ctypes.pointer(activation_params), ctypes.POINTER(ctypes.c_ubyte)),
    )
    handler = _ActivationHandler()
    operation = ctypes.c_void_p()
    _check_hresult(
        mmdevapi.ActivateAudioInterfaceAsync(
            PROCESS_LOOPBACK_DEVICE,
            ctypes.byref(IID_AUDIO_CLIENT),
            ctypes.byref(prop_variant),
            ctypes.byref(handler.struct),
            ctypes.byref(operation),
        ),
        "Не удалось запросить звук выбранного приложения",
    )
    try:
        if not handler.completed.wait(5):
            raise AudioCaptureError("Windows не ответила на запрос звука выбранного приложения")
        _check_hresult(handler.call_result, "Не удалось завершить подключение к приложению")
        _check_hresult(handler.activation_result, "Windows не разрешила захват выбранного приложения")
        if not handler.audio_interface:
            raise AudioCaptureError("Windows не вернула звуковой поток выбранного приложения")
        audio_client = _query_interface(handler.audio_interface, IID_AUDIO_CLIENT)
        return audio_client
    finally:
        if handler.audio_interface:
            _release_com(handler.audio_interface)
        if operation:
            _release_com(operation)


def _process_wave_format() -> _WaveFormatExtensible:
    return _WaveFormatExtensible(
        WAVE_FORMAT_EXTENSIBLE,
        2,
        48_000,
        48_000 * 8,
        8,
        32,
        22,
        32,
        3,
        KSDATAFORMAT_SUBTYPE_IEEE_FLOAT,
    )


def _query_interface(pointer: ctypes.c_void_p, interface_id: _Guid) -> ctypes.c_void_p:
    result = ctypes.c_void_p()
    query_interface = _com_method(
        pointer,
        0,
        ctypes.c_long,
        ctypes.POINTER(_Guid),
        ctypes.POINTER(ctypes.c_void_p),
    )
    _check_hresult(
        query_interface(pointer, ctypes.byref(interface_id), ctypes.byref(result)),
        "Windows не вернула требуемый звуковой интерфейс",
    )
    return result


def _release_com(pointer: ctypes.c_void_p) -> None:
    if pointer:
        release = _com_method(pointer, 2, wintypes.ULONG)
        release(pointer)


def _com_method(pointer: Any, index: int, result_type: Any, *argument_types: Any) -> Any:
    address = int(ctypes.cast(pointer, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents[index])
    prototype = ctypes.WINFUNCTYPE(result_type, ctypes.c_void_p, *argument_types)
    return prototype(address)


def _check_hresult(result: int, message: str) -> None:
    value = int(result)
    if value < 0:
        raise AudioCaptureError(f"{message}: {hex(value & 0xFFFFFFFF)}")


def _signed_hresult(value: int) -> int:
    return ctypes.c_long(value).value
