"""Best-effort host memory measurements in bytes, without optional packages."""
from __future__ import annotations

import ctypes
import os
import platform
from typing import Any


def memory_snapshot() -> dict[str, Any]:
    result: dict[str, Any] = {"rss_bytes": None, "peak_rss_bytes": None, "available_memory_bytes": None, "total_memory_bytes": None}
    try:
        if platform.system() == "Windows":
            class MemoryStatus(ctypes.Structure):
                _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [(name, ctypes.c_ulonglong) for name in ("total", "available", "page_total", "page_available", "virtual_total", "virtual_available", "extended")]
            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                raise OSError("GlobalMemoryStatusEx failed")
            result.update(total_memory_bytes=status.total, available_memory_bytes=status.available)
            class ProcessMemory(ctypes.Structure):
                _fields_ = [("cb", ctypes.c_ulong), ("faults", ctypes.c_ulong)] + [(name, ctypes.c_size_t) for name in ("peak", "working", "paged_peak", "paged", "nonpaged_peak", "nonpaged", "pagefile", "pagefile_peak")]
            counters = ProcessMemory()
            counters.cb = ctypes.sizeof(counters)
            kernel = ctypes.windll.kernel32
            kernel.GetCurrentProcess.restype = ctypes.c_void_p
            fn = ctypes.windll.psapi.GetProcessMemoryInfo
            fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong]
            if not fn(kernel.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
                raise OSError("GetProcessMemoryInfo failed")
            result.update(rss_bytes=counters.working, peak_rss_bytes=counters.peak)
        else:
            import resource
            peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            result["peak_rss_bytes"] = int(peak if platform.system() == "Darwin" else peak * 1024)
            if hasattr(os, "sysconf"):
                result["total_memory_bytes"] = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
    except (OSError, ValueError, AttributeError) as exc:
        result["measurement_error"] = str(exc)
    return result
