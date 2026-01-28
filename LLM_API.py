"""
LLM Inference Profiling API
============================
A unified Python API for GPU profiling tools across all granularity levels.

This module provides abstract interfaces and concrete implementations for:
- Level 1: GPU Microarchitecture Profiling
- Level 2: Memory Hierarchy Profiling  
- Level 3: Operator/Layer-Level Profiling
- Level 4: Intra-Node Multi-GPU Profiling
- Level 5: Inter-Node Communication Profiling
- Level 6: Batch Scheduling Profiling
- Level 7: Cluster-Wide Profiling
- Level 8: Production Monitoring

Usage:
    from llm_profiler_api import UnifiedProfiler
    
    profiler = UnifiedProfiler()
    with profiler.profile(levels=[1, 2, 3]):
        model.generate(inputs)
    
    report = profiler.get_report()
"""

import os
import json
import subprocess
import time
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable, Union
from contextlib import contextmanager
from enum import Enum, auto
from pathlib import Path
import tempfile


# =============================================================================
# Data Classes for Profiling Results
# =============================================================================

@dataclass
class MetricResult:
    """Single metric measurement result."""
    name: str
    value: float
    unit: str
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KernelProfile:
    """Kernel-level profiling result."""
    kernel_name: str
    duration_us: float
    grid_size: tuple
    block_size: tuple
    registers_per_thread: int
    shared_memory_bytes: int
    metrics: Dict[str, MetricResult] = field(default_factory=dict)


@dataclass
class OperatorProfile:
    """Operator/layer-level profiling result."""
    operator_name: str
    cpu_time_us: float
    cuda_time_us: float
    memory_allocated_bytes: int
    input_shapes: List[tuple]
    call_count: int = 1


@dataclass
class GPUStatus:
    """GPU status snapshot."""
    gpu_id: int
    name: str
    utilization_percent: float
    memory_used_bytes: int
    memory_total_bytes: int
    temperature_celsius: float
    power_watts: float
    nvlink_tx_bytes: int = 0
    nvlink_rx_bytes: int = 0


@dataclass
class CollectiveProfile:
    """NCCL/RCCL collective operation profile."""
    operation: str  # AllReduce, AllGather, etc.
    size_bytes: int
    duration_us: float
    algorithm: str
    bus_bandwidth_gbps: float
    ranks: List[int]


@dataclass
class ProfilingReport:
    """Complete profiling report across all levels."""
    levels_profiled: List[int]
    duration_seconds: float
    kernel_profiles: List[KernelProfile] = field(default_factory=list)
    operator_profiles: List[OperatorProfile] = field(default_factory=list)
    gpu_statuses: List[GPUStatus] = field(default_factory=list)
    collective_profiles: List[CollectiveProfile] = field(default_factory=list)
    metrics: Dict[str, MetricResult] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """Convert report to dictionary for serialization."""
        return {
            "levels_profiled": self.levels_profiled,
            "duration_seconds": self.duration_seconds,
            "kernel_profiles": [vars(k) for k in self.kernel_profiles],
            "operator_profiles": [vars(o) for o in self.operator_profiles],
            "gpu_statuses": [vars(g) for g in self.gpu_statuses],
            "collective_profiles": [vars(c) for c in self.collective_profiles],
            "metrics": {k: vars(v) for k, v in self.metrics.items()},
            "metadata": self.metadata
        }
    
    def to_json(self, path: str):
        """Export report to JSON file."""
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2, default=str)


# =============================================================================
# Abstract Base Classes for Each Profiling Level
# =============================================================================

class BaseProfiler(ABC):
    """Abstract base class for all profilers."""
    
    @property
    @abstractmethod
    def level(self) -> int:
        """Return the profiling level (1-8)."""
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Return the profiler name."""
        pass
    
    @abstractmethod
    def start(self) -> None:
        """Start profiling."""
        pass
    
    @abstractmethod
    def stop(self) -> None:
        """Stop profiling."""
        pass
    
    @abstractmethod
    def get_results(self) -> Dict[str, Any]:
        """Get profiling results."""
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the profiler is available on this system."""
        pass
    
    @contextmanager
    def profile(self):
        """Context manager for profiling."""
        self.start()
        try:
            yield self
        finally:
            self.stop()


# =============================================================================
# Level 1: GPU Microarchitecture Profiling
# =============================================================================

class NsightComputeProfiler(BaseProfiler):
    """
    NVIDIA Nsight Compute (ncu) profiler wrapper.
    
    Provides kernel-level analysis including:
    - Roofline analysis
    - Tensor core utilization
    - Warp stall analysis
    - SM throughput metrics
    
    Example:
        profiler = NsightComputeProfiler(
            metrics=["sm__throughput.avg.pct_of_peak_sustained_active"],
            kernel_filter="attention"
        )
        with profiler.profile():
            model.forward(inputs)
        results = profiler.get_results()
    """
    
    # Common LLM inference metrics
    LLM_METRICS = [
        "sm__throughput.avg.pct_of_peak_sustained_active",
        "dram__throughput.avg.pct_of_peak_sustained_elapsed",
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
        "l1tex__t_sector_hit_rate.pct",
        "lts__t_sector_hit_rate.pct",
    ]
    
    def __init__(
        self,
        output_dir: str = "./ncu_profiles",
        metrics: Optional[List[str]] = None,
        kernel_filter: Optional[str] = None,
        sections: Optional[List[str]] = None,
        replay_mode: str = "kernel"  # kernel, application, range
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics = metrics or self.LLM_METRICS
        self.kernel_filter = kernel_filter
        self.sections = sections or ["SpeedOfLight", "MemoryWorkloadAnalysis"]
        self.replay_mode = replay_mode
        self._process = None
        self._output_file = None
        self._results = {}
    
    @property
    def level(self) -> int:
        return 1
    
    @property
    def name(self) -> str:
        return "NsightCompute"
    
    def is_available(self) -> bool:
        """Check if ncu is installed."""
        try:
            result = subprocess.run(["ncu", "--version"], capture_output=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def build_command(self, target_command: List[str]) -> List[str]:
        """Build the ncu command line."""
        cmd = ["ncu"]
        
        # Output file
        self._output_file = self.output_dir / f"profile_{int(time.time())}.ncu-rep"
        cmd.extend(["-o", str(self._output_file)])
        
        # Sections
        for section in self.sections:
            cmd.extend(["--section", section])
        
        # Metrics
        if self.metrics:
            cmd.extend(["--metrics", ",".join(self.metrics)])
        
        # Kernel filter
        if self.kernel_filter:
            cmd.extend(["--kernel-name", self.kernel_filter])
        
        # Replay mode
        cmd.extend(["--replay-mode", self.replay_mode])
        
        # Target command
        cmd.extend(target_command)
        
        return cmd
    
    def profile_command(self, command: List[str]) -> Dict[str, Any]:
        """Profile an external command."""
        cmd = self.build_command(command)
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"ncu failed: {result.stderr}")
        
        return self._parse_output()
    
    def start(self) -> None:
        """Start profiling (requires CUPTI for programmatic control)."""
        # For programmatic profiling, we need to use CUPTI
        # ncu is primarily a command-line tool
        self._start_time = time.time()
        os.environ["CUDA_PROFILE"] = "1"
    
    def stop(self) -> None:
        """Stop profiling."""
        os.environ.pop("CUDA_PROFILE", None)
        self._end_time = time.time()
    
    def get_results(self) -> Dict[str, Any]:
        """Get profiling results."""
        return self._results
    
    def _parse_output(self) -> Dict[str, Any]:
        """Parse ncu output file."""
        # ncu outputs binary .ncu-rep files
        # Use ncu --import to convert to CSV/JSON
        if self._output_file and self._output_file.exists():
            csv_file = self._output_file.with_suffix(".csv")
            subprocess.run([
                "ncu", "--import", str(self._output_file),
                "--csv", "--page", "raw",
                "-o", str(csv_file)
            ])
            # Parse CSV...
            return {"output_file": str(self._output_file)}
        return {}
    
    @staticmethod
    def query_available_metrics() -> List[str]:
        """Query all available metrics from ncu."""
        result = subprocess.run(
            ["ncu", "--query-metrics"],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            return result.stdout.strip().split('\n')
        return []


class ROCmComputeProfiler(BaseProfiler):
    """
    AMD ROCm Compute Profiler (rocprof-compute/Omniperf) wrapper.
    
    Provides kernel-level analysis for AMD MI-series GPUs including:
    - Hierarchical roofline analysis
    - Hardware block counters (SQ, TCC, TA, TD)
    - Per-kernel dispatch profiling
    
    Example:
        profiler = ROCmComputeProfiler(
            workload_name="llm_inference",
            blocks=["SQ", "TCC"]
        )
        profiler.profile_command(["python", "inference.py"])
        results = profiler.get_results()
    """
    
    def __init__(
        self,
        output_dir: str = "./rocprof_profiles",
        workload_name: str = "workload",
        blocks: Optional[List[str]] = None,
        kernel_filter: Optional[str] = None
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.workload_name = workload_name
        self.blocks = blocks or ["SQ", "TCC", "TD", "TA"]
        self.kernel_filter = kernel_filter
        self._results = {}
    
    @property
    def level(self) -> int:
        return 1
    
    @property
    def name(self) -> str:
        return "ROCmCompute"
    
    def is_available(self) -> bool:
        """Check if rocprof-compute is installed."""
        try:
            result = subprocess.run(
                ["rocprof-compute", "--version"],
                capture_output=True
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def profile_command(self, command: List[str]) -> Dict[str, Any]:
        """Profile an external command."""
        cmd = [
            "rocprof-compute", "profile",
            "-n", self.workload_name,
            "-b", *self.blocks,
            "--"
        ] + command
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"rocprof-compute failed: {result.stderr}")
        
        return self._parse_output()
    
    def analyze(self, gui: bool = False) -> Dict[str, Any]:
        """Analyze collected profiles."""
        workload_path = self.output_dir / "workloads" / self.workload_name
        
        cmd = ["rocprof-compute", "analyze", "-p", str(workload_path)]
        if gui:
            cmd.append("--gui")
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        return {"analysis": result.stdout}
    
    def start(self) -> None:
        self._start_time = time.time()
    
    def stop(self) -> None:
        self._end_time = time.time()
    
    def get_results(self) -> Dict[str, Any]:
        return self._results
    
    def _parse_output(self) -> Dict[str, Any]:
        return {"workload_name": self.workload_name}


class CUPTIProfiler(BaseProfiler):
    """
    NVIDIA CUPTI (CUDA Profiling Tools Interface) wrapper.
    
    Provides programmatic profiling API for:
    - Activity tracing (kernel, memcpy, memset)
    - Callback API for CUDA API interception
    - Range profiling for specific code regions
    - PC Sampling for hotspot identification
    
    Note: Requires pycupti or direct ctypes bindings.
    
    Example:
        profiler = CUPTIProfiler(
            activities=["kernel", "memcpy"],
            metrics=["achieved_occupancy"]
        )
        with profiler.profile():
            model.forward(inputs)
        results = profiler.get_results()
    """
    
    def __init__(
        self,
        activities: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
        callback_domains: Optional[List[str]] = None
    ):
        self.activities = activities or ["kernel", "memcpy", "memset"]
        self.metrics = metrics or []
        self.callback_domains = callback_domains or []
        self._results = {}
        self._cupti = None
    
    @property
    def level(self) -> int:
        return 1
    
    @property
    def name(self) -> str:
        return "CUPTI"
    
    def is_available(self) -> bool:
        """Check if CUPTI is available."""
        try:
            # Try to import pycupti or check for libcupti
            import ctypes
            ctypes.CDLL("libcupti.so")
            return True
        except (ImportError, OSError):
            return False
    
    def start(self) -> None:
        """Start CUPTI profiling."""
        self._start_time = time.time()
        self._activities = []
        # Initialize CUPTI subscriber and enable activities
        # This requires ctypes bindings to libcupti
    
    def stop(self) -> None:
        """Stop CUPTI profiling."""
        self._end_time = time.time()
        # Disable activities and flush buffers
    
    def get_results(self) -> Dict[str, Any]:
        """Get collected activities and metrics."""
        return {
            "duration": self._end_time - self._start_time,
            "activities": self._activities,
            "metrics": self._results
        }
    
    def add_range(self, name: str) -> "CUPTIRange":
        """Create a named profiling range."""
        return CUPTIRange(self, name)


class CUPTIRange:
    """Context manager for CUPTI range profiling."""
    
    def __init__(self, profiler: CUPTIProfiler, name: str):
        self.profiler = profiler
        self.name = name
    
    def __enter__(self):
        # Push range
        return self
    
    def __exit__(self, *args):
        # Pop range
        pass


# =============================================================================
# Level 2: Memory Hierarchy Profiling
# =============================================================================

class MemoryProfiler(BaseProfiler):
    """
    Memory hierarchy profiler combining multiple tools.
    
    Tracks:
    - L1/L2 cache hit rates
    - HBM bandwidth utilization
    - Memory coalescing efficiency
    - KV-cache access patterns
    
    Example:
        profiler = MemoryProfiler()
        with profiler.profile():
            model.generate(inputs)
        
        print(f"L2 hit rate: {profiler.l2_hit_rate}%")
        print(f"HBM bandwidth: {profiler.hbm_bandwidth_gbps} GB/s")
    """
    
    MEMORY_METRICS = {
        "nvidia": [
            "l1tex__t_sector_hit_rate.pct",
            "lts__t_sector_hit_rate.pct",
            "dram__bytes_read.sum",
            "dram__bytes_write.sum",
            "l1tex__data_pipe_lsu_wavefronts_mem_shared.avg.pct_of_peak_sustained_active",
        ],
        "amd": [
            "TCP_TCC_READ_REQ_sum",
            "TCP_TCC_WRITE_REQ_sum",
            "TCC_HIT_sum",
            "TCC_MISS_sum",
        ]
    }
    
    def __init__(self, vendor: str = "auto"):
        self.vendor = self._detect_vendor() if vendor == "auto" else vendor
        self._results = {}
        self._start_time = None
        self._end_time = None
    
    @property
    def level(self) -> int:
        return 2
    
    @property
    def name(self) -> str:
        return "MemoryHierarchy"
    
    def _detect_vendor(self) -> str:
        """Detect GPU vendor."""
        try:
            subprocess.run(["nvidia-smi"], capture_output=True, check=True)
            return "nvidia"
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        try:
            subprocess.run(["rocm-smi"], capture_output=True, check=True)
            return "amd"
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
        
        return "unknown"
    
    def is_available(self) -> bool:
        return self.vendor in ["nvidia", "amd"]
    
    def start(self) -> None:
        self._start_time = time.time()
    
    def stop(self) -> None:
        self._end_time = time.time()
    
    def get_results(self) -> Dict[str, Any]:
        return self._results
    
    @property
    def l1_hit_rate(self) -> Optional[float]:
        """Get L1 cache hit rate percentage."""
        return self._results.get("l1_hit_rate")
    
    @property
    def l2_hit_rate(self) -> Optional[float]:
        """Get L2 cache hit rate percentage."""
        return self._results.get("l2_hit_rate")
    
    @property
    def hbm_bandwidth_gbps(self) -> Optional[float]:
        """Get HBM bandwidth in GB/s."""
        return self._results.get("hbm_bandwidth_gbps")
    
    def profile_with_ncu(self, command: List[str]) -> Dict[str, Any]:
        """Profile using Nsight Compute for detailed memory analysis."""
        ncu = NsightComputeProfiler(
            metrics=self.MEMORY_METRICS["nvidia"],
            sections=["MemoryWorkloadAnalysis", "MemoryWorkloadAnalysis_Chart"]
        )
        return ncu.profile_command(command)


# =============================================================================
# Level 3: Operator/Layer-Level Profiling
# =============================================================================

class PyTorchProfiler(BaseProfiler):
    """
    PyTorch native profiler wrapper.
    
    Provides:
    - Operator-level timing
    - Memory allocation tracking
    - CUDA kernel correlation
    - Chrome trace export
    - TensorBoard integration
    
    Example:
        profiler = PyTorchProfiler(
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        )
        
        with profiler.profile():
            output = model(inputs)
        
        # Print top operators by CUDA time
        profiler.print_summary(sort_by="cuda_time_total", limit=20)
        
        # Export trace for Chrome
        profiler.export_chrome_trace("trace.json")
    """
    
    def __init__(
        self,
        activities: Optional[List[str]] = None,
        record_shapes: bool = True,
        profile_memory: bool = True,
        with_stack: bool = False,
        with_flops: bool = True,
        with_modules: bool = True,
        output_dir: str = "./pytorch_profiles"
    ):
        self.activities = activities or ["cpu", "cuda"]
        self.record_shapes = record_shapes
        self.profile_memory = profile_memory
        self.with_stack = with_stack
        self.with_flops = with_flops
        self.with_modules = with_modules
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self._profiler = None
        self._results = None
    
    @property
    def level(self) -> int:
        return 3
    
    @property
    def name(self) -> str:
        return "PyTorchProfiler"
    
    def is_available(self) -> bool:
        try:
            import torch
            from torch.profiler import profile
            return True
        except ImportError:
            return False
    
    def start(self) -> None:
        """Start profiling."""
        import torch
        from torch.profiler import profile, ProfilerActivity
        
        activities = []
        if "cpu" in self.activities:
            activities.append(ProfilerActivity.CPU)
        if "cuda" in self.activities:
            activities.append(ProfilerActivity.CUDA)
        
        self._profiler = profile(
            activities=activities,
            record_shapes=self.record_shapes,
            profile_memory=self.profile_memory,
            with_stack=self.with_stack,
            with_flops=self.with_flops,
            with_modules=self.with_modules
        )
        self._profiler.__enter__()
    
    def stop(self) -> None:
        """Stop profiling."""
        if self._profiler:
            self._profiler.__exit__(None, None, None)
            self._results = self._profiler
    
    def get_results(self) -> Dict[str, Any]:
        """Get profiling results as structured data."""
        if not self._results:
            return {}
        
        events = self._results.key_averages()
        return {
            "operators": [
                {
                    "name": e.key,
                    "cpu_time_us": e.cpu_time_total,
                    "cuda_time_us": e.cuda_time_total,
                    "count": e.count,
                    "cpu_memory_usage": e.cpu_memory_usage,
                    "cuda_memory_usage": e.cuda_memory_usage,
                    "flops": getattr(e, 'flops', None),
                    "input_shapes": getattr(e, 'input_shapes', None)
                }
                for e in events
            ]
        }
    
    def print_summary(
        self,
        sort_by: str = "cuda_time_total",
        limit: int = 20,
        header: Optional[str] = None
    ) -> None:
        """Print operator summary table."""
        if self._results:
            print(self._results.key_averages().table(
                sort_by=sort_by,
                row_limit=limit,
                header=header
            ))
    
    def export_chrome_trace(self, path: Optional[str] = None) -> str:
        """Export trace in Chrome tracing format."""
        if not self._results:
            raise RuntimeError("No profiling results available")
        
        path = path or str(self.output_dir / f"trace_{int(time.time())}.json")
        self._results.export_chrome_trace(path)
        return path
    
    def export_stacks(self, path: Optional[str] = None) -> str:
        """Export flame graph data."""
        if not self._results:
            raise RuntimeError("No profiling results available")
        
        path = path or str(self.output_dir / f"stacks_{int(time.time())}.txt")
        self._results.export_stacks(path)
        return path
    
    @contextmanager
    def record_function(self, name: str):
        """Context manager for recording a named function."""
        from torch.profiler import record_function
        with record_function(name):
            yield


class NsightSystemsProfiler(BaseProfiler):
    """
    NVIDIA Nsight Systems (nsys) profiler wrapper.
    
    Provides system-wide timeline profiling:
    - CPU-GPU timeline visualization
    - CUDA API calls and kernel execution
    - NVTX marker support
    - Multi-GPU profiling
    
    Example:
        profiler = NsightSystemsProfiler(
            trace_types=["cuda", "nvtx", "osrt"],
            gpu_metrics=True
        )
        profiler.profile_command(["python", "inference.py"])
    """
    
    def __init__(
        self,
        output_dir: str = "./nsys_profiles",
        trace_types: Optional[List[str]] = None,
        gpu_metrics: bool = True,
        cuda_graph_trace: str = "node",
        sample_cpu: bool = False
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trace_types = trace_types or ["cuda", "nvtx", "osrt"]
        self.gpu_metrics = gpu_metrics
        self.cuda_graph_trace = cuda_graph_trace
        self.sample_cpu = sample_cpu
        self._output_file = None
    
    @property
    def level(self) -> int:
        return 3
    
    @property
    def name(self) -> str:
        return "NsightSystems"
    
    def is_available(self) -> bool:
        try:
            result = subprocess.run(["nsys", "--version"], capture_output=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def build_command(self, target_command: List[str]) -> List[str]:
        """Build nsys command line."""
        self._output_file = self.output_dir / f"report_{int(time.time())}"
        
        cmd = [
            "nsys", "profile",
            "-o", str(self._output_file),
            "-f", "true",
            "-t", ",".join(self.trace_types)
        ]
        
        if self.gpu_metrics:
            cmd.extend(["--gpu-metrics-device", "all"])
        
        cmd.extend(["--cuda-graph-trace", self.cuda_graph_trace])
        
        if self.sample_cpu:
            cmd.extend(["--sample", "cpu"])
        
        cmd.extend(target_command)
        return cmd
    
    def profile_command(self, command: List[str]) -> Dict[str, Any]:
        """Profile an external command."""
        cmd = self.build_command(command)
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            raise RuntimeError(f"nsys failed: {result.stderr}")
        
        return {
            "output_file": str(self._output_file) + ".nsys-rep",
            "stdout": result.stdout
        }
    
    def export_sqlite(self, nsys_rep_path: str) -> str:
        """Export nsys report to SQLite for analysis."""
        sqlite_path = Path(nsys_rep_path).with_suffix(".sqlite")
        subprocess.run([
            "nsys", "export",
            "-t", "sqlite",
            "-o", str(sqlite_path),
            nsys_rep_path
        ])
        return str(sqlite_path)
    
    def start(self) -> None:
        """Start profiling (use cudaProfilerStart for programmatic control)."""
        self._start_time = time.time()
    
    def stop(self) -> None:
        self._end_time = time.time()
    
    def get_results(self) -> Dict[str, Any]:
        return {"output_file": str(self._output_file)}


class NVTXAnnotator:
    """
    NVTX (NVIDIA Tools Extension) annotation helper.
    
    Provides custom markers for Nsight Systems/Compute visualization.
    
    Example:
        nvtx = NVTXAnnotator()
        
        with nvtx.range("attention_layer"):
            attention_output = self.attention(x)
        
        nvtx.mark("checkpoint_saved")
    """
    
    def __init__(self):
        self._nvtx = None
        try:
            import nvtx
            self._nvtx = nvtx
        except ImportError:
            try:
                import torch.cuda.nvtx as nvtx
                self._nvtx = nvtx
            except ImportError:
                pass
    
    def is_available(self) -> bool:
        return self._nvtx is not None
    
    @contextmanager
    def range(self, name: str, color: Optional[str] = None):
        """Create an NVTX range for a code block."""
        if self._nvtx:
            if hasattr(self._nvtx, 'range'):
                with self._nvtx.range(name, color=color):
                    yield
            else:
                self._nvtx.range_push(name)
                try:
                    yield
                finally:
                    self._nvtx.range_pop()
        else:
            yield
    
    def mark(self, message: str):
        """Create an NVTX marker (instant event)."""
        if self._nvtx and hasattr(self._nvtx, 'mark'):
            self._nvtx.mark(message)


# =============================================================================
# Level 4: Intra-Node Multi-GPU Profiling
# =============================================================================

class DCGMProfiler(BaseProfiler):
    """
    NVIDIA DCGM (Data Center GPU Manager) profiler wrapper.
    
    Provides multi-GPU monitoring:
    - GPU utilization and health
    - NVLink bandwidth tracking
    - Power and thermal monitoring
    - XID error detection
    
    Example:
        profiler = DCGMProfiler(
            fields=["gpu_util", "memory_used", "power", "nvlink_tx"],
            sample_interval_ms=100
        )
        
        with profiler.profile():
            model.generate(inputs)
        
        for gpu in profiler.get_gpu_statuses():
            print(f"GPU {gpu.gpu_id}: {gpu.utilization_percent}% util")
    """
    
    # DCGM field IDs
    FIELD_IDS = {
        "gpu_util": 203,          # DCGM_FI_DEV_GPU_UTIL
        "memory_used": 204,       # DCGM_FI_DEV_FB_USED
        "memory_free": 205,       # DCGM_FI_DEV_FB_FREE
        "power": 206,             # DCGM_FI_DEV_POWER_USAGE
        "temperature": 207,       # DCGM_FI_DEV_GPU_TEMP
        "nvlink_tx": 1011,        # DCGM_FI_PROF_NVLINK_TX_BYTES
        "nvlink_rx": 1012,        # DCGM_FI_PROF_NVLINK_RX_BYTES
        "pcie_tx": 1009,          # DCGM_FI_PROF_PCIE_TX_BYTES
        "pcie_rx": 1010,          # DCGM_FI_PROF_PCIE_RX_BYTES
        "sm_active": 1002,        # DCGM_FI_PROF_SM_ACTIVE
        "sm_occupancy": 1003,     # DCGM_FI_PROF_SM_OCCUPANCY
        "tensor_active": 1004,    # DCGM_FI_PROF_PIPE_TENSOR_ACTIVE
        "dram_active": 1005,      # DCGM_FI_PROF_DRAM_ACTIVE
        "xid_errors": 230,        # DCGM_FI_DEV_XID_ERRORS
    }
    
    def __init__(
        self,
        fields: Optional[List[str]] = None,
        sample_interval_ms: int = 100,
        gpu_ids: Optional[List[int]] = None
    ):
        self.fields = fields or ["gpu_util", "memory_used", "power", "temperature"]
        self.sample_interval_ms = sample_interval_ms
        self.gpu_ids = gpu_ids
        self._samples = []
        self._monitor_thread = None
        self._stop_event = threading.Event()
    
    @property
    def level(self) -> int:
        return 4
    
    @property
    def name(self) -> str:
        return "DCGM"
    
    def is_available(self) -> bool:
        """Check if DCGM is available."""
        try:
            result = subprocess.run(["dcgmi", "discovery", "-l"], capture_output=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def start(self) -> None:
        """Start continuous monitoring in background thread."""
        self._stop_event.clear()
        self._samples = []
        self._monitor_thread = threading.Thread(target=self._monitor_loop)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
    
    def stop(self) -> None:
        """Stop monitoring."""
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
    
    def _monitor_loop(self):
        """Background monitoring loop using dcgmi."""
        field_ids = [str(self.FIELD_IDS[f]) for f in self.fields if f in self.FIELD_IDS]
        
        while not self._stop_event.is_set():
            try:
                sample = self._collect_sample(field_ids)
                self._samples.append(sample)
            except Exception as e:
                print(f"DCGM sampling error: {e}")
            
            time.sleep(self.sample_interval_ms / 1000)
    
    def _collect_sample(self, field_ids: List[str]) -> Dict:
        """Collect a single sample from DCGM."""
        result = subprocess.run(
            ["dcgmi", "dmon", "-e", ",".join(field_ids), "-c", "1"],
            capture_output=True, text=True
        )
        
        return {
            "timestamp": time.time(),
            "raw_output": result.stdout
        }
    
    def get_results(self) -> Dict[str, Any]:
        """Get collected samples."""
        return {"samples": self._samples}
    
    def get_gpu_statuses(self) -> List[GPUStatus]:
        """Get current GPU status for all GPUs."""
        statuses = []
        
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        
        if result.returncode == 0:
            for line in result.stdout.strip().split('\n'):
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 7:
                    statuses.append(GPUStatus(
                        gpu_id=int(parts[0]),
                        name=parts[1],
                        utilization_percent=float(parts[2]),
                        memory_used_bytes=int(float(parts[3]) * 1024 * 1024),
                        memory_total_bytes=int(float(parts[4]) * 1024 * 1024),
                        temperature_celsius=float(parts[5]),
                        power_watts=float(parts[6]) if parts[6] != '[N/A]' else 0
                    ))
        
        return statuses


class NvidiaSMIProfiler(BaseProfiler):
    """
    nvidia-smi wrapper for basic GPU monitoring.
    
    Provides:
    - GPU utilization
    - Memory usage
    - Temperature and power
    - NVLink status
    - Topology information
    
    Example:
        profiler = NvidiaSMIProfiler(interval_ms=500)
        
        with profiler.profile():
            model.generate(inputs)
        
        stats = profiler.get_statistics()
        print(f"Average GPU util: {stats['avg_gpu_util']}%")
    """
    
    def __init__(self, interval_ms: int = 1000):
        self.interval_ms = interval_ms
        self._samples = []
        self._monitor_thread = None
        self._stop_event = threading.Event()
    
    @property
    def level(self) -> int:
        return 4
    
    @property
    def name(self) -> str:
        return "nvidia-smi"
    
    def is_available(self) -> bool:
        try:
            result = subprocess.run(["nvidia-smi"], capture_output=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def start(self) -> None:
        self._stop_event.clear()
        self._samples = []
        self._monitor_thread = threading.Thread(target=self._monitor_loop)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
    
    def stop(self) -> None:
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
    
    def _monitor_loop(self):
        """Continuous monitoring using nvidia-smi."""
        while not self._stop_event.is_set():
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True
            )
            
            if result.returncode == 0:
                self._samples.append({
                    "timestamp": time.time(),
                    "data": result.stdout.strip()
                })
            
            time.sleep(self.interval_ms / 1000)
    
    def get_results(self) -> Dict[str, Any]:
        return {"samples": self._samples}
    
    def get_statistics(self) -> Dict[str, float]:
        """Calculate statistics from collected samples."""
        if not self._samples:
            return {}
        
        gpu_utils = []
        mem_usages = []
        
        for sample in self._samples:
            for line in sample["data"].split('\n'):
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 4:
                    gpu_utils.append(float(parts[1]))
                    mem_usages.append(float(parts[2]))
        
        return {
            "avg_gpu_util": sum(gpu_utils) / len(gpu_utils) if gpu_utils else 0,
            "max_gpu_util": max(gpu_utils) if gpu_utils else 0,
            "avg_memory_mb": sum(mem_usages) / len(mem_usages) if mem_usages else 0,
            "max_memory_mb": max(mem_usages) if mem_usages else 0,
        }
    
    def get_topology(self) -> str:
        """Get GPU topology information."""
        result = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            capture_output=True, text=True
        )
        return result.stdout
    
    def get_nvlink_status(self) -> str:
        """Get NVLink status."""
        result = subprocess.run(
            ["nvidia-smi", "nvlink", "-s"],
            capture_output=True, text=True
        )
        return result.stdout


class ROCmSMIProfiler(BaseProfiler):
    """
    AMD rocm-smi wrapper for MI-series GPU monitoring.
    
    Example:
        profiler = ROCmSMIProfiler()
        gpu_status = profiler.get_gpu_status()
    """
    
    def __init__(self, interval_ms: int = 1000):
        self.interval_ms = interval_ms
        self._samples = []
    
    @property
    def level(self) -> int:
        return 4
    
    @property
    def name(self) -> str:
        return "rocm-smi"
    
    def is_available(self) -> bool:
        try:
            result = subprocess.run(["rocm-smi"], capture_output=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def start(self) -> None:
        self._start_time = time.time()
    
    def stop(self) -> None:
        self._end_time = time.time()
    
    def get_results(self) -> Dict[str, Any]:
        return {"samples": self._samples}
    
    def get_gpu_status(self) -> str:
        """Get current GPU status."""
        result = subprocess.run(
            ["rocm-smi", "--showuse", "--showmemuse", "--showtemp", "--showpower"],
            capture_output=True, text=True
        )
        return result.stdout
    
    def get_topology(self) -> str:
        """Get GPU topology."""
        result = subprocess.run(
            ["rocm-smi", "--showtopo"],
            capture_output=True, text=True
        )
        return result.stdout


# =============================================================================
# Level 5: Inter-Node Communication Profiling
# =============================================================================

class NCCLProfiler(BaseProfiler):
    """
    NCCL Inspector and NCCL tests wrapper.
    
    Profiles collective operations:
    - AllReduce, AllGather, ReduceScatter, Broadcast
    - Algorithmic and bus bandwidth
    - Per-communicator performance
    
    Example:
        profiler = NCCLProfiler(
            output_dir="./nccl_profiles",
            interval_ms=1000
        )
        
        # Run NCCL Inspector during inference
        with profiler.profile():
            distributed_model.generate(inputs)
        
        # Or run NCCL benchmarks
        bandwidth = profiler.run_allreduce_benchmark(
            min_bytes=8, max_bytes=256*1024*1024
        )
    """
    
    def __init__(
        self,
        output_dir: str = "./nccl_profiles",
        interval_ms: int = 1000,
        inspector_path: Optional[str] = None
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.interval_ms = interval_ms
        self.inspector_path = inspector_path
        self._original_env = {}
    
    @property
    def level(self) -> int:
        return 5
    
    @property
    def name(self) -> str:
        return "NCCL"
    
    def is_available(self) -> bool:
        # Check for nccl-tests
        try:
            result = subprocess.run(["which", "all_reduce_perf"], capture_output=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def start(self) -> None:
        """Enable NCCL Inspector via environment variables."""
        self._original_env = {
            "NCCL_DEBUG": os.environ.get("NCCL_DEBUG"),
            "NCCL_PROFILER_PLUGIN": os.environ.get("NCCL_PROFILER_PLUGIN"),
            "NCCL_INSPECTOR_ENABLE": os.environ.get("NCCL_INSPECTOR_ENABLE"),
            "NCCL_INSPECTOR_OUTPUT_DIR": os.environ.get("NCCL_INSPECTOR_OUTPUT_DIR"),
            "NCCL_INSPECTOR_INTERVAL": os.environ.get("NCCL_INSPECTOR_INTERVAL"),
        }
        
        os.environ["NCCL_DEBUG"] = "INFO"
        if self.inspector_path:
            os.environ["NCCL_PROFILER_PLUGIN"] = self.inspector_path
        os.environ["NCCL_INSPECTOR_ENABLE"] = "1"
        os.environ["NCCL_INSPECTOR_OUTPUT_DIR"] = str(self.output_dir)
        os.environ["NCCL_INSPECTOR_INTERVAL"] = str(self.interval_ms)
    
    def stop(self) -> None:
        """Restore original environment."""
        for key, value in self._original_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    
    def get_results(self) -> Dict[str, Any]:
        """Parse NCCL Inspector output files."""
        results = []
        for jsonl_file in self.output_dir.glob("*.jsonl"):
            with open(jsonl_file) as f:
                for line in f:
                    results.append(json.loads(line))
        return {"collectives": results}
    
    def run_allreduce_benchmark(
        self,
        min_bytes: int = 8,
        max_bytes: int = 256 * 1024 * 1024,
        factor: int = 2,
        num_gpus: int = 8,
        num_iterations: int = 20
    ) -> Dict[str, Any]:
        """Run all_reduce_perf benchmark."""
        result = subprocess.run([
            "all_reduce_perf",
            "-b", str(min_bytes),
            "-e", str(max_bytes),
            "-f", str(factor),
            "-g", str(num_gpus),
            "-n", str(num_iterations)
        ], capture_output=True, text=True)
        
        return self._parse_nccl_test_output(result.stdout)
    
    def run_allgather_benchmark(self, **kwargs) -> Dict[str, Any]:
        """Run all_gather_perf benchmark."""
        # Similar to allreduce
        pass
    
    def _parse_nccl_test_output(self, output: str) -> Dict[str, Any]:
        """Parse nccl-tests output format."""
        results = {
            "size_bytes": [],
            "time_us": [],
            "algbw_gbps": [],
            "busbw_gbps": []
        }
        
        for line in output.split('\n'):
            # Parse the tabular output
            parts = line.split()
            if len(parts) >= 8 and parts[0].isdigit():
                results["size_bytes"].append(int(parts[0]))
                results["time_us"].append(float(parts[3]))
                results["algbw_gbps"].append(float(parts[4]))
                results["busbw_gbps"].append(float(parts[5]))
        
        return results


class InfiniBandProfiler(BaseProfiler):
    """
    InfiniBand/RDMA profiler using perftest suite.
    
    Benchmarks:
    - ib_write_bw / ib_read_bw - RDMA bandwidth
    - ib_write_lat / ib_read_lat - RDMA latency
    - GPU Direct RDMA support
    
    Example:
        profiler = InfiniBandProfiler(device="mlx5_0")
        
        # Run bandwidth test (requires server/client)
        bw = profiler.run_write_bandwidth(
            server_ip="10.0.0.1",
            use_cuda=0  # GPU ID for GPUDirect
        )
        print(f"Peak bandwidth: {bw['peak_bandwidth_gbps']} Gb/s")
    """
    
    def __init__(self, device: str = "mlx5_0"):
        self.device = device
    
    @property
    def level(self) -> int:
        return 5
    
    @property
    def name(self) -> str:
        return "InfiniBand"
    
    def is_available(self) -> bool:
        try:
            result = subprocess.run(["ibv_devinfo"], capture_output=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def start(self) -> None:
        self._start_time = time.time()
    
    def stop(self) -> None:
        self._end_time = time.time()
    
    def get_results(self) -> Dict[str, Any]:
        return {}
    
    def get_device_info(self) -> str:
        """Get InfiniBand device information."""
        result = subprocess.run(["ibv_devinfo"], capture_output=True, text=True)
        return result.stdout
    
    def get_port_status(self) -> str:
        """Get port status using ibstat."""
        result = subprocess.run(["ibstat"], capture_output=True, text=True)
        return result.stdout
    
    def run_write_bandwidth(
        self,
        server_ip: Optional[str] = None,
        size: int = 65536,
        iterations: int = 1000,
        use_cuda: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run ib_write_bw benchmark.
        
        If server_ip is None, starts as server.
        """
        cmd = ["ib_write_bw", "-d", self.device, "-s", str(size), "-n", str(iterations)]
        
        if use_cuda is not None:
            cmd.extend(["--use_cuda", str(use_cuda)])
        
        if server_ip:
            cmd.append(server_ip)
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        return self._parse_perftest_output(result.stdout)
    
    def run_read_latency(
        self,
        server_ip: Optional[str] = None,
        iterations: int = 1000
    ) -> Dict[str, Any]:
        """Run ib_read_lat benchmark."""
        cmd = ["ib_read_lat", "-d", self.device, "-n", str(iterations)]
        
        if server_ip:
            cmd.append(server_ip)
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        return self._parse_perftest_output(result.stdout)
    
    def _parse_perftest_output(self, output: str) -> Dict[str, Any]:
        """Parse perftest output format."""
        results = {}
        
        for line in output.split('\n'):
            if "BW peak" in line or "BW average" in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if "Gb/sec" in part or "GB/sec" in part:
                        results["bandwidth_gbps"] = float(parts[i-1])
            elif "t_typical" in line or "t_avg" in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if "usec" in part:
                        results["latency_us"] = float(parts[i-1])
        
        return results


# =============================================================================
# Level 6: Batch Scheduling Profiling
# =============================================================================

class VLLMMetricsProfiler(BaseProfiler):
    """
    vLLM metrics profiler via Prometheus endpoint.
    
    Collects:
    - Request queue metrics
    - KV cache utilization
    - TTFT/ITL histograms
    - Throughput metrics
    
    Example:
        profiler = VLLMMetricsProfiler(endpoint="http://localhost:8000/metrics")
        
        with profiler.profile():
            # Send requests to vLLM server
            requests.post(...)
        
        metrics = profiler.get_summary()
        print(f"p99 TTFT: {metrics['ttft_p99_seconds']}s")
        print(f"KV cache usage: {metrics['kv_cache_usage_percent']}%")
    """
    
    KEY_METRICS = [
        "vllm:num_requests_running",
        "vllm:num_requests_waiting",
        "vllm:num_requests_swapped",
        "vllm:gpu_cache_usage_perc",
        "vllm:cpu_cache_usage_perc",
        "vllm:time_to_first_token_seconds",
        "vllm:time_per_output_token_seconds",
        "vllm:e2e_request_latency_seconds",
        "vllm:request_prompt_tokens",
        "vllm:request_generation_tokens",
        "vllm:avg_prompt_throughput_toks_per_s",
        "vllm:avg_generation_throughput_toks_per_s",
        "vllm:prefix_cache_hit_rate",
    ]
    
    def __init__(
        self,
        endpoint: str = "http://localhost:8000/metrics",
        sample_interval_ms: int = 1000
    ):
        self.endpoint = endpoint
        self.sample_interval_ms = sample_interval_ms
        self._samples = []
        self._monitor_thread = None
        self._stop_event = threading.Event()
    
    @property
    def level(self) -> int:
        return 6
    
    @property
    def name(self) -> str:
        return "vLLMMetrics"
    
    def is_available(self) -> bool:
        try:
            import requests
            response = requests.get(self.endpoint, timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def start(self) -> None:
        """Start background metrics collection."""
        self._stop_event.clear()
        self._samples = []
        self._monitor_thread = threading.Thread(target=self._monitor_loop)
        self._monitor_thread.daemon = True
        self._monitor_thread.start()
    
    def stop(self) -> None:
        """Stop metrics collection."""
        self._stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5)
    
    def _monitor_loop(self):
        """Background monitoring loop."""
        import requests
        
        while not self._stop_event.is_set():
            try:
                response = requests.get(self.endpoint, timeout=5)
                if response.status_code == 200:
                    self._samples.append({
                        "timestamp": time.time(),
                        "metrics": self._parse_prometheus(response.text)
                    })
            except Exception as e:
                print(f"vLLM metrics error: {e}")
            
            time.sleep(self.sample_interval_ms / 1000)
    
    def _parse_prometheus(self, text: str) -> Dict[str, float]:
        """Parse Prometheus text format."""
        metrics = {}
        for line in text.split('\n'):
            if line and not line.startswith('#'):
                parts = line.split()
                if len(parts) >= 2:
                    name = parts[0].split('{')[0]
                    try:
                        value = float(parts[-1])
                        metrics[name] = value
                    except ValueError:
                        pass
        return metrics
    
    def get_results(self) -> Dict[str, Any]:
        """Get all collected samples."""
        return {"samples": self._samples}
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary statistics from collected samples."""
        if not self._samples:
            return {}
        
        # Get latest sample
        latest = self._samples[-1]["metrics"]
        
        return {
            "requests_running": latest.get("vllm:num_requests_running", 0),
            "requests_waiting": latest.get("vllm:num_requests_waiting", 0),
            "kv_cache_usage_percent": latest.get("vllm:gpu_cache_usage_perc", 0) * 100,
            "prompt_throughput_tps": latest.get("vllm:avg_prompt_throughput_toks_per_s", 0),
            "generation_throughput_tps": latest.get("vllm:avg_generation_throughput_toks_per_s", 0),
            "prefix_cache_hit_rate": latest.get("vllm:prefix_cache_hit_rate", 0),
        }
    
    def get_latency_percentiles(self) -> Dict[str, float]:
        """Extract latency percentiles from histogram metrics."""
        # Parse histogram buckets from samples
        # This requires more complex parsing of _bucket metrics
        return {}


class TRTLLMBenchmarkProfiler(BaseProfiler):
    """
    TensorRT-LLM benchmark profiler.
    
    Example:
        profiler = TRTLLMBenchmarkProfiler(
            model="meta-llama/Llama-3.1-8B-Instruct",
            tp=2
        )
        
        results = profiler.run_throughput_benchmark(
            dataset="dataset.jsonl",
            concurrency=64
        )
    """
    
    def __init__(
        self,
        model: str,
        tp: int = 1,
        pp: int = 1,
        backend: str = "pytorch"
    ):
        self.model = model
        self.tp = tp
        self.pp = pp
        self.backend = backend
    
    @property
    def level(self) -> int:
        return 6
    
    @property
    def name(self) -> str:
        return "TRTLLMBenchmark"
    
    def is_available(self) -> bool:
        try:
            result = subprocess.run(["trtllm-bench", "--help"], capture_output=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def start(self) -> None:
        self._start_time = time.time()
    
    def stop(self) -> None:
        self._end_time = time.time()
    
    def get_results(self) -> Dict[str, Any]:
        return {}
    
    def run_throughput_benchmark(
        self,
        dataset: str,
        concurrency: int = 32,
        streaming: bool = True,
        output_json: Optional[str] = None
    ) -> Dict[str, Any]:
        """Run throughput benchmark."""
        cmd = [
            "trtllm-bench", "throughput",
            "--model", self.model,
            "--dataset", dataset,
            "--tp", str(self.tp),
            "--backend", self.backend,
            "--concurrency", str(concurrency)
        ]
        
        if streaming:
            cmd.append("--streaming")
        
        if output_json:
            cmd.extend(["--report_json", output_json])
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if output_json and Path(output_json).exists():
            with open(output_json) as f:
                return json.load(f)
        
        return {"stdout": result.stdout}


# =============================================================================
# Level 7: Cluster-Wide Profiling
# =============================================================================

class RayDashboardProfiler(BaseProfiler):
    """
    Ray dashboard and metrics profiler.
    
    Example:
        profiler = RayDashboardProfiler(dashboard_url="http://localhost:8265")
        
        # Get cluster status
        status = profiler.get_cluster_status()
        
        # Get deployment metrics
        metrics = profiler.get_serve_metrics()
    """
    
    def __init__(
        self,
        dashboard_url: str = "http://localhost:8265",
        prometheus_port: int = 8080
    ):
        self.dashboard_url = dashboard_url
        self.prometheus_port = prometheus_port
    
    @property
    def level(self) -> int:
        return 7
    
    @property
    def name(self) -> str:
        return "RayDashboard"
    
    def is_available(self) -> bool:
        try:
            import requests
            response = requests.get(f"{self.dashboard_url}/api/cluster_status", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def start(self) -> None:
        self._start_time = time.time()
    
    def stop(self) -> None:
        self._end_time = time.time()
    
    def get_results(self) -> Dict[str, Any]:
        return {}
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get Ray cluster status."""
        import requests
        response = requests.get(f"{self.dashboard_url}/api/cluster_status")
        return response.json()
    
    def get_nodes(self) -> List[Dict[str, Any]]:
        """Get all nodes in the cluster."""
        import requests
        response = requests.get(f"{self.dashboard_url}/nodes?view=summary")
        return response.json().get("data", {}).get("summary", [])
    
    def get_serve_metrics(self) -> Dict[str, Any]:
        """Get Ray Serve deployment metrics."""
        import requests
        response = requests.get(f"{self.dashboard_url}/api/serve/deployments/")
        return response.json()


class PrometheusProfiler(BaseProfiler):
    """
    Prometheus metrics aggregator for cluster-wide monitoring.
    
    Example:
        profiler = PrometheusProfiler(prometheus_url="http://prometheus:9090")
        
        # Query GPU utilization across cluster
        results = profiler.query("avg(DCGM_FI_DEV_GPU_UTIL)")
        
        # Query range
        results = profiler.query_range(
            "sum(rate(vllm:request_success_total[5m]))",
            start="-1h",
            step="1m"
        )
    """
    
    def __init__(self, prometheus_url: str = "http://localhost:9090"):
        self.prometheus_url = prometheus_url
    
    @property
    def level(self) -> int:
        return 7
    
    @property
    def name(self) -> str:
        return "Prometheus"
    
    def is_available(self) -> bool:
        try:
            import requests
            response = requests.get(f"{self.prometheus_url}/-/healthy", timeout=5)
            return response.status_code == 200
        except Exception:
            return False
    
    def start(self) -> None:
        self._start_time = time.time()
    
    def stop(self) -> None:
        self._end_time = time.time()
    
    def get_results(self) -> Dict[str, Any]:
        return {}
    
    def query(self, promql: str) -> Dict[str, Any]:
        """Execute instant PromQL query."""
        import requests
        response = requests.get(
            f"{self.prometheus_url}/api/v1/query",
            params={"query": promql}
        )
        return response.json()
    
    def query_range(
        self,
        promql: str,
        start: str,
        end: str = "now",
        step: str = "1m"
    ) -> Dict[str, Any]:
        """Execute range PromQL query."""
        import requests
        response = requests.get(
            f"{self.prometheus_url}/api/v1/query_range",
            params={
                "query": promql,
                "start": start,
                "end": end,
                "step": step
            }
        )
        return response.json()


# =============================================================================
# Level 8: Production Monitoring
# =============================================================================

class ProductionMonitor(BaseProfiler):
    """
    Production monitoring combining multiple sources.
    
    Integrates:
    - DCGM for GPU health
    - vLLM metrics for inference
    - Prometheus for aggregation
    - OpenTelemetry for tracing
    
    Example:
        monitor = ProductionMonitor(
            dcgm_exporter="http://localhost:9400",
            vllm_endpoint="http://localhost:8000",
            prometheus_url="http://prometheus:9090"
        )
        
        # Check SLO compliance
        slo_status = monitor.check_slos({
            "ttft_p99_seconds": 2.0,
            "error_rate_percent": 1.0,
            "throughput_tps": 1000
        })
    """
    
    def __init__(
        self,
        dcgm_exporter: str = "http://localhost:9400",
        vllm_endpoint: str = "http://localhost:8000/metrics",
        prometheus_url: Optional[str] = None
    ):
        self.dcgm_exporter = dcgm_exporter
        self.vllm_endpoint = vllm_endpoint
        self.prometheus_url = prometheus_url
        
        self.dcgm_profiler = DCGMProfiler()
        self.vllm_profiler = VLLMMetricsProfiler(endpoint=vllm_endpoint)
        if prometheus_url:
            self.prometheus_profiler = PrometheusProfiler(prometheus_url)
    
    @property
    def level(self) -> int:
        return 8
    
    @property
    def name(self) -> str:
        return "ProductionMonitor"
    
    def is_available(self) -> bool:
        return self.dcgm_profiler.is_available() or self.vllm_profiler.is_available()
    
    def start(self) -> None:
        self.dcgm_profiler.start()
        self.vllm_profiler.start()
    
    def stop(self) -> None:
        self.dcgm_profiler.stop()
        self.vllm_profiler.stop()
    
    def get_results(self) -> Dict[str, Any]:
        return {
            "dcgm": self.dcgm_profiler.get_results(),
            "vllm": self.vllm_profiler.get_results()
        }
    
    def check_slos(self, slo_targets: Dict[str, float]) -> Dict[str, Dict]:
        """
        Check current metrics against SLO targets.
        
        Returns status for each SLO (passing/failing with current value).
        """
        results = {}
        vllm_summary = self.vllm_profiler.get_summary()
        
        for slo_name, target in slo_targets.items():
            current_value = vllm_summary.get(slo_name)
            if current_value is not None:
                results[slo_name] = {
                    "target": target,
                    "current": current_value,
                    "status": "passing" if current_value <= target else "failing"
                }
        
        return results
    
    def get_gpu_health(self) -> List[Dict[str, Any]]:
        """Get GPU health status including XID errors."""
        statuses = self.dcgm_profiler.get_gpu_statuses()
        return [
            {
                "gpu_id": s.gpu_id,
                "name": s.name,
                "healthy": s.temperature_celsius < 85,  # Example threshold
                "utilization": s.utilization_percent,
                "memory_usage_percent": s.memory_used_bytes / s.memory_total_bytes * 100,
                "temperature": s.temperature_celsius,
                "power": s.power_watts
            }
            for s in statuses
        ]


# =============================================================================
# Unified Profiler API
# =============================================================================

class UnifiedProfiler:
    """
    Unified profiling API that orchestrates all profiling levels.
    
    Example:
        profiler = UnifiedProfiler()
        
        # Profile specific levels
        with profiler.profile(levels=[3, 4, 6]):
            model.generate(inputs)
        
        # Get comprehensive report
        report = profiler.get_report()
        report.to_json("profile_report.json")
        
        # Print summary
        profiler.print_summary()
    """
    
    LEVEL_NAMES = {
        1: "Microarchitecture",
        2: "Memory Hierarchy",
        3: "Operator/Layer",
        4: "Intra-Node",
        5: "Inter-Node",
        6: "Batch Scheduling",
        7: "Cluster-Wide",
        8: "Production"
    }
    
    def __init__(self, auto_detect: bool = True):
        self.profilers: Dict[int, List[BaseProfiler]] = {i: [] for i in range(1, 9)}
        self._active_levels: List[int] = []
        self._start_time: Optional[float] = None
        self._end_time: Optional[float] = None
        
        if auto_detect:
            self._auto_register_profilers()
    
    def _auto_register_profilers(self):
        """Automatically register available profilers."""
        # Level 1: Microarchitecture
        ncu = NsightComputeProfiler()
        if ncu.is_available():
            self.register_profiler(ncu)
        
        rocm = ROCmComputeProfiler()
        if rocm.is_available():
            self.register_profiler(rocm)
        
        # Level 2: Memory
        mem = MemoryProfiler()
        if mem.is_available():
            self.register_profiler(mem)
        
        # Level 3: Operator/Layer
        pytorch = PyTorchProfiler()
        if pytorch.is_available():
            self.register_profiler(pytorch)
        
        nsys = NsightSystemsProfiler()
        if nsys.is_available():
            self.register_profiler(nsys)
        
        # Level 4: Intra-Node
        dcgm = DCGMProfiler()
        if dcgm.is_available():
            self.register_profiler(dcgm)
        
        smi = NvidiaSMIProfiler()
        if smi.is_available():
            self.register_profiler(smi)
        
        rocm_smi = ROCmSMIProfiler()
        if rocm_smi.is_available():
            self.register_profiler(rocm_smi)
        
        # Level 5: Inter-Node
        nccl = NCCLProfiler()
        if nccl.is_available():
            self.register_profiler(nccl)
        
        ib = InfiniBandProfiler()
        if ib.is_available():
            self.register_profiler(ib)
        
        # Level 6: Batch Scheduling
        vllm = VLLMMetricsProfiler()
        if vllm.is_available():
            self.register_profiler(vllm)
        
        # Level 7: Cluster-Wide
        ray = RayDashboardProfiler()
        if ray.is_available():
            self.register_profiler(ray)
        
        prometheus = PrometheusProfiler()
        if prometheus.is_available():
            self.register_profiler(prometheus)
    
    def register_profiler(self, profiler: BaseProfiler):
        """Register a profiler for its level."""
        self.profilers[profiler.level].append(profiler)
    
    def get_available_profilers(self) -> Dict[int, List[str]]:
        """Get available profilers by level."""
        return {
            level: [p.name for p in profilers]
            for level, profilers in self.profilers.items()
            if profilers
        }
    
    @contextmanager
    def profile(self, levels: Optional[List[int]] = None):
        """
        Context manager for multi-level profiling.
        
        Args:
            levels: List of levels to profile (1-8). If None, profiles all available.
        """
        self._active_levels = levels or list(range(1, 9))
        self._start_time = time.time()
        
        # Start all profilers for active levels
        for level in self._active_levels:
            for profiler in self.profilers[level]:
                try:
                    profiler.start()
                except Exception as e:
                    print(f"Warning: Failed to start {profiler.name}: {e}")
        
        try:
            yield self
        finally:
            # Stop all profilers
            for level in self._active_levels:
                for profiler in self.profilers[level]:
                    try:
                        profiler.stop()
                    except Exception as e:
                        print(f"Warning: Failed to stop {profiler.name}: {e}")
            
            self._end_time = time.time()
    
    def get_report(self) -> ProfilingReport:
        """Generate comprehensive profiling report."""
        report = ProfilingReport(
            levels_profiled=self._active_levels,
            duration_seconds=(self._end_time - self._start_time) if self._end_time else 0
        )
        
        # Collect results from all profilers
        for level in self._active_levels:
            for profiler in self.profilers[level]:
                try:
                    results = profiler.get_results()
                    report.metadata[f"{profiler.name}_level{level}"] = results
                except Exception as e:
                    report.metadata[f"{profiler.name}_error"] = str(e)
        
        # Extract GPU statuses from DCGM
        for profiler in self.profilers[4]:
            if isinstance(profiler, (DCGMProfiler, NvidiaSMIProfiler)):
                try:
                    if hasattr(profiler, 'get_gpu_statuses'):
                        report.gpu_statuses = profiler.get_gpu_statuses()
                except Exception:
                    pass
        
        return report
    
    def print_summary(self):
        """Print profiling summary to console."""
        print("\n" + "="*60)
        print("LLM INFERENCE PROFILING SUMMARY")
        print("="*60)
        
        if self._start_time and self._end_time:
            print(f"Duration: {self._end_time - self._start_time:.2f}s")
        
        print(f"\nLevels profiled: {self._active_levels}")
        
        available = self.get_available_profilers()
        print("\nProfilers used:")
        for level, names in available.items():
            if level in self._active_levels and names:
                print(f"  Level {level} ({self.LEVEL_NAMES[level]}): {', '.join(names)}")
        
        # Print key metrics from each level
        for level in self._active_levels:
            for profiler in self.profilers[level]:
                try:
                    results = profiler.get_results()
                    if results:
                        print(f"\n{profiler.name} (Level {level}):")
                        # Print first few keys
                        for key in list(results.keys())[:5]:
                            print(f"  {key}: {results[key]}")
                except Exception as e:
                    print(f"\n{profiler.name}: Error getting results - {e}")
        
        print("\n" + "="*60)


# =============================================================================
# Convenience Functions
# =============================================================================

def profile_llm_inference(
    inference_fn: Callable,
    levels: List[int] = [3, 4, 6],
    output_path: Optional[str] = None
) -> ProfilingReport:
    """
    Convenience function to profile LLM inference.
    
    Example:
        def run_inference():
            model.generate(["Hello, world!"])
        
        report = profile_llm_inference(run_inference, levels=[3, 4, 6])
    """
    profiler = UnifiedProfiler()
    
    with profiler.profile(levels=levels):
        inference_fn()
    
    report = profiler.get_report()
    
    if output_path:
        report.to_json(output_path)
    
    profiler.print_summary()
    return report


def quick_gpu_status() -> List[GPUStatus]:
    """Quick function to get current GPU status."""
    smi = NvidiaSMIProfiler()
    if smi.is_available():
        dcgm = DCGMProfiler()
        return dcgm.get_gpu_statuses()
    return []


def benchmark_nccl(
    operation: str = "allreduce",
    min_bytes: int = 8,
    max_bytes: int = 256 * 1024 * 1024,
    num_gpus: int = 8
) -> Dict[str, Any]:
    """Quick function to benchmark NCCL operations."""
    nccl = NCCLProfiler()
    if operation == "allreduce":
        return nccl.run_allreduce_benchmark(
            min_bytes=min_bytes,
            max_bytes=max_bytes,
            num_gpus=num_gpus
        )
    return {}


# =============================================================================
# Module Exports
# =============================================================================

__all__ = [
    # Data classes
    "MetricResult",
    "KernelProfile", 
    "OperatorProfile",
    "GPUStatus",
    "CollectiveProfile",
    "ProfilingReport",
    
    # Base classes
    "BaseProfiler",
    
    # Level 1: Microarchitecture
    "NsightComputeProfiler",
    "ROCmComputeProfiler",
    "CUPTIProfiler",
    
    # Level 2: Memory
    "MemoryProfiler",
    
    # Level 3: Operator/Layer
    "PyTorchProfiler",
    "NsightSystemsProfiler",
    "NVTXAnnotator",
    
    # Level 4: Intra-Node
    "DCGMProfiler",
    "NvidiaSMIProfiler",
    "ROCmSMIProfiler",
    
    # Level 5: Inter-Node
    "NCCLProfiler",
    "InfiniBandProfiler",
    
    # Level 6: Batch Scheduling
    "VLLMMetricsProfiler",
    "TRTLLMBenchmarkProfiler",
    
    # Level 7: Cluster-Wide
    "RayDashboardProfiler",
    "PrometheusProfiler",
    
    # Level 8: Production
    "ProductionMonitor",
    
    # Unified API
    "UnifiedProfiler",
    
    # Convenience functions
    "profile_llm_inference",
    "quick_gpu_status",
    "benchmark_nccl",
]


if __name__ == "__main__":
    # Demo usage
    print("LLM Inference Profiling API")
    print("="*40)
    
    profiler = UnifiedProfiler()
    available = profiler.get_available_profilers()
    
    print("\nAvailable profilers by level:")
    for level, names in available.items():
        if names:
            print(f"  Level {level}: {', '.join(names)}")
