import torch
import torch.distributed as dist
from datetime import timedelta
from ..util import log_tp


# Finite default timeout for the process group: rendezvous and every collective are
# bounded, so a failed peer surfaces as a timeout instead of a permanent hang.
DEFAULT_TIMEOUT_S = 600.0
# Teardown barrier is best-effort and strictly shorter than the group timeout: after a
# worker failure, close() must not stall the surviving ranks for the full timeout.
CLOSE_BARRIER_TIMEOUT_S = 30.0


class TPBackendRCCL:

    def __init__(
        self,
        device: int,
        active_devices: list[int],
        output_device: int,
        init_method: str,
        master: bool,
        uuid: str,
        shbuf_size: int = 0,
        dist_backend: str | None = None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        close_barrier_timeout_s: float = CLOSE_BARRIER_TIMEOUT_S,
    ):
        """
        Tensor-parallel communication backend implemented entirely on torch.distributed
        collectives. On HIP builds the backend name "nccl" is served by RCCL, so this is
        the ROCm TP transport.

        Unlike the CUDA TPBackendNCCL, nothing is delegated to the native pg_* shared-memory
        collectives: broadcast, all_reduce and both gather variants are complete torch.distributed
        operations here. master, uuid and shbuf_size are accepted for interface compatibility
        with the other backends (the model loader passes them to every backend type) but no
        shared-memory segments are created. The shared-arena host registration used to import
        module weights is HIP-translated independently of this backend and is unaffected.

        Device IDs map to ranks by position in active_devices, which is ordered with the output
        device last (the in-process pseudo-worker) and may otherwise be arbitrary. The CPU helper
        process (device < 0) joins nothing: it initializes no process group, touches no
        accelerator and runs no collective; its only backend interactions are the
        run_cpu_reduce_jobs / end_cpu_reduce_jobs no-ops.

        Correctness notes:
        - all_reduce keeps the tensor's own dtype. Float32 reductions stay float32; the CUDA
          NCCL backend's bf16 round-trip is deliberately not reproduced.
        - contribution=False zeroes the contribution buffer before the reduction, so a rank
          that contributes nothing sends exact zeros regardless of whether its buffer held
          uninitialized or NaN data.
        - Noncontiguous tensors are reduced/broadcast through a contiguous staging buffer and
          copied back with copy_, which handles strided outputs.
        - gather/gather_small are matched point-to-point transfers between each contributor and
          the output rank (a local copy for the output rank's own slice), so ranks outside
          gather_devices never enter the operation. Each contributor sends its payload with one
          send, and the output rank receives in gather_devices order, so latency scales with the
          number of contributors, not the number of collectives: for TP2 this is a single
          send/recv pair per gather.
        """
        self.device = device
        self.output_device = output_device
        self.active_devices = []
        self.rank = -1
        self.world_size = 0
        self.timeout_s = float(timeout_s)
        self.close_barrier_timeout_s = float(close_barrier_timeout_s)
        self.dist_backend = dist_backend or "nccl"
        # Partial init safety: close() only tears down a group that was actually created here,
        # and is idempotent. Both flags stay in their safe state if init below raises.
        self.pg_active = False
        self.closed = True

        if device < 0:
            log_tp(device, "RCCL init: skip CPU process")
            return

        self.active_devices = list(active_devices)
        self.world_size = len(self.active_devices)
        self.rank = self.active_devices.index(device)
        output_rank = self.active_devices.index(output_device)
        assert self.timeout_s > 0 and self.close_barrier_timeout_s > 0, \
            "RCCL backend timeouts must be finite and positive"

        log_tp(device, f"RCCL init: world_size {self.world_size}, rank {self.rank}, device {device}, "
                       f"output rank {output_rank}, backend {self.dist_backend}, init_method {init_method}")
        print(f" -- RCCL init: world_size {self.world_size}, rank {self.rank}, device {device}, "
              f"init_method {init_method}")
        if not dist.is_initialized():
            try:
                dist.init_process_group(
                    self.dist_backend,
                    rank = self.rank,
                    world_size = self.world_size,
                    init_method = init_method,
                    timeout = timedelta(seconds = self.timeout_s),
                )
            except Exception:
                log_tp(device, "RCCL init: process group init failed")
                raise
            self.pg_active = True
        elif dist.get_world_size() != self.world_size or dist.get_rank() != self.rank:
            raise RuntimeError(
                "RCCL backend: a process group is already initialized with rank/world different from "
                f"this TP split (existing rank {dist.get_rank()}, world {dist.get_world_size()})"
            )
        self.closed = False
        self.mp_warmup_rccl()


    def mp_warmup_rccl(self):
        """
        RCCL, like NCCL, initializes its internal state lazily on the first collective, which
        can take tens of seconds. Force that here, while every rank has just joined the group
        and no forward pass is in flight.
        """
        if self.world_size < 2:
            return
        print(f" -- RCCL warmup, device {self.device}, please wait...")
        if self.dist_backend == "nccl" and torch.cuda.is_available():
            x = torch.ones((6,), device = self.device)
        else:
            x = torch.ones((6,))
        dist.all_reduce(x)
        print(f" -- Finished RCCL warmup, device {self.device}")


    def close(self):
        if self.closed:
            return
        self.closed = True
        if not self.pg_active:
            # Never got as far as creating a group (partial init): nothing to tear down.
            return
        self.pg_active = False
        if not dist.is_initialized():
            return
        try:
            try:
                dist.barrier(timeout = timedelta(seconds = self.close_barrier_timeout_s))
            except TypeError:
                # torch without per-op timeout support: skip the best-effort drain barrier
                # rather than reintroducing an unbounded teardown stall.
                log_tp(self.device, "RCCL close: no per-op barrier timeout, skipping barrier")
        except Exception as e:
            log_tp(self.device, f"RCCL close: barrier failed ({e}), destroying process group anyway")
        finally:
            dist.destroy_process_group()


    def fwd_barrier(self):
        self._check_live("fwd_barrier")
        if self.world_size > 1:
            dist.barrier()


    def broadcast(self, tensor: torch.Tensor, src_device: int):
        self._check_live("broadcast")
        src_rank = self._rank_of(src_device)
        buf = self._stage_in(tensor)
        self._wait(dist.broadcast(buf, src = src_rank))
        self._stage_out(tensor, buf)


    def all_reduce(self, tensor: torch.Tensor, contribution: bool = True):
        # Every rank of the TP split enters this collective every time, in the same sequence
        # (the loader's contract); contribution=False means "sum the others, not me". The
        # tensor's dtype is passed to the collective unchanged: float32 stays float32.
        self._check_live("all_reduce")
        if self.world_size <= 1:
            if not contribution:
                tensor.zero_()
            return
        buf = self._stage_in(tensor)
        if not contribution:
            buf.zero_()
        self._wait(dist.all_reduce(buf))
        self._stage_out(tensor, buf)


    def gather(
        self,
        tensor: torch.Tensor,
        out_tensor: torch.Tensor | None,
        gather_devices: torch.Tensor | None,
        out_device: int,
        ldims: list[int]
    ):
        self._gather(tensor, out_tensor, gather_devices, out_device, ldims, "gather")


    def gather_small(
        self,
        tensor: torch.Tensor,
        out_tensor: torch.Tensor | None,
        gather_devices: torch.Tensor | None,
        out_device: int,
        ldims: list[int]
    ):
        self._gather(tensor, out_tensor, gather_devices, out_device, ldims, "gather_small")


    def run_cpu_reduce_jobs(self):
        # CPU-assisted reduction is a native-backend mechanism. RCCL all-reduce runs on the
        # GPU ranks and the CPU helper participates in nothing, so there is no work here.
        pass


    def end_cpu_reduce_jobs(self):
        pass


    def _check_live(self, op: str):
        if self.rank < 0:
            raise AssertionError(f"RCCL backend: {op} called from the CPU helper process")
        if self.closed:
            raise AssertionError(f"RCCL backend: {op} called after close()")


    def _rank_of(self, device: int) -> int:
        try:
            return self.active_devices.index(device)
        except ValueError:
            raise ValueError(
                f"RCCL backend: device {device} is not part of this TP split {self.active_devices}"
            ) from None


    def _stage_in(self, tensor: torch.Tensor) -> torch.Tensor:
        # torch.distributed collectives operate on contiguous storage. A noncontiguous
        # tensor is staged into a contiguous buffer; _stage_out copies the result back
        # with copy_, which is also what writes strided output slices correctly.
        if tensor.is_contiguous():
            return tensor
        buf = torch.empty(
            tensor.shape,
            dtype = tensor.dtype,
            device = tensor.device,
            memory_format = torch.contiguous_format,
        )
        buf.copy_(tensor)
        return buf


    def _stage_out(self, tensor: torch.Tensor, buf: torch.Tensor):
        if buf is not tensor:
            tensor.copy_(buf)


    def _wait(self, work):
        # Sync-mode returns are not always Work objects: dist.send yields None and
        # dist.recv yields the resolved source rank; both already blocked until
        # completion. Only genuine Work handles need an explicit wait.
        if work is not None and hasattr(work, "wait"):
            work.wait()


    def _gather(
        self,
        tensor: torch.Tensor,
        out_tensor: torch.Tensor | None,
        gather_devices,
        out_device: int,
        ldims: list[int],
        op: str,
    ):
        self._check_live(op)
        if gather_devices is None:
            raise AssertionError(f"RCCL backend: {op} requires the list of participating device IDs")
        devices = [int(d) for d in gather_devices]
        widths = [int(m) for m in ldims]
        assert len(devices) == len(widths), \
            f"{op}: gather_devices and ldims must have the same length"
        out_rank = self._rank_of(out_device)

        if self.rank == out_rank:
            assert out_tensor is not None, \
                f"{op}: Output device must supply output tensor"
            assert out_tensor.shape[-1] == sum(widths), \
                f"{op}: Output tensor must match size of concatenated slices: {sum(widths)}"
            offset = 0
            for src_device, width in zip(devices, widths):
                if width > 0:
                    dst = out_tensor.narrow(-1, offset, width)
                    if src_device == self.device:
                        assert tensor.shape[-1] == width, \
                            f"{op}: output rank tensor width {tensor.shape[-1]} != expected {width}"
                        dst.copy_(tensor)
                    else:
                        buf = torch.empty(dst.shape, dtype = dst.dtype, device = dst.device)
                        self._wait(dist.recv(buf, src = self._rank_of(src_device)))
                        dst.copy_(buf)
                offset += width
        else:
            if self.device not in devices:
                raise AssertionError(
                    f"{op}: rank device {self.device} is neither the output device nor a contributor"
                )
            width = widths[devices.index(self.device)]
            assert tensor.shape[-1] == width, \
                f"{op}: contributor width {tensor.shape[-1]} != expected {width}"
            if width > 0:
                buf = self._stage_in(tensor)
                self._wait(dist.send(buf, dst = out_rank))
