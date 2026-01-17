import ctypes
from pathlib import Path


class F18Sim:
    def __init__(self, lib_path=None):
        self._handle = None
        self._lib = None
        self._lib = ctypes.CDLL(str(self._resolve_lib_path(lib_path)))
        self._bind()
        self._handle = self._lib.f18_create()
        if not self._handle:
            raise RuntimeError("Failed to create F18 simulator handle.")
        self.initialize()

    def _resolve_lib_path(self, lib_path):
        if lib_path:
            return Path(lib_path)
        env_path = Path.cwd() / "python" / "f18sim" / "libf18sim.dylib"
        if env_path.exists():
            return env_path
        here = Path(__file__).resolve().parent / "libf18sim.dylib"
        if here.exists():
            return here
        raise FileNotFoundError("libf18sim.dylib not found. Build with f18_wrap.mk.")

    def _bind(self):
        self._lib.f18_create.restype = ctypes.c_void_p
        self._lib.f18_destroy.argtypes = [ctypes.c_void_p]
        self._lib.f18_initialize.argtypes = [ctypes.c_void_p]
        self._lib.f18_reset.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
        ]
        self._lib.f18_get_num_states.restype = ctypes.c_int
        self._lib.f18_get_num_actions.restype = ctypes.c_int
        self._lib.f18_set_action.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
        ]
        self._lib.f18_set_action.restype = ctypes.c_int
        self._lib.f18_get_action.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
        ]
        self._lib.f18_get_action.restype = ctypes.c_int
        self._lib.f18_set_state.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
        ]
        self._lib.f18_set_state.restype = ctypes.c_int
        self._lib.f18_get_state.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
        ]
        self._lib.f18_get_state.restype = ctypes.c_int
        self._lib.f18_step.argtypes = [ctypes.c_void_p]
        self._lib.f18_step_u.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_double),
            ctypes.c_int,
        ]
        self._lib.f18_step_u.restype = None
        self._lib.f18_get_time.argtypes = [ctypes.c_void_p]
        self._lib.f18_get_time.restype = ctypes.c_double
        self._lib.f18_get_step_size.argtypes = [ctypes.c_void_p]
        self._lib.f18_get_step_size.restype = ctypes.c_double
        self._lib.f18_set_step_size.argtypes = [ctypes.c_void_p, ctypes.c_double]

    def initialize(self):
        self._lib.f18_initialize(self._handle)

    def reset(self, state=None):
        if state is None:
            self._lib.f18_reset(self._handle, None, 0)
            return
        buf = (ctypes.c_double * len(state))(*state)
        self._lib.f18_reset(self._handle, buf, len(state))

    def step(self):
        self._lib.f18_step(self._handle)

    def get_state(self):
        n = self._lib.f18_get_num_states()
        buf = (ctypes.c_double * n)()
        self._lib.f18_get_state(self._handle, buf, n)
        return [buf[i] for i in range(n)]

    def set_state(self, state):
        buf = (ctypes.c_double * len(state))(*state)
        return self._lib.f18_set_state(self._handle, buf, len(state))

    def get_num_states(self):
        return int(self._lib.f18_get_num_states())

    def get_num_actions(self):
        return int(self._lib.f18_get_num_actions())

    def set_action(self, u):
        # Action order: [ail, rud, elev, T]
        buf = (ctypes.c_double * len(u))(*u)
        return self._lib.f18_set_action(self._handle, buf, len(u))

    def get_action(self):
        n = self._lib.f18_get_num_actions()
        buf = (ctypes.c_double * n)()
        self._lib.f18_get_action(self._handle, buf, n)
        return [buf[i] for i in range(n)]

    def get_time(self):
        return self._lib.f18_get_time(self._handle)

    def get_step_size(self):
        return self._lib.f18_get_step_size(self._handle)

    def set_step_size(self, step_size):
        self._lib.f18_set_step_size(self._handle, float(step_size))

    def step(self, u=None):
        if u is None:
            self._lib.f18_step(self._handle)
            return
        buf = (ctypes.c_double * len(u))(*u)
        self._lib.f18_step_u(self._handle, buf, len(u))

    def close(self):
        if getattr(self, "_handle", None):
            self._lib.f18_destroy(self._handle)
            self._handle = None

    def __del__(self):
        self.close()
