# OpenNR-IFのpythonラッパー
# Open以外の命令は基本的にOpenNR-IFのC++取説のように使えるようにする
import ctypes
from ctypes import wintypes

# CONNECT SPEC
NR_OBJECT_INTERNAL = 0  # Internal
NR_OBJECT_EXTERNAL_TCP = 1  # External(TCP)
NR_OBJECT_EXTERNAL_UDP = 2  # External(UDP)
NR_MODE_COMM_SERVER = 0  # Server For Communication
NR_MODE_COMM_CLIENT = 1  # Client For Communication
NR_DATA_REAL = 0  # Server For Communication
NR_DATA_XML = 1  # Client For Communication

# DATA ACCESS MODE
NR_ACCESS_WAIT = 0  # Wait
NR_ACCESS_NO_WAIT = 1  # No Wait

# Run/Stop
# For CtrlMotor
NR_STANBY_OFF_REQ = 0  # Stanby OFF
NR_STANBY_ON_REQ = 1  # Stanby ON
# For CtrlRun
NR_STOP_REQ = 0  # Stop
NR_RUN_REQ = 1  # Run

# SubId MAX（ For Set()/Get()）
NR_MAX_AXIS_STD = 8
NR_MAX_AXIS = 7
NR_MAX_XYZRPY = 6
NR_MAX_DIGITAL_SIG = 64
NR_MAX_ANALOG_CH = 16

# Priority
NR_PULL_MODE = -1
NR_PUSH_MODE_5 = 0
NR_PUSH_MODE_10 = 1
NR_PUSH_MODE_50 = 2
NR_PUSH_MODE_100 = 3
NR_PUSH_MODE_200 = 4
NR_PUSH_MODE_500 = 5
NR_PUSH_MODE_1000 = 6

# I/F RET CODE
NR_E_NORMAL = 0  # Normal
NR_E_ALREADY = 1001  # Al ready
NR_E_NONE_SEND = 1002  # None Send (Already Send)
NR_E_EXIT = 1003  # Process Exit

# I/F ERROR
NR_E_PARAM = -1001  # I/F Parameter Error
NR_E_INVALID = -1002  # I/F Invalid Error
NR_E_NONE_SUPPORT = -1003  # I/F None Support
NR_E_EXEC_FAIL = -1004  # Exec Error
NR_E_SEQ = -1005  # Sequence Error
NR_E_LICENSE = -1006  # LICENSE Error
NR_E_XML_SCHEMA = -1007  # XML schema Error
NR_E_VERSION_NONE_SUPPORT = -1010  # VERSION NONE SUPPORT

# I/F CONNECT ERROR
NR_E_SEND = -2001  # Send Error
NR_E_SEND_TIME_OUT = -2002  # Response Recv TimeOut After Send Error
NR_E_RECV = -2003  # Recv Error
NR_E_RETRY_OVER = -2004  # Retry Over
NR_E_RECV_TIME_OUT = -2005  # Recv TimeOut

# I/F UPDATE ERROR
NR_E_UDATE_FATAL = -3001  # fatal error
NR_E_UDATE_READONLY = -3002  # read only
NR_E_UDATE_MISMATCH = -3003  # mode	mismatch
NR_E_UDATE_MISSING = -3004  # missing data
NR_E_UDATE_UNITINVALID = -3005  # unit invalid

# I/F CTRL ERROR
NR_E_CTRL_FATAL = -4001  # fatal error
NR_E_CTRL_FORMAT = -4002  # format error
NR_E_CTRL_LIMIT = -4003  # limit error
NR_E_CTRL_EXECFAIL = -4004  # exec	error


# インターフェイス構造体


class NACHI_COMMIF_INFO(ctypes.Structure):
    _fields_ = [
        ("pcAddrs", ctypes.c_char_p),  # LPTSTR
        ("lPortNo", ctypes.c_long),
        ("lRetry", ctypes.c_long),
        ("lSendTimeOut", ctypes.c_long),
        ("lCommSide", ctypes.c_long),
        ("lMode", ctypes.c_long),
        ("lKind", ctypes.c_long),
    ]


class NR_NOTIFICATION(ctypes.Structure):
    _fields_ = [
        ("nErrCode", ctypes.c_int),
        ("nUnitNo", ctypes.c_int),
        ("nMechNo", ctypes.c_int),
        ("nAxisNo", ctypes.c_int),
        ("csProg", ctypes.c_char_p),  # LPTSTR
        ("nStepNo", ctypes.c_int),
    ]


class NR_SHIFT(ctypes.Structure):
    _fields_ = [
        ("fX", ctypes.c_float),
        ("fY", ctypes.c_float),
        ("fZ", ctypes.c_float),
        ("fRoll", ctypes.c_float),
        ("fPitch", ctypes.c_float),
        ("fYaw", ctypes.c_float),
    ]


class NR_POSE(ctypes.Structure):
    _fields_ = [
        ("fX", ctypes.c_float),
        ("fY", ctypes.c_float),
        ("fZ", ctypes.c_float),
        ("fRoll", ctypes.c_float),
        ("fPitch", ctypes.c_float),
        ("fYaw", ctypes.c_float),
    ]


class NR_PALLETREG(ctypes.Structure):
    _fields_ = [
        ("PalletID", ctypes.c_int),
        ("Execute", ctypes.c_int),
        ("Kind", ctypes.c_int),
        ("Layer", ctypes.c_int),
        ("Work", ctypes.c_int),
    ]


class NR_PALLETWORK(ctypes.Structure):
    _fields_ = [
        ("PalletID", ctypes.c_float),
        ("Length", ctypes.c_float),
        ("Width", ctypes.c_float),
        ("Height", ctypes.c_float),
        ("dL", ctypes.c_float),
        ("dW", ctypes.c_float),
        ("Xa", ctypes.c_float),
        ("Ya", ctypes.c_float),
    ]


class Layer(ctypes.Structure):
    _fields_ = [
        ("PleneID", ctypes.c_float),
        ("Height", ctypes.c_float),
    ]


class NR_PALLETLAYER(ctypes.Structure):
    _fields_ = [
        ("LayerNum", ctypes.c_float),
        ("TotalHeight", ctypes.c_float),
        ("LayerType", ctypes.c_float),
        ("Layer", Layer * 50),
    ]


class Work(ctypes.Structure):
    _fields_ = [
        ("PosX", ctypes.c_float),
        ("PosY", ctypes.c_float),
        ("PosZ", ctypes.c_float),
        ("ThetaZ", ctypes.c_float),
        ("Approach", ctypes.c_float),
    ]


class NR_PALLETPLENE(ctypes.Structure):
    _fields_ = [
        ("Type", ctypes.c_float),
        ("WorkNum", ctypes.c_float * 2),
        ("Shift", ctypes.c_float * 2),
        ("Work", Work * 99),
    ]


class NR_POSE_CONF(ctypes.Structure):
    _fields_ = [
        ("fX", ctypes.c_float),
        ("fY", ctypes.c_float),
        ("fZ", ctypes.c_float),
        ("fRoll", ctypes.c_float),
        ("fPitch", ctypes.c_float),
        ("fYaw", ctypes.c_float),
        ("unConfig", ctypes.c_uint),
    ]


class NR_ANGLE(ctypes.Structure):
    _fields_ = [
        ("fJ1", ctypes.c_float),
        ("fJ2", ctypes.c_float),
        ("fJ3", ctypes.c_float),
        ("fJ4", ctypes.c_float),
        ("fJ5", ctypes.c_float),
        ("fJ6", ctypes.c_float),
    ]


class NR_USERCOORD_ORG(ctypes.Structure):
    _fields_ = [
        ("fX", ctypes.c_float),
        ("fY", ctypes.c_float),
        ("fZ", ctypes.c_float),
        ("fAngX", ctypes.c_float),
        ("fAngY", ctypes.c_float),
        ("fAngZ", ctypes.c_float),
    ]


class NR_TOOL_PARAM(ctypes.Structure):
    _fields_ = [
        ("fX", ctypes.c_float),
        ("fY", ctypes.c_float),
        ("fZ", ctypes.c_float),
        ("fAngX", ctypes.c_float),
        ("fAngY", ctypes.c_float),
        ("fAngZ", ctypes.c_float),
    ]


# 外部トラッキング


class NR_GET_CTRL_INFO(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("ushEstopBit", ctypes.c_ushort, 1),
        ("ushPlaybkBit", ctypes.c_ushort, 1),
        ("ushConnectBit", ctypes.c_ushort, 1),
        ("ushErrorBit", ctypes.c_ushort, 5),
        ("ushMotorBit", ctypes.c_ushort, 1),
        ("ushRsv", ctypes.c_ushort, 6),
        ("ushProtcolBit", ctypes.c_ushort, 1),  # Protocol should be Protocol
    ]


class NR_GET_REAL_DATA_BODY_EXT(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("fCurTcpPos", ctypes.c_float * NR_MAX_AXIS),
        ("fComTcpPos", ctypes.c_float * NR_MAX_AXIS),
        ("fWorkCoord", ctypes.c_float * NR_MAX_XYZRPY),
        ("fToolCoord", ctypes.c_float * NR_MAX_XYZRPY),
        ("fCurAngle", ctypes.c_float * NR_MAX_AXIS),
        ("fComAngle", ctypes.c_float * NR_MAX_AXIS),
        ("fTorque", ctypes.c_float * NR_MAX_AXIS),
        ("fCurrent", ctypes.c_float * NR_MAX_AXIS),
        ("bDigOut", ctypes.c_bool * NR_MAX_DIGITAL_SIG),
        ("bDigIn", ctypes.c_bool * NR_MAX_DIGITAL_SIG),
        ("fAnaOut", ctypes.c_float * NR_MAX_ANALOG_CH),
        ("fAnaIn", ctypes.c_float * NR_MAX_ANALOG_CH),
    ]


class NR_GET_REAL_DATA_BODY_STD(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("fCurTcpPos", ctypes.c_float * NR_MAX_AXIS_STD),
        ("fCurAngle", ctypes.c_float * NR_MAX_AXIS_STD),
        ("fCurrent", ctypes.c_float * NR_MAX_AXIS_STD),
        ("bDigOut", ctypes.c_bool * NR_MAX_DIGITAL_SIG),
        ("bDigIn", ctypes.c_bool * NR_MAX_DIGITAL_SIG),
    ]


class _NR_GET_REAL_DATA_UNION(ctypes.Union):
    _pack_ = 1
    _fields_ = [
        ("stExt", NR_GET_REAL_DATA_BODY_EXT),
        ("stStd", NR_GET_REAL_DATA_BODY_STD),
    ]


class NR_GET_REAL_DATA_ALL(ctypes.Structure):
    _pack_ = 1
    _anonymous_ = ("ustData",)
    _fields_ = [
        ("stCtrl", NR_GET_CTRL_INFO),
        ("nTime", ctypes.c_int),
        ("ustData", _NR_GET_REAL_DATA_UNION),
    ]


class NR_SET_CTRL_INFO(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("ushEstopBit", ctypes.c_ushort, 1),
        ("ushFinishBit", ctypes.c_ushort, 1),
        ("ushOrderBit", ctypes.c_ushort, 1),
        ("ushRsv", ctypes.c_ushort, 12),
        ("ushProtcolBit", ctypes.c_ushort, 1),  # Protocol should be Protocol
    ]


class NR_SET_REAL_DATA_BODY_EXT(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("fComTcpPos", ctypes.c_float * NR_MAX_AXIS),
        ("fWorkCoord", ctypes.c_float * NR_MAX_XYZRPY),
        ("fToolCoord", ctypes.c_float * NR_MAX_XYZRPY),
        ("fComAngle", ctypes.c_float * NR_MAX_AXIS),
        ("bDigOut", ctypes.c_bool * NR_MAX_DIGITAL_SIG),
        ("fAnaOut", ctypes.c_float * NR_MAX_ANALOG_CH),
    ]


class NR_SET_REAL_DATA_BODY_STD(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("fComTcpPos", ctypes.c_float * NR_MAX_AXIS_STD),
        ("fWorkCoord", ctypes.c_float * NR_MAX_XYZRPY),
        ("fToolCoord", ctypes.c_float * NR_MAX_XYZRPY),
        ("fComAngle", ctypes.c_float * NR_MAX_AXIS_STD),
        ("bDigOut", ctypes.c_bool * NR_MAX_DIGITAL_SIG),
    ]


class _NR_SET_REAL_DATA_UNION(ctypes.Union):
    _pack_ = 1
    _fields_ = [
        ("stExt", NR_SET_REAL_DATA_BODY_EXT),
        ("stStd", NR_SET_REAL_DATA_BODY_STD),
    ]


class NR_SET_REAL_DATA_ALL(ctypes.Structure):
    _pack_ = 1
    _anonymous_ = ("ustData",)
    _fields_ = [
        ("stCtrl", NR_SET_CTRL_INFO),
        ("nTime", ctypes.c_int),
        ("ustData", _NR_SET_REAL_DATA_UNION),
    ]


class OpenNRIF:
    def __init__(
        self,
        dllpath: str | None = None,
    ):
        if dllpath is not None:
            self.LoadDLL(dllpath)

    # DLL 読み込み
    def LoadDLL(
        self,
        dllpath: str | None = "C:/Program Files/nachi/OpenNR-IF/OpenNR-IF.dll",
    ):
        if dllpath is not None:
            # dll読み込み
            self.dllpath = dllpath
            self.dll = ctypes.cdll.LoadLibrary(self.dllpath)
        else:
            raise ValueError("No DLL path given")

    # 内部関数
    # BOOL関係(boolではなくwintypes.BOOLなので変換する)
    def _BoolSingle(
        self,
        func,
        value: list[bool],
        nPriority: int,
        fThreshold: float,
        nOpenId: int | None = None,
    ) -> int:
        func.argtypes = (
            ctypes.c_int,
            ctypes.POINTER(wintypes.BOOL),
            ctypes.c_int,
            ctypes.c_float,
        )
        func.restype = ctypes.c_int
        if nOpenId is None:
            nOpenId = self.nOpenId
        ints = [1 if v else 0 for v in value]
        BoolArray = wintypes.BOOL * len(ints)
        c_arr = BoolArray(*ints)
        ret = func(
            ctypes.c_int(nOpenId),
            c_arr,
            ctypes.c_int(nPriority),
            ctypes.c_float(fThreshold),
        )
        value[:] = [bool(x) for x in c_arr]
        return ret

    def _BoolSingleUpdate(
        self,
        func,
        value: list[bool],
        bUpdate: bool,
        nPriority: int,
        fThreshold: float,
        nOpenId: int | None = None,
    ) -> int:
        func.argtypes = (
            ctypes.c_int,
            ctypes.POINTER(wintypes.BOOL),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
        )
        func.restype = ctypes.c_int
        if nOpenId is None:
            nOpenId = self.nOpenId
        ints = [1 if v else 0 for v in value]
        BoolArray = wintypes.BOOL * len(ints)
        c_arr = BoolArray(*ints)
        ret = func(
            ctypes.c_int(nOpenId),
            c_arr,
            wintypes.BOOL(bUpdate),
            ctypes.c_int(nPriority),
            ctypes.c_float(fThreshold),
        )
        value[:] = [bool(x) for x in c_arr]
        return ret

    def _BoolSingleUnitUpdate(
        self,
        func,
        value: list[bool],
        bUpdate: bool,
        nUnitId: int,
        nPriority: int,
        fThreshold: float,
        nOpenId: int | None = None,
    ) -> int:
        func.argtypes = (
            ctypes.c_int,
            ctypes.POINTER(wintypes.BOOL),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
        )
        func.restype = ctypes.c_int
        if nOpenId is None:
            nOpenId = self.nOpenId
        ints = [1 if v else 0 for v in value]
        BoolArray = wintypes.BOOL * len(ints)
        c_arr = BoolArray(*ints)
        ret = func(
            ctypes.c_int(nOpenId),
            c_arr,
            wintypes.BOOL(bUpdate),
            ctypes.c_int(nUnitId),
            ctypes.c_int(nPriority),
            ctypes.c_float(fThreshold),
        )
        value[:] = [bool(x) for x in c_arr]
        return ret

    def _BoolMultiple(
        self,
        func,
        value: list[bool],
        nSubId: int,
        nCount: int,
        nPriority: int,
        fThreshold: float,
        nOpenId: int | None = None,
    ) -> int:
        func.argtypes = (
            ctypes.c_int,
            ctypes.POINTER(wintypes.BOOL),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
        )
        func.restype = ctypes.c_int
        if nOpenId is None:
            nOpenId = self.nOpenId
        ints = [1 if v else 0 for v in value]
        BoolArray = wintypes.BOOL * len(ints)
        c_arr = BoolArray(*ints)
        ret = func(
            ctypes.c_int(nOpenId),
            c_arr,
            ctypes.c_int(nSubId),
            ctypes.c_int(nCount),
            ctypes.c_int(nPriority),
            ctypes.c_float(fThreshold),
        )
        value[:] = [bool(x) for x in c_arr]
        return ret

    def _BoolMultipleUpdate(
        self,
        func,
        value: list[bool],
        bUpdate: bool,
        nSubId: int,
        nCount: int,
        nPriority: int,
        fThreshold: float,
        nOpenId: int | None = None,
    ) -> int:
        func.argtypes = (
            ctypes.c_int,
            ctypes.POINTER(wintypes.BOOL),
            wintypes.BOOL,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
        )
        func.restype = ctypes.c_int
        if nOpenId is None:
            nOpenId = self.nOpenId
        ints = [1 if v else 0 for v in value]
        BoolArray = wintypes.BOOL * len(ints)
        c_arr = BoolArray(*ints)
        ret = func(
            ctypes.c_int(nOpenId),
            c_arr,
            wintypes.BOOL(bUpdate),
            ctypes.c_int(nSubId),
            ctypes.c_int(nCount),
            ctypes.c_int(nPriority),
            ctypes.c_float(fThreshold),
        )
        value[:] = [bool(x) for x in c_arr]
        return ret

    # Int関係
    def _IntSingle(
        self,
        func,
        value: list[int],
        nPriority: int,
        fThreshold: float,
        nOpenId: int | None = None,
    ) -> int:
        func.argtypes = (
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_float,
        )
        func.restype = ctypes.c_int
        if nOpenId is None:
            nOpenId = self.nOpenId
        IntArray = ctypes.c_int * len(value)
        c_arr = IntArray(*value)
        ret = func(
            ctypes.c_int(nOpenId),
            c_arr,
            ctypes.c_int(nPriority),
            ctypes.c_float(fThreshold),
        )
        value[:] = list(c_arr)
        return ret

    def _IntSingleUpdate(
        self,
        func,
        value: list[int],
        bUpdate: bool,
        nPriority: int,
        fThreshold: float,
        nOpenId: int | None = None,
    ) -> int:
        func.argtypes = (
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            wintypes.BOOL,
            ctypes.c_int,
            ctypes.c_float,
        )
        func.restype = ctypes.c_int
        if nOpenId is None:
            nOpenId = self.nOpenId
        IntArray = ctypes.c_int * len(value)
        c_arr = IntArray(*value)
        ret = func(
            ctypes.c_int(nOpenId),
            c_arr,
            wintypes.BOOL(bUpdate),
            ctypes.c_int(nPriority),
            ctypes.c_float(fThreshold),
        )
        value[:] = list(c_arr)
        return ret

    def _IntSingleUnit(
        self,
        func,
        value: list[int],
        nUnitId: int,
        nPriority: int,
        fThreshold: float,
        nOpenId: int | None = None,
    ) -> int:
        func.argtypes = (
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
        )
        func.restype = ctypes.c_int
        if nOpenId is None:
            nOpenId = self.nOpenId
        IntArray = ctypes.c_int * len(value)
        c_arr = IntArray(*value)
        ret = func(
            ctypes.c_int(nOpenId),
            c_arr,
            ctypes.c_int(nUnitId),
            ctypes.c_int(nPriority),
            ctypes.c_float(fThreshold),
        )
        value[:] = list(c_arr)
        return ret

    def _IntSingleUnitUpdate(
        self,
        func,
        value: list[int],
        bUpdate: bool,
        nUnitId: int,
        nPriority: int,
        fThreshold: float,
        nOpenId: int | None = None,
    ) -> int:
        func.argtypes = (
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            wintypes.BOOL,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
        )
        func.restype = ctypes.c_int
        if nOpenId is None:
            nOpenId = self.nOpenId
        IntArray = ctypes.c_int * len(value)
        c_arr = IntArray(*value)
        ret = func(
            ctypes.c_int(nOpenId),
            c_arr,
            wintypes.BOOL(bUpdate),
            ctypes.c_int(nUnitId),
            ctypes.c_int(nPriority),
            ctypes.c_float(fThreshold),
        )
        value[:] = list(c_arr)
        return ret

    def _IntMultiple(
        self,
        func,
        value: list[int],
        nSubId: int,
        nCount: int,
        nPriority: int,
        fThreshold: float,
        nOpenId: int | None = None,
    ) -> int:
        func.argtypes = (
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
        )
        func.restype = ctypes.c_int
        if nOpenId is None:
            nOpenId = self.nOpenId
        IntArray = ctypes.c_int * len(value)
        c_arr = IntArray(*value)
        ret = func(
            ctypes.c_int(nOpenId),
            c_arr,
            ctypes.c_int(nSubId),
            ctypes.c_int(nCount),
            ctypes.c_int(nPriority),
            ctypes.c_float(fThreshold),
        )
        value[:] = list(c_arr)
        return ret

    def _IntMultipleUpdate(
        self,
        func,
        value: list[int],
        bUpdate: bool,
        nSubId: int,
        nCount: int,
        nPriority: int,
        fThreshold: float,
        nOpenId: int | None = None,
    ) -> int:
        func.argtypes = (
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            wintypes.BOOL,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
        )
        func.restype = ctypes.c_int
        if nOpenId is None:
            nOpenId = self.nOpenId
        IntArray = ctypes.c_int * len(value)
        c_arr = IntArray(*value)
        ret = func(
            ctypes.c_int(nOpenId),
            c_arr,
            wintypes.BOOL(bUpdate),
            ctypes.c_int(nSubId),
            ctypes.c_int(nCount),
            ctypes.c_int(nPriority),
            ctypes.c_float(fThreshold),
        )
        value[:] = list(c_arr)
        return ret

    def _IntSingleSubUpdate(
        self,
        func,
        value: list[int],
        bUpdate: bool,
        nSubId: int,
        nPriority: int,
        fThreshold: float,
        nOpenId: int | None = None,
    ) -> int:
        func.argtypes = (
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            wintypes.BOOL,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
        )
        func.restype = ctypes.c_int
        if nOpenId is None:
            nOpenId = self.nOpenId
        IntArray = ctypes.c_int * len(value)
        c_arr = IntArray(*value)
        ret = func(
            ctypes.c_int(nOpenId),
            c_arr,
            wintypes.BOOL(bUpdate),
            ctypes.c_int(nSubId),
            ctypes.c_int(nPriority),
            ctypes.c_float(fThreshold),
        )
        value[:] = list(c_arr)
        return ret

    # Float関係
    def _FloatSingle(
        self,
        func,
        value: list[float],
        nPriority: int,
        fThreshold: float,
        nOpenId: int | None = None,
    ) -> int:
        func.argtypes = (
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_float,
        )
        func.restype = ctypes.c_int
        if nOpenId is None:
            nOpenId = self.nOpenId
        FloatArray = ctypes.c_float * len(value)
        c_arr = FloatArray(*value)
        ret = func(
            ctypes.c_int(nOpenId),
            c_arr,
            ctypes.c_int(nPriority),
            ctypes.c_float(fThreshold),
        )
        value[:] = list(c_arr)
        return ret

    def _FloatSingleUnit(
        self,
        func,
        value: list[float],
        nUnitId: int,
        nPriority: int,
        fThreshold: float,
        nOpenId: int | None = None,
    ) -> int:
        func.argtypes = (
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
        )
        func.restype = ctypes.c_int
        if nOpenId is None:
            nOpenId = self.nOpenId
        FloatArray = ctypes.c_float * len(value)
        c_arr = FloatArray(*value)
        ret = func(
            ctypes.c_int(nOpenId),
            c_arr,
            ctypes.c_int(nUnitId),
            ctypes.c_int(nPriority),
            ctypes.c_float(fThreshold),
        )
        value[:] = list(c_arr)
        return ret

    def _FloatMultiple(
        self,
        func,
        value: list[float],
        nSubId: int,
        nCount: int,
        nPriority: int,
        fThreshold: float,
        nOpenId: int | None = None,
    ) -> int:
        func.argtypes = (
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
        )
        func.restype = ctypes.c_int
        if nOpenId is None:
            nOpenId = self.nOpenId
        FloatArray = ctypes.c_float * len(value)
        c_arr = FloatArray(*value)
        ret = func(
            ctypes.c_int(nOpenId),
            c_arr,
            ctypes.c_int(nSubId),
            ctypes.c_int(nCount),
            ctypes.c_int(nPriority),
            ctypes.c_float(fThreshold),
        )
        value[:] = list(c_arr)
        return ret

    def _FloatMultipleUpdate(
        self,
        func,
        value: list[float],
        bUpdate: bool,
        nSubId: int,
        nCount: int,
        nPriority: int,
        fThreshold: float,
        nOpenId: int | None = None,
    ) -> int:
        func.argtypes = (
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            wintypes.BOOL,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
        )
        func.restype = ctypes.c_int
        if nOpenId is None:
            nOpenId = self.nOpenId
        FloatArray = ctypes.c_float * len(value)
        c_arr = FloatArray(*value)
        ret = func(
            ctypes.c_int(nOpenId),
            c_arr,
            wintypes.BOOL(bUpdate),
            ctypes.c_int(nSubId),
            ctypes.c_int(nCount),
            ctypes.c_int(nPriority),
            ctypes.c_float(fThreshold),
        )
        value[:] = list(c_arr)
        return ret

    def _FloatMultipleUnit(
        self,
        func,
        value: list[float],
        nSubId: int,
        nCount: int,
        nUnitId: int,
        nPriority: int,
        fThreshold: float,
        nOpenId: int | None = None,
    ) -> int:
        func.argtypes = (
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
        )
        func.restype = ctypes.c_int
        if nOpenId is None:
            nOpenId = self.nOpenId
        FloatArray = ctypes.c_float * len(value)
        c_arr = FloatArray(*value)
        ret = func(
            ctypes.c_int(nOpenId),
            c_arr,
            ctypes.c_int(nSubId),
            ctypes.c_int(nCount),
            ctypes.c_int(nUnitId),
            ctypes.c_int(nPriority),
            ctypes.c_float(fThreshold),
        )
        value[:] = list(c_arr)
        return ret

    # String関係
    def _StringSingleUnit(
        self,
        func,
        value: list[str],  # 要素数1限定
        szBufSize: int,
        nSubId: int,
        nUnitId: int,
        nPriority: int,
        fThreshold: float,
        nOpenId: int | None = None,
    ) -> int:
        func.argtypes = (
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
        )
        func.restype = ctypes.c_int
        if nOpenId is None:
            nOpenId = self.nOpenId
        if not isinstance(value, list) or len(value) == 0:  # 要素数確認
            raise ValueError(
                "value must be a list with at least one element (value[0])"
            )
        buf_size = int(szBufSize) + 1  # NULL 終端分を確保
        # 入力文字列をバッファにコピー（切り詰め）
        buf = ctypes.create_string_buffer(buf_size)
        if value[0] is not None:
            b = value[0].encode("utf-8")[: buf_size - 1]
            ctypes.memmove(buf, b, len(b))
            buf[len(b)] = 0

        # 呼び出し
        ret = func(
            nOpenId,
            buf,
            ctypes.c_size_t(buf_size),
            ctypes.c_int(nSubId),
            ctypes.c_int(nUnitId),
            ctypes.c_int(nPriority),
            ctypes.c_float(fThreshold),
        )

        # バッファの中身を取り出して文字列化して返す
        try:
            val = buf.value.decode("utf-8", errors="ignore")
        except Exception:
            val = buf.value.decode("latin-1", errors="ignore")
        value[0] = val
        return ret

    def _StringMulti(
        self,
        func,
        value: list[str],
        szBufSize: int,
        nSubId: int,
        nCount: int,
        nPriority: int,
        fThreshold: float,
        nOpenId: int | None = None,
    ) -> int:
        func.argtypes = (
            ctypes.c_int,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_char)),
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
        )
        func.restype = ctypes.c_int
        if nOpenId is None:
            nOpenId = self.nOpenId

        buf_size = int(szBufSize) + 1  # バッファ作成Null終端のため+1
        num = max(1, int(nCount))  # C に渡す要素数
        # truncate or pad the input list to num
        vals = (value or [])[:num] + [""] * max(0, num - len(value or []))

        # create buffers for each element
        buffers = [ctypes.create_string_buffer(buf_size) for _ in range(num)]
        for i, s in enumerate(vals):
            if s is not None:
                b = s.encode("utf-8")[: buf_size - 1]
                ctypes.memmove(buffers[i], b, len(b))
                buffers[i][len(b)] = 0

        ptr_type = (
            ctypes.POINTER(ctypes.c_char) * num
        )  # build array of char* (pointer to each buffer)
        ptrs = ptr_type(
            *[ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)) for buf in buffers]
        )
        char_pp = ctypes.cast(
            ptrs, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))
        )  # cast to char** (pointer to pointer) expected by argtypes

        ret = func(
            ctypes.c_int(nOpenId),
            char_pp,
            ctypes.c_size_t(buf_size),
            ctypes.c_int(nSubId),
            ctypes.c_int(nCount),
            ctypes.c_int(nPriority),
            ctypes.c_float(fThreshold),
        )
        for i in range(num):
            try:
                vals[i] = buffers[i].value.decode("utf-8", errors="ignore")
            except Exception:
                vals[i] = buffers[i].value.decode("latin-1", errors="ignore")
        # replace original list contents (caller-visible mutation)
        value[:] = vals
        return ret

    def _StringMultiUpdate(
        self,
        func,
        value: list[str],
        szBufSize: int,
        bUpdate: bool,
        nSubId: int,
        nCount: int,
        nPriority: int,
        fThreshold: float,
        nOpenId: int | None = None,
    ) -> int:
        func.argtypes = (
            ctypes.c_int,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_char)),
            ctypes.c_size_t,
            wintypes.BOOL,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
        )
        func.restype = ctypes.c_int
        if nOpenId is None:
            nOpenId = self.nOpenId

        buf_size = int(szBufSize) + 1  # バッファ作成Null終端のため+1
        num = max(1, int(nCount))  # C に渡す要素数
        # truncate or pad the input list to num
        vals = (value or [])[:num] + [""] * max(0, num - len(value or []))

        # create buffers for each element
        buffers = [ctypes.create_string_buffer(buf_size) for _ in range(num)]
        for i, s in enumerate(vals):
            if s is not None:
                b = s.encode("utf-8")[: buf_size - 1]
                ctypes.memmove(buffers[i], b, len(b))
                buffers[i][len(b)] = 0

        ptr_type = (
            ctypes.POINTER(ctypes.c_char) * num
        )  # build array of char* (pointer to each buffer)
        ptrs = ptr_type(
            *[ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)) for buf in buffers]
        )
        char_pp = ctypes.cast(
            ptrs, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))
        )  # cast to char** (pointer to pointer) expected by argtypes

        ret = func(
            ctypes.c_int(nOpenId),
            char_pp,
            ctypes.c_size_t(buf_size),
            wintypes.BOOL(bUpdate),
            ctypes.c_int(nSubId),
            ctypes.c_int(nCount),
            ctypes.c_int(nPriority),
            ctypes.c_float(fThreshold),
        )
        for i in range(num):
            try:
                vals[i] = buffers[i].value.decode("utf-8", errors="ignore")
            except Exception:
                vals[i] = buffers[i].value.decode("latin-1", errors="ignore")
        # replace original list contents (caller-visible mutation)
        value[:] = vals
        return ret

    def _StringMultiUpdateUnit(
        self,
        func,
        value: list[str],
        szBufSize: int,
        bUpdate: bool,
        nSubId: int,
        nCount: int,
        nUnitId: int,
        nPriority: int,
        fThreshold: float,
        nOpenId: int | None = None,
    ) -> int:
        func.argtypes = (
            ctypes.c_int,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_char)),
            ctypes.c_size_t,
            wintypes.BOOL,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
        )
        func.restype = ctypes.c_int
        if nOpenId is None:
            nOpenId = self.nOpenId

        buf_size = int(szBufSize) + 1  # バッファ作成Null終端のため+1
        num = max(1, int(nCount))  # C に渡す要素数
        # truncate or pad the input list to num
        vals = (value or [])[:num] + [""] * max(0, num - len(value or []))

        # create buffers for each element
        buffers = [ctypes.create_string_buffer(buf_size) for _ in range(num)]
        for i, s in enumerate(vals):
            if s is not None:
                b = s.encode("utf-8")[: buf_size - 1]
                ctypes.memmove(buffers[i], b, len(b))
                buffers[i][len(b)] = 0

        ptr_type = (
            ctypes.POINTER(ctypes.c_char) * num
        )  # build array of char* (pointer to each buffer)
        ptrs = ptr_type(
            *[ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)) for buf in buffers]
        )
        char_pp = ctypes.cast(
            ptrs, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))
        )  # cast to char** (pointer to pointer) expected by argtypes

        ret = func(
            ctypes.c_int(nOpenId),
            char_pp,
            ctypes.c_size_t(buf_size),
            wintypes.BOOL(bUpdate),
            ctypes.c_int(nSubId),
            ctypes.c_int(nCount),
            ctypes.c_int(nUnitId),
            ctypes.c_int(nPriority),
            ctypes.c_float(fThreshold),
        )
        for i in range(num):
            try:
                vals[i] = buffers[i].value.decode("utf-8", errors="ignore")
            except Exception:
                vals[i] = buffers[i].value.decode("latin-1", errors="ignore")
        # replace original list contents (caller-visible mutation)
        value[:] = vals
        return ret

    # 環境/制御グループ
    def Open(
        self,
        pcAddrs,
        lPortNo=0,
        lRetry=0,
        lSendTimeOut=5000,
        lCommSide=0,
        lMode=0,
        lKind=NR_DATA_XML,
    ) -> int:  # 接続 (NACHI_COMMIF_INFO + dllのパス)
        # open
        self.Info = NACHI_COMMIF_INFO(
            pcAddrs.encode(), lPortNo, lRetry, lSendTimeOut, lCommSide, lMode, lKind
        )
        self.nOpenId = self.dll.NR_Open(ctypes.pointer(self.Info))
        return self.nOpenId

    def Close(
        self,
        nOpenId: int | None = None,
    ) -> int:
        if nOpenId is None:
            nOpenId = self.nOpenId
        return self.dll.NR_Close(nOpenId)

    def SetNotify(
        self,
        Notify: NR_NOTIFICATION,
        nOpenId: int | None = None,
    ) -> int:
        if nOpenId is None:
            nOpenId = self.nOpenId
        return self.dll.NR_SetNotify(nOpenId, ctypes.pointer(self.Notify))

    def Send(
        self,
        sendBuf: str,
        nBufSize: int | None = None,
        nOpenId: int | None = None,
    ) -> int:
        self.dll.NR_Send.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
        )
        self.dll.NR_Send.restype = ctypes.c_int

        if nOpenId is None:
            nOpenId = 0
        buf_bytes = sendBuf.encode("utf-8")
        if nBufSize is None:
            nBufSize = len(buf_bytes)

        if nBufSize > len(buf_bytes):
            buf_bytes = buf_bytes + b"\x00" * (nBufSize - len(buf_bytes))
        return self.dll.NR_Send(
            ctypes.c_int(nOpenId), ctypes.c_char_p(buf_bytes), ctypes.c_int(nBufSize)
        )

    def Recv(
        self,
        recvBuf: list[str],  # 要素数1限定
        nBufSize: int,
        nTimeOut: int,
        nOpenId: int | None = None,
    ) -> int:
        self.dll.NR_Recv.argtypes = (
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_int,
            ctypes.c_int,
        )
        self.dll.NR_Recv.restype = ctypes.c_int
        if nOpenId is None:
            nOpenId = self.nOpenId
        if not isinstance(recvBuf, list) or len(recvBuf) == 0:  # 要素数確認
            raise ValueError(
                "value must be a list with at least one element (recvBuf[0])"
            )
        buf_size = int(szBufSize) + 1  # NULL 終端分を確保
        # 入力文字列をバッファにコピー（切り詰め）
        buf = ctypes.create_string_buffer(buf_size)
        if recvBuf[0] is not None:
            b = recvBuf[0].encode("utf-8")[: buf_size - 1]
            ctypes.memmove(buf, b, len(b))
            buf[len(b)] = 0
        # 呼び出し
        ret = self.dll.NR_Recv(
            nOpenId,
            buf,
            ctypes.c_int(nBufSize),
            ctypes.c_int(nTimeOut),
        )
        # バッファの中身を取り出して文字列化して返す
        try:
            val = buf.value.decode("utf-8", errors="ignore")
        except Exception:
            val = buf.value.decode("latin-1", errors="ignore")
        recvBuf[0] = val
        return ret

    # 固定入出力グループ
    def AcsFixedIOInputSignal(
        self,
        value: list[bool],
        nSubId: int,
        nCount: int,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._BoolMultiple(
            self.dll.NR_AcsFixedIOInputSignal,
            value,
            nSubId,
            nCount,
            nPriority,
            fThreshold,
            nOpenId,
        )

    def AcsFixedIOMotorsOn(
        self,
        value: list[bool],
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._BoolSingle(
            self.dll.NR_AcsFixedIOMotorsOn, value, nPriority, fThreshold, nOpenId
        )

    def AcsFixedIOGStop1(
        self,
        value: list[bool],
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._BoolSingle(
            self.dll.NR_AcsFixedIOGStop1, value, nPriority, fThreshold, nOpenId
        )

    def AcsFixedIOStart1(
        self,
        value: list[bool],
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._BoolSingle(
            self.dll.NR_AcsFixedIOStart1, value, nPriority, fThreshold, nOpenId
        )

    def AcsFixedIOStart2(
        self,
        value: list[bool],
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._BoolSingle(
            self.dll.NR_AcsFixedIOStart2, value, nPriority, fThreshold, nOpenId
        )

    def AcsFixedIOStart3(
        self,
        value: list[bool],
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._BoolSingle(
            self.dll.NR_AcsFixedIOStart3, value, nPriority, fThreshold, nOpenId
        )

    def AcsFixedIOStart4(
        self,
        value: list[bool],
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._BoolSingle(
            self.dll.NR_AcsFixedIOStart4, value, nPriority, fThreshold, nOpenId
        )

    def AcsFixedIOStop(
        self,
        value: list[bool],
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._BoolSingle(
            self.dll.NR_AcsFixedIOStop, value, nPriority, fThreshold, nOpenId
        )

    def AcsFixedIOPlayBack(
        self,
        value: list[bool],
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._BoolSingle(
            self.dll.NR_AcsFixedIOPlayBack, value, nPriority, fThreshold, nOpenId
        )

    def AcsFixedIOMatSwitch(
        self,
        value: list[bool],
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._BoolSingle(
            self.dll.NR_AcsFixedIOMatSwitch, value, nPriority, fThreshold, nOpenId
        )

    def AcsFixedIOHighSpeedTeach(
        self,
        value: list[bool],
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._BoolSingle(
            self.dll.NR_AcsFixedIOHighSpeedTeach, value, nPriority, fThreshold, nOpenId
        )

    def AcsFixedIOExtStop(
        self,
        value: list[bool],
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._BoolSingle(
            self.dll.NR_AcsFixedIOExtStop, value, nPriority, fThreshold, nOpenId
        )

    def AcsFixedIOEStop(
        self,
        value: list[bool],
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._BoolSingle(
            self.dll.NR_AcsFixedIOEStop, value, nPriority, fThreshold, nOpenId
        )

    def AcsFixedIOSafetyPlug(
        self,
        value: list[bool],
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._BoolSingle(
            self.dll.NR_AcsFixedIOSafetyPlug, value, nPriority, fThreshold, nOpenId
        )

    def AcsFixedIOConfirmMotorsOn(
        self,
        value: list[bool],
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._BoolSingle(
            self.dll.NR_AcsFixedIOConfirmMotorsOn, value, nPriority, fThreshold, nOpenId
        )

    def AcsFixedIOTPEStop(
        self,
        value: list[bool],
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._BoolSingle(
            self.dll.NR_AcsFixedIOTPEStop, value, nPriority, fThreshold, nOpenId
        )

    def AcsFixedIOTeach(
        self,
        value: list[bool],
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._BoolSingle(
            self.dll.NR_AcsFixedIOTeach, value, nPriority, fThreshold, nOpenId
        )

    def AcsFixedIOTPEnableSW(
        self,
        value: list[bool],
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._BoolSingle(
            self.dll.NR_AcsFixedIOTPEnableSW, value, nPriority, fThreshold, nOpenId
        )

    def AcsFixedIOServoOn(
        self,
        value: list[bool],
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._BoolSingle(
            self.dll.NR_AcsFixedIOServoOn, value, nPriority, fThreshold, nOpenId
        )

    def AcsFixedIOServoEnable(
        self,
        value: list[bool],
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._BoolSingle(
            self.dll.NR_AcsFixedIOServoEnable, value, nPriority, fThreshold, nOpenId
        )

    def AcsFixedIOMagentON(
        self,
        value: list[bool],
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._BoolSingle(
            self.dll.NR_AcsFixedIOMagentON, value, nPriority, fThreshold, nOpenId
        )

    def AcsFixedIOWeldDetection(
        self,
        value: list[bool],
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._BoolSingle(
            self.dll.NR_AcsFixedIOWeldDetection, value, nPriority, fThreshold, nOpenId
        )

    def AcsFixedIOInConsistency(
        self,
        value: list[bool],
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._BoolSingle(
            self.dll.NR_AcsFixedIOInConsistency, value, nPriority, fThreshold, nOpenId
        )

    def AcsFixedIOOutputSignal(
        self,
        value: list[bool],
        nSubId: int,
        nCount: int,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._BoolMultiple(
            self.dll.NR_AcsFixedIOInputSignal,
            value,
            nSubId,
            nCount,
            nPriority,
            fThreshold,
            nOpenId,
        )

    def AcsFixedIOMotorsOnLAMP(
        self,
        value: list[bool],
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._BoolSingle(
            self.dll.NR_AcsFixedIOMotorsOnLAMP, value, nPriority, fThreshold, nOpenId
        )

    def AcsFixedIOMotorsOnRequest(
        self,
        value: list[bool],
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._BoolSingle(
            self.dll.NR_AcsFixedIOMotorsOnRequest, value, nPriority, fThreshold, nOpenId
        )

    def AcsFixedIOStartDisplay1(
        self,
        value: list[bool],
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._BoolSingle(
            self.dll.NR_AcsFixedIOStartDisplay1, value, nPriority, fThreshold, nOpenId
        )

    def AcsFixedIOStartDisplay2(
        self,
        value: list[bool],
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._BoolSingle(
            self.dll.NR_AcsFixedIOStartDisplay2, value, nPriority, fThreshold, nOpenId
        )

    def AcsFixedIOStartDisplay3(
        self,
        value: list[bool],
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._BoolSingle(
            self.dll.NR_AcsFixedIOStartDisplay3, value, nPriority, fThreshold, nOpenId
        )

    def AcsFixedIOStartDisplay4(
        self,
        value: list[bool],
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._BoolSingle(
            self.dll.NR_AcsFixedIOStartDisplay4, value, nPriority, fThreshold, nOpenId
        )

    def AcsFixedIOStopDisplay(
        self,
        value: list[bool],
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._BoolSingle(
            self.dll.NR_AcsFixedIOStopDisplay, value, nPriority, fThreshold, nOpenId
        )

    def AcsFixedIOTPEnableRelease(
        self,
        value: list[bool],
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._BoolSingle(
            self.dll.NR_AcsFixedIOTPEnableRelease, value, nPriority, fThreshold, nOpenId
        )

    def AcsFixedIOMotorsOnEnable(
        self,
        value: list[bool],
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._BoolSingle(
            self.dll.NR_AcsFixedIOMotorsOnEnable, value, nPriority, fThreshold, nOpenId
        )

    def AcsFixedIOMagnetOnEnable(
        self,
        value: list[bool],
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._BoolSingle(
            self.dll.NR_AcsFixedIOMagnetOnEnable, value, nPriority, fThreshold, nOpenId
        )

    def AcsFixedIOInternalExternal(
        self,
        value: list[bool],
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._BoolSingle(
            self.dll.NR_AcsFixedIOInternalExternal,
            value,
            nPriority,
            fThreshold,
            nOpenId,
        )

    def AcsFixedIOWPSEStop(
        self,
        value: list[bool],
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._BoolSingle(
            self.dll.NR_AcsFixedIOWPSEStop, value, nPriority, fThreshold, nOpenId
        )

    def AcsFixedIOCPUFailure(
        self,
        value: list[bool],
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._BoolSingle(
            self.dll.NR_AcsFixedIOCPUFailure, value, nPriority, fThreshold, nOpenId
        )

    def AcsFixedIOTPMode(
        self,
        value: list[bool],
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._BoolSingle(
            self.dll.NR_AcsFixedIOTPMode, value, nPriority, fThreshold, nOpenId
        )

    def AcsFixedIOEXTMotorsOn(
        self,
        value: list[bool],
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._BoolSingle(
            self.dll.NR_AcsFixedIOEXTMotorsOn, value, nPriority, fThreshold, nOpenId
        )

    # 汎用入出力グループ
    def AcsGeneralInputSignal(
        self,
        value: list[bool],
        nSubId: int,
        nCount: int,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._BoolMultiple(
            self.dll.NR_AcsGeneralInputSignal,
            value,
            nSubId,
            nCount,
            nPriority,
            fThreshold,
            nOpenId,
        )

    def AcsGeneralInputSignalB(
        self,
        value: list[int],
        nSubId: int,
        nCount: int,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._IntMultiple(
            self.dll.NR_AcsGeneralInputSignalB,
            value,
            nSubId,
            nCount,
            nPriority,
            fThreshold,
            nOpenId,
        )

    def AcsGeneralOutputSignal(
        self,
        value: list[bool],
        bUpdate: bool,
        nSubId: int,
        nCount: int,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._BoolMultipleUpdate(
            self.dll.NR_AcsGeneralOutputSignal,
            value,
            bUpdate,
            nSubId,
            nCount,
            nPriority,
            fThreshold,
            nOpenId,
        )

    def AcsGeneralOutputSignalB(
        self,
        value: list[int],
        bUpdate: bool,
        nSubId: int,
        nCount: int,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._IntMultipleUpdate(
            self.dll.NR_AcsGeneralOutputSignalB,
            value,
            bUpdate,
            nSubId,
            nCount,
            nPriority,
            fThreshold,
            nOpenId,
        )

    def AcsGeneralInputSignalOndesk(
        self,
        value: list[bool],
        bUpdate: bool,
        nSubId: int,
        nCount: int,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        raise ValueError("未実装です")  # TODO ondeskのみの機能

    def AcsGeneralInputSignalBOndesk(
        self,
        value: list[int],
        bUpdate: bool,
        nSubId: int,
        nCount: int,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        raise ValueError("未実装です")  # TODO ondeskのみの機能

    # 信号名称グループ
    def AcsStrInputSignalName(
        self,
        value: list[str],
        szBufSize: int,
        nSubId: int,
        nCount: int,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._StringMulti(
            self.dll.NR_AcsStrInputSignalNameA,
            value,
            szBufSize,
            nSubId,
            nCount,
            nPriority,
            fThreshold,
            nOpenId,
        )

    def AcsStrOutputSignalName(
        self,
        value: list[str],
        szBufSize: int,
        nSubId: int,
        nCount: int,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._StringMulti(
            self.dll.NR_AcsStrOutputSignalNameA,
            value,
            szBufSize,
            nSubId,
            nCount,
            nPriority,
            fThreshold,
            nOpenId,
        )

    # 変数グループ
    def AcsGlobalInt(
        self,
        value: list[int],
        bUpdate: bool,
        nSubId: int,
        nCount: int,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._IntMultipleUpdate(
            self.dll.NR_AcsGlobalInt,
            value,
            bUpdate,
            nSubId,
            nCount,
            nPriority,
            fThreshold,
            nOpenId,
        )

    def AcsGlobalFloat(
        self,
        value: list[float],
        bUpdate: bool,
        nSubId: int,
        nCount: int,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._FloatMultipleUpdate(
            self.dll.NR_AcsGlobalFloat,
            value,
            bUpdate,
            nSubId,
            nCount,
            nPriority,
            fThreshold,
            nOpenId,
        )

    def AcsGlobalString(
        self,
        value: list[str],
        szBufSize: int,
        bUpdate: bool,
        nSubId: int,
        nCount: int,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._StringMultiUpdate(
            self.dll.NR_AcsGlobalStringA,
            value,
            szBufSize,
            bUpdate,
            nSubId,
            nCount,
            nPriority,
            fThreshold,
            nOpenId,
        )

    def AcsLocalInt(
        self,
        value: list[int],
        bUpdate: bool,
        nSubId: int,
        nCount: int,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._IntMultipleUpdate(
            self.dll.NR_AcsLocalInt,
            value,
            bUpdate,
            nSubId,
            nCount,
            nPriority,
            fThreshold,
            nOpenId,
        )

    def AcsLocalFloat(
        self,
        value: list[float],
        bUpdate: bool,
        nSubId: int,
        nCount: int,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._FloatMultipleUpdate(
            self.dll.NR_AcsLocalFloat,
            value,
            bUpdate,
            nSubId,
            nCount,
            nPriority,
            fThreshold,
            nOpenId,
        )

    def AcsLocalString(
        self,
        value: list[str],
        szBufSize: int,
        bUpdate: bool,
        nSubId: int,
        nCount: int,
        nUnitId: int = 1,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._StringMultiUpdateUnit(
            self.dll.NR_AcsLocalStringA,
            value,
            szBufSize,
            bUpdate,
            nSubId,
            nCount,
            nUnitId,
            nPriority,
            fThreshold,
            nOpenId,
        )

    def AcsShift(
        self,
        value: list[NR_SHIFT],
        bUpdate: bool,
        nSubId: int,
        nCount: int,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        raise ValueError("未実装です")  # TODO

    # システム情報グループ
    def AcsVersion(
        self,
        value: list[str],  # 要素数1限定
        szBufSize: int,
        nOpenId: int | None = None,
    ) -> int:
        self.dll.NR_AcsVersion.argtypes = (
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_char),
            ctypes.c_int,
            ctypes.c_size_t,
        )
        self.dll.NR_AcsVersion.restype = ctypes.c_int
        if nOpenId is None:
            nOpenId = self.nOpenId
        if not isinstance(value, list) or len(value) == 0:  # 要素数確認
            raise ValueError(
                "value must be a list with at least one element (value[0])"
            )
        buf_size = int(szBufSize) + 1  # NULL 終端分を確保
        # 入力文字列をバッファにコピー（切り詰め）
        buf = ctypes.create_string_buffer(buf_size)
        if value[0] is not None:
            b = value[0].encode("utf-8")[: buf_size - 1]
            ctypes.memmove(buf, b, len(b))
            buf[len(b)] = 0
        # 呼び出し
        ret = self.dll.NR_AcsVersion(
            nOpenId,
            buf,
            ctypes.c_size_t(buf_size),
        )
        # バッファの中身を取り出して文字列化して返す
        try:
            val = buf.value.decode("utf-8", errors="ignore")
        except Exception:
            val = buf.value.decode("latin-1", errors="ignore")
        value[0] = val
        return ret

    def AcsUnitName(
        self,
        value: list[str],
        szBufSize: int,
        nSubId: int,
        nCount: int,
        nOpenId: int | None = None,
    ) -> int:  # TODO
        self.dll.NR_AcsUnitName.argtypes = (
            ctypes.c_int,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_char)),
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
        )
        self.dll.NR_AcsUnitName.restype = ctypes.c_int
        if nOpenId is None:
            nOpenId = self.nOpenId

        buf_size = int(szBufSize) + 1  # バッファ作成Null終端のため+1
        num = max(1, int(nCount))  # C に渡す要素数
        # truncate or pad the input list to num
        vals = (value or [])[:num] + [""] * max(0, num - len(value or []))

        # create buffers for each element
        buffers = [ctypes.create_string_buffer(buf_size) for _ in range(num)]
        for i, s in enumerate(vals):
            if s is not None:
                b = s.encode("utf-8")[: buf_size - 1]
                ctypes.memmove(buffers[i], b, len(b))
                buffers[i][len(b)] = 0

        ptr_type = (
            ctypes.POINTER(ctypes.c_char) * num
        )  # build array of char* (pointer to each buffer)
        ptrs = ptr_type(
            *[ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)) for buf in buffers]
        )
        char_pp = ctypes.cast(
            ptrs, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))
        )  # cast to char** (pointer to pointer) expected by argtypes

        ret = self.dll.NR_AcsUnitName(
            ctypes.c_int(nOpenId),
            char_pp,
            ctypes.c_size_t(buf_size),
            ctypes.c_int(nSubId),
            ctypes.c_int(nCount),
            ctypes.c_float(fThreshold),
        )
        for i in range(num):
            try:
                vals[i] = buffers[i].value.decode("utf-8", errors="ignore")
            except Exception:
                vals[i] = buffers[i].value.decode("latin-1", errors="ignore")
        # replace original list contents (caller-visible mutation)
        value[:] = vals
        return ret

    def AcsUnitAxisCnt(
        self,
        value: list[int],
        nSubId: int,
        nCount: int,
        nOpenId: int | None = None,
    ) -> int:
        self.dll.NR_AcsUnitAxisCnt.argtypes = (
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
        )
        self.dll.NR_AcsUnitAxisCnt.restype = ctypes.c_int
        if nOpenId is None:
            nOpenId = self.nOpenId
        IntArray = ctypes.c_int * len(value)
        c_arr = IntArray(*value)
        ret = self.dll.NR_AcsUnitAxisCnt(
            ctypes.c_int(nOpenId),
            c_arr,
            ctypes.c_int(nSubId),
            ctypes.c_int(nCount),
        )
        value[:] = list(c_arr)
        return ret

    def AcsCurrentUnitNo(
        self,
        value: list[int],
        bUpdate: bool,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._IntSingleUpdate(
            self.dll.NR_AcsCurrentUnitNo, value, bUpdate, nPriority, fThreshold, nOpenId
        )

    def AcsCurrentMechaNo(
        self,
        value: list[int],
        bUpdate: bool,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._IntSingleUpdate(
            self.dll.NR_AcsCurrentMechaNo,
            value,
            bUpdate,
            nPriority,
            fThreshold,
            nOpenId,
        )

    def AcsRemoteMode(
        self,
        value: list[bool],
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._BoolSingle(
            self.dll.NR_AcsRemoteMode, value, nPriority, fThreshold, nOpenId
        )

    def AcsPrgNo(
        self,
        value: list[int],
        nCount: int,
        nUnitId: int = 1,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        self.dll.NR_AcsPrgNo.argtypes = (
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
        )
        self.dll.NR_AcsPrgNo.restype = ctypes.c_int
        if nOpenId is None:
            nOpenId = self.nOpenId
        IntArray = ctypes.c_int * len(value)
        c_arr = IntArray(*value)
        ret = self.dll.NR_AcsPrgNo(
            ctypes.c_int(nOpenId),
            c_arr,
            ctypes.c_int(nCount),
            ctypes.c_int(nUnitId),
            ctypes.c_int(nPriority),
            ctypes.c_float(fThreshold),
        )
        value[:] = list(c_arr)
        return ret

    def AcsStepNo(
        self,
        value: list[int],
        nCount: int,
        nUnitId: int = 1,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        self.dll.NR_AcsStepNo.argtypes = (
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
        )
        self.dll.NR_AcsStepNo.restype = ctypes.c_int
        if nOpenId is None:
            nOpenId = self.nOpenId
        IntArray = ctypes.c_int * len(value)
        c_arr = IntArray(*value)
        ret = self.dll.NR_AcsStepNo(
            ctypes.c_int(nOpenId),
            c_arr,
            ctypes.c_int(nCount),
            ctypes.c_int(nUnitId),
            ctypes.c_int(nPriority),
            ctypes.c_float(fThreshold),
        )
        value[:] = list(c_arr)
        return ret

    def AcsUserLevel(
        self,
        value: int,  # pointer
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        raise ValueError("未実装です")  # TODO

    def AcsErrInfo(
        self,
        value: list[int],
        nSubId: int,
        nCount: int,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        self.dll.NR_AcsErrInfo.argtypes = (
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_float,
        )
        self.dll.NR_AcsErrInfo.restype = ctypes.c_int
        if nOpenId is None:
            nOpenId = self.nOpenId
        IntArray = ctypes.c_int * len(value)
        c_arr = IntArray(*value)
        ret = self.dll.NR_AcsErrInfo(
            ctypes.c_int(nOpenId),
            c_arr,
            ctypes.c_int(nSubId),
            ctypes.c_int(nCount),
            ctypes.c_int(nPriority),
            ctypes.c_float(fThreshold),
        )
        value[:] = list(c_arr)
        return ret

    def AcsCPULoad(
        self,
        value: list[float],
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._FloatSingle(
            self.dll.NR_AcsCPULoad, value, nPriority, fThreshold, nOpenId
        )

    def AcsCPUVoltage3(
        self,
        value: list[float],
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._FloatSingle(
            self.dll.NR_AcsCPUVoltage3, value, nPriority, fThreshold, nOpenId
        )

    def AcsCPUVoltage5(
        self,
        value: list[float],
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._FloatSingle(
            self.dll.NR_AcsCPUVoltage5, value, nPriority, fThreshold, nOpenId
        )

    def AcsCPUTempe1(
        self,
        value: list[float],
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._FloatSingle(
            self.dll.NR_AcsCPUTempe1, value, nPriority, fThreshold, nOpenId
        )

    def AcsCPUTempe2(
        self,
        value: list[float],
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._FloatSingle(
            self.dll.NR_AcsCPUTempe2, value, nPriority, fThreshold, nOpenId
        )

    def AcsServoCommErr(
        self,
        value: list[int],  # pointer list
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        raise ValueError("未実装です")  # TODO

    # モニタ情報グループ
    def AcsAxisEncode(
        self,
        value: list[int],
        nSubId: int,
        nCount: int,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._IntMultiple(
            self.dll.NR_AcsAxisEncode,
            value,
            nSubId,
            nCount,
            nPriority,
            fThreshold,
            nOpenId,
        )

    def AcsAxisAmpValue(
        self,
        value: list[float],
        nSubId: int,
        nCount: int,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._FloatMultiple(
            self.dll.NR_AcsAxisAmpValue,
            value,
            nSubId,
            nCount,
            nPriority,
            fThreshold,
            nOpenId,
        )

    def AcsAxisOrderAmpValue(
        self,
        value: list[float],
        nSubId: int,
        nCount: int,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._FloatMultiple(
            self.dll.NR_AcsAxisOrderAmpValue,
            value,
            nSubId,
            nCount,
            nPriority,
            fThreshold,
            nOpenId,
        )

    def AcsAxisSpeed(
        self,
        value: list[float],
        nSubId: int,
        nCount: int,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._FloatMultiple(
            self.dll.NR_AcsAxisSpeed,
            value,
            nSubId,
            nCount,
            nPriority,
            fThreshold,
            nOpenId,
        )

    def AcsAxisOrderSpeed(
        self,
        value: list[float],
        nSubId: int,
        nCount: int,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._FloatMultiple(
            self.dll.NR_AcsAxisOrderSpeed,
            value,
            nSubId,
            nCount,
            nPriority,
            fThreshold,
            nOpenId,
        )

    def AcsAxisTheta(
        self,
        value: list[float],
        nSubId: int,
        nCount: int,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._FloatMultiple(
            self.dll.NR_AcsAxisTheta,
            value,
            nSubId,
            nCount,
            nPriority,
            fThreshold,
            nOpenId,
        )

    def AcsAxisThetaOrder(
        self,
        value: list[float],
        nSubId: int,
        nCount: int,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._FloatMultiple(
            self.dll.NR_AcsAxisThetaOrder,
            value,
            nSubId,
            nCount,
            nPriority,
            fThreshold,
            nOpenId,
        )

    def AcsAxisTorque(
        self,
        value: list[float],
        nSubId: int,
        nCount: int,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._FloatMultiple(
            self.dll.NR_AcsAxisTorque,
            value,
            nSubId,
            nCount,
            nPriority,
            fThreshold,
            nOpenId,
        )

    def AcsToolTipPos(
        self,
        value: list[float],
        nSubId: int,
        nCount: int,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._FloatMultiple(
            self.dll.NR_AcsToolTipPos,
            value,
            nSubId,
            nCount,
            nPriority,
            fThreshold,
            nOpenId,
        )

    def AcsOrderToolTipPos(
        self,
        value: list[float],
        nSubId: int,
        nCount: int,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._FloatMultiple(
            self.dll.NR_AcsOrderToolTipPos,
            value,
            nSubId,
            nCount,
            nPriority,
            fThreshold,
            nOpenId,
        )

    def AcsTcpSpeed(
        self,
        value: list[float],
        nSubId: int,
        nCount: int,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._FloatMultiple(
            self.dll.NR_AcsTcpSpeed,
            value,
            nSubId,
            nCount,
            nPriority,
            fThreshold,
            nOpenId,
        )

    def AcsOrderTcpSpeed(
        self,
        value: list[float],
        nSubId: int,
        nCount: int,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._FloatMultiple(
            self.dll.NR_AcsOrderTcpSpeed,
            value,
            nSubId,
            nCount,
            nPriority,
            fThreshold,
            nOpenId,
        )

    def AcsProportionTcpSpeedDigital(
        self,
        value: list[float],
        nSubId: int,
        nCount: int,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._FloatMultiple(
            self.dll.NR_AcsProportionTcpSpeedDigital,
            value,
            nSubId,
            nCount,
            nPriority,
            fThreshold,
            nOpenId,
        )

    def AcsTorqueImbalance(
        self,
        value: list[float],
        nSubId: int,
        nCount: int,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._FloatMultiple(
            self.dll.NR_AcsTorqueImbalance,
            value,
            nSubId,
            nCount,
            nPriority,
            fThreshold,
            nOpenId,
        )

    def AcsShiftOnlineCurrent(
        self,
        value: list[NR_SHIFT],
        nUnitId: int = 1,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        # TODO 実装
        raise ValueError("未実装です")  # TODO

    def AcsShiftRobotCurrent(
        self,
        value: list[NR_SHIFT],
        bUpdate: bool,
        nUnitId: int = 1,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        raise ValueError("未実装です")  # TODO

    def AcsShiftAllCurrent(
        self,
        value: list[NR_SHIFT],
        nUnitId: int = 1,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        # TODO 実装
        raise ValueError("未実装です")  # TODO

    def AcsElePowerCost(
        self,
        value: list[float],
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._FloatSingle(
            self.dll.NR_AcsElePowerCost, value, nPriority, fThreshold, nOpenId
        )

    def AcsEncErrCntTrans(
        self,
        value: list[int],
        nSubId: int,
        nCount: int,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._IntMultiple(
            self.dll.NR_AcsEncErrCntTrans,
            value,
            nSubId,
            nCount,
            nPriority,
            fThreshold,
            nOpenId,
        )

    def AcsEncErrCntBitJmp(
        self,
        value: list[int],
        nSubId: int,
        nCount: int,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._IntMultiple(
            self.dll.NR_AcsEncErrCntBitJmp,
            value,
            nSubId,
            nCount,
            nPriority,
            fThreshold,
            nOpenId,
        )

    def AcsEncErrCntStatus(
        self,
        value: list[int],
        nSubId: int,
        nCount: int,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._IntMultiple(
            self.dll.NR_AcsEncErrCntStatus,
            value,
            nSubId,
            nCount,
            nPriority,
            fThreshold,
            nOpenId,
        )

    def AcsEncErrCntAny(
        self,
        value: list[int],
        nSubId: int,
        nCount: int,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._IntMultiple(
            self.dll.NR_AcsEncErrCntAny,
            value,
            nSubId,
            nCount,
            nPriority,
            fThreshold,
            nOpenId,
        )

    def AcsOverHaul(
        self,
        value: list[float],
        nSubId: int,
        nCount: int,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._FloatMultiple(
            self.dll.NR_AcsOverHaul,
            value,
            nSubId,
            nCount,
            nPriority,
            fThreshold,
            nOpenId,
        )

    def AcsLifeSpan(
        self,
        value: list[float],
        nSubId: int,
        nCount: int,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._FloatMultiple(
            self.dll.NR_AcsLifeSpan,
            value,
            nSubId,
            nCount,
            nPriority,
            fThreshold,
            nOpenId,
        )

    # デジタル情報グループ
    def AcsInputDigital(
        self,
        value: list[int],
        nSubId: int,
        nCount: int,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._IntMultiple(
            self.dll.NR_AcsInputDigital,
            value,
            nSubId,
            nCount,
            nPriority,
            fThreshold,
            nOpenId,
        )

    def AcsOutputDigital(
        self,
        value: list[int],
        nSubId: int,
        nCount: int,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._IntMultiple(
            self.dll.NR_AcsOutputDigital,
            value,
            nSubId,
            nCount,
            nPriority,
            fThreshold,
            nOpenId,
        )

    # PLC情報グループ
    def AcsInputSignalPhysicsPlc(
        self,
        value: list[bool],
        nSubId: int,
        nCount: int,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._BoolMultiple(
            self.dll.NR_AcsInputSignalPhysicsPlc,
            value,
            nSubId,
            nCount,
            nPriority,
            fThreshold,
            nOpenId,
        )

    def AcsOutputSignalPhysicsPlc(
        self,
        value: list[bool],
        nSubId: int,
        nCount: int,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._BoolMultiple(
            self.dll.NR_AcsOutputSignalPhysicsPlc,
            value,
            nSubId,
            nCount,
            nPriority,
            fThreshold,
            nOpenId,
        )

    def AcsBoolValuePlc(
        self,
        value: list[bool],
        bUpdate: bool,
        nSubId: int,
        nCount: int,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._BoolMultipleUpdate(
            self.dll.NR_AcsBoolValuePlc,
            value,
            bUpdate,
            nSubId,
            nCount,
            nPriority,
            fThreshold,
            nOpenId,
        )

    def AcsRealValuePlc(
        self,
        value: list[float],
        bUpdate: bool,
        nSubId: int,
        nCount: int,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._FloatMultipleUpdate(
            self.dll.NR_AcsRealValuePlc,
            value,
            bUpdate,
            nSubId,
            nCount,
            nPriority,
            fThreshold,
            nOpenId,
        )

    def AcsDintValuePlc(
        self,
        value: list[int],
        bUpdate: bool,
        nSubId: int,
        nCount: int,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._IntMultipleUpdate(
            self.dll.NR_AcsDintValuePlc,
            value,
            bUpdate,
            nSubId,
            nCount,
            nPriority,
            fThreshold,
            nOpenId,
        )

    def AcsSintValuePlc(
        self,
        value: list[int],
        bUpdate: bool,
        nSubId: int,
        nCount: int,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._IntMultipleUpdate(
            self.dll.NR_AcsSintValuePlc,
            value,
            bUpdate,
            nSubId,
            nCount,
            nPriority,
            fThreshold,
            nOpenId,
        )

    def AcsTimerValuePlc(
        self,
        value: list[int],
        bUpdate: bool,
        nSubId: int,
        nCount: int,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._IntMultipleUpdate(
            self.dll.NR_AcsTimerValuePlc,
            value,
            bUpdate,
            nSubId,
            nCount,
            nPriority,
            fThreshold,
            nOpenId,
        )

    def AcsStringValuePlc(
        self,
        value: list[str],
        szBufSize: int,
        bUpdate: bool,
        nSubId: int,
        nCount: int,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._StringMultiUpdate(
            self.dll.NR_AcsStringValuePlcA,
            value,
            szBufSize,
            bUpdate,
            nSubId,
            nCount,
            nPriority,
            fThreshold,
            nOpenId,
        )

    # パレタイズ情報グループ
    def AcsPalletizingStringValue(
        self,
        value: list[str],
        szBufSize: int,
        bUpdate: bool,
        nSubId: int,
        nCount: int,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._StringMultiUpdate(
            self.dll.NR_AcsPalletizingStringValueA,
            value,
            szBufSize,
            bUpdate,
            nSubId,
            nCount,
            nPriority,
            fThreshold,
            nOpenId,
        )

    def AcsPalletizingCounter(
        self,
        value: list[int],
        bUpdate: bool,
        nSubId: int,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._IntSingleSubUpdate(
            self.dll.NR_AcsPalletizingCounter,
            value,
            bUpdate,
            nSubId,
            nPriority,
            fThreshold,
            nOpenId,
        )

    def AcsPalletizingReg(
        self,
        value: list[NR_PALLETREG],
        szBufSize: int,
        bUpdate: bool,
        nSubId: int,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        raise ValueError("未実装です")  # TODO

    def AcsPalletizingWork(
        self,
        value: NR_PALLETWORK,  # pointer
        bUpdate: bool,
        nSubId: int,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        raise ValueError("未実装です")  # TODO

    def AcsPalletizingLayer(
        self,
        value: NR_PALLETLAYER,  # pointer
        bUpdate: bool,
        nSubId: int,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        raise ValueError("未実装です")  # TODO

    def AcsPalletizingPlene(
        self,
        value: NR_PALLETPLENE,  # pointer
        bUpdate: bool,
        nSubId: int,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        raise ValueError("未実装です")  # TODO

    # ティーチング情報グループ
    def AcsRecordSpeed(
        self,
        value: list[int],
        bUpdate: bool,
        nUnitId: int = 1,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._IntSingleUnitUpdate(
            self.dll.NR_AcsRecordSpeed,
            value,
            bUpdate,
            nUnitId,
            nPriority,
            fThreshold,
            nOpenId,
        )

    def AcsRecordToolNo(
        self,
        value: list[int],
        bUpdate: bool,
        nUnitId: int = 1,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._IntSingleUnitUpdate(
            self.dll.NR_AcsRecordToolNo,
            value,
            bUpdate,
            nUnitId,
            nPriority,
            fThreshold,
            nOpenId,
        )

    def AcsRecordAccuracyNo(
        self,
        value: list[int],
        bUpdate: bool,
        nUnitId: int = 1,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._IntSingleUnitUpdate(
            self.dll.NR_AcsRecordAccuracyNo,
            value,
            bUpdate,
            nUnitId,
            nPriority,
            fThreshold,
            nOpenId,
        )

    def AcsRecordSmoothNo(
        self,
        value: list[int],
        bUpdate: bool,
        nUnitId: int = 1,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._IntSingleUnitUpdate(
            self.dll.NR_AcsRecordSmoothNo,
            value,
            bUpdate,
            nUnitId,
            nPriority,
            fThreshold,
            nOpenId,
        )

    def AcsAccelerationNo(
        self,
        value: list[int],
        bUpdate: bool,
        nUnitId: int = 1,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._IntSingleUnitUpdate(
            self.dll.NR_AcsAccelerationNo,
            value,
            bUpdate,
            nUnitId,
            nPriority,
            fThreshold,
            nOpenId,
        )

    def AcsInterpolationKind(
        self,
        value: list[int],
        bUpdate: bool,
        nUnitId: int = 1,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._IntSingleUnitUpdate(
            self.dll.NR_AcsInterpolationKind,
            value,
            bUpdate,
            nUnitId,
            nPriority,
            fThreshold,
            nOpenId,
        )

    def AcsPoseValue(
        self,
        value: list[NR_POSE],
        bUpdate: bool,
        nId: int,
        nSubId: int,
        nUnitId: int = 1,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        raise ValueError("未実装です")  # TODO

    def AcsPoseValueConfig(
        self,
        value: NR_POSE_CONF,
        bUpdate: bool,
        nId: int,
        nSubId: int,
        nUnitId: int = 1,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        raise ValueError("未実装です")  # TODO

    def AcsPoseValueXYZ(
        self,
        value: NR_POSE_CONF,  # pointer
        bUpdate: bool,
        nPoseFileNo: int,
        nPoseNo: int,
        nPoseFormat: int = 0,
        nUserCoordNo: int = 0,
        nUnitId: int = 1,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        raise ValueError("未実装です")  # TODO

    def AcsPoseValueAngle(
        self,
        value: NR_ANGLE,  # pointer
        bUpdate: bool,
        nPoseFileNo: int,
        nPoseNo: int,
        nPoseFormat: int = 0,
        nUserCoordNo: int = 0,
        nUnitId: int = 1,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        raise ValueError("未実装です")  # TODO

    def AcsUserCoordOrg(
        self,
        value: NR_USERCOORD_ORG,  # pointer
        bUpdate: bool,
        nUserCoodNo: int,
        nUnitId: int = 1,
        nOpenId: int | None = None,
    ) -> int:
        raise ValueError("未実装です")  # TODO

    def AcsToolParam(
        self,
        value: NR_TOOL_PARAM,
        bUpdate: bool,
        nMechNo: int,
        nToolNo: int,
        nOpenId: int | None = None,
    ) -> int:
        raise ValueError("未実装です")  # TODO

    # 手動操作グループ
    def AcsEntryManualCoordinateType(
        self,
        value: list[str],
        szBufSize: int,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:

        self.dll.NR_AcsEntryManualCoordinateType.argtypes = (
            ctypes.c_int,
            ctypes.POINTER(ctypes.POINTER(ctypes.c_char)),
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_float,
        )
        self.dll.NR_AcsEntryManualCoordinateType.restype = ctypes.c_int
        if nOpenId is None:
            nOpenId = self.nOpenId

        buf_size = int(szBufSize) + 1  # バッファ作成Null終端のため+1
        num = max(1, int(nCount))  # C に渡す要素数 TODO エラー
        # truncate or pad the input list to num
        vals = (value or [])[:num] + [""] * max(0, num - len(value or []))

        # create buffers for each element
        buffers = [ctypes.create_string_buffer(buf_size) for _ in range(num)]
        for i, s in enumerate(vals):
            if s is not None:
                b = s.encode("utf-8")[: buf_size - 1]
                ctypes.memmove(buffers[i], b, len(b))
                buffers[i][len(b)] = 0

        ptr_type = (
            ctypes.POINTER(ctypes.c_char) * num
        )  # build array of char* (pointer to each buffer)
        ptrs = ptr_type(
            *[ctypes.cast(buf, ctypes.POINTER(ctypes.c_char)) for buf in buffers]
        )
        char_pp = ctypes.cast(
            ptrs, ctypes.POINTER(ctypes.POINTER(ctypes.c_char))
        )  # cast to char** (pointer to pointer) expected by argtypes

        ret = self.dll.NR_AcsEntryManualCoordinateType(
            ctypes.c_int(nOpenId),
            char_pp,
            ctypes.c_size_t(buf_size),
            ctypes.c_int(nPriority),
            ctypes.c_float(fThreshold),
        )
        for i in range(num):
            try:
                vals[i] = buffers[i].value.decode("utf-8", errors="ignore")
            except Exception:
                vals[i] = buffers[i].value.decode("latin-1", errors="ignore")
        # replace original list contents (caller-visible mutation)
        value[:] = vals
        return ret

    def AcsManualCoordinateType(
        self,
        value: list[bool],
        bUpdate: bool,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._BoolSingleUpdate(
            self.dll.NR_AcsManualCoordinateType,
            value,
            bUpdate,
            nPriority,
            fThreshold,
            nOpenId,
        )

    def AcsManualSpeed(
        self,
        value: list[int],
        bUpdate: bool,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._IntSingleUpdate(
            self.dll.NR_AcsManualSpeed, value, bUpdate, nPriority, fThreshold, nOpenId
        )

    # チェック操作グループ
    def AcsCheckSpeed(
        self,
        value: list[int],
        bUpdate: bool,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._IntSingleUpdate(
            self.dll.NR_AcsCheckSpeed, value, bUpdate, nPriority, fThreshold, nOpenId
        )

    def AcsCheckMode(
        self,
        value: list[bool],
        bUpdate: bool,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._BoolSingleUpdate(
            self.dll.NR_AcsCheckMode, value, bUpdate, nPriority, fThreshold, nOpenId
        )

    def AcsCheckOperationProgress(
        self,
        value: list[float],
        nUnitId: int = 1,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._FloatSingleUnit(
            self.dll.NR_AcsCheckOperationProgress,
            value,
            nUnitId,
            nPriority,
            fThreshold,
            nOpenId,
        )

    # 再生情報グループ
    def AcsOperationModePlayback(
        self,
        value: list[int],
        bUpdate: bool,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._IntSingleUpdate(
            self.dll.NR_AcsOperationModePlayback,
            value,
            bUpdate,
            nPriority,
            fThreshold,
            nOpenId,
        )

    def AcsWaittingStatusSignal(
        self,
        value: list[bool],
        bUpdate: bool,
        nUnitId: int = 1,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._BoolSingleUnitUpdate(
            self.dll.NR_AcsWaittingStatusSignal,
            value,
            bUpdate,
            nUnitId,
            nPriority,
            fThreshold,
            nOpenId,
        )

    def AcsStackDepthPrg(
        self,
        value: list[int],
        nUnitId: int = 1,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._IntSingleUnit(
            self.dll.NR_AcsStackDepthPrg, value, nUnitId, nPriority, fThreshold, nOpenId
        )

    def AcsOverRideSpeed(
        self,
        value: list[float],
        nUnitId: int = 1,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._FloatSingleUnit(
            self.dll.NR_AcsOverRideSpeed, value, nUnitId, nPriority, fThreshold, nOpenId
        )

    def AcsStatusSlowPlayback(
        self,
        value: list[int],
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._IntSingle(
            self.dll.NR_AcsStatusSlowPlayback, value, nPriority, fThreshold, nOpenId
        )

    def AcsStatusSavingEnergy(
        self,
        value: list[int],
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._IntSingle(
            self.dll.NR_AcsStatusSavingEnergy, value, nPriority, fThreshold, nOpenId
        )

    # 力センサ情報グループ
    def AcsForceCtrl(
        self,
        value: list[float],
        nSubId: int,
        nCount: int,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._FloatMultiple(
            self.dll.NR_AcsForceCtrl,
            value,
            nSubId,
            nCount,
            nPriority,
            fThreshold,
            nOpenId,
        )

    def AcsForceShift(
        self,
        value: list[float],
        nSubId: int,
        nCount: int,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._FloatMultiple(
            self.dll.NR_AcsForceShift,
            value,
            nSubId,
            nCount,
            nPriority,
            fThreshold,
            nOpenId,
        )

    # コンベア情報グループ
    def AcsConveyerRegister(
        self,
        value: list[float],
        nSubId: int,
        nCount: int,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._FloatMultiple(
            self.dll.NR_AcsConveyerRegister,
            value,
            nSubId,
            nCount,
            nPriority,
            fThreshold,
            nOpenId,
        )

    def AcsConveyerMode(
        self,
        value: list[int],
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._IntSingle(
            self.dll.NR_AcsConveyerMode, value, nPriority, fThreshold, nOpenId
        )

    def AcsConveyerSpeed(
        self,
        value: list[float],
        nSubId: int,
        nCount: int,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._FloatMultiple(
            self.dll.NR_AcsConveyerSpeed,
            value,
            nSubId,
            nCount,
            nPriority,
            fThreshold,
            nOpenId,
        )

    def AcsConveyerPulse(
        self,
        value: list[float],
        nSubId: int,
        nCount: int,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._FloatMultiple(
            self.dll.NR_AcsConveyerPulse,
            value,
            nSubId,
            nCount,
            nPriority,
            fThreshold,
            nOpenId,
        )

    # 操作グループ
    def CtrlMotor(
        self,
        lCtrlSW: int,
        nOpenId: int | None = None,
    ) -> int:
        self.dll.NR_CtrlMotor.argtypes = (ctypes.c_int, ctypes.c_long)
        self.dll.NR_CtrlMotor.restype = ctypes.c_int
        if nOpenId is None:
            nOpenId = self.nOpenId
        return self.dll.NR_CtrlMotor(nOpenId, lCtrlSW)

    def CtrlRun(
        self,
        lCtrlSW: int,
        lUnitId: int = 1,
        nOpenId: int | None = None,
    ) -> int:
        self.dll.NR_CtrlRun.argtypes = (ctypes.c_int, ctypes.c_long, ctypes.c_long)
        self.dll.NR_CtrlRun.restype = ctypes.c_int
        if nOpenId is None:
            nOpenId = self.nOpenId
        return self.dll.NR_CtrlRun(nOpenId, lCtrlSW, lUnitId)

    def CtrlProgram(
        self,
        nProgNo: int,
        nOpenId: int | None = None,
    ) -> int:
        self.dll.NR_CtrlProgram.argtypes = (ctypes.c_int, ctypes.c_int)
        self.dll.NR_CtrlProgram.restype = ctypes.c_int
        if nOpenId is None:
            nOpenId = self.nOpenId
        return self.dll.NR_CtrlProgram(nOpenId, nProgNo)

    def CtrlStep(
        self,
        nStepNo: int,
        nOpenId: int | None = None,
    ) -> int:
        self.dll.NR_CtrlStep.argtypes = (ctypes.c_int, ctypes.c_int)
        self.dll.NR_CtrlStep.restype = ctypes.c_int
        if nOpenId is None:
            nOpenId = self.nOpenId
        return self.dll.NR_CtrlStep(nOpenId, nStepNo)

    def CtrlJog(
        self,
        pose: NR_POSE,
        nUnitId: int = 1,
        nOpenId: int | None = None,
    ) -> int:
        self.dll.NR_CtrlJog.argtypes = (
            ctypes.c_int,
            ctypes.POINTER(NR_POSE),
            ctypes.c_int,
        )
        self.dll.NR_CtrlJog.restype = ctypes.c_int
        if nOpenId is None:
            nOpenId = self.nOpenId
        return self.dll.NR_CtrlJog(nOpenId, ctypes.byref(pose), nUnitId)

    def CtrlMoveX(
        self,
        pose: NR_POSE,
        nType: int = 0,
        nUnitId: int = 1,
        nConf: int = 0,
        fExtPos: list[float] | None = None,
        nExtPosSize: int | None = None,
        nOpenId: int | None = None,
    ) -> int:
        self.dll.NR_CtrlMoveX.argtypes = (
            ctypes.c_int,
            ctypes.POINTER(NR_POSE),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        )
        self.dll.NR_CtrlMoveX.restype = ctypes.c_int

        if nOpenId is None:
            nOpenId = self.nOpenId

        if fExtPos is None:
            _zero_arr = (ctypes.c_float * 1)(0.0)
            ext_ptr = _zero_arr
            ext_size = 0 if nExtPosSize is None else nExtPosSize
        else:
            size = nExtPosSize if nExtPosSize is not None else len(fExtPos)
            arr = (ctypes.c_float * size)(*fExtPos[:size])
            ext_ptr = arr
            ext_size = size

        return self.dll.NR_CtrlMoveX(
            nOpenId,
            ctypes.byref(pose),
            nType,
            nUnitId,
            nConf,
            ext_ptr,
            ext_size,
        )

    def CtrlMoveJ(
        self,
        fAngle: list[float],
        nAngleSize: int,
        nType: int = 0,
        nUnitId: int = 1,
        nOpenId: int | None = None,
    ) -> int:
        self.dll.NR_CtrlMoveJ.argtypes = (
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        )
        self.dll.NR_CtrlMoveJ.restype = ctypes.c_int
        if nOpenId is None:
            nOpenId = self.nOpenId
        arr = (ctypes.c_float * nAngleSize)(*fAngle)
        return self.dll.NR_CtrlMoveJ(
            nOpenId,
            arr,
            nAngleSize,
            nType,
            nUnitId,
        )

    def CtrlMoveXR(
        self,
        pose: NR_POSE,
        nType: int = 0,
        nUnitId: int = 1,
        nConf: int = 0,
        fExtPos: list[float] | None = None,
        nExtPosSize: int | None = None,
        nOpenId: int | None = None,
    ) -> int:
        self.dll.NR_CtrlMoveXR.argtypes = (
            ctypes.c_int,
            ctypes.POINTER(NR_POSE),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        )
        self.dll.NR_CtrlMoveXR.restype = ctypes.c_int

        if nOpenId is None:
            nOpenId = self.nOpenId

        if fExtPos is None:
            _zero_arr = (ctypes.c_float * 1)(0.0)
            ext_ptr = _zero_arr
            ext_size = 0 if nExtPosSize is None else nExtPosSize
        else:
            size = nExtPosSize if nExtPosSize is not None else len(fExtPos)
            arr = (ctypes.c_float * size)(*fExtPos[:size])
            ext_ptr = arr
            ext_size = size

        return self.dll.NR_CtrlMoveXR(
            nOpenId,
            ctypes.byref(pose),
            nType,
            nUnitId,
            nConf,
            ext_ptr,
            ext_size,
        )

    def CtrlMoveXT(
        self,
        pose: NR_POSE,
        nType: int = 0,
        nUnitId: int = 1,
        nConf: int = 0,
        fExtPos: list[float] | None = None,
        nExtPosSize: int | None = None,
        nOpenId: int | None = None,
    ) -> int:
        self.dll.NR_CtrlMoveXT.argtypes = (
            ctypes.c_int,
            ctypes.POINTER(NR_POSE),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        )
        self.dll.NR_CtrlMoveXT.restype = ctypes.c_int

        if nOpenId is None:
            nOpenId = self.nOpenId

        if fExtPos is None:
            _zero_arr = (ctypes.c_float * 1)(0.0)
            ext_ptr = _zero_arr
            ext_size = 0 if nExtPosSize is None else nExtPosSize
        else:
            size = nExtPosSize if nExtPosSize is not None else len(fExtPos)
            arr = (ctypes.c_float * size)(*fExtPos[:size])
            ext_ptr = arr
            ext_size = size

        return self.dll.NR_CtrlMoveXT(
            nOpenId,
            ctypes.byref(pose),
            nType,
            nUnitId,
            nConf,
            ext_ptr,
            ext_size,
        )

    def CtrlMoveJA(
        self,
        fAngle: list[float],
        nAngleSize: int,
        nType: int = 0,
        nUnitId: int = 1,
        nOpenId: int | None = None,
    ) -> int:
        self.dll.NR_CtrlMoveJA.argtypes = (
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
        )
        self.dll.NR_CtrlMoveJA.restype = ctypes.c_int
        if nOpenId is None:
            nOpenId = self.nOpenId
        arr = (ctypes.c_float * nAngleSize)(*fAngle)
        return self.dll.NR_CtrlMoveJA(
            nOpenId,
            arr,
            nAngleSize,
            nType,
            nUnitId,
        )

    # プログラム診断グループ
    def AcsProgStartTime(
        self,
        value: list[str],  # 要素数1
        szBufSize: int,
        nSubId: int,
        nUnitId: int = 1,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._StringSingleUnit(
            self.dll.NR_AcsProgStartTime,
            value,
            szBufSize,
            nSubId,
            nUnitId,
            nPriority,
            fThreshold,
            nOpenId,
        )

    def AcsProgCycleTime(
        self,
        value: list[str],  # 要素数1
        szBufSize: int,
        nSubId: int,
        nUnitId: int = 1,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._StringSingleUnit(
            self.dll.NR_AcsProgCycleTime,
            value,
            szBufSize,
            nSubId,
            nUnitId,
            nPriority,
            fThreshold,
            nOpenId,
        )

    def AcsProgCycleCnt(
        self,
        value: int,  # pointer
        nSubId: int,
        nUnitId: int = 1,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        raise ValueError("未実装です")  # TODO

    def AcsProgIWaitTime(
        self,
        value: list[str],  # 要素数1
        szBufSize: int,
        nSubId: int,
        nUnitId: int = 1,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._StringSingleUnit(
            self.dll.NR_AcsProgIWaitTime,
            value,
            szBufSize,
            nSubId,
            nUnitId,
            nPriority,
            fThreshold,
            nOpenId,
        )

    def AcsProgDelayTime(
        self,
        value: list[str],  # 要素数1
        szBufSize: int,
        nSubId: int,
        nUnitId: int = 1,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._StringSingleUnit(
            self.dll.NR_AcsProgDelayTime,
            value,
            szBufSize,
            nSubId,
            nUnitId,
            nPriority,
            fThreshold,
            nOpenId,
        )

    def AcsProgAveSpeed(
        self,
        value: list[float],
        nSubId: int,
        nCount: int,
        nUnitId: int = 1,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._FloatMultipleUnit(
            self.dll.NR_AcsProgAveSpeed,
            value,
            nSubId,
            nCount,
            nUnitId,
            nPriority,
            fThreshold,
            nOpenId,
        )

    def AcsProgAveTorque(
        self,
        value: list[float],
        nSubId: int,
        nCount: int,
        nUnitId: int = 1,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._FloatMultipleUnit(
            self.dll.NR_AcsProgAveTorque,
            value,
            nSubId,
            nCount,
            nUnitId,
            nPriority,
            fThreshold,
            nOpenId,
        )

    def AcsProgAveCurrent(
        self,
        value: list[float],
        nSubId: int,
        nCount: int,
        nUnitId: int = 1,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._FloatMultipleUnit(
            self.dll.NR_AcsProgAveCurrent,
            value,
            nSubId,
            nCount,
            nUnitId,
            nPriority,
            fThreshold,
            nOpenId,
        )

    def AcsProgMaxSpeed(
        self,
        value: list[float],
        nSubId: int,
        nCount: int,
        nUnitId: int = 1,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._FloatMultipleUnit(
            self.dll.NR_AcsProgMaxSpeed,
            value,
            nSubId,
            nCount,
            nUnitId,
            nPriority,
            fThreshold,
            nOpenId,
        )

    def AcsProgMaxTorque(
        self,
        value: list[float],
        nSubId: int,
        nCount: int,
        nUnitId: int = 1,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._FloatMultipleUnit(
            self.dll.NR_AcsProgMaxTorque,
            value,
            nSubId,
            nCount,
            nUnitId,
            nPriority,
            fThreshold,
            nOpenId,
        )

    def AcsProgMaxCurrent(
        self,
        value: list[float],
        nSubId: int,
        nCount: int,
        nUnitId: int = 1,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._FloatMultipleUnit(
            self.dll.NR_AcsProgMaxCurrent,
            value,
            nSubId,
            nCount,
            nUnitId,
            nPriority,
            fThreshold,
            nOpenId,
        )

    def AcsProgLifeSpan(
        self,
        value: list[float],
        nSubId: int,
        nCount: int,
        nUnitId: int = 1,
        nPriority: int = NR_PULL_MODE,
        fThreshold: float = 0.01,
        nOpenId: int | None = None,
    ) -> int:
        return self._FloatMultipleUnit(
            self.dll.NR_AcsProgLifeSpan,
            value,
            nSubId,
            nCount,
            nUnitId,
            nPriority,
            fThreshold,
            nOpenId,
        )

    # 外部トラッキング
    def GetAll(
        self,
        pData: list[
            NR_GET_REAL_DATA_ALL
        ],  # pointer - list length must be 1, element will be updated in-place
        lAccessMode: int = NR_ACCESS_NO_WAIT,
        nOpenId: int | None = None,
    ) -> int:
        if (
            not isinstance(pData, list)
            or len(pData) != 1
            or not isinstance(pData[0], NR_GET_REAL_DATA_ALL)
        ):
            raise ValueError(
                "pData must be a list of one NR_GET_REAL_DATA_ALL instance"
            )
        if nOpenId is None:
            nOpenId = self.nOpenId
        # configure dll prototype
        self.dll.NR_GetAll.argtypes = (
            ctypes.c_int,
            ctypes.POINTER(NR_GET_REAL_DATA_ALL),
            ctypes.c_long,
        )
        self.dll.NR_GetAll.restype = ctypes.c_int
        # call and return result; pData[0] is updated by the call
        return self.dll.NR_GetAll(
            ctypes.c_int(nOpenId), ctypes.byref(pData[0]), ctypes.c_long(lAccessMode)
        )

    def SetAll(
        self,
        pData: NR_SET_REAL_DATA_ALL,  # pointer - will be passed by reference
        lAccessMode: int = NR_ACCESS_NO_WAIT,
        nOpenId: int | None = None,
    ) -> int:
        if not isinstance(pData, NR_SET_REAL_DATA_ALL):
            raise ValueError("pData must be an instance of NR_SET_REAL_DATA_ALL")
        if nOpenId is None:
            nOpenId = self.nOpenId
        # configure dll prototype
        self.dll.NR_SetAll.argtypes = (
            ctypes.c_int,
            ctypes.POINTER(NR_SET_REAL_DATA_ALL),
            ctypes.c_long,
        )
        self.dll.NR_SetAll.restype = ctypes.c_int
        # call and return result
        return self.dll.NR_SetAll(
            ctypes.c_int(nOpenId), ctypes.byref(pData), ctypes.c_long(lAccessMode)
        )


# グローバルなインスタンス
global_nr = OpenNRIF()
