"""Reader for DICHASUS tfrecords without TensorFlow.

Each feature holds a serialized tf TensorProto (tf.io.serialize_tensor). We decode the
protobuf wire format directly: field 1 = dtype, field 2 = TensorShapeProto, field 4 = raw content.
"""
import numpy as np
from tfrecord.reader import tfrecord_iterator
from tfrecord import example_pb2

_DT = {1: np.float32, 2: np.float64, 3: np.int32, 9: np.int64}


def _varint(b, i):
    r = s = 0
    while True:
        c = b[i]; i += 1
        r |= (c & 0x7F) << s; s += 7
        if c < 0x80:
            return r, i


def _fields(b):
    i, n = 0, len(b)
    while i < n:
        key, i = _varint(b, i)
        f, wt = key >> 3, key & 7
        if wt == 0:
            v, i = _varint(b, i)
        elif wt == 2:
            L, i = _varint(b, i); v = b[i:i + L]; i += L
        elif wt == 5:
            v = b[i:i + 4]; i += 4
        elif wt == 1:
            v = b[i:i + 8]; i += 8
        else:
            raise ValueError(wt)
        yield f, v


def parse_tensor(b):
    dtype, shape, content = None, [], b""
    for f, v in _fields(b):
        if f == 1:
            dtype = v
        elif f == 2:
            for f2, dim in _fields(v):
                if f2 == 2:
                    for f3, sz in _fields(dim):
                        if f3 == 1:
                            shape.append(sz)
        elif f == 4:
            content = v
    return np.frombuffer(content, dtype=_DT[dtype]).reshape(shape)


def iterate(path, keys=("csi", "pos-tachy", "time", "snr")):
    for rec in tfrecord_iterator(path):
        ex = example_pb2.Example(); ex.ParseFromString(rec)
        out = {}
        for k in keys:
            v = ex.features.feature[k]
            kind = v.WhichOneof("kind")
            out[k] = parse_tensor(v.bytes_list.value[0]) if kind == "bytes_list" else np.asarray(getattr(v, kind).value)
        yield out


if __name__ == "__main__":
    import sys
    for i, r in enumerate(iterate(sys.argv[1])):
        print({k: (v.shape, v.dtype) for k, v in r.items()}, r["pos-tachy"], r["time"])
        if i == 2:
            break
