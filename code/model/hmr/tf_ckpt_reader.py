"""
Minimal reader for TensorFlow 1.x checkpoints (.index + .data-00000-of-00001)
without needing TensorFlow installed.

The .index file is a LevelDB-style SSTable whose values are BundleEntryProto
messages describing where each tensor lives inside the .data shard.
"""

import struct
import numpy as np

_TABLE_MAGIC = 0xdb4775248b80fb57
_DTYPES = {1: np.float32, 2: np.float64, 3: np.int32, 9: np.int64}


def _varint(buf, pos):
    result = shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            return result, pos
        shift += 7


def _proto_fields(buf):
    """Yield (field_number, wire_type, value) for a protobuf message."""
    pos = 0
    while pos < len(buf):
        tag, pos = _varint(buf, pos)
        field, wire = tag >> 3, tag & 7
        if wire == 0:
            val, pos = _varint(buf, pos)
        elif wire == 1:
            val = buf[pos:pos + 8]; pos += 8
        elif wire == 2:
            ln, pos = _varint(buf, pos)
            val = buf[pos:pos + ln]; pos += ln
        elif wire == 5:
            val = buf[pos:pos + 4]; pos += 4
        else:
            raise ValueError(f'unsupported wire type {wire}')
        yield field, wire, val


def _block_entries(buf):
    """Decode a LevelDB block into (key, value) pairs."""
    num_restarts = struct.unpack_from('<I', buf, len(buf) - 4)[0]
    end = len(buf) - 4 - 4 * num_restarts
    pos, key = 0, b''
    while pos < end:
        shared, pos = _varint(buf, pos)
        non_shared, pos = _varint(buf, pos)
        vlen, pos = _varint(buf, pos)
        key = key[:shared] + buf[pos:pos + non_shared]
        pos += non_shared
        yield key, buf[pos:pos + vlen]
        pos += vlen


def _read_block(data, handle):
    off, pos = _varint(handle, 0)
    size, _ = _varint(handle, pos)
    if data[off + size] != 0:
        raise ValueError('compressed checkpoint blocks are not supported')
    return data[off:off + size]


def _parse_entry(val):
    entry = {'dtype': 1, 'shape': [], 'offset': 0, 'size': 0}
    for f, _, v in _proto_fields(val):
        if f == 1:
            entry['dtype'] = v
        elif f == 2:
            for df, _, dv in _proto_fields(v):
                if df == 2:
                    for sf, _, sv in _proto_fields(dv):
                        if sf == 1:
                            entry['shape'].append(sv)
        elif f == 4:
            entry['offset'] = v
        elif f == 5:
            entry['size'] = v
    return entry


def load_checkpoint(prefix):
    """Return {variable_name: np.ndarray} for a TF1 checkpoint prefix."""
    with open(prefix + '.index', 'rb') as fh:
        index = fh.read()
    footer = index[-48:]
    if struct.unpack('<Q', footer[-8:])[0] != _TABLE_MAGIC:
        raise ValueError('not a TensorFlow checkpoint index')
    _, pos = _varint(footer, 0)          # metaindex offset
    _, pos = _varint(footer, pos)        # metaindex size
    index_handle = footer[pos:]

    entries = {}
    for _, handle in _block_entries(_read_block(index, index_handle)):
        for key, val in _block_entries(_read_block(index, handle)):
            if key:                       # empty key = bundle header
                entries[key.decode()] = _parse_entry(val)

    tensors = {}
    with open(prefix + '.data-00000-of-00001', 'rb') as fh:
        for name, e in entries.items():
            dt = _DTYPES.get(e['dtype'])
            if dt is None:
                continue
            fh.seek(e['offset'])
            arr = np.frombuffer(fh.read(e['size']), dtype=dt)
            tensors[name] = arr.reshape(e['shape']) if e['shape'] else arr.reshape(())
    return tensors
