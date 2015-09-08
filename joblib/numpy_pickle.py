"""
Utilities for fast persistence of big data, with optional compression.
"""

# Author: Gael Varoquaux <gael dot varoquaux at normalesup dot org>
# Copyright (c) 2009 Gael Varoquaux
# License: BSD Style, 3 clauses.

import pickle
import traceback
import sys
import os
import zlib
import warnings
import struct
import codecs

from ._compat import _basestring

from io import BytesIO

PY3 = sys.version_info[0] >= 3

if PY3:
    Unpickler = pickle._Unpickler
    Pickler = pickle._Pickler
    from io.StringIO import StringIO

    def asbytes(s):
        if isinstance(s, bytes):
            return s
        return s.encode('latin1')
else:
    Unpickler = pickle.Unpickler
    Pickler = pickle.Pickler
    asbytes = str
    from cStringIO import StringIO


def hex_str(an_int):
    """Converts an int to an hexadecimal string
    """
    return '{0:#x}'.format(an_int)


_MEGA = 2 ** 20

# Compressed pickle header format: _ZFILE_PREFIX followed by _MAX_LEN
# bytes which contains the length of the zlib compressed data as an
# hexadecimal string. For example: 'ZF0x139              '
_ZFILE_PREFIX = asbytes('ZF')
_MAX_LEN = len(hex_str(2 ** 64))

_IFILE_PREFIX = asbytes('IF')
_IFILE_PREFIX_LEN = len(_IFILE_PREFIX)
_HEADER_LEN = _IFILE_PREFIX_LEN + _MAX_LEN

###############################################################################
# Compressed file with Zlib

def _read_magic(file_handle):
    """ Utility to check the magic signature of a file identifying it as a
        Zfile
    """
    magic = file_handle.read(len(_ZFILE_PREFIX))
    # Pickling needs file-handles at the beginning of the file
    file_handle.seek(0)
    return magic


def read_zfile(file_handle):
    """Read the z-file and return the content as a string

    Z-files are raw data compressed with zlib used internally by joblib
    for persistence. Backward compatibility is not guaranteed. Do not
    use for external purposes.
    """
    file_handle.seek(0)
    assert _read_magic(file_handle) == _ZFILE_PREFIX, \
        "File does not have the right magic"
    header_length = len(_ZFILE_PREFIX) + _MAX_LEN
    length = file_handle.read(header_length)
    length = length[len(_ZFILE_PREFIX):]
    length = int(length, 16)

    # With python2 and joblib version <= 0.8.4 compressed pickle header is one
    # character wider so we need to ignore an additional space if present.
    # Note: the first byte of the zlib data is guaranteed not to be a
    # space according to
    # https://tools.ietf.org/html/rfc6713#section-2.1
    next_byte = file_handle.read(1)
    if next_byte != b' ':
        # The zlib compressed data has started and we need to go back
        # one byte
        file_handle.seek(header_length)

    # We use the known length of the data to tell Zlib the size of the
    # buffer to allocate.
    data = zlib.decompress(file_handle.read(), 15, length)
    assert len(data) == length, (
        "Incorrect data length while decompressing %s."
        "The file could be corrupted." % file_handle)
    return data


def write_zfile(file_handle, data, compress=1):
    """Write the data in the given file as a Z-file.

    Z-files are raw data compressed with zlib used internally by joblib
    for persistence. Backward compatibility is not guarantied. Do not
    use for external purposes.
    """
    file_handle.write(_ZFILE_PREFIX)
    length = hex_str(len(data))
    # Store the length of the data
    file_handle.write(asbytes(length.ljust(_MAX_LEN)))
    file_handle.write(zlib.compress(asbytes(data), compress))


###############################################################################
# Ifile util
def read_ifile_offsets(file_handle):
    file_handle.seek(0)
    assert _read_magic(file_handle) == _IFILE_PREFIX, \
        "File does not have the right magic"
    length = file_handle.read(_HEADER_LEN)
    length = length[_IFILE_PREFIX_LEN:]
    length = int(length, 16)
    file_handle.seek(0)
    return _HEADER_LEN + length


def open_memmap(fp, np):
    version = np.lib.format.read_magic(fp)
    np.lib.format._check_version(version)
    shape, fortran_order, dtype = np.lib.format._read_array_header(fp, version)
    if dtype.hasobject:
        msg = "Array can't be memory-mapped: Python objects in dtype."
        raise ValueError(msg)
    offset = fp.tell()
    if fortran_order:
        order = 'F'
    else:
        order = 'C'

    marray = np.memmap(fp.name, dtype=dtype, shape=shape, order=order,
                       mode='r', offset=offset)
    return marray

###############################################################################
# Utility objects for persistence.

class NDArrayWrapper(object):
    """ An object to be persisted instead of numpy arrays.

        The only thing this object does, is to carry the filename in which
        the array has been persisted, and the array subclass.
    """
    def __init__(self, filename, subclass, allow_mmap=True):
        "Store the useful information for later"
        self.filename = filename
        self.subclass = subclass
        self.allow_mmap = allow_mmap

    def read(self, unpickler):
        "Reconstruct the array"
        filename = os.path.join(unpickler._dirname, self.filename)
        # Load the array from the disk

        # use getattr instead of self.allow_mmap to ensure backward compat
        # with NDArrayWrapper instances pickled with joblib < 0.9.0
        allow_mmap = getattr(self, 'allow_mmap', True)
        memmap_kwargs = ({} if not allow_mmap
                         else {'mmap_mode': unpickler.mmap_mode})
        array = unpickler.np.load(filename, **memmap_kwargs)
        # Reconstruct subclasses. This does not work with old
        # versions of numpy
        if (hasattr(array, '__array_prepare__')
                and not self.subclass in (unpickler.np.ndarray,
                                      unpickler.np.memmap)):
            # We need to reconstruct another subclass
            new_array = unpickler.np.core.multiarray._reconstruct(
                    self.subclass, (0,), 'b')
            new_array.__array_prepare__(array)
            array = new_array
        return array

    #def __reduce__(self):
    #    return None


class InlineNDArrayWrapper(NDArrayWrapper):
    """ An object to be persisted instead of numpy arrays.

        The only thing this object does, is to carry the offset in which
        the array has been persisted, and the array subclass.
    """
    def __init__(self, subclass, offset, allow_mmap=True):
        "Store the useful information for later"
        self.subclass = subclass
        self.offset = offset
        self.allow_mmap = allow_mmap

    def read(self, unpickler):
        "Reconstruct the array"

        np_ver = [int(x) for x in unpickler.np.__version__.split('.', 2)[:2]]

        mmap_mode = unpickler.mmap_mode

        fd = unpickler.file_handle
        pos = fd.tell()
        fd.seek(unpickler._array_segment_offset + self.offset)
        try:
            if mmap_mode:
                array = open_memmap(fd, unpickler.np)
            else:
                array = unpickler.np.load(fd)
        finally:
            fd.seek(pos)
        # Reconstruct subclasses. This does not work with old
        # versions of numpy
        if (hasattr(array, '__array_prepare__')
                and not self.subclass in (unpickler.np.ndarray,
                                      unpickler.np.memmap)):
            # We need to reconstruct another subclass
            new_array = unpickler.np.core.multiarray._reconstruct(
                    self.subclass, (0,), 'b')
            new_array.__array_prepare__(array)
            array = new_array
        return array

    #def __reduce__(self):
    #    return None


class ZNDArrayWrapper(NDArrayWrapper):
    """An object to be persisted instead of numpy arrays.

    This object store the Zfile filename in which
    the data array has been persisted, and the meta information to
    retrieve it.

    The reason that we store the raw buffer data of the array and
    the meta information, rather than array representation routine
    (tostring) is that it enables us to use completely the strided
    model to avoid memory copies (a and a.T store as fast). In
    addition saving the heavy information separately can avoid
    creating large temporary buffers when unpickling data with
    large arrays.
    """
    def __init__(self, filename, init_args, state):
        "Store the useful information for later"
        self.filename = filename
        self.state = state
        self.init_args = init_args

    def read(self, unpickler):
        "Reconstruct the array from the meta-information and the z-file"
        # Here we a simply reproducing the unpickling mechanism for numpy
        # arrays
        filename = os.path.join(unpickler._dirname, self.filename)
        array = unpickler.np.core.multiarray._reconstruct(*self.init_args)
        with open(filename, 'rb') as f:
            data = read_zfile(f)
        state = self.state + (data,)
        array.__setstate__(state)
        return array


###############################################################################
# Pickler classes

class NumpyPickler(Pickler):
    """A pickler to persist of big data efficiently.

        The main features of this object are:

         * persistence of numpy arrays in separate .npy files, for which
           I/O is fast.

         * optional compression using Zlib, with a special care on avoid
           temporaries.
    """
    dispatch = Pickler.dispatch.copy()

    def __init__(self, filename, compress=0, cache_size=10):
        self._filename = filename
        self._filenames = [filename, ]
        self.cache_size = cache_size
        self.compress = compress
        if not self.compress:
            self.file = open(filename, 'wb')
        else:
            self.file = BytesIO()
        # Count the number of npy files that we have created:
        self._npy_counter = 0
        highest_python_2_3_compatible_protocol = 2
        Pickler.__init__(self, self.file,
                         protocol=highest_python_2_3_compatible_protocol)
        # delayed import of numpy, to avoid tight coupling
        try:
            import numpy as np
        except ImportError:
            np = None
        self.np = np

    def _write_array(self, array, filename):
        if not self.compress:
            self.np.save(filename, array)
            allow_mmap = not array.dtype.hasobject
            container = NDArrayWrapper(os.path.basename(filename),
                                       type(array),
                                       allow_mmap=allow_mmap)
        else:
            filename += '.z'
            # Efficient compressed storage:
            # The meta data is stored in the container, and the core
            # numerics in a z-file
            _, init_args, state = array.__reduce__()
            # the last entry of 'state' is the data itself
            with open(filename, 'wb') as zfile:
                write_zfile(zfile, state[-1], compress=self.compress)
            state = state[:-1]
            container = ZNDArrayWrapper(os.path.basename(filename),
                                            init_args, state)
        return container, filename

    def save(self, obj):
        """ Subclass the save method, to save ndarray subclasses in npy
            files, rather than pickling them. Of course, this is a
            total abuse of the Pickler class.
        """
        if self.np is not None and type(obj) in (self.np.ndarray,
                                            self.np.matrix, self.np.memmap):
            size = obj.size * obj.itemsize
            if self.compress and size < self.cache_size * _MEGA:
                # When compressing, as we are not writing directly to the
                # disk, it is more efficient to use standard pickling
                if type(obj) is self.np.memmap:
                    # Pickling doesn't work with memmaped arrays
                    obj = self.np.asarray(obj)
                return Pickler.save(self, obj)
            self._npy_counter += 1
            try:
                filename = '%s_%02i.npy' % (self._filename,
                                            self._npy_counter)
                # This converts the array in a container
                obj, filename = self._write_array(obj, filename)
                self._filenames.append(filename)
            except:
                self._npy_counter -= 1
                # XXX: We should have a logging mechanism
                print('Failed to save %s to .npy file:\n%s' % (
                        type(obj),
                        traceback.format_exc()))
        return Pickler.save(self, obj)

    def save_bytes(self, obj):
        """Strongly inspired from python 2.7 pickle.Pickler.save_string"""
        if self.bin:
            n = len(obj)
            if n < 256:
                self.write(pickle.SHORT_BINSTRING + asbytes(chr(n)) + obj)
            else:
                self.write(pickle.BINSTRING + struct.pack("<i", n) + obj)
            self.memoize(obj)
        else:
            Pickler.save_bytes(self, obj)

    # We need to override save_bytes for python 3. We are using
    # protocol=2 for python 2/3 compatibility and save_bytes for
    # protocol < 3 ends up creating a unicode string which is very
    # inefficient resulting in pickles up to 1.5 times the size you
    # would get with protocol=4 or protocol=2 with python 2.7. This
    # cause severe slowdowns in joblib.dump and joblib.load. See
    # https://github.com/joblib/joblib/issues/194 for more details.
    if PY3:
        dispatch[bytes] = save_bytes

    def close(self):
        if self.compress:
            with open(self._filename, 'wb') as zfile:
                write_zfile(zfile, self.file.getvalue(), self.compress)


class InlineNumpyPickler(Pickler):
    """A pickler to persist of big data efficiently.

        The main features of this object are:

         * persistence of numpy arrays by postfixing them to the pickle file.

       One can think of each inline pickle file as containing 3 segments:

           Metadata segment | Pickle segment | Array segment

       The metadata segment is fixed lenght and contains a magic prefix (`_IFILE_PREFIX`)
       and the lenght of the Pickle segment (stored in a 19 byte hex string repr).

       The pickle segment is a regular pickle dump but with all numpy array objects
       replaced by InlineNDArrayWrapper objects that hold offsets (in bytes) to the
       Array segment.
       The array segment contains a sequence of np.save dumps.
    """
    dispatch = Pickler.dispatch.copy()

    def __init__(self, filename):
        self._filename = filename
        self._filenames = [filename]

        self.file = open(filename, 'wb')

        # write header
        self.file.write(_IFILE_PREFIX)
        # use dummy length
        self.file.write(asbytes(hex_str(0).ljust(_MAX_LEN)))

        # Count the number of npy arrays
        self._npy_counter = 0
        self._npy_arrays = []
        self._npy_offset = 0

        highest_python_2_3_compatible_protocol = 2
        Pickler.__init__(self, self.file,
                         protocol=highest_python_2_3_compatible_protocol)
        # delayed import of numpy, to avoid tight coupling
        try:
            import numpy as np
        except ImportError:
            np = None
        self.np = np

    def _write_array(self, obj):
        """Write array will return a container for the numpy array that holds the
        offset of the serialized array in the postfix segment.

        The offset is computed by taking the old offset and adding the size of the header
        (`np.lib.format._write_array_header`) and the size of the data (`obj.nbytes`).

        The actual arrays are written in `self.close` .
        """
        # get header size
        buf = StringIO()
        self.np.lib.format._write_array_header(
            buf, self.np.lib.format.header_data_from_array_1_0(obj), version=None)
        header_len = buf.tell()
        # size = header size + data size
        size = header_len + obj.nbytes

        self._npy_arrays.append(obj)
        container = InlineNDArrayWrapper(type(obj), self._npy_offset)
        self._npy_offset += size
        return container

    def save(self, obj):
        """ Subclass the save method, to postfix ndarray subclasses in
        the pickle file. Of course, this is a total abuse of the Pickler class.
        """
        if (self.np is not None and type(obj) in (self.np.ndarray, self.np.matrix, self.np.memmap)
            and obj.dtype not in (self.np.dtype('O'), )):
            self._npy_counter += 1
            obj = self._write_array(obj)
        return Pickler.save(self, obj)

    def save_bytes(self, obj):
        """Strongly inspired from python 2.7 pickle.Pickler.save_string"""
        if self.bin:
            n = len(obj)
            if n < 256:
                self.write(pickle.SHORT_BINSTRING + asbytes(chr(n)) + obj)
            else:
                self.write(pickle.BINSTRING + struct.pack("<i", n) + obj)
            self.memoize(obj)
        else:
            Pickler.save_bytes(self, obj)

    # We need to override save_bytes for python 3. We are using
    # protocol=2 for python 2/3 compatibility and save_bytes for
    # protocol < 3 ends up creating a unicode string which is very
    # inefficient resulting in pickles up to 1.5 times the size you
    # would get with protocol=4 or protocol=2 with python 2.7. This
    # cause severe slowdowns in joblib.dump and joblib.load. See
    # https://github.com/joblib/joblib/issues/194 for more details.
    if PY3:
        dispatch[bytes] = save_bytes

    def close(self):
        """Close writes the postfix segment and updates the len of the pickle segment. """
        # length of pickled content
        pkl_length = self.file.tell() - _HEADER_LEN
        # append arrays
        for obj in self._npy_arrays:
            #t0 = self.file.tell()
            self.np.save(self.file, obj)
            #print('close size: {}'.format(self.file.tell() - t0))
        # seek to begin and update header length
        self.file.seek(_IFILE_PREFIX_LEN)
        self.file.write(asbytes(hex_str(pkl_length).ljust(_MAX_LEN)))


class NumpyUnpickler(Unpickler):
    """A subclass of the Unpickler to unpickle our numpy pickles.
    """
    dispatch = Unpickler.dispatch.copy()

    def __init__(self, filename, file_handle, mmap_mode=None):
        self._filename = os.path.basename(filename)
        self._dirname = os.path.dirname(filename)
        self.mmap_mode = mmap_mode
        self.file_handle = self._open_pickle(file_handle)
        Unpickler.__init__(self, self.file_handle)
        try:
            import numpy as np
        except ImportError:
            np = None
        self.np = np

        if PY3:
            self.encoding = 'bytes'

    # Python 3.2 and 3.3 do not support encoding=bytes so I copied
    # _decode_string, load_string, load_binstring and
    # load_short_binstring from python 3.4 to emulate this
    # functionality
    if PY3 and sys.version_info.minor < 4:
        def _decode_string(self, value):
            """Copied from python 3.4 pickle.Unpickler._decode_string"""
            # Used to allow strings from Python 2 to be decoded either as
            # bytes or Unicode strings.  This should be used only with the
            # STRING, BINSTRING and SHORT_BINSTRING opcodes.
            if self.encoding == "bytes":
                return value
            else:
                return value.decode(self.encoding, self.errors)

        def load_string(self):
            """Copied from python 3.4 pickle.Unpickler.load_string"""
            data = self.readline()[:-1]
            # Strip outermost quotes
            if len(data) >= 2 and data[0] == data[-1] and data[0] in b'"\'':
                data = data[1:-1]
            else:
                raise pickle.UnpicklingError(
                    "the STRING opcode argument must be quoted")
            self.append(self._decode_string(codecs.escape_decode(data)[0]))
        dispatch[pickle.STRING[0]] = load_string

        def load_binstring(self):
            """Copied from python 3.4 pickle.Unpickler.load_binstring"""
            # Deprecated BINSTRING uses signed 32-bit length
            len, = struct.unpack('<i', self.read(4))
            if len < 0:
                raise pickle.UnpicklingError(
                    "BINSTRING pickle has negative byte count")
            data = self.read(len)
            self.append(self._decode_string(data))
        dispatch[pickle.BINSTRING[0]] = load_binstring

        def load_short_binstring(self):
            """Copied from python 3.4 pickle.Unpickler.load_short_binstring"""
            len = self.read(1)[0]
            data = self.read(len)
            self.append(self._decode_string(data))
        dispatch[pickle.SHORT_BINSTRING[0]] = load_short_binstring

    def _open_pickle(self, file_handle):
        return file_handle

    def load_build(self):
        """ This method is called to set the state of a newly created
            object.

            We capture it to replace our place-holder objects,
            NDArrayWrapper, by the array we are interested in. We
            replace them directly in the stack of pickler.
        """
        Unpickler.load_build(self)
        if isinstance(self.stack[-1], NDArrayWrapper):
            if self.np is None:
                raise ImportError('Trying to unpickle an ndarray, '
                        "but numpy didn't import correctly")
            nd_array_wrapper = self.stack.pop()
            array = nd_array_wrapper.read(self)
            self.stack.append(array)

    # Be careful to register our new method.
    if PY3:
        dispatch[pickle.BUILD[0]] = load_build
    else:
        dispatch[pickle.BUILD] = load_build


class ZipNumpyUnpickler(NumpyUnpickler):
    """A subclass of our Unpickler to unpickle on the fly from
    compressed storage."""

    def __init__(self, filename, file_handle):
        NumpyUnpickler.__init__(self, filename,
                                file_handle,
                                mmap_mode=None)

    def _open_pickle(self, file_handle):
        return BytesIO(read_zfile(file_handle))


class InlineNumpyUnpickler(NumpyUnpickler):
    """A subclass of the Unpickler to unpickle our numpy pickles.
    """

    def __init__(self, filename, file_handle, array_segment_offset, mmap_mode=None):
        self._filename = os.path.basename(filename)
        self._array_segment_offset = array_segment_offset
        self.mmap_mode = mmap_mode
        self.file_handle = self._open_pickle(file_handle)
        Unpickler.__init__(self, self.file_handle)
        try:
            import numpy as np
        except ImportError:
            np = None
        self.np = np

        if PY3:
            self.encoding = 'bytes'


###############################################################################
# Utility functions

def dump(value, filename, compress=0, cache_size=100, inline=False):
    """Fast persistence of an arbitrary Python object into a files, with
    dedicated storage for numpy arrays.

    Parameters
    -----------
    value: any Python object
        The object to store to disk
    filename: string
        The name of the file in which it is to be stored
    compress: integer for 0 to 9, optional
        Optional compression level for the data. 0 is no compression.
        Higher means more compression, but also slower read and
        write times. Using a value of 3 is often a good compromise.
        See the notes for more details.
    cache_size: positive number, optional
        Fixes the order of magnitude (in megabytes) of the cache used
        for in-memory compression. Note that this is just an order of
        magnitude estimate and that for big arrays, the code will go
        over this value at dump and at load time.
    inline: bool
        Whether or not to store numpy arrays within the pickle file or externally.

    Returns
    -------
    filenames: list of strings
        The list of file names in which the data is stored. If
        compress is false, each array is stored in a different file.

    See Also
    --------
    joblib.load : corresponding loader

    Notes
    -----
    Memmapping on load cannot be used for compressed files. Thus
    using compression can significantly slow down loading. In
    addition, compressed files take extra extra memory during
    dump and load.
    """
    if compress is True:
        # By default, if compress is enabled, we want to be using 3 by
        # default
        compress = 3
    if not isinstance(filename, _basestring):
        # People keep inverting arguments, and the resulting error is
        # incomprehensible
        raise ValueError(
              'Second argument should be a filename, %s (type %s) was given'
              % (filename, type(filename))
            )
    try:
        if inline:
            pickler = InlineNumpyPickler(filename)
        else:
            pickler = NumpyPickler(filename, compress=compress,
                                   cache_size=cache_size)
        pickler.dump(value)
        pickler.close()
    finally:
        if 'pickler' in locals() and hasattr(pickler, 'file'):
            pickler.file.flush()
            pickler.file.close()
    return pickler._filenames


def load(filename, mmap_mode=None):
    """Reconstruct a Python object from a file persisted with joblib.dump.

    Parameters
    -----------
    filename: string
        The name of the file from which to load the object
    mmap_mode: {None, 'r+', 'r', 'w+', 'c'}, optional
        If not None, the arrays are memory-mapped from the disk. This
        mode has no effect for compressed files. Note that in this
        case the reconstructed object might not longer match exactly
        the originally pickled object.

    Returns
    -------
    result: any Python object
        The object stored in the file.

    See Also
    --------
    joblib.dump : function to save an object

    Notes
    -----

    This function can load numpy array files saved separately during the
    dump. If the mmap_mode argument is given, it is passed to np.load and
    arrays are loaded as memmaps. As a consequence, the reconstructed
    object might not match the original pickled object. Note that if the
    file was saved with compression, the arrays cannot be memmaped.
    """
    with open(filename, 'rb') as file_handle:
        # We are careful to open the file handle early and keep it open to
        # avoid race-conditions on renames. That said, if data are stored in
        # companion files, moving the directory will create a race when
        # joblib tries to access the companion files.
        magic = _read_magic(file_handle)
        if magic == _ZFILE_PREFIX:
            if mmap_mode is not None:
                warnings.warn('file "%(filename)s" appears to be a zip, '
                              'ignoring mmap_mode "%(mmap_mode)s" flag passed'
                              % locals(), Warning, stacklevel=2)
            unpickler = ZipNumpyUnpickler(filename, file_handle=file_handle)
        elif magic == _IFILE_PREFIX:
            array_segment_offset = read_ifile_offsets(file_handle)
            file_handle.seek(_HEADER_LEN)
            unpickler = InlineNumpyUnpickler(filename, file_handle=file_handle,
                                             array_segment_offset=array_segment_offset,
                                             mmap_mode=mmap_mode)
        else:
            unpickler = NumpyUnpickler(filename, file_handle=file_handle,
                                       mmap_mode=mmap_mode)

        try:
            obj = unpickler.load()
        finally:
            if hasattr(unpickler, 'file_handle'):
                unpickler.file_handle.close()
        return obj
