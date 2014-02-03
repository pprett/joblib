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
import numpy as np
import struct

from marshal import loads as mloads

from copy import copy
from numpy.lib.format import magic
from numpy.lib.format import write_array_header_1_0
from numpy.lib.format import header_data_from_array_1_0
from numpy.lib.format import isfileobj

from ._compat import _basestring

from io import BytesIO

if sys.version_info[0] >= 3:
    Unpickler = pickle._Unpickler
    Pickler = pickle._Pickler

else:
    Unpickler = pickle.Unpickler
    Pickler = pickle.Pickler


_MEGA = 2 ** 20
_MAX_LEN = len(hex(2 ** 64))

# To detect file types
_ZFILE_PREFIX = b'ZF'
_CHUNK_SIZE = 64 * 1024

BINARRAY = 'A'

###############################################################################
# Compressed file with Zlib

def _read_magic(file_handle):
    """ Utility to check the magic signature of a file identifying it as a
        Zfile
    """
    magic_ = file_handle.read(len(_ZFILE_PREFIX))
    # Pickling needs file-handles at the beginning of the file
    file_handle.seek(0)
    return magic_


def read_zfile(file_handle):
    """Read the z-file and return the content as a string

    Z-files are raw data compressed with zlib used internally by joblib
    for persistence. Backward compatibility is not guaranteed. Do not
    use for external purposes.
    """
    file_handle.seek(0)
    assert _read_magic(file_handle) == _ZFILE_PREFIX, \
        "File does not have the right magic"
    length = file_handle.read(len(_ZFILE_PREFIX) + _MAX_LEN)
    length = length[len(_ZFILE_PREFIX):]
    length = int(length, 16)
    # We use the known length of the data to tell Zlib the size of the
    # buffer to allocate.

    decompresser= zlib.decompressobj()
    data = ''
    while True:
        chunk = file_handle.read(_CHUNK_SIZE)
        if not chunk:
            break
        data += decompresser.decompress(chunk)
    data += decompresser.flush()  # Read the remainder


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
    compresser = zlib.compressobj(compress)
    file_handle.write(_ZFILE_PREFIX)
    length = hex(len(data)).encode('ascii')
    # If python 2.x, we need to remove the trailing 'L' in the hex representation
    length = length.rstrip(b'L')

    file_handle.write(length.ljust(_MAX_LEN))

    # Write the data out in chunks
    for i in xrange(len(data)//_CHUNK_SIZE+1):
        chunk = data[i*_CHUNK_SIZE:(i+1)*_CHUNK_SIZE]
        file_handle.write(compresser.compress(chunk))
    tail = compresser.flush()
    if tail: # Write the remainder
        file_handle.write(tail)


def np_write_array(fp, array, version=(1, 0)):
    """
    Write an array to an NPY file, including a header.

    If the array is neither C-contiguous nor Fortran-contiguous AND the
    file_like object is not a real file object, this function will have to
    copy data in memory.

    Parameters
    ----------
    fp : file_like object
        An open, writable file object, or similar object with a ``.write()``
        method.
    array : ndarray
        The array to write to disk.
    version : (int, int), optional
        The version number of the format.  Default: (1, 0)

    Raises
    ------
    ValueError
        If the array cannot be persisted.
    Various other errors
        If the array contains Python objects as part of its dtype, the
        process of pickling them may raise various errors if the objects
        are not picklable.

    """
    if version != (1, 0):
        msg = "we only support format version (1,0), not %s"
        raise ValueError(msg % (version,))
    fp.write(magic(*version))
    write_array_header_1_0(fp, header_data_from_array_1_0(array))

    # Set buffer size to 16 MiB to hide the Python loop overhead.
    buffersize = max(16 * 1024 ** 2 // array.itemsize, 1)

    if array.dtype.hasobject:
        # We contain Python objects so we cannot write out the data directly.
        # Instead, we will pickle it out with version 2 of the pickle protocol.
        pickle.dump(array, fp, protocol=2)
    elif array.flags.f_contiguous and not array.flags.c_contiguous:
        if isfileobj(fp):
            array.T.tofile(fp)
        else:
            for chunk in np.nditer(
                    array, flags=['external_loop', 'buffered', 'zerosize_ok'],
                    buffersize=buffersize, order='F'):
                fp.write(chunk.tostring('C'))
    else:
        if isfileobj(fp):
            array.tofile(fp)
        else:
            for chunk in np.nditer(
                    array, flags=['external_loop', 'buffered', 'zerosize_ok'],
                    buffersize=buffersize, order='C'):
                fp.write(chunk.tostring('C'))


def np_save(file, arr):
    """
    Save an array to a binary file in NumPy ``.npy`` format.

    Parameters
    ----------
    file : file or str
        File or filename to which the data is saved.  If file is a file-object,
        then the filename is unchanged.  If file is a string, a ``.npy``
        extension will be appended to the file name if it does not already
        have one.
    arr : array_like
        Array data to be saved.

    See Also
    --------
    savez : Save several arrays into a ``.npz`` archive
    savetxt, load

    Notes
    -----
    For a description of the ``.npy`` format, see `format`.

    Examples
    --------
    >>> from tempfile import TemporaryFile
    >>> outfile = TemporaryFile()

    >>> x = np.arange(10)
    >>> np.save(outfile, x)

    >>> outfile.seek(0) # Only needed here to simulate closing & reopening file
    >>> np.load(outfile)
    array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    """
    own_fid = False
    if isinstance(file, basestring):
        if not file.endswith('.npy'):
            file = file + '.npy'
        fid = open(file, "wb")
        own_fid = True
    else:
        fid = file
    try:
        arr = np.asanyarray(arr)
        np_write_array(fid, arr)
    finally:
        if own_fid:
            fid.close()


###############################################################################
# Utility objects for persistence.

class NDArrayWrapper(object):
    """ An object to be persisted instead of numpy arrays. """
    def __init__(self, array):
        "Store the useful information for later"
        self.subclass = type(array)
        self.array = array


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
    def __init__(self, array):
        "Store the useful information for later"
        super(ZNDArrayWrapper, self).__init__(array)


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
        self.file = open(filename, 'wb')

        Pickler.__init__(self, self.file,
                                protocol=pickle.HIGHEST_PROTOCOL)

    def save(self, obj):
        """ Subclass the save method, to save ndarray subclasses in npy
            files, rather than pickling them. Of course, this is a
            total abuse of the Pickler class.
        """
        if np is not None and type(obj) in (np.ndarray,
                                            np.matrix, np.memmap):
            if self.compress == 0:
                obj = NDArrayWrapper(obj)
            else:
                obj = ZNDArrayWrapper(obj)
        return Pickler.save(self, obj)

    def close(self):
        pass

    def save_nd_array(self, obj, pack=struct.pack):
        n = obj.array.nbytes
        self.write(BINARRAY + pack("<i", n))
        np_write_array(self.file, obj.array)
    dispatch[NDArrayWrapper] = save_nd_array

    def save_znp_array(self, obj, pack=struct.pack):
        n = obj.array.nbytes
        self.write(BINARRAY + pack("<i", n))
        np_write_array(self.file, obj.array)
    dispatch[ZNDArrayWrapper] = save_znd_array



class NumpyUnpickler(Unpickler):
    """A subclass of the Unpickler to unpickle our numpy pickles.
    """
    dispatch = Unpickler.dispatch.copy()

    def __init__(self, filename, file_handle):
        self.file_handle = self._open_pickle(file_handle)
        Unpickler.__init__(self, self.file_handle)

    def _open_pickle(self, file_handle):
        return file_handle

    def load_np_array(self):
        size = mloads('i' + self.read(4))
        arr = np.lib.format.read_array(self.file_handle)
        self.append(arr)
    dispatch[BINARRAY] = load_np_array



class ZipNumpyUnpickler(NumpyUnpickler):
    """A subclass of our Unpickler to unpickle on the fly from
    compressed storage."""

    def __init__(self, filename, file_handle):
        NumpyUnpickler.__init__(self, filename,
                                file_handle)

    def _open_pickle(self, file_handle):
        return BytesIO(read_zfile(file_handle))


###############################################################################
# Utility functions

def dump(value, filename, compress=0, cache_size=100):
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
    """Reconstruct a Python object from a file persisted with joblib.load.

    Parameters
    -----------
    filename: string
        The name of the file from which to load the object
    mmap_mode: {None, 'r+', 'r', 'w+', 'c'}, optional
        If not None, the arrays are memory-mapped from the disk. This
        mode has not effect for compressed files. Note that in this
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
    file_handle = open(filename, 'rb')
    # We are careful to open the file handle early and keep it open to
    # avoid race-conditions on renames. That said, if data are stored in
    # companion files, moving the directory will create a race when
    # joblib tries to access the companion files.
    if _read_magic(file_handle) == _ZFILE_PREFIX:
        if mmap_mode is not None:
            warnings.warn('file "%(filename)s" appears to be a zip, '
                    'ignoring mmap_mode "%(mmap_mode)s" flag passed'
                    % locals(), Warning, stacklevel=2)
        unpickler = ZipNumpyUnpickler(filename, file_handle=file_handle)
    else:
        unpickler = NumpyUnpickler(filename,
                                   file_handle=file_handle)

    try:
        obj = unpickler.load()
    finally:
        if hasattr(unpickler, 'file_handle'):
            unpickler.file_handle.close()
    return obj
