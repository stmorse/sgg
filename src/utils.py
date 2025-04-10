import json
import os

import bz2                # for .bz2
import lzma               # for .xz
import zstandard as zstd  # for .zst
import zarr

import numpy as np

DATA_PATH = '/sciclone/data10/twford/reddit/reddit/comments/'
EMBED_PATH = '/sciclone/geograd/stmorse/reddit/embeddings'

def open_compressed(file_path):
    if file_path.endswith('.bz2'):
        return bz2.BZ2File(file_path, 'rb')
    elif file_path.endswith('.xz'):
        return lzma.open(file_path, 'rb')
    elif file_path.endswith('.zst'):
        # For .zst, return a stream reader
        f = open(file_path, 'rb')  # Open file in binary mode
        dctx = zstd.ZstdDecompressor(max_window_size=2**31)
        return dctx.stream_reader(f)
        # return dctx.read_to_iter(f)
    else:
        raise ValueError('Unsupported file extension.')


def parse_line(line, return_type='metadata', metadata=['id'], 
               filter=True, truncate=0):
    entry = json.loads(line.decode('utf-8'))

    if filter:
        if 'body' not in entry or entry['author'] == '[deleted]':
            return None
    
    if return_type == 'metadata':
        res = {k: entry[k] for k in metadata}
        return res
    elif return_type == 'sentences':
        body = entry['body']
        if truncate > 0:
            body = body[:truncate]
        return body
    else: 
        print('Error: return_type not recognized. Must be metadata or sentences.')

    return None


def read_file(
        year, month, 
        return_type='metadata', 
        metadata=['id'],
        chunk_size=10000,
        data_path=DATA_PATH):
    """
    Read JSON entries from a compressed file, extract fields,
    and yield chunks of size `chunk_size`.

    return_type : 'metadata' or 'sentences'
    """

    # TODO: Add support for .xz and .zst files
    file_path = os.path.join(
        data_path,
        f'RC_{year}-{month:02}.bz2'
    )

    buffer = []         # To store 'body' fields in chunks
    byte_buffer = b""   # For handling partial lines in `.zst` files

    with open_compressed(file_path) as f:
        # Iterate over the file
        for chunk in iter(lambda: f.read(8192), b""):  # Read file in binary chunks
            byte_buffer += chunk

            # Process each line in the byte buffer
            while b"\n" in byte_buffer:
                line, byte_buffer = byte_buffer.split(b"\n", 1)

                res = parse_line(
                    line, 
                    return_type=return_type,
                    metadata=metadata,
                    filter=True,
                    truncate=2000
                )

                if res is None:
                    continue

                # Add to the chunk buffer
                buffer.append(res)
                if len(buffer) >= chunk_size:
                    yield buffer
                    buffer = []

        # Handle any remaining partial JSON line
        if byte_buffer:
            res = parse_line(
                line, 
                return_type=return_type,
                metadata=metadata,
                filter=True,
                truncate=2000
            )
            buffer.append(res)

        # Yield any leftovers in the chunk buffer
        if buffer:
            yield buffer

def load_embeddings(year, month, embed_path=EMBED_PATH):
    file_path = os.path.join(
        embed_path,
        f'embeddings_{year}-{month:02}.npz'
    )
    with open(file_path, 'rb') as f:
        embeddings = np.load(f)['embeddings']
        return embeddings