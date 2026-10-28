"""Lazy local Parquet reading with a canonical global sample order."""
from bisect import bisect_right
import glob
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch


class OrderedParquetSource:
    """Footer-only index; token pages are read on demand without an Arrow cache."""

    def __init__(self, data_files, data_dir=None):
        if isinstance(data_files, str):
            patterns = data_files.split(',')
        elif isinstance(data_files, (tuple, list)):
            patterns = data_files
        else:
            raise ValueError('Ordered streaming requires explicit local Parquet data_files')
        files = []
        for pattern in patterns:
            if data_dir and not Path(pattern).is_absolute():
                pattern = str(Path(data_dir) / pattern)
            files.extend(sorted(glob.glob(pattern)))
        if not files or len(set(files)) != len(files):
            raise ValueError('Ordered Parquet files are missing or duplicated')
        self.files = files
        self.groups, self.starts = [], []
        total, signature = 0, []
        self.seq_len = None
        for file_index, file in enumerate(files):
            pf = pq.ParquetFile(file)
            typ = pf.schema_arrow.field('input_ids').type
            if not pa.types.is_fixed_size_list(typ) or not pa.types.is_integer(typ.value_type):
                raise ValueError('Ordered input_ids must be a fixed-size integer list')
            if self.seq_len is None:
                self.seq_len = typ.list_size
            if typ.list_size != self.seq_len:
                raise ValueError('Inconsistent sequence lengths')
            for group_index in range(pf.num_row_groups):
                rows = pf.metadata.row_group(group_index).num_rows
                if rows:
                    self.starts.append(total)
                    self.groups.append((file_index, group_index, rows))
                    total += rows
            signature.append((Path(file).name, Path(file).stat().st_size, pf.metadata.num_rows))
            pf.close()
        if not total:
            raise ValueError('Ordered dataset is empty')
        self.rows = total
        manifest = Path(files[0]).parent / 'metadata.json'
        manifest_hash = None
        if manifest.exists():
            contents = manifest.read_bytes()
            record = json.loads(contents)
            if record.get('format') == 'flame-ordered-parquet-v1':
                expected = [(e['file'], e['bytes'], e['rows']) for e in record['files']]
                if signature != expected or total != record['sequences']:
                    raise ValueError('Ordered files do not match their manifest')
                manifest_hash = hashlib.sha256(contents).hexdigest()
        self.fingerprint = hashlib.sha256(json.dumps([signature, manifest_hash]).encode()).hexdigest()


class OrderedParquetDataset(torch.utils.data.IterableDataset):
    """Each microbatch consumes consecutive rows of one global ordered stream.

    Workers receive microbatch numbers in round-robin order. StatefulDataLoader's
    ordered delivery restores that order irrespective of worker count. The union
    across ranks and accumulation microbatches is therefore the same at each
    optimizer step when global batch size is unchanged.
    """

    def __init__(self, source, seq_len, rank, world_size, batch_size):
        super().__init__()
        if seq_len != source.seq_len:
            raise ValueError(f'Expected seq_len={seq_len}, found {source.seq_len}')
        if batch_size <= 0 or world_size <= 0 or not 0 <= rank < world_size:
            raise ValueError('Invalid ordered data geometry')
        if source.rows % (world_size * batch_size):
            raise ValueError('Ordered dataset rows must be divisible by the global microbatch size')
        self.source, self.rank, self.world_size, self.batch_size = source, rank, world_size, batch_size
        self.next_batch, self.offset, self.worker_geometry = None, 0, None

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        geometry = (worker.id, worker.num_workers) if worker else (0, 1)
        if self.worker_geometry is not None and self.worker_geometry != geometry:
            raise ValueError('Ordered data resume requires the original worker geometry')
        self.worker_geometry = geometry
        if self.next_batch is None:
            self.next_batch = geometry[0]
        group_id, values, file_id, pf = None, None, None, None
        try:
            while True:
                base = (self.next_batch * self.world_size * self.batch_size) % self.source.rows
                while self.offset < self.batch_size:
                    row = base + self.rank * self.batch_size + self.offset
                    selected_group = bisect_right(self.source.starts, row) - 1
                    if selected_group != group_id:
                        selected_file, row_group, _ = self.source.groups[selected_group]
                        if selected_file != file_id:
                            if pf is not None:
                                pf.close()
                            pf = pq.ParquetFile(self.source.files[selected_file])
                            file_id = selected_file
                        column = pf.read_row_group(row_group, columns=['input_ids'], use_threads=False).column(0).combine_chunks()
                        values = column.flatten().to_numpy(zero_copy_only=False).reshape(-1, self.source.seq_len)
                        group_id = selected_group
                    tensor = torch.tensor(values[row-self.source.starts[group_id]].astype(np.int64, copy=False))
                    self.offset += 1
                    if self.offset == self.batch_size:
                        self.next_batch += geometry[1]
                        self.offset = 0
                        yield {'input_ids': tensor}
                        break
                    yield {'input_ids': tensor}
        finally:
            if pf is not None:
                pf.close()

    def state_dict(self):
        return dict(fingerprint=self.source.fingerprint, rank=self.rank, world_size=self.world_size,
                    batch_size=self.batch_size, next_batch=self.next_batch, offset=self.offset,
                    worker_geometry=self.worker_geometry)

    def load_state_dict(self, state):
        if 'fingerprint' not in state:
            raise ValueError('Checkpoint is not from the ordered Parquet loader')
        for key in ('rank', 'world_size', 'batch_size'):
            if state[key] != getattr(self, key):
                raise ValueError(f'Ordered data resume changed {key}')
        if state['fingerprint'] != self.source.fingerprint:
            raise ValueError('Ordered data resume changed the dataset')
        self.next_batch, self.offset = state['next_batch'], state['offset']
        # StatefulDataLoader's incremental snapshots can serialize tuples as lists.
        geometry = state['worker_geometry']
        self.worker_geometry = tuple(geometry) if geometry is not None else None
