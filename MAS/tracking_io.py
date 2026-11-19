"""Normalize DLC CSVs by body-part name, independent of exported column order."""
import csv
import pandas as pd

DAM = ('Nose', 'Ear_left', 'Ear_right', 'Center', 'Spine1', 'Spine2', 'Tail_base')
PUP = ('PupNose', 'PEar_left', 'pEar_right', 'PupNeck', 'PupCenter', 'PupTail')


def read_tracking_csv(filename):
    with open(filename, newline='', encoding='utf-8-sig') as stream:
        first = [next(csv.reader(stream), []) for _ in range(4)]
    depth = 4 if first[1] and first[1][0] == 'individuals' else 3
    if not first[0] or first[0][0] != 'scorer' or first[depth-1][0] != 'coords':
        raise ValueError(f'Unrecognized DLC CSV header: {filename}')
    data = pd.read_csv(filename, header=list(range(depth)), index_col=0)
    expected_index = list(range(len(data)))
    if data.index.tolist() != expected_index:
        raise ValueError('DLC frame indices must be consecutive from zero; missing frames cannot be silently renumbered.')
    columns = []
    for part in DAM + PUP:
        for coordinate in ('x', 'y', 'likelihood'):
            candidates = [i for i, col in enumerate(data.columns)
                          if col[-2:] == (part, coordinate)]
            if len(candidates) != 1:
                raise ValueError(f'Expected exactly one {part}/{coordinate}; found {len(candidates)}. Check animal identities and model layout.')
            columns.append(candidates[0])
    normalized = data.iloc[:, columns].apply(pd.to_numeric, errors='raise').copy()
    normalized.columns = range(1, 40)
    normalized.insert(0, 0, expected_index)
    return normalized.reset_index(drop=True)
