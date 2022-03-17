import json
import os
from collections import Counter

import numpy as np
import pandas as pd
from pqdm.processes import pqdm
from pymatgen.core import Structure, PeriodicSite
from pymatgen.util.coord import pbc_shortest_vectors
from sklearn.ensemble import RandomForestRegressor


def read_pymatgen_dict(file):
    with open(file, "r") as f:
        d = json.load(f)
    return Structure.from_dict(d)


structure_paths_private = os.listdir('data/dichalcogenides_private/structures/')
structures_private = [read_pymatgen_dict(f"data/dichalcogenides_private/structures/{p}") for p in
                      structure_paths_private]

structure_paths = os.listdir('data/dichalcogenides_public/structures/')
structures_public = [read_pymatgen_dict(f"data/dichalcogenides_public/structures/{p}") for p in structure_paths]

structure_sites = []

for s in structures_public[:20]:
    structure_sites += s.sites

cnt = Counter(structure_sites)

reference_sites = [k for k, v in cnt.items() if v > 15]

reference_structure = Structure.from_sites(reference_sites)
reference_structure_sites_set = set(reference_structure.sites)


def get_difference_from_reference(s):
    diff = []
    for site in reference_sites:
        neighbour_sites = s.get_sites_in_sphere(site.coords, 0.05)
        if len(neighbour_sites) != 1 or site != neighbour_sites[0]:
            diff.append(site)
    diff_sites_new = []
    diff = sorted(diff, key=lambda x: x.species_string)
    for site in diff:
        neighbour_sites = s.get_sites_in_sphere(site.coords, 0.05)
        if len(neighbour_sites) == 1:
            diff_sites_new += neighbour_sites
        elif len(neighbour_sites) == 0:
            if site.species_string == 'Mo':
                diff_sites_new.append(PeriodicSite('H', site.frac_coords, site.lattice))
            elif site.species_string == 'S':
                diff_sites_new.append(PeriodicSite('He', site.frac_coords, site.lattice))
        else:
            raise ValueError('Incorrect number of sites found')

    return Structure.from_sites(diff_sites_new)


def get_distance_features(s1, s2):
    lattice = s1.lattice
    x1, x2, x3 = pbc_shortest_vectors(lattice, s1.frac_coords, s2.frac_coords)[0, 0]

    return np.array(
        [np.abs(x1), np.abs(x2), np.abs(x3), np.sqrt(x1 ** 2 + x2 ** 2), np.sqrt(x1 ** 2 + x2 ** 2 + x3 ** 2)])


def compute_features_per_structure(s):
    features = []
    sites = s.sites
    for i in range(len(sites)):
        features.append(get_distance_features(sites[i], sites[i - 1]))
    return np.concatenate(features)


df = pd.DataFrame({'id': structure_paths, 's': structures_public})
df['id'] = df['id'].str.split('.').str[0]

target = pd.read_csv('data/dichalcogenides_public/targets.csv', names=['id', 'band_gap'], skiprows=1)
df = df.merge(target, on='id')
df = df.reset_index()

df['composition'] = df['s'].apply(lambda x: str(x.composition))
df['composition'].value_counts()

df_private = pd.DataFrame({'id': structure_paths_private, 's': structures_private})
df_private['id'] = df_private['id'].str.split('.').str[0]
df_private['composition'] = df_private['s'].apply(lambda x: str(x.composition))

diffs_from_ref = pqdm(df_private['s'], get_difference_from_reference, n_jobs=20)
df_private['diff'] = diffs_from_ref

diffs_from_ref = pqdm(df['s'], get_difference_from_reference, n_jobs=20)
df['diff'] = diffs_from_ref

features_public = pqdm(df['diff'], compute_features_per_structure, n_jobs=20)
df['features'] = features_public

features_private = pqdm(df_private['diff'], compute_features_per_structure, n_jobs=20)
df_private['features'] = features_private

model_dict = {}

for comp in df['composition'].unique():
    print(f"Processing {comp}:")
    X, y = df[df['composition'] == comp][['features', 'band_gap']].values.T
    X = np.array(list(X))

    rf = RandomForestRegressor().fit(X, y)

    model_dict[comp] = rf


df_private['predictions'] = df_private.apply(
    lambda x: model_dict[x['composition']].predict(x['features'].reshape(1, -1))[0] if x['composition'] in model_dict.keys() else 1, axis=1)

df_private['band_gap'] = df_private['predictions']

submission = df_private[['id', 'predictions']]

submission.to_csv('submission.csv', index=False)
