# Authors: Jonathan Siegel, Adam M. Krajewski
#
# Calculates the descriptor first introduced by Ward and Wolverton.
#
# Please Cite:
# L. Ward, R. Liu, A. Krishna, V. I. Hegde, A. Agrawal, A. Choudhary, and C. Wolverton,
# “Including crystal structure attributes in machine learning models of formation energies
# via Voronoi tessellations,” Physical Review B, vol. 96, no. 2, 7 2017.

import math
import time
import numpy as np
import os
from pymatgen.core import Structure, Element
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import json
from tqdm import tqdm
from collections import Counter

citation = 'L. Ward, R. Liu, A. Krishna, V. I. Hegde, A. Agrawal, A. Choudhary, and C. Wolverton, “Including crystal structure attributes in machine learning models of formation energies via Voronoi tessellations,” Physical Review B, vol. 96, no. 2, 7 2017.'

periodic_table_size = 112
attribute_matrix = np.loadtxt(os.path.join(os.path.dirname(__file__), 'Magpie_element_properties.csv'), delimiter=',')
attribute_matrix = np.nan_to_num(attribute_matrix)
# Only select attributes actually used in Magpie.
attribute_matrix = attribute_matrix[:,
                   [45, 33, 2, 32, 5, 48, 6, 10, 44, 42, 38, 40, 36, 43, 41, 37, 39, 35, 18, 13, 17]]

# A prototype function which computes a weighted average over neighbors,
# weighted by the area of the voronoi cell between them.
def local_env_function(local_env, site, struct):
    local_attributes = np.zeros(attribute_matrix.shape[1])
    for key, value in site.species.get_el_amt_dict().items():
        local_attributes += value * attribute_matrix[Element(key).Z - 1, :]
    diff_attributes = np.zeros(attribute_matrix.shape[1])
    total_weight = 0
    volume = 0
    for ind, neighbor_site in local_env.items():
        neighbor_attributes = np.zeros(attribute_matrix.shape[1])
        for key, value in neighbor_site['site'].species.get_el_amt_dict().items():
            neighbor_attributes += value * attribute_matrix[Element(key).Z - 1, :]
        diff_attributes += np.abs(local_attributes - neighbor_attributes) * neighbor_site['area']
        total_weight += neighbor_site['area']
        volume += neighbor_site['volume']
    elemental_properties_attributes = [diff_attributes / total_weight, local_attributes]
    # Calculate coordination number attribute
    average = 0
    variance = 0
    for neighbor_site in local_env.values():
        average += neighbor_site['area']
        variance += neighbor_site['area'] * neighbor_site['area']
    eff_coord_num = average * average / variance
    # Calculate Bond Length Attributes
    # AVG
    blen_average = 0
    for neighbor_site in local_env.values():
        blen_average += neighbor_site['area'] * 2 * neighbor_site['face_dist']
    blen_average /= total_weight
    # VAR
    blen_var = 0
    for neighbor_site in local_env.values():
        blen_var += neighbor_site['area'] * abs(2 * neighbor_site['face_dist'] - blen_average)
    blen_var /= total_weight * blen_average
    # Calculate Packing Efficiency info
    sphere_rad = min(neighbor_site['face_dist'] for neighbor_site in local_env.values())
    sphere_volume = (4.0 / 3.0) * math.pi * math.pow(sphere_rad, 3.0)
    return [np.concatenate(
        ([eff_coord_num, blen_average, blen_var, volume, sphere_volume], elemental_properties_attributes[0])),
            elemental_properties_attributes[1]]

def findDilute(struct):
    spoList = struct.species_and_occu
    spCount = dict(Counter(spoList))
    spDilute = [spoList.index(sp) for sp in spCount if spCount[sp]==1]
    if len(spCount)-len(spDilute)==1:
        return spDilute[0]
    else:
        print('Custom dilute structure descriptor calculation is defined only one dilute species in a single element matrix')
        raise RuntimeError

def generate_voronoi_attributes(struct, baseStruct='pure', local_funct=local_env_function):
    local_generator = LocalAttributeGenerator(struct, local_funct)

    # Generate a base structure of pure elemental solid or take one as input
    if isinstance(baseStruct, Structure):
        diluteSite = []
        for i, (s1s, s2s) in enumerate(zip(baseStruct.sites, struct.sites)):
            if s1s != s2s:
                diluteSite.append(i)
                continue
        if len(diluteSite)==1:
            diluteSite = diluteSite[0]
        else:
            print('Sites in the provided base structure matched the investigated one exactly')
            raise TypeError
    elif baseStruct=='pure':
        baseStruct = struct.copy()
        for sp in set(baseStruct.species):
            baseStruct.replace_species({sp: 'A'})
        # Find position of the 1 dilute atom and calculate output for it
        diluteSite = findDilute(struct)
    else:
        raise TypeError

    # Find equivalent positions in the original base structure
    spgAbase = SpacegroupAnalyzer(baseStruct, symprec=0.001, angle_tolerance=0.1)
    originalEquivalents = list(spgAbase.get_symmetry_dataset()['equivalent_atoms'])

    # Output list
    attribute_list = list()
    attribute_list.append(local_generator.generate_local_attributes_diluteSite(diluteSite))

    # Based on the dilute atom output, identify its neighbors
    neighborsFacesDict = attribute_list[0][2]

    # Create a dictionary of LCE parameters to determine equivalency in a dilute case
    siteLCEparams = dict(zip(range(len(originalEquivalents)), [[e] for e in originalEquivalents]))
    siteLCEparams[diluteSite] = 'dilute'
    for siteN in neighborsFacesDict:
        siteLCEparams[siteN].append(neighborsFacesDict[siteN])

    # Group into equivalents and remove the dilute atom, already calcualted
    equivalentGroups = {}
    for siteN in siteLCEparams:
        params = ''.join(str(siteLCEparams[siteN]))
        if params in equivalentGroups:
            equivalentGroups[params].append(siteN)
        else:
            equivalentGroups.update({params: [siteN]})
    del equivalentGroups['dilute']

    equivalentSitesMultiplicities = dict(
        zip([g[0] for g in equivalentGroups.values()],
            [len(g) for g in equivalentGroups.values()]))

    for siteN in equivalentSitesMultiplicities:
        localAttributes = [local_generator.generate_local_attributes(siteN)]
        attribute_list += localAttributes*equivalentSitesMultiplicities[siteN]

    return np.array([value[0] for value in attribute_list]), np.array([value[1] for value in attribute_list])

# A wrapper class which contains an instance of an NN generator (the default is a VoronoiNN), a structure, and
# a function which computes the local environment attributes.
class LocalAttributeGenerator:
    def __init__(self, struct, local_env_func, nn_generator=VoronoiNN(compute_adj_neighbors=False, extra_nn_info=False)):
        self.generator = nn_generator
        self.struct = struct
        self.function = local_env_func

    def generate_local_attributes(self, n):
        local_env = self.generator.get_voronoi_polyhedra(self.struct, n)
        return self.function(local_env, self.struct[n], self.struct)

    def generate_local_attributes_diluteSite(self, n):
        local_env = self.generator.get_voronoi_polyhedra(self.struct, n)
        local_env_result = self.function(local_env, self.struct[n], self.struct)

        neighbor_dict = {value['site'].index:
                             [str(value['site'].species),
                              round(value['face_dist'], 2),
                              round(value['area'], 2),
                              value['n_verts']]
                         for value in local_env.values()}

        local_env_result.append(neighbor_dict)

        return local_env_result

# Calculates the attributes corresponding to the most common elements.
def magpie_mode(attribute_properties, axis=0):
    scores = np.unique(np.ravel(attribute_properties[:, 0]))  # get all unique atomic numbers
    max_occurrence = 0
    top_elements = []
    for score in scores:
        template = (attribute_properties[:, 0] == score)
        count = np.expand_dims(np.sum(template, axis), axis)[0]
        if count > max_occurrence:
            top_elements.clear()
            top_elements.append(score)
            max_occurrence = count
        elif count == max_occurrence:
            top_elements.append(score)
    output = np.zeros_like(attribute_properties[0, :])
    for elem in top_elements:
        output += attribute_matrix[int(elem) - 1, :]
    return output / len(top_elements)

def generate_descriptor(struct: Structure, baseStruct='pure'):
    diff_properties, attribute_properties = generate_voronoi_attributes(struct, baseStruct=baseStruct)
    properties = np.concatenate(
        (np.stack(
            (np.mean(diff_properties, axis=0),
             np.mean(np.abs(diff_properties - np.mean(diff_properties, axis=0)), axis=0),
             np.min(diff_properties, axis=0),
             np.max(diff_properties, axis=0),
             np.max(diff_properties, axis=0) - np.min(diff_properties, axis=0)), axis=-1).reshape((-1)),
        np.stack(
            (np.mean(attribute_properties, axis=0),
             np.max(attribute_properties, axis=0) - np.min(attribute_properties, axis=0),
             np.mean(np.abs(attribute_properties - np.mean(attribute_properties, axis=0)), axis=0),
             np.max(attribute_properties, axis=0),
             np.min(attribute_properties, axis=0),
             magpie_mode(attribute_properties)), axis=-1).reshape((-1))))
    # Normalize Bond Length properties.
    properties[6] /= properties[5]
    properties[7] /= properties[5]
    properties[8] /= properties[5]
    # Normalize the Cell Volume Deviation.
    properties[16] /= properties[15]
    # Remove properties not in magpie.
    properties = np.delete(properties, [4, 5, 9, 14, 15, 17, 18, 19, 21, 22, 23, 24])
    # Renormalize the packing efficiency.
    properties[12] *= len(attribute_properties) / struct.volume
    # Calculate and insert stoichiometry attributes.
    element_dict = {}
    for composition in struct.species_and_occu:
        for key, value in composition.get_el_amt_dict().items():
            if key in element_dict:
                element_dict[key] += value / len(struct.species_and_occu)
            else:
                element_dict[key] = value / len(struct.species_and_occu)
    position = 118
    for p in [10, 7, 5, 3, 2]:
        properties = np.insert(properties, position,
                               math.pow(sum(math.pow(value, p) for value in element_dict.values()), 1.0 / p))
    properties = np.insert(properties, position, len(element_dict))
    # Calculate Valence Electron Statistics
    electron_occupation_dict = {'s': 0, 'p': 0, 'd': 0, 'f': 0}
    total_valence_factor = 0
    for key, value in element_dict.items():
        electron_occupation_dict['s'] += value * attribute_matrix[Element(key).Z - 1][8]
        electron_occupation_dict['p'] += value * attribute_matrix[Element(key).Z - 1][9]
        electron_occupation_dict['d'] += value * attribute_matrix[Element(key).Z - 1][10]
        electron_occupation_dict['f'] += value * attribute_matrix[Element(key).Z - 1][11]
    total_valence_factor = sum([val for (key, val) in electron_occupation_dict.items()])
    for orb in ['s', 'p', 'd', 'f']:
        properties = np.append(properties, electron_occupation_dict[orb] / total_valence_factor)
    # Calculate ionic compound attributes.
    max_ionic_char = 0
    av_ionic_char = 0
    for key1, value1 in element_dict.items():
        for key2, value2 in element_dict.items():
            ionic_char = 1.0 - math.exp(-0.25 * (Element(key1).X - Element(key2).X) ** 2)
            if ionic_char > max_ionic_char:
                max_ionic_char = ionic_char
            av_ionic_char += ionic_char * value1 * value2
    properties = np.append(properties, max_ionic_char)
    properties = np.append(properties, av_ionic_char)
    properties = properties.astype(np.float32)
    return properties

def cite():
    return citation

def profile(test='JVASP-10001', nRuns=10):
    if test == 'diluteNiAlloy':
        print(f'KS2022 profiling/testing task will calculate a descriptor for a dilute Ni alloy {nRuns} times in series.')
        matStr = '{"@module": "pymatgen.core.structure", "@class": "Structure", "charge": null, "lattice": {"matrix": [[6.995692, 0.0, 0.0], [0.0, 6.995692, 0.0], [0.0, 0.0, 6.995692]], "a": 6.995692, "b": 6.995692, "c": 6.995692, "alpha": 90.0, "beta": 90.0, "gamma": 90.0, "volume": 342.36711365619243}, "sites": [{"species": [{"element": "Cr", "occu": 1}], "abc": [0.0, 0.0, 0.0], "xyz": [0.0, 0.0, 0.0], "label": "Cr", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.0, 0.0, 0.5], "xyz": [0.0, 0.0, 3.497846], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.0, 0.5, 0.0], "xyz": [0.0, 3.497846, 0.0], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.0, 0.5, 0.5], "xyz": [0.0, 3.497846, 3.497846], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.5, 0.0, 0.0], "xyz": [3.497846, 0.0, 0.0], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.5, 0.0, 0.5], "xyz": [3.497846, 0.0, 3.497846], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.5, 0.5, 0.0], "xyz": [3.497846, 3.497846, 0.0], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.5, 0.5, 0.5], "xyz": [3.497846, 3.497846, 3.497846], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.25, 0.25, 0.0], "xyz": [1.748923, 1.748923, 0.0], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.25, 0.25, 0.5], "xyz": [1.748923, 1.748923, 3.497846], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.25, 0.7500000000000001, 0.0], "xyz": [1.748923, 5.2467690000000005, 0.0], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.25, 0.7500000000000001, 0.5], "xyz": [1.748923, 5.2467690000000005, 3.497846], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.7500000000000001, 0.25, 0.0], "xyz": [5.2467690000000005, 1.748923, 0.0], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.7500000000000001, 0.25, 0.5], "xyz": [5.2467690000000005, 1.748923, 3.497846], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.7500000000000001, 0.7500000000000001, 0.0], "xyz": [5.2467690000000005, 5.2467690000000005, 0.0], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.7500000000000001, 0.7500000000000001, 0.5], "xyz": [5.2467690000000005, 5.2467690000000005, 3.497846], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.25, 0.0, 0.25], "xyz": [1.748923, 0.0, 1.748923], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.25, 0.0, 0.7500000000000001], "xyz": [1.748923, 0.0, 5.2467690000000005], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.25, 0.5, 0.25], "xyz": [1.748923, 3.497846, 1.748923], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.25, 0.5, 0.7500000000000001], "xyz": [1.748923, 3.497846, 5.2467690000000005], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.7500000000000001, 0.0, 0.25], "xyz": [5.2467690000000005, 0.0, 1.748923], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.7500000000000001, 0.0, 0.7500000000000001], "xyz": [5.2467690000000005, 0.0, 5.2467690000000005], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.7500000000000001, 0.5, 0.25], "xyz": [5.2467690000000005, 3.497846, 1.748923], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.7500000000000001, 0.5, 0.7500000000000001], "xyz": [5.2467690000000005, 3.497846, 5.2467690000000005], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.0, 0.25, 0.25], "xyz": [0.0, 1.748923, 1.748923], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.0, 0.25, 0.7500000000000001], "xyz": [0.0, 1.748923, 5.2467690000000005], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.0, 0.7500000000000001, 0.25], "xyz": [0.0, 5.2467690000000005, 1.748923], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.0, 0.7500000000000001, 0.7500000000000001], "xyz": [0.0, 5.2467690000000005, 5.2467690000000005], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.5, 0.25, 0.25], "xyz": [3.497846, 1.748923, 1.748923], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.5, 0.25, 0.7500000000000001], "xyz": [3.497846, 1.748923, 5.2467690000000005], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.5, 0.7500000000000001, 0.25], "xyz": [3.497846, 5.2467690000000005, 1.748923], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.5, 0.7500000000000001, 0.7500000000000001], "xyz": [3.497846, 5.2467690000000005, 5.2467690000000005], "label": "Ni", "properties": {}}], "@version": null}'
    else:
        print('Unrecognized test name.')
        return None
    sList = [Structure.from_dict(json.loads(matStr))] * nRuns
    for s in tqdm(sList):
        d = generate_descriptor(s)
    with open('KS2022_dilute_TestResult.csv', 'w+') as f:
        f.writelines([f'{v}\n' for v in d])
    print('Done!')

def profileParallel(test='JVASP-10001', nRuns=1000):
    from tqdm.contrib.concurrent import process_map
    if test == 'diluteNiAlloy':
        print(f'KS2022 profiling/testing task will calculate a descriptor for a dilute Ni alloy {nRuns} times in parallel with 8 workers.')
        matStr = '{"@module": "pymatgen.core.structure", "@class": "Structure", "charge": null, "lattice": {"matrix": [[6.995692, 0.0, 0.0], [0.0, 6.995692, 0.0], [0.0, 0.0, 6.995692]], "a": 6.995692, "b": 6.995692, "c": 6.995692, "alpha": 90.0, "beta": 90.0, "gamma": 90.0, "volume": 342.36711365619243}, "sites": [{"species": [{"element": "Cr", "occu": 1}], "abc": [0.0, 0.0, 0.0], "xyz": [0.0, 0.0, 0.0], "label": "Cr", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.0, 0.0, 0.5], "xyz": [0.0, 0.0, 3.497846], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.0, 0.5, 0.0], "xyz": [0.0, 3.497846, 0.0], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.0, 0.5, 0.5], "xyz": [0.0, 3.497846, 3.497846], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.5, 0.0, 0.0], "xyz": [3.497846, 0.0, 0.0], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.5, 0.0, 0.5], "xyz": [3.497846, 0.0, 3.497846], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.5, 0.5, 0.0], "xyz": [3.497846, 3.497846, 0.0], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.5, 0.5, 0.5], "xyz": [3.497846, 3.497846, 3.497846], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.25, 0.25, 0.0], "xyz": [1.748923, 1.748923, 0.0], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.25, 0.25, 0.5], "xyz": [1.748923, 1.748923, 3.497846], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.25, 0.7500000000000001, 0.0], "xyz": [1.748923, 5.2467690000000005, 0.0], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.25, 0.7500000000000001, 0.5], "xyz": [1.748923, 5.2467690000000005, 3.497846], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.7500000000000001, 0.25, 0.0], "xyz": [5.2467690000000005, 1.748923, 0.0], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.7500000000000001, 0.25, 0.5], "xyz": [5.2467690000000005, 1.748923, 3.497846], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.7500000000000001, 0.7500000000000001, 0.0], "xyz": [5.2467690000000005, 5.2467690000000005, 0.0], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.7500000000000001, 0.7500000000000001, 0.5], "xyz": [5.2467690000000005, 5.2467690000000005, 3.497846], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.25, 0.0, 0.25], "xyz": [1.748923, 0.0, 1.748923], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.25, 0.0, 0.7500000000000001], "xyz": [1.748923, 0.0, 5.2467690000000005], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.25, 0.5, 0.25], "xyz": [1.748923, 3.497846, 1.748923], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.25, 0.5, 0.7500000000000001], "xyz": [1.748923, 3.497846, 5.2467690000000005], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.7500000000000001, 0.0, 0.25], "xyz": [5.2467690000000005, 0.0, 1.748923], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.7500000000000001, 0.0, 0.7500000000000001], "xyz": [5.2467690000000005, 0.0, 5.2467690000000005], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.7500000000000001, 0.5, 0.25], "xyz": [5.2467690000000005, 3.497846, 1.748923], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.7500000000000001, 0.5, 0.7500000000000001], "xyz": [5.2467690000000005, 3.497846, 5.2467690000000005], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.0, 0.25, 0.25], "xyz": [0.0, 1.748923, 1.748923], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.0, 0.25, 0.7500000000000001], "xyz": [0.0, 1.748923, 5.2467690000000005], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.0, 0.7500000000000001, 0.25], "xyz": [0.0, 5.2467690000000005, 1.748923], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.0, 0.7500000000000001, 0.7500000000000001], "xyz": [0.0, 5.2467690000000005, 5.2467690000000005], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.5, 0.25, 0.25], "xyz": [3.497846, 1.748923, 1.748923], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.5, 0.25, 0.7500000000000001], "xyz": [3.497846, 1.748923, 5.2467690000000005], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.5, 0.7500000000000001, 0.25], "xyz": [3.497846, 5.2467690000000005, 1.748923], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.5, 0.7500000000000001, 0.7500000000000001], "xyz": [3.497846, 5.2467690000000005, 5.2467690000000005], "label": "Ni", "properties": {}}], "@version": null}'
    else:
        print('Unrecognized test name.')
        return None
    s = Structure.from_dict(json.loads(matStr))
    #s.make_supercell(scaling_matrix=[2,2,2])
    sList = [s] * nRuns
    descList = process_map(generate_descriptor, sList, max_workers=8)
    print('Done!')

if __name__ == "__main__":
    profile(test='diluteNiAlloy')
    profileParallel(test='diluteNiAlloy')