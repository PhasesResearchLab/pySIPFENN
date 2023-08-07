# Authors: Jonathan Siegel, Adam M. Krajewski

import math
import numpy as np
import os
from pymatgen.core import Structure, Element, Composition
from pymatgen.analysis.local_env import VoronoiNN
import json
from collections import Counter
from typing import List, Union, Tuple
import random
from importlib import resources
import pandas as pd

citations = [
    'Adam M. Krajewski, Jonathan W. Siegel, Jinchao Xu, Zi-Kui Liu, Extensible Structure-Informed Prediction of '
    'Formation Energy with improved accuracy and usability employing neural networks, Computational '
    'Materials Science, Volume 208, 2022, 111254'
]

periodic_table_size = 112
attribute_matrix = np.loadtxt(os.path.join(os.path.dirname(__file__), 'Magpie_element_properties.csv'), delimiter=',')
attribute_matrix = np.nan_to_num(attribute_matrix)
# Only select attributes actually used in Magpie.
attribute_matrix = attribute_matrix[:,
                   [45, 33, 2, 32, 5, 48, 6, 10, 44, 42, 38, 40, 36, 43, 41, 37, 39, 35, 18, 13, 17]]

maxFeaturesInOQMD = np.array((13.1239, 5.01819, 12.0, 35.9918, 0.305284, 1.0, 1.89776, 0.61604, 0.251582, 0.505835,
                              0.671142, 0.648262, 0.74048, 89.0, 26.4054, 89.0, 93.0, 90.2828, 95.0, 30.0847, 95.0,
                              95.0, 90.6205, 233.189, 68.4697, 233.189, 242.992, 235.263, 3631.95, 1395.25, 3631.95,
                              3808.99, 3785.18, 16.0, 5.77718, 16.0, 16.0, 16.0, 5.99048, 1.66782, 5.99048, 6.0, 5.8207,
                              213.0, 56.9856, 213.0, 213.0, 194.519, 3.19, 0.971532, 3.19, 3.19, 3.01435, 2.0, 0.749908,
                              2.0, 2.0, 2.0, 5.0, 1.80537, 5.0, 5.25839, 5.0, 10.0, 3.85266, 10.0, 10.0, 10.0, 14.0,
                              5.61254, 14.0, 14.0, 14.0, 26.0, 8.0159, 26.0, 28.0, 27.0, 1.0, 0.406792, 1.0, 1.0, 1.0,
                              5.0, 1.56393, 5.0, 5.0, 5.0, 9.0, 3.49493, 9.0, 9.0, 9.0, 13.0, 4.6277, 13.0, 13.0, 13.0,
                              20.0, 6.30236, 20.0, 22.0, 22.0, 109.15, 36.0677, 109.15, 110.125, 108.956, 7.853,
                              2.78409, 7.853, 7.853, 7.853, 2.09127, 0.82173, 2.09127, 2.11066, 2.11066, 7.0, 1.0, 1.0,
                              1.0, 1.0, 1.0, 94.0, 93.0, 46.5, 94.0, 94.0, 94.0, 102.0, 97.0, 47.5, 102.0, 102.0, 102.0,
                              244.0, 242.992, 121.496, 244.0, 244.0, 244.0, 3823.0, 3808.99, 1904.5, 3823.0, 3823.0,
                              3823.0, 18.0, 17.0, 8.0, 18.0, 18.0, 18.0, 7.0, 6.0, 3.0, 7.0, 7.0, 7.0, 244.0, 213.0,
                              106.5, 244.0, 244.0, 244.0, 3.98, 3.19, 1.595, 3.98, 3.98, 3.98, 2.0, 2.0, 1.0, 2.0, 2.0,
                              2.0, 6.0, 6.0, 2.5, 6.0, 6.0, 6.0, 10.0, 10.0, 5.0, 10.0, 10.0, 10.0, 14.0, 14.0, 7.0,
                              14.0, 14.0, 14.0, 29.0, 28.0, 14.0, 29.0, 29.0, 29.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 5.0,
                              5.0, 2.5, 5.0, 5.0, 5.0, 9.0, 9.0, 4.5, 9.0, 9.0, 9.0, 13.0, 13.0, 6.5, 13.0, 13.0, 13.0,
                              22.0, 22.0, 11.0, 22.0, 22.0, 22.0, 115.765, 110.125, 55.0625, 115.765, 115.765, 115.765,
                              7.853, 7.853, 3.9265, 7.853, 7.853, 7.853, 2.11066, 2.11066, 1.05533, 2.11066, 2.11066,
                              2.11066, 1.0, 0.714286, 1.0, 0.875, 0.92145, 0.460725))


def local_env_function(local_env, site):
    """A prototype function which computes a weighted average over neighbors, weighted by the area of the voronoi cell
        between them.

        Args:
            local_env: A dictionary of the local environment of a site, as returned by a VoronoiNN generator.
            site: The site number for which the local environment is being computed.
            element_dict: A dictionary of the elements in the structure.

        Returns:
            A list of the local environment attributes.
    """
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


class LocalAttributeGenerator:
    """A wrapper class which contains an instance of an NN generator (the default is a VoronoiNN), a structure, and
    a function which computes the local environment attributes.
    """

    def __init__(self, struct, local_env_func,
                 nn_generator=VoronoiNN(compute_adj_neighbors=False, extra_nn_info=False)):
        self.generator = nn_generator
        self.struct = struct
        self.function = local_env_func

    def generate_local_attributes(self, n):
        local_env = self.generator.get_voronoi_polyhedra(self.struct, n)
        return self.function(local_env, self.struct[n])


def generate_voronoi_attributes(struct, local_funct=local_env_function):
    """Generates the local environment attributes for a given structure using a VoronoiNN generator.

        Args:
            struct: A pymatgen Structure object.
            local_funct: A function which computes the local environment attributes for a given site.
    """
    local_generator = LocalAttributeGenerator(struct, local_funct)
    attribute_list = list(map(local_generator.generate_local_attributes, range(len(struct.sites))))
    return np.array([value[0] for value in attribute_list]), np.array([value[1] for value in attribute_list])


def magpie_mode(attribute_properties, axis=0):
    """Calculates the attributes corresponding to the most common elements."""
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


def generate_descriptor(struct: Structure,
                        comp: Composition,
                        minimumSitesPerExpansion: int = 50,
                        featureConvergenceCriterion: float = 0.01,
                        compositionConvergenceCriterion: float = 0.01,
                        minimumElementOccurances: int = 10,
                        plotParameters: bool = False,
                        printProgress: bool = True,
                        returnMeta: bool = False,
                        ) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
    """Main functionality. Generates the KS2022 descriptor for a given composition randomly distributed on a given
    structure until the convergence criteria are met. The descriptor is KS2022 which is compatible with all KS2022
    models and approaches values that would be reached by infinite supercell size.
    
    Args:
        struct: A pymatgen Structure object that will be used as the basis for the structure to be generated. It can
            be occupied by any species without affecting the result since all will be replaced by the composition.
        comp: A pymatgen Composition object that will be randomly distributed on the structure within accuracy
            determined by the compositionConvergenceCriterion.
        minimumSitesPerExpansion: The minimum number of sites that the base structure will be expanded to (doubling dimension-by-dimension) before it will
            be used as expansion step in each iteration adding local chemical environment information to the global pool.
            Optimal value will depend on the number of species and their relative fractions in the composition.
            Generally, low values will result in slower convergence (<20ish) and too high values (>150ish) will result 
            in slower computation. The default value is 50.
        featureConvergenceCriterion: The maximum difference between any feature belonging to the current iteration (statistics based on the
            global ensemble of local chemical environments) and the previous iteration (before last expansion) 
            expressed as a fraction of the maximum value of each feature found in the OQMD database at the time of 
            SIPFENN publication (see maxFeaturesInOQMD array). The default value is 0.01, corresponding to 1% of the 
            maximum value.
        compositionConvergenceCriterion: The maximum average difference between any element fraction belonging in the current
            composition (all expansions) and the the target composition (comp). The default value is 0.01, corresponding
            to deviation depending on the number of elements in the composition.
        minimumElementOccurances: The minimum number of times all elements must occur in the composition before it is
            considered converged. This is to prevent the algorithm from converging before very dilute elements have
            had a chance to occur. The default value is 10.
        plotParameters: If True, the convergence history will be plotted using plotly. The default value is False.
        printProgress: If True, the progress will be printed to the console. The default value is True.

    Returns:
        A numpy array containing the KS2022 descriptor. Please note the stochastic nature of the algorithm and that
        the result may vary slightly between runs and parameters.
    """

    # Obtain the elemental frequencies
    elementalFrequencies = dict(comp.fractional_composition.get_el_amt_dict())

    # If the number of sites is lower than 50, keep doubling up the structure, iterating over one of the 3 dimensions
    # at a time until there are at least 50 sites.
    adjustedStruct = struct.copy()
    i = 0
    while len(adjustedStruct.sites) < minimumSitesPerExpansion:
        scaling = [1, 1, 1]
        scaling[i] = 2
        adjustedStruct.make_supercell(scaling)
        i = (i + 1) % 3

    diff_properties = np.ndarray(shape=(0, 26))
    attribute_properties = np.ndarray(shape=(0, 21))
    propHistory = []
    diffHistory = []
    allOccupations = []
    maxDiff = 1
    compositionDistance = 0
    minOccupationCount = 0
    properties = None

    if printProgress:
        print(f'#Atoms | Comp. Distance AVG | Convergence Crit. MAX | Occupation Count MIN')

    while maxDiff > featureConvergenceCriterion \
            or compositionDistance > compositionConvergenceCriterion \
            or minOccupationCount < minimumElementOccurances:
        # Choose random structure occupation
        randomOccupation = random.choices(list(elementalFrequencies.keys()),
                                          weights=elementalFrequencies.values(),
                                          k=adjustedStruct.num_sites)
        allOccupations += randomOccupation
        occupationCount = dict(Counter(allOccupations))
        minOccupationCount = min(occupationCount.values())
        currentComposition = Composition.from_dict(occupationCount)
        # Adjust current elemental frequencies to push the current composition towards the target composition

        compositionDistance = 0
        for element in elementalFrequencies.keys():
            difference = currentComposition.fractional_composition.get_atomic_fraction(element) \
                         - comp.fractional_composition.get_atomic_fraction(element)
            compositionDistance += abs(difference)
            elementalFrequencies[element] -= difference * 0.1
        compositionDistance /= len(elementalFrequencies.keys())

        for site, occupation in zip(adjustedStruct.sites, randomOccupation):
            site.species = occupation

        diff_properties_instance, attribute_properties_instance = generate_voronoi_attributes(struct=adjustedStruct)

        diff_properties = np.concatenate((diff_properties, diff_properties_instance), axis=0)
        attribute_properties = np.concatenate((attribute_properties, attribute_properties_instance), axis=0)

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
        properties[12] *= adjustedStruct.num_sites / adjustedStruct.volume
        # Calculate and insert stoichiometry attributes.
        element_dict = currentComposition.fractional_composition.as_dict()
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

        propHistory.append(properties)
        # Calculate the difference between the current step and the previous step and divide it by maximum value of
        # each feature found in OQMD to normalize the difference.
        if len(propHistory) > 1:
            diff = np.subtract(properties, propHistory[-2])
            diff /= maxFeaturesInOQMD
            diffHistory.append(diff)
            maxDiff = np.max(np.abs(diff))
            if printProgress:
                print(f'{attribute_properties.shape[0]:^6} | '
                      f'{compositionDistance: 18.6f} | '
                      f'{maxDiff: 21.6f} | '
                      f'{minOccupationCount:^4}')
        else:
            if printProgress:
                print(f'{attribute_properties.shape[0]:^6} | '
                      f'{compositionDistance: 18.6f} | '
                      f'{"(init)":^21} | '
                      f'{minOccupationCount:^4}')

    if returnMeta:
        metaData = {'diffHistory': diffHistory,
                    'propHistory': propHistory,
                    'finalAtomsN': attribute_properties.shape[0],
                    'finalCompositionDistance': compositionDistance
                    }

    if plotParameters:
        import plotly.express as px
        import pandas as pd
        import warnings
        warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

        diffArray = np.array(diffHistory)

        # Plot the parameters as lines. Add hover text to show the parameter name based on the labels_KS2022.csv file.
        with resources.files('pysipfenn').joinpath('descriptorDefinitions/labels_KS2022.csv').open() as f:
            labels = f.readlines()
        fig = px.line(pd.DataFrame(diffArray, columns=labels), title='KS2022 Descriptor Parameters',
                      range_y=[-0.5, 0.5])
        # Add horizontal lines at convergence criteria of +/- 0.01.
        fig.add_hline(y=0.01, line_dash='dash', line_color='red')
        fig.add_hline(y=-0.01, line_dash='dash', line_color='red')

        fig.show()

    # print(f'Target: {comp.fractional_composition}')
    # print(f'Final:  {currentComposition.fractional_composition}')
    if properties is not None:
        assert properties.shape == (256,)
        assert isinstance(properties, np.ndarray)
        if returnMeta:
            return properties, metaData
        else:
            return properties
    else:
        raise RuntimeError('KS2022_randomSolution descriptor failed to converge.')


def cite() -> List[str]:
    """Citation/s for the descriptor."""
    return citations


def onlyStructural(descriptor: np.ndarray) -> np.ndarray:
    """Returns the structure-dependent part of the KS2022descriptor.

    Args:
        descriptor: A 256-length numpy array of the KS2022 descriptor.

    Returns:
        A 103-length numpy array of the structure-dependent part of the KS2022 descriptor. Useful in cases where the
        descriptor is used as a fingerprint to compare polymorphs of the same compound.

    """
    assert isinstance(descriptor, np.ndarray)
    assert descriptor.shape == (256,)
    descriptorSplit = np.split(descriptor, [68, 73, 93, 98, 113])
    ks2022_structural = np.concatenate((
        descriptorSplit[0],
        descriptorSplit[2],
        descriptorSplit[4]
    ), axis=-1, dtype=np.float32)
    assert ks2022_structural.shape == (103,)

    return ks2022_structural


def profile(test: str = 'FCC',
            comp: Composition = Composition('Cr12.8 Fe12.8 Co12.8 Ni12.8 Cu12.8 Al35.9'),
            nIterations: int = 1,
            plotParameters: bool = False,
            returnDescriptor: bool = False):
    """Profiles the descriptor in series using one of the test structures."""
    if test == 'FCC':
        print(
            f'KS2022 Random Solid Solution profiling/testing task will calculate a descriptor for a random FCC alloy.')
        matStr = '{"@module": "pymatgen.core.structure", "@class": "Structure", "charge": 0, "lattice": {"matrix": [[3.475145865948011, 0.0, 2.1279131306516942e-16], [5.588460777961125e-16, 3.475145865948011, 2.1279131306516942e-16], [0.0, 0.0, 3.475145865948011]], "pbc": [true, true, true], "a": 3.475145865948011, "b": 3.475145865948011, "c": 3.475145865948011, "alpha": 90.0, "beta": 90.0, "gamma": 90.0, "volume": 41.968081364279875}, "sites": [{"species": [{"element": "Ni", "occu": 1}], "abc": [0.0, 0.0, 0.0], "xyz": [0.0, 0.0, 0.0], "properties": {}, "label": "Ni"}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.0, 0.5, 0.5], "xyz": [2.7942303889805623e-16, 1.7375729329740055, 1.7375729329740055], "properties": {}, "label": "Ni"}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.5, 0.0, 0.5], "xyz": [1.7375729329740055, 0.0, 1.7375729329740055], "properties": {}, "label": "Ni"}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.5, 0.5, 0.0], "xyz": [1.7375729329740057, 1.7375729329740055, 2.1279131306516942e-16], "properties": {}, "label": "Ni"}]}'
    elif test == 'BCC':
        print('KS2022 Random Solution profiling/testing taks will calculate the descriptor for a random BCC alloy.')
        matStr = '{"@module": "pymatgen.core.structure", "@class": "Structure", "charge": 0, "lattice": {"matrix": [[2.863035498949916, 0.0, 1.75310362981713e-16], [4.60411223268961e-16, 2.863035498949916, 1.75310362981713e-16], [0.0, 0.0, 2.863035498949916]], "pbc": [true, true, true], "a": 2.863035498949916, "b": 2.863035498949916, "c": 2.863035498949916, "alpha": 90.0, "beta": 90.0, "gamma": 90.0, "volume": 23.468222587900303}, "sites": [{"species": [{"element": "Fe", "occu": 1}], "abc": [0.0, 0.0, 0.0], "xyz": [0.0, 0.0, 0.0], "properties": {}, "label": "Fe"}, {"species": [{"element": "Fe", "occu": 1}], "abc": [0.5, 0.5, 0.5], "xyz": [1.4315177494749582, 1.431517749474958, 1.4315177494749582], "properties": {}, "label": "Fe"}]}'
    elif test == 'HCP':
        print('KS2022 Random Solution profiling/testing taks will calculate the descriptor for a random HCP alloy.')
        matStr = '{"@module": "pymatgen.core.structure", "@class": "Structure", "charge": 0, "lattice": {"matrix": [[1.4678659615336875, -2.54241842407729, 0.0], [1.4678659615336875, 2.54241842407729, 0.0], [0.0, 0.0, 4.64085615]], "pbc": [true, true, true], "a": 2.9357319230673746, "b": 2.9357319230673746, "c": 4.64085615, "alpha": 90.0, "beta": 90.0, "gamma": 120.00000000000001, "volume": 34.6386956150451}, "sites": [{"species": [{"element": "Ti", "occu": 1}], "abc": [0.3333333333333333, 0.6666666666666666, 0.25], "xyz": [1.4678659615336875, 0.8474728080257632, 1.1602140375], "properties": {}, "label": "Ti"}, {"species": [{"element": "Ti", "occu": 1}], "abc": [0.6666666666666667, 0.33333333333333337, 0.75], "xyz": [1.4678659615336878, -0.8474728080257634, 3.4806421125], "properties": {}, "label": "Ti"}]}'
    else:
        print('Unrecognized test name.')
        return None

    if nIterations == 1:
        s = Structure.from_dict(json.loads(matStr))
        d = generate_descriptor(s, comp, plotParameters=plotParameters)
    elif nIterations > 1:
        print(f'Running {nIterations} iterations in parallel...')
        s = Structure.from_dict(json.loads(matStr))
        from tqdm.contrib.concurrent import process_map
        d = process_map(generate_descriptor,
                        [s for i in range(nIterations)],
                        [comp for i in range(nIterations)],
                        chunksize=1,
                        max_workers=8)
    else:
        d = None

    if d is None:
        print('No descriptors generated.')
        return None
    else:
        name = f'TestResult_KS2022_randomSolution_{test}_{nIterations}iter.csv'
        if nIterations == 1:
            with open(name, 'w+') as f:
                f.writelines([f'{v}\n' for v in d])
            if returnDescriptor:
                return d
        else:
            with open(name, 'w+') as f:
                f.writelines([f'{",".join([str(v) for v in di])}\n' for di in d])
    print('Done!')

if __name__ == "__main__":
    profile(test='FCC', plotParameters=True)
    profile(test='BCC', plotParameters=True)
    profile(test='HCP', plotParameters=True)
    profile(test='BCC', nIterations=6)
