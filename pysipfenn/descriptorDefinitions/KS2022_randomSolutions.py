# This file is part of pySIPFENN and is licensed under the terms of the LGPLv3 or later.
# Copyright (C) 2023 Adam M. Krajewski, Jonathan Siegel

"""This ``KS2022`` feature vector calculator is a **special-input converter** modification of our the base ``KS2022``. Unlike the
``KS2022_dilute`` which reduces the computation, this one is designed to take an anonymous ``Structure`` and ``Composition`` pair and 
obtain vales of the ``KS2022`` features at infinite random supercell size.

It does that by expanding the ensamble of local chemical environments by iteratively adding supercells of the structure until the
features and composition converge. If you use this code, plese cite (as in ``KS2022.cite()``):

- Adam M. Krajewski, Jonathan W. Siegel, Jinchao Xu, Zi-Kui Liu, "Extensible Structure-Informed Prediction of Formation Energy with 
  improved accuracy and usability employing neural networks", Computational Materials Science, Volume 208, 2022, 111254

The core purpose of this module is to calculate numpy ``ndarray`` with ``256`` features constructed by considering all local chemical 
environments existing in an atomic structure. Their list is available in the ``labels_KS2022.csv`` and will be discussed in our upcoming
publication (Spring 2024).
"""

# pySIPFENN (for handling prototype library with high-level API at lower-level here)
import pysipfenn

# Standard Library Imports
import random
import math
import time
from collections import Counter
from typing import List, Union, Tuple
from importlib import resources

# Third Party Dependencies
from tqdm.contrib.concurrent import process_map
import numpy as np
from pymatgen.core import Structure, Element, Composition, PeriodicSite
from pymatgen.analysis.local_env import VoronoiNN

# Certain hard-coded basic elemental properties used in the featurization (attribute_matrix is compatible with Magpie references,
# and maxFeaturesInOQMD is based on the 2017 snapshot of OQMD, which was current when we started and will be retained in KS2022, but
# change in the future).
periodic_table_size = 112
f = resources.files('pysipfenn.descriptorDefinitions').joinpath("element_properties_Ward2017KS2022.csv")
attribute_matrix = np.loadtxt(f, delimiter=',')
attribute_matrix = np.nan_to_num(attribute_matrix)
attribute_matrix = attribute_matrix[:,[45, 33, 2, 32, 5, 48, 6, 10, 44, 42, 38, 40, 36, 43, 41, 37, 39, 35, 18, 13, 17]]
maxFeaturesInOQMD = np.array([
    13.1239, 5.01819, 12.0, 35.9918, 0.305284, 1.0, 1.89776, 0.61604, 0.251582, 0.505835, 0.671142, 0.648262, 0.74048, 
    89.0, 26.4054, 89.0, 93.0, 90.2828, 95.0, 30.0847, 95.0, 95.0, 90.6205, 233.189, 68.4697, 233.189, 242.992, 235.263, 
    3631.95, 1395.25, 3631.95, 3808.99, 3785.18, 16.0, 5.77718, 16.0, 16.0, 16.0, 5.99048, 1.66782, 5.99048, 6.0, 5.8207,
    213.0, 56.9856, 213.0, 213.0, 194.519, 3.19, 0.971532, 3.19, 3.19, 3.01435, 2.0, 0.749908, 2.0, 2.0, 2.0, 5.0, 1.80537, 
    5.0, 5.25839, 5.0, 10.0, 3.85266, 10.0, 10.0, 10.0, 14.0, 5.61254, 14.0, 14.0, 14.0, 26.0, 8.0159, 26.0, 28.0, 27.0, 
    1.0, 0.406792, 1.0, 1.0, 1.0, 5.0, 1.56393, 5.0, 5.0, 5.0, 9.0, 3.49493, 9.0, 9.0, 9.0, 13.0, 4.6277, 13.0, 13.0, 13.0,
    20.0, 6.30236, 20.0, 22.0, 22.0, 109.15, 36.0677, 109.15, 110.125, 108.956, 7.853, 2.78409, 7.853, 7.853, 7.853, 2.09127, 
    0.82173, 2.09127, 2.11066, 2.11066, 7.0, 1.0, 1.0, 1.0, 1.0, 1.0, 94.0, 93.0, 46.5, 94.0, 94.0, 94.0, 102.0, 97.0, 47.5, 
    102.0, 102.0, 102.0, 244.0, 242.992, 121.496, 244.0, 244.0, 244.0, 3823.0, 3808.99, 1904.5, 3823.0, 3823.0, 3823.0, 18.0, 
    17.0, 8.0, 18.0, 18.0, 18.0, 7.0, 6.0, 3.0, 7.0, 7.0, 7.0, 244.0, 213.0, 106.5, 244.0, 244.0, 244.0, 3.98, 3.19, 1.595, 
    3.98, 3.98, 3.98, 2.0, 2.0, 1.0, 2.0, 2.0, 2.0, 6.0, 6.0, 2.5, 6.0, 6.0, 6.0, 10.0, 10.0, 5.0, 10.0, 10.0, 10.0, 14.0, 
    14.0, 7.0, 14.0, 14.0, 14.0, 29.0, 28.0, 14.0, 29.0, 29.0, 29.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 5.0, 5.0, 2.5, 5.0, 5.0, 
    5.0, 9.0, 9.0, 4.5, 9.0, 9.0, 9.0, 13.0, 13.0, 6.5, 13.0, 13.0, 13.0, 22.0, 22.0, 11.0, 22.0, 22.0, 22.0, 115.765, 110.125, 
    55.0625, 115.765, 115.765, 115.765, 7.853, 7.853, 3.9265, 7.853, 7.853, 7.853, 2.11066, 2.11066, 1.05533, 2.11066, 2.11066,
    2.11066, 1.0, 0.714286, 1.0, 0.875, 0.92145, 0.460725])


def local_env_function(
    local_env: dict,
    site: PeriodicSite
) -> List[np.ndarray]:
    """A prototype function which computes a weighted average over neighbors, weighted by the area of the Voronoi cell
    between them. This allows concurrently capturing impact of neighbor-neighbor interactions and geometric effects. 
    Critically, in contrast to cut-off based methods, the interaction is `guaranteed` to be continous as a function of 
    displacement.

    Args:
        local_env: A dictionary of the local environment of a site, as returned by a ``VoronoiNN`` generator. Contains 
            a number of critical geometric attributes like face distances, face areas, and corresponding face-bound volumes.
        site: The ``Site`` number for which the local environment is being computed.

    Returns:
        A nested list of ``np.ndarray``s. Contains several geometric attributes concatenated with gometry weighted neighbor-neighbor
        elemental attributes, and (2) a list of ``np.ndarray`` of geometry independent elemental attributes of the site.
    """
    local_attributes = np.zeros(attribute_matrix.shape[1])
    for key, value in site.species.get_el_amt_dict().items():
        local_attributes += value * attribute_matrix[Element(key).Z - 1, :]
    diff_attributes = np.zeros(attribute_matrix.shape[1])
    total_weight = 0
    volume = 0
    for _, neighbor_site in local_env.items():
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
    """A wrapper class which contains an instance of an NN generator (the default is a ``VoronoiNN``), a structure, and
    a function which computes the local environment attributes.
    
    Args:
        struct: A pymatgen ``Structure`` object.
        local_env_func: A function which computes the local environment attributes for a given site.
        nn_generator: A ``VoronoiNN`` generator object.
    """

    def __init__(
        self, 
        struct: Structure,
        local_env_func,
        nn_generator: VoronoiNN = VoronoiNN(
            compute_adj_neighbors=False, 
            extra_nn_info=False)
        ):
        self.generator = nn_generator
        self.struct = struct
        self.function = local_env_func

    def generate_local_attributes(self, n: int):
        """Wrapper pointing to a given ``Site`` index.
        
        Args:
            n: The index of the site for which the local environment attributes are being computed.
            
        Returns:
            A list of the local environment attributes for the site. The type will depend on the function used to compute the
            attributes. By default, this is a list of two numpy arrays computed by ``local_env_function``.
        """
        local_env = self.generator.get_voronoi_polyhedra(self.struct, n)
        return self.function(local_env, self.struct[n])


def generate_voronoi_attributes(
    struct: Structure, 
    local_funct=local_env_function
    ) -> tuple[np.ndarray, np.ndarray]:
    """Generates the local environment attributes for a given structure using a VoronoiNN generator.

    Args:
        struct: A pymatgen ``Structure`` object.
        local_funct: A function which computes the local environment attributes for a given site. By default, this is
            the prototype function ``local_env_function``, but you can neatly customize this to your own needs at this 
            level, if you so desire (e.g. to use a compiled alternative you have written).
            
    Returns:
        A tuple of two numpy arrays. Each contains concatenated outputs of respecive tuples from ``local_env_function``. Please note
        that, at this stage, the order of rows `does not` have to correspond to the order of sites in the structure and usually does not.
    """
    local_generator = LocalAttributeGenerator(struct, local_funct)
    attribute_list = list(
        map(local_generator.generate_local_attributes, 
            range(len(struct.sites))))
    return np.array([value[0] for value in attribute_list]), np.array([value[1] for value in attribute_list])


def most_common(
    attribute_properties: np.ndarray
    ) -> np.ndarray:
    """Calculates the attributes corresponding to the most common elements.
    
    Args:
        attribute_properties: A numpy array of the local environment attributes generated from ``generate_voronoi_attributes``.
        
    Returns:
        A numpy array of the attributes corresponding to the most common elements.
    """
    scores = np.unique(np.ravel(attribute_properties[:, 0]))  # get all unique atomic numbers
    max_occurrence = 0
    top_elements = []
    for score in scores:
        template = (attribute_properties[:, 0] == score)
        count = np.expand_dims(np.sum(template, 0), 0)[0]
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
                        featureConvergenceCriterion: float = 0.005,
                        compositionConvergenceCriterion: float = 0.01,
                        minimumElementOccurrences: int = 10,
                        plotParameters: bool = False,
                        printProgress: bool = True,
                        returnMeta: bool = False,
                        ) -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
    """**Main functionality.** Generates the KS2022 descriptor for a given composition randomly distributed on a given
    structure until the convergence criteria are met. The descriptor is **KS2022** which is compatible with all KS2022
    models. It approaches values that would be reached by infinite supercell size.
    
    Args:
        struct: A pymatgen `Structure` object that will be used as the basis for the structure to be generated. It can
            be occupied by any species without affecting the result since all will be replaced by the composition.
        comp: A pymatgen `Composition` object that will be randomly distributed on the structure within accuracy
            determined by the `compositionConvergenceCriterion`.
        minimumSitesPerExpansion: The minimum number of sites that the base structure will be expanded to (doubling
            dimension-by-dimension) before it is used as an expansion step in each iteration adding local chemical
            environment information to the global pool.
            Optimal value will depend on the number of species and their relative fractions in the composition.
            Generally, low values will result in slower convergence (<20ish) and too high values (>150ish) will result
            in slower computation. The default value is 50.
        featureConvergenceCriterion: **The maximum difference between any feature belonging to the current iteration
            (statistics based on the global ensemble of local chemical environments) and the previous two iterations
            (before the last expansion, and the one before that)** expressed as a fraction of the maximum value of each
            structure-dependent KS2022 feature found in the OQMD database at the time of SIPFENN creation
            (see `maxFeaturesInOQMD` array). The default value is 0.005, corresponding to 0.5% of the maximum value.
        compositionConvergenceCriterion: The maximum average difference between any element fraction belonging in the
            current composition (superposition of all expansions) and the target composition (comp). The default value
            is 0.01, corresponding to deviation depending on the number of elements in the composition.
        minimumElementOccurrences: The minimum number of times all elements must occur in the composition before it is
            considered converged. This is to prevent the algorithm from converging before very dilute elements have
            had a chance to occur. The default value is 10.
        plotParameters: If True, the convergence history will be plotted using plotly and, by default, will display as an 
            interactive plot in your default web browser, allowing you to zoom and pan. The figure below shows an example
            of such plot for a complex BCC 6-component high entropy alloy. The default value is False.
            
            .. image:: https://raw.githubusercontent.com/PhasesResearchLab/pySIPFENN/main/pysipfenn/descriptorDefinitions/assets/KS2022_randomSolution_ConvergencePlot.png
                :alt: KS2022_randomSolution_ConvergencePlot
                :width: 800
            
        printProgress: If True, the progress will be printed to the console. The default value is True.
        returnMeta: If True, a dictionary containing the convergence history will be returned in addition to the
            descriptor. The default value is False.

    Returns: By default, a numpy array containing the KS2022 descriptor. Please note the stochastic nature of the
    algorithm, and that the result may vary slightly between runs and parameters. If returnMeta is True,
    a tuple containing the descriptor and a dictionary containing the convergence history will be returned.
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
    maxDiff = 5
    compositionDistance = 1
    minOccupationCount = 0
    properties: np.ndarray = None
    currentComposition: Composition = None

    if printProgress:
        print(f'#Atoms | Comp. Distance AVG | Convergence Crit. MAX | Occupation Count MIN')

    if maxDiff < featureConvergenceCriterion:
        raise AssertionError('Invalid convergence criteria (maxDiff < featureConvergenceCriterion).')
    if compositionDistance < compositionConvergenceCriterion:
        raise AssertionError('Invalid convergence criteria (compositionDistance > compositionConvergenceCriterion).')
    if minOccupationCount > minimumElementOccurrences:
        raise AssertionError('Invalid convergence criteria (minOccupationCount > minimumElementOccurrences).')

    while maxDiff > featureConvergenceCriterion \
            or compositionDistance > compositionConvergenceCriterion \
            or minOccupationCount < minimumElementOccurrences:
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
                  most_common(attribute_properties)), axis=-1).reshape((-1))))
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
        if len(propHistory) > 2:
            # Current iteration diff
            diff = np.subtract(properties, propHistory[-2])
            diff /= maxFeaturesInOQMD
            diffHistory.append(diff)
            # Calculate the additional diff to one level older iteration
            diff2 = np.subtract(properties, propHistory[-3])
            diff2 /= maxFeaturesInOQMD
            # Calculate the maximum difference across both differences
            maxDiff = max(np.concatenate((diff, diff2), axis=0))
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
    # ^^^ End of the long while-loop above

    if plotParameters:
        import plotly.express as px
        import pandas as pd
        import warnings
        warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

        diffArray = np.array(diffHistory)

        # Plot the parameters as lines. Add hover text to show the parameter name based on the labels_KS2022.csv file.
        with resources.files('pysipfenn.descriptorDefinitions').joinpath('labels_KS2022.csv').open() as f:
            labels = f.readlines()
        fig = px.line(pd.DataFrame(diffArray, columns=labels), title='KS2022 Descriptor Parameters',
                      range_y=[-0.5, 0.5])
        # Add horizontal lines at convergence criteria of +/- 0.01.
        fig.add_hline(y=featureConvergenceCriterion, line_dash='dash', line_color='red')
        fig.add_hline(y=-featureConvergenceCriterion, line_dash='dash', line_color='red')

        fig.show()

    # print(f'Target: {comp.fractional_composition}')
    # print(f'Final:  {currentComposition.fractional_composition}')
    if properties is not None:
        assert properties.shape == (256,)
        assert isinstance(properties, np.ndarray)
        if returnMeta:
            return properties, {
                'diffHistory': diffHistory,
                'propHistory': propHistory,
                'finalAtomsN': attribute_properties.shape[0],
                'finalCompositionDistance': compositionDistance,
                'finalComposition': currentComposition.fractional_composition
            }
        else:
            return properties
    else:
        raise RuntimeError('KS2022_randomSolution descriptor failed to converge.')


def cite() -> List[str]:
    """Citation/s for the descriptor."""
    return [
        'Adam M. Krajewski, Jonathan W. Siegel, Jinchao Xu, Zi-Kui Liu, Extensible Structure-Informed Prediction of '
        'Formation Energy with improved accuracy and usability employing neural networks, Computational '
        'Materials Science, Volume 208, 2022, 111254'
    ]


def onlyStructural(descriptor: np.ndarray) -> np.ndarray:
    """Returns only the **part of the KS2022 descriptor that has to depend on structure**, useful in cases where the descriptor is used 
    as a fingerprint to compare polymorphs of the same compound. **Please note, this does not mean it selects all structure-dependent 
    features which span nearly entire descriptor, but only the part of the descriptor which is explicitly structure-dependent.** 

    Args:
        descriptor: A ``256``-length numpy ``ndarray`` of the KS2022 descriptor. Generated by the ``generate_descriptor`` function.

    Returns:
        A ``103``-length numpy ``ndarray`` of the structure-dependent part of the KS2022 descriptor. 
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
            returnDescriptorAndMeta: bool = False) -> Union[None, Tuple[np.ndarray, dict]]:
    """Profiles the descriptor in parallel using one of the test structures.

    Args:
        test: The test structure to use. Options are 'FCC', 'BCC', and 'HCP'.
        comp: The composition to use. Should be a Composition pymatgen object.
        nIterations: The number of iterations to run. If 1, the descriptor will be calculated once and may be available
            in the return value. If >1, the descriptor will be calculated nIterations times and the result will
            be persisted in `f'TestResult_KS2022_randomSolution_{test}_{nIterations}iter.csv'` file in the current
            working directory.
        plotParameters: If True, the convergence history will be plotted using plotly. The default value is False.
        returnDescriptorAndMeta: If True, a tuple containing the descriptor and a dictionary containing the convergence
            history will be returned. The default value is False.

    Returns:
        Depending on the value of nIterations and returnDescriptorAndMeta, the return value will be either a tuple of
        the descriptor and a dictionary containing the convergence history, or None. In either case, the descriptor
        will be persisted in `f'TestResult_KS2022_randomSolution_{test}_{nIterations}iter.csv'` file.
    """
    c = pysipfenn.Calculator(autoLoad=False)

    try:
        s = c.prototypeLibrary[test]['structure']
    except KeyError:
        raise NotImplementedError(f'Unrecognized test name: {test}')

    name = f'TestResult_KS2022_randomSolution_{test}_{nIterations}iter.csv'

    if nIterations == 1:
        d, meta = generate_descriptor(s, comp, plotParameters=plotParameters, returnMeta=True)
        print(f"Got meta with :{meta.keys()} keys")
        with open(name, 'w+') as f:
            f.writelines([f'{v}\n' for v in d])
        if returnDescriptorAndMeta:
            return d, meta
    elif nIterations > 1:
        print(f'Running {nIterations} iterations in parallel...')
        d = process_map(generate_descriptor,
                        [s for _ in range(nIterations)],
                        [comp for _ in range(nIterations)],
                        chunksize=1,
                        max_workers=8)
        with open(name, 'w+') as f:
            f.writelines([f'{",".join([str(v) for v in di])}\n' for di in d])
        return None
    else:
        print('No descriptors generated.')
        return None

    print('Done!')


if __name__ == "__main__":
    print('You are running the KS2022_randomSolutions.py file directly. It is intended to be used as a module. '
          'A profiling task will now commence, going over several cases. This will take a while.')
    t0 = time.time()
    profile(test='FCC', plotParameters=True)
    profile(test='BCC', plotParameters=True)
    profile(test='HCP', plotParameters=True)
    profile(test='BCC', nIterations=6)
    print(f"All profiling tasks completed in {time.time() - t0:.2f} seconds. The results have been saved to the current working directory.")
    print(f"Average of {((time.time() - t0) / 4):.2f} seconds per task.")
