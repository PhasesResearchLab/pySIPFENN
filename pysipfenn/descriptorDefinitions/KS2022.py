# This file is part of pySIPFENN and is licensed under the terms of the LGPLv3 or later.
# Copyright (C) 2023 Adam M. Krajewski, Jonathan Siegel

"""This ``KS2022`` feature vector calculator is a highly optimized modification of our previous work calculating ``Ward2017`` features. It 
is **improved in terms of both feature selection and speed**. 

Most critically, it uses symmetry analysis to avoid redundant calculations of chemically equivalent atoms,
typically speeding up the calculation by a factor of 3 to 10 for ordered structures. Feature-optimization was also carried out to remove
several unphysical features or representations, such as space group being represented as a real number, rather than category. If
you use this code, plese cite (as in ``KS2022.cite()``):

- Adam M. Krajewski, Jonathan W. Siegel, Jinchao Xu, Zi-Kui Liu, "Extensible Structure-Informed Prediction of Formation Energy with 
  improved accuracy and usability employing neural networks", Computational Materials Science, Volume 208, 2022, 111254

The core purpose of this module is to calculate numpy ``ndarray`` with ``256`` features constructed by considering all local chemical 
environments existing in an atomic structure. Their list is available in the ``labels_KS2022.csv`` and will be discussed in our upcoming
publication (Spring 2024).
"""

# Standard Library Imports
import math
import time
import json
from collections import Counter
from typing import List, Dict
from importlib import resources

# Third Party Dependencies
from tqdm import tqdm
import numpy as np
from pymatgen.core import Structure, Element, PeriodicSite
from pymatgen.analysis.local_env import VoronoiNN
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

# Certain hard-coded basic elemental properties used in the featurization (compatible with Magpie references).
periodic_table_size = 112
f = resources.files('pysipfenn.descriptorDefinitions').joinpath("element_properties_Ward2017KS2022.csv")
attribute_matrix = np.loadtxt(f, delimiter=',')
attribute_matrix = np.nan_to_num(attribute_matrix)
attribute_matrix = attribute_matrix[:,[45, 33, 2, 32, 5, 48, 6, 10, 44, 42, 38, 40, 36, 43, 41, 37, 39, 35, 18, 13, 17]]


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
    """Generates the local environment attributes for a given structure using a VoronoiNN generator. **Critically, this 
    implementation uses ``get_equivalentSitesMultiplicities`` to skip the expensive ``generate_local_attributes`` call when
    it is not guaranteed to be needed based on the symmetry of the structure.**

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
    attribute_list = list()
    equivalentSitesMultiplicities = get_equivalentSitesMultiplicities(struct)
    for siteN in equivalentSitesMultiplicities:
        localAttributes = [local_generator.generate_local_attributes(siteN)]
        attribute_list += localAttributes * equivalentSitesMultiplicities[siteN]
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


def get_equivalentSitesMultiplicities(
    struct: Structure,
    angle_tolerance: float = 0.1, 
    symprec: float = 0.001
    ) -> Dict[int, int]:
    """Takes a ``pymatgen`` ``Structure`` object, obtains an ``spglib`` symmetry dataset through the ``SpacegroupAnalyzer`` wrapper of 
    ```pymatgen```, and returns a dictionary of the multiplicities of equivalent sites.```
    
    Args:
        struct: A ``pymatgen`` ``Structure`` object.
        angle_tolerance: The angle tolerance for the symmetry analysis. By default, this is ``0.1``.
        symprec: The symmetry precision for the symmetry analysis. By default, this is ``0.001`` `when called from here`.
        
    Returns:
        A dictionary of the multiplicities of equivalent sites. The keys are the indices of the sites in the structure, and the values 
        are the integer multiplicities.
    """
    spgA = SpacegroupAnalyzer(
        struct, 
        angle_tolerance=angle_tolerance, 
        symprec=symprec)
    return dict(
        Counter(
            list(spgA.get_symmetry_dataset()['equivalent_atoms'])))


def generate_descriptor(struct: Structure) -> np.ndarray:
    """Main functionality sharing API with every other featurizer in ``pySIPEFNN``. Generates the KS2022 descriptor for a given structure.
    As explained in the top-level documentation, this descriptor is very efficient for ordered structures, and is minimizes expensive 
    computation by considering site equivalences due to symmetry.

    Args:
        struct: A pymatgen ``Structure`` object. It can be any ordered (e.g., crystal) or disordered (e.g., glass) structure with collapsed
            (defined) occupancies.
    Returns:
        A ``256``-length numpy ``ndarray`` of the descriptor. See ``labels_KS2022.csv`` for the meaning of each element of the array.
    """
    diff_properties, attribute_properties = generate_voronoi_attributes(struct)
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
    properties[12] *= len(attribute_properties) / struct.volume
    # Calculate and insert stoichiometry attributes.
    element_dict = struct.composition.fractional_composition.as_dict()
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


def profile(
    test: str = 'JVASP-10001', 
    nRuns: int = 10,
    persistResult: bool = True
    ) -> None:
    """Profiles the descriptor in `series` using one of the test structures.
    
    Args:
        test: The name of the test structure. By default, this is ``'JVASP-10001'``. Currently implemented tests are: ``'JVASP-10001'`` and 
            ``'diluteNiAlloy'``.
        nRuns: The number of runs. By default, this is ``10``.
        persistResult: Whether to persist the result to a file (``'KS2022_TestResult.csv'``) to allow for inspection. By default, this is
            ``True``.
    """
    if test == 'JVASP-10001':
        print(
            f'KS2022 profiling/testing task will calculate a descriptor for Li2 Zr1 Te1 O6 (JVASP-10001) {nRuns} times in series.')
        matStr = '{"@module": "pymatgen.core.structure", "@class": "Structure", "charge": null, "lattice": {"matrix": [[4.599305652662459, 0.0098015076998823, 3.1052612865443736], [1.6553257726204653, 4.291108475854712, 3.1052602938979565], [0.0142541214919749, 0.0098025099996131, 5.549419141866351]], "a": 5.549446478152326, "b": 5.549446536179343, "c": 5.549446105810423, "alpha": 55.82714459985832, "beta": 55.82714014289371, "gamma": 55.82713972779092, "volume": 109.15484625642743}, "sites": [{"species": [{"element": "Li", "occu": 1.0}], "abc": [0.2738784872669924, 0.2738784872670407, 0.2738784872673032], "xyz": [1.7169128904007063, 1.1806114167777613, 3.220794775377278], "label": "Li", "properties": {}}, {"species": [{"element": "Li", "occu": 1.0}], "abc": [0.7852272010728069, 0.7852272010728856, 0.785227201073315], "xyz": [4.922499451739965, 3.3848887059434927, 9.234225338163633], "label": "Li", "properties": {}}, {"species": [{"element": "O", "occu": 1.0}], "abc": [0.8669964454661124, 0.604089882092114, 0.241821769873143], "xyz": [4.990994160164061, 2.603083545876856, 5.910077181137658], "label": "O", "properties": {}}, {"species": [{"element": "O", "occu": 1.0}], "abc": [0.717840894529788, 0.1213675889628683, 0.393537009186973], "xyz": [3.508082106234713, 0.531695063215412, 4.789863306469278], "label": "O", "properties": {}}, {"species": [{"element": "O", "occu": 1.0}], "abc": [0.1213675889638402, 0.3935370091873943, 0.7178408945283384], "xyz": [1.2198707830817896, 1.6969362235910317, 5.58251292516964], "label": "O", "properties": {}}, {"species": [{"element": "O", "occu": 1.0}], "abc": [0.3935370091861915, 0.7178408945293856, 0.1213675889634014], "xyz": [2.999987512595622, 3.085380109860344, 4.124637687962297], "label": "O", "properties": {}}, {"species": [{"element": "O", "occu": 1.0}], "abc": [0.2418217698721573, 0.8669964454671221, 0.6040898820921513], "xyz": [2.555984564633329, 3.728667610729149, 6.795517372377343], "label": "O", "properties": {}}, {"species": [{"element": "O", "occu": 1.0}], "abc": [0.6040898820933115, 0.2418217698723637, 0.8669964454664059], "xyz": [3.191046090145173, 1.0521031793025595, 7.4381031350436535], "label": "O", "properties": {}}, {"species": [{"element": "Te", "occu": 1.0}], "abc": [0.4965905610507353, 0.4965905610507355, 0.4965905610507361], "xyz": [3.113069390835793, 2.1406591357024984, 5.8398755612146624], "label": "Te", "properties": {}}, {"species": [{"element": "Zr", "occu": 1.0}], "abc": [0.0006501604980668, 0.0006501604980928, 0.0006501604982344], "xyz": [0.00407578174946036, 0.002802654981945192, 0.007645848918263076], "label": "Zr", "properties": {}}]}'
    elif test == 'diluteNiAlloy':
        print(
            f'KS2022 profiling/testing task will calculate a descriptor for a dilute FCC Ni31Cr1 alloy {nRuns} times in series.')
        matStr = '{"@module": "pymatgen.core.structure", "@class": "Structure", "charge": null, "lattice": {"matrix": [[6.995692, 0.0, 0.0], [0.0, 6.995692, 0.0], [0.0, 0.0, 6.995692]], "a": 6.995692, "b": 6.995692, "c": 6.995692, "alpha": 90.0, "beta": 90.0, "gamma": 90.0, "volume": 342.36711365619243}, "sites": [{"species": [{"element": "Cr", "occu": 1}], "abc": [0.0, 0.0, 0.0], "xyz": [0.0, 0.0, 0.0], "label": "Cr", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.0, 0.0, 0.5], "xyz": [0.0, 0.0, 3.497846], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.0, 0.5, 0.0], "xyz": [0.0, 3.497846, 0.0], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.0, 0.5, 0.5], "xyz": [0.0, 3.497846, 3.497846], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.5, 0.0, 0.0], "xyz": [3.497846, 0.0, 0.0], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.5, 0.0, 0.5], "xyz": [3.497846, 0.0, 3.497846], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.5, 0.5, 0.0], "xyz": [3.497846, 3.497846, 0.0], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.5, 0.5, 0.5], "xyz": [3.497846, 3.497846, 3.497846], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.25, 0.25, 0.0], "xyz": [1.748923, 1.748923, 0.0], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.25, 0.25, 0.5], "xyz": [1.748923, 1.748923, 3.497846], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.25, 0.7500000000000001, 0.0], "xyz": [1.748923, 5.2467690000000005, 0.0], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.25, 0.7500000000000001, 0.5], "xyz": [1.748923, 5.2467690000000005, 3.497846], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.7500000000000001, 0.25, 0.0], "xyz": [5.2467690000000005, 1.748923, 0.0], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.7500000000000001, 0.25, 0.5], "xyz": [5.2467690000000005, 1.748923, 3.497846], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.7500000000000001, 0.7500000000000001, 0.0], "xyz": [5.2467690000000005, 5.2467690000000005, 0.0], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.7500000000000001, 0.7500000000000001, 0.5], "xyz": [5.2467690000000005, 5.2467690000000005, 3.497846], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.25, 0.0, 0.25], "xyz": [1.748923, 0.0, 1.748923], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.25, 0.0, 0.7500000000000001], "xyz": [1.748923, 0.0, 5.2467690000000005], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.25, 0.5, 0.25], "xyz": [1.748923, 3.497846, 1.748923], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.25, 0.5, 0.7500000000000001], "xyz": [1.748923, 3.497846, 5.2467690000000005], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.7500000000000001, 0.0, 0.25], "xyz": [5.2467690000000005, 0.0, 1.748923], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.7500000000000001, 0.0, 0.7500000000000001], "xyz": [5.2467690000000005, 0.0, 5.2467690000000005], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.7500000000000001, 0.5, 0.25], "xyz": [5.2467690000000005, 3.497846, 1.748923], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.7500000000000001, 0.5, 0.7500000000000001], "xyz": [5.2467690000000005, 3.497846, 5.2467690000000005], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.0, 0.25, 0.25], "xyz": [0.0, 1.748923, 1.748923], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.0, 0.25, 0.7500000000000001], "xyz": [0.0, 1.748923, 5.2467690000000005], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.0, 0.7500000000000001, 0.25], "xyz": [0.0, 5.2467690000000005, 1.748923], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.0, 0.7500000000000001, 0.7500000000000001], "xyz": [0.0, 5.2467690000000005, 5.2467690000000005], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.5, 0.25, 0.25], "xyz": [3.497846, 1.748923, 1.748923], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.5, 0.25, 0.7500000000000001], "xyz": [3.497846, 1.748923, 5.2467690000000005], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.5, 0.7500000000000001, 0.25], "xyz": [3.497846, 5.2467690000000005, 1.748923], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.5, 0.7500000000000001, 0.7500000000000001], "xyz": [3.497846, 5.2467690000000005, 5.2467690000000005], "label": "Ni", "properties": {}}], "@version": null}'
    else:
        print('Unrecognized test name.')
        return None
    t0 = time.time()
    structList = [Structure.from_dict(json.loads(matStr))] * nRuns
    for s in tqdm(structList):
        d = generate_descriptor(s)
    if persistResult:
        with open('KS2022_TestResult.csv', 'w+') as f:
            f.writelines([f'{v}\n' for v in d])
    print(f"Done in {time.time() - t0} seconds.")
    print(f"Average time per run: {(time.time() - t0) / nRuns} seconds.")
    return None


def profileParallel(
    test: str = 'JVASP-10001', 
    nRuns: int = 1000,
    makeSupercell222: bool = False
    ) -> None:
    """Profiles the descriptor in `parallel` using one of the test structures.
    
    Args:
        test: The name of the test structure. By default, this is ``'JVASP-10001'``. Currently implemented tests are: ``'JVASP-10001'`` and 
            ``'diluteNiAlloy'``.
        nRuns: The number of total runs done in parallel by 8 workers. By default, this is ``1000``.
        makeSupercell222: Whether to make a 2x2x2 supercell of the structure before profiling, increasing the number of atoms by a factor
            of 8, but should not increase time thanks to the symmetry consierations. By default, this is ``False``.
    """
    from tqdm.contrib.concurrent import process_map
    if test == 'JVASP-10001':
        print(
            f'KS2022 profiling/testing task will calculate a descriptor for Li2 Zr1 Te1 O6 (JVASP-10001) {nRuns} times in parallel with 8 workers.')
        matStr = '{"@module": "pymatgen.core.structure", "@class": "Structure", "charge": null, "lattice": {"matrix": [[4.599305652662459, 0.0098015076998823, 3.1052612865443736], [1.6553257726204653, 4.291108475854712, 3.1052602938979565], [0.0142541214919749, 0.0098025099996131, 5.549419141866351]], "a": 5.549446478152326, "b": 5.549446536179343, "c": 5.549446105810423, "alpha": 55.82714459985832, "beta": 55.82714014289371, "gamma": 55.82713972779092, "volume": 109.15484625642743}, "sites": [{"species": [{"element": "Li", "occu": 1.0}], "abc": [0.2738784872669924, 0.2738784872670407, 0.2738784872673032], "xyz": [1.7169128904007063, 1.1806114167777613, 3.220794775377278], "label": "Li", "properties": {}}, {"species": [{"element": "Li", "occu": 1.0}], "abc": [0.7852272010728069, 0.7852272010728856, 0.785227201073315], "xyz": [4.922499451739965, 3.3848887059434927, 9.234225338163633], "label": "Li", "properties": {}}, {"species": [{"element": "O", "occu": 1.0}], "abc": [0.8669964454661124, 0.604089882092114, 0.241821769873143], "xyz": [4.990994160164061, 2.603083545876856, 5.910077181137658], "label": "O", "properties": {}}, {"species": [{"element": "O", "occu": 1.0}], "abc": [0.717840894529788, 0.1213675889628683, 0.393537009186973], "xyz": [3.508082106234713, 0.531695063215412, 4.789863306469278], "label": "O", "properties": {}}, {"species": [{"element": "O", "occu": 1.0}], "abc": [0.1213675889638402, 0.3935370091873943, 0.7178408945283384], "xyz": [1.2198707830817896, 1.6969362235910317, 5.58251292516964], "label": "O", "properties": {}}, {"species": [{"element": "O", "occu": 1.0}], "abc": [0.3935370091861915, 0.7178408945293856, 0.1213675889634014], "xyz": [2.999987512595622, 3.085380109860344, 4.124637687962297], "label": "O", "properties": {}}, {"species": [{"element": "O", "occu": 1.0}], "abc": [0.2418217698721573, 0.8669964454671221, 0.6040898820921513], "xyz": [2.555984564633329, 3.728667610729149, 6.795517372377343], "label": "O", "properties": {}}, {"species": [{"element": "O", "occu": 1.0}], "abc": [0.6040898820933115, 0.2418217698723637, 0.8669964454664059], "xyz": [3.191046090145173, 1.0521031793025595, 7.4381031350436535], "label": "O", "properties": {}}, {"species": [{"element": "Te", "occu": 1.0}], "abc": [0.4965905610507353, 0.4965905610507355, 0.4965905610507361], "xyz": [3.113069390835793, 2.1406591357024984, 5.8398755612146624], "label": "Te", "properties": {}}, {"species": [{"element": "Zr", "occu": 1.0}], "abc": [0.0006501604980668, 0.0006501604980928, 0.0006501604982344], "xyz": [0.00407578174946036, 0.002802654981945192, 0.007645848918263076], "label": "Zr", "properties": {}}]}'
    elif test == 'diluteNiAlloy':
        print(
            f'KS2022 profiling/testing task will calculate a descriptor for a dilute FCC Ni31Cr1 alloy {nRuns} times in parallel with 8 workers.')
        matStr = '{"@module": "pymatgen.core.structure", "@class": "Structure", "charge": null, "lattice": {"matrix": [[6.995692, 0.0, 0.0], [0.0, 6.995692, 0.0], [0.0, 0.0, 6.995692]], "a": 6.995692, "b": 6.995692, "c": 6.995692, "alpha": 90.0, "beta": 90.0, "gamma": 90.0, "volume": 342.36711365619243}, "sites": [{"species": [{"element": "Cr", "occu": 1}], "abc": [0.0, 0.0, 0.0], "xyz": [0.0, 0.0, 0.0], "label": "Cr", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.0, 0.0, 0.5], "xyz": [0.0, 0.0, 3.497846], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.0, 0.5, 0.0], "xyz": [0.0, 3.497846, 0.0], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.0, 0.5, 0.5], "xyz": [0.0, 3.497846, 3.497846], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.5, 0.0, 0.0], "xyz": [3.497846, 0.0, 0.0], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.5, 0.0, 0.5], "xyz": [3.497846, 0.0, 3.497846], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.5, 0.5, 0.0], "xyz": [3.497846, 3.497846, 0.0], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.5, 0.5, 0.5], "xyz": [3.497846, 3.497846, 3.497846], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.25, 0.25, 0.0], "xyz": [1.748923, 1.748923, 0.0], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.25, 0.25, 0.5], "xyz": [1.748923, 1.748923, 3.497846], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.25, 0.7500000000000001, 0.0], "xyz": [1.748923, 5.2467690000000005, 0.0], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.25, 0.7500000000000001, 0.5], "xyz": [1.748923, 5.2467690000000005, 3.497846], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.7500000000000001, 0.25, 0.0], "xyz": [5.2467690000000005, 1.748923, 0.0], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.7500000000000001, 0.25, 0.5], "xyz": [5.2467690000000005, 1.748923, 3.497846], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.7500000000000001, 0.7500000000000001, 0.0], "xyz": [5.2467690000000005, 5.2467690000000005, 0.0], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.7500000000000001, 0.7500000000000001, 0.5], "xyz": [5.2467690000000005, 5.2467690000000005, 3.497846], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.25, 0.0, 0.25], "xyz": [1.748923, 0.0, 1.748923], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.25, 0.0, 0.7500000000000001], "xyz": [1.748923, 0.0, 5.2467690000000005], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.25, 0.5, 0.25], "xyz": [1.748923, 3.497846, 1.748923], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.25, 0.5, 0.7500000000000001], "xyz": [1.748923, 3.497846, 5.2467690000000005], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.7500000000000001, 0.0, 0.25], "xyz": [5.2467690000000005, 0.0, 1.748923], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.7500000000000001, 0.0, 0.7500000000000001], "xyz": [5.2467690000000005, 0.0, 5.2467690000000005], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.7500000000000001, 0.5, 0.25], "xyz": [5.2467690000000005, 3.497846, 1.748923], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.7500000000000001, 0.5, 0.7500000000000001], "xyz": [5.2467690000000005, 3.497846, 5.2467690000000005], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.0, 0.25, 0.25], "xyz": [0.0, 1.748923, 1.748923], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.0, 0.25, 0.7500000000000001], "xyz": [0.0, 1.748923, 5.2467690000000005], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.0, 0.7500000000000001, 0.25], "xyz": [0.0, 5.2467690000000005, 1.748923], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.0, 0.7500000000000001, 0.7500000000000001], "xyz": [0.0, 5.2467690000000005, 5.2467690000000005], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.5, 0.25, 0.25], "xyz": [3.497846, 1.748923, 1.748923], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.5, 0.25, 0.7500000000000001], "xyz": [3.497846, 1.748923, 5.2467690000000005], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.5, 0.7500000000000001, 0.25], "xyz": [3.497846, 5.2467690000000005, 1.748923], "label": "Ni", "properties": {}}, {"species": [{"element": "Ni", "occu": 1}], "abc": [0.5, 0.7500000000000001, 0.7500000000000001], "xyz": [3.497846, 5.2467690000000005, 5.2467690000000005], "label": "Ni", "properties": {}}], "@version": null}'
    else:
        print('Unrecognized test name.')
        return None
    t0 = time.time()
    s = Structure.from_dict(json.loads(matStr))
    if makeSupercell222:
        s.make_supercell(scaling_matrix=[2,2,2])
    structList = [s] * nRuns
    process_map(generate_descriptor, structList, max_workers=8)
    print(f"Done in {time.time() - t0} seconds.")
    print(f"Average time per run: {(time.time() - t0) / nRuns} seconds.")
    return None


if __name__ == "__main__":
    profile(test='JVASP-10001')
    profile(test='diluteNiAlloy')
    profileParallel(test='JVASP-10001')
    profileParallel(test='diluteNiAlloy')
