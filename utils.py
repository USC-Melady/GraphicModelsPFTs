# Created by xunannancy at 2021/1/4

import numpy as np
from pyseqlab.attributes_extraction import GenericAttributeExtractor
from pyseqlab.utilities import TemplateGenerator, SequenceStruct
import os
import json

atchley_factors_dict = {
    '*': [0, 0, 0, 0, 0],
    'A': [-0.591, -1.302, -0.733, 1.570, -0.146],
    'C': [-1.343, 0.465, -0.862, -1.020, -0.255],
    'D': [1.050, 0.302, -3.656, -0.259, -3.242],
    'E': [1.357, -1.453, 1.477, 0.113, -0.837],
    'F': [-1.006, -0.590, 1.891, -0.397, 0.412],
    'G': [-0.384, 1.652, 1.330, 1.045, 2.064],
    'H': [0.336, -0.417, -1.673, -1.474, -0.078],
    'I': [-1.239, -0.547, 2.131, 0.393, 0.816],
    'K': [1.831, -0.561, 0.533, -0.277, 1.648],
    'L': [-1.019, -0.987, -1.505, 1.266, -0.912],
    'M': [-0.663, -1.524, 2.219, -1.005, 1.212],
    'N': [0.945, 0.828, 1.299, -0.169, 0.933],
    'P': [0.189, 2.081, -1.628, 0.421, -1.392],
    'Q': [0.931, -0.179, -3.005, -0.503, -1.853],
    'R': [1.538, -0.055, 1.502, 0.440, 2.897],
    'S': [-0.228, 1.399, -4.760, 0.670, -2.647],
    'T': [-0.032, 0.326, 2.213, 0.908, 1.313],
    'V': [-1.337, -0.279, -0.544, 1.242, -1.262],
    'W': [-0.595, 0.009, 0.672, -2.128, -0.184],
    'Y': [0.260, 0.830, 3.097, -0.838, 1.512],
}

one_hot_dict = dict()
amini_acid_num = len(atchley_factors_dict.keys())
for index, key in enumerate(atchley_factors_dict.keys()):
    one_hot_dict[key] = np.zeros(amini_acid_num)
    one_hot_dict[key][index] = 1


s3_str_mapping = {
    'C': str(0),
    'H': str(2),
    'E': str(1),
}

s8_str_mapping = {
    'G': str(0),
    'H': str(1),
    'I': str(2),
    'E': str(3),
    'B': str(4),
    'T': str(5),
    'S': str(6),
    'L': str(7)
}


class SeqSegAttributeExtractor(GenericAttributeExtractor):
    def __init__(self):
        """

        :param seg_length: compute segment length as feature or not
        :param ss_percentile: compute ratios of H:E:C in 3-state secondary structure (predictions)
        """
        attr_desc = self.generate_attributes_desc()
        super().__init__(attr_desc)

    def generate_attributes_desc(self):
        attr_desc = {
            'atchley_factor_0':
                {
                    'description': '1st of 5-digit vector representation of the amino acid',
                    'encoding': 'continuous'},
            'atchley_factor_1':
                {
                    'description': '2nd of 5-digit vector representation of the amino acid',
                    'encoding': 'continuous'},
            'atchley_factor_2':
                {
                    'description': '3rd of 5-digit vector representation of the amino acid',
                    'encoding': 'continuous'},
            'atchley_factor_3':
                {
                    'description': '4th of 5-digit vector representation of the amino acid',
                    'encoding': 'continuous'},
            'atchley_factor_4':
                {
                    'description': '5th of 5-digit vector representation of the amino acid',
                    'encoding': 'continuous'},
            'one_hot':
                {
                    'description': 'one-hot representation of the amino acid',
                    'encoding': 'categorical',
                },

            'ss3':
                {
                    'description': 'ss3 of the amino acid',
                    'encoding': 'categorical',
                },
            'ss8':
                {
                    'description': 'ss8 of the amino acid',
                    'encoding': 'categorical',
                },

        }

        return attr_desc


def template_config(group_index):
    if group_index in ['1', '2']:
        X_Y_tmp = ('1-gram:2-gram:3-gram', range(-2, 3))
    elif group_index == '3':
        X_Y_tmp = ('1-gram:2-gram:3-gram:4-gram:5-gram:6-gram', range(-5, 6))


    template_generator = TemplateGenerator()
    templateXY = {}
    for factor_index in range(5):
        template_generator.generate_template_XY('atchley_factor_%d'%factor_index, X_Y_tmp, '1-state:2-states:3-states', templateXY)
    template_generator.generate_template_XY('one_hot', X_Y_tmp, '1-state:2-states:3-states', templateXY)
    template_generator.generate_template_XY('ss3', X_Y_tmp, '1-state:2-states:3-states', templateXY)
    template_generator.generate_template_XY('ss8', X_Y_tmp, '1-state:2-states:3-states', templateXY)

    templateY = template_generator.generate_template_Y('3-states')
    return templateXY, templateY


segment_states = {
    '1': {
        'alpha helix': [],
        'beta strand': ['S1', 'S2', 'S3', 'S4', 'S5'],
    },
    '2': {
        'alpha helix': [],
        'beta strand': ['S1', 'S2', 'S3', 'S4', 'S5'],
    },
    '3': {
        'alpha helix': [],
        'beta strand': ['S1', 'S2', 'S3', 'S4'],
    },
}

def summarize_segments(group_index, testing_predictions):
    results_to_save = dict()
    for protein_index, cur_predictions in enumerate(testing_predictions):
        results_to_save[protein_index] = dict()

        loc_info = dict()
        # get the maximum length for each segments
        for beta_strand_name in segment_states[group_index]['beta strand']:
            loc_info[beta_strand_name] = list()
        flag = -1
        for residue_index, residue_label in enumerate(cur_predictions):
            if residue_label == '0':
                if flag != -1:
                    loc_info['S%s' % flag][-1][-1] = residue_index
                flag = -1
            elif flag == -1:
                loc_info['S%s' % residue_label].append([residue_index, -1])
                flag = residue_label

        for beta_strand_name in segment_states[group_index]['beta strand']:
            tt = list()
            for one_segment in loc_info[beta_strand_name]:
                if one_segment[-1] == -1:
                    tt.append([one_segment[0], len(cur_predictions)])
                else:
                    tt.append(one_segment)
            if len(tt) == 0:
                continue
            results_to_save[protein_index][beta_strand_name] = tt[np.argsort(np.array(tt)[:, 1] - np.array(tt)[:, 0])[-1]]

    return results_to_save

# majority of positive proteins have HB for that segment
segmentation_property_majority = {
    '1': {
        'S1<=>S2': 'anti-parallel',
        'S3<=>S4': 'anti-parallel',
        'S4<=>S5': 'anti-parallel',
    },
    '2': {
        'S2<=>S3': 'anti-parallel',
        'S2<=>S4': 'anti-parallel',
        'S1<=>S4': 'anti-parallel',
        'S4<=>S5': 'anti-parallel',
    },
    '3': {
        'S1<=>S2': 'anti-parallel',
        'S2<=>S3': 'anti-parallel',
        'S2<=>S4': 'anti-parallel',
    },
}

amino_acid_relative_frequency = {
    'parallel': {
        'A-A': 0.77,
        'A-C': 1.84,
        'A-D': 0.75,
        'A-E': 0.67,
        'A-F': 2.89,
        'A-G': 1.38,
        'A-H': 1.08,
        'A-I': 5.01,
        'A-K': 0.48,
        'A-L': 2.64,
        'A-M': 1.94,
        'A-N': 0.83,
        'A-P': 0.50,
        'A-Q': 0.75,
        'A-R': 1.17,
        'A-S': 1.11,
        'A-T': 2.10,
        'A-V': 5.35,
        'A-W': 2.32,
        'A-Y': 2.44,
        'A-X': 1.801, # mean

        'C-C': 2.52,
        'C-D': 1.56,
        'C-E': 0.83,
        'C-F': 3.62,
        'C-G': 1.96,
        'C-H': 4.71,
        'C-I': 4.65,
        'C-K': 0.66,
        'C-L': 2.45,
        'C-M': 2.68,
        'C-N': 0.88,
        'C-P': 0.71,
        'C-Q': 1.00,
        'C-R': 0.53,
        'C-S': 1.54,
        'C-T': 1.63,
        'C-V': 5.30,
        'C-W': 6.10,
        'C-Y': 4.59,
        'C-X': 2.488, # mean

        'D-D': 0.43,
        'D-E': 0.57,
        'D-F': 0.82,
        'D-G': 0.90,
        'D-H': 1.71,
        'D-I': 1.22,
        'D-K': 1.21,
        'D-L': 0.92,
        'D-M': 0.60,
        'D-N': 1.22,
        'D-P': 0.33,
        'D-Q': 0.46,
        'D-R': 1.83,
        'D-S': 1.26,
        'D-T': 1.36,
        'D-V': 1.62,
        'D-W': 0.94,
        'D-Y': 0.82,
        'D-X': 1.0265, # mean

        'E-E': 0.04,
        'E-F': 1.05,
        'E-G': 0.54,
        'E-H': 1.22,
        'E-I': 1.30,
        'E-K': 1.25,
        'E-L': 0.83,
        'E-M': 0.74,
        'E-N': 0.49,
        'E-P': 0.20,
        'E-Q': 0.59,
        'E-R': 1.93,
        'E-S': 0.73,
        'E-T': 0.98,
        'E-V': 1.55,
        'E-W': 0.89,
        'E-Y': 1.44,
        'E-X': 0.892, # mean

        'F-F': 2.77,
        'F-G': 2.34,
        'F-H': 2.10,
        'F-I': 7.09,
        'F-K': 0.88,
        'F-L': 3.10,
        'F-M': 2.93,
        'F-N': 1.28,
        'F-P': 0.43,
        'F-Q': 1.24,
        'F-R': 1.02,
        'F-S': 1.32,
        'F-T': 2.16,
        'F-V': 8.72,
        'F-W': 2.59,
        'F-Y': 3.70,
        'F-X': 2.6025, # mean

        'G-G': 0.76,
        'G-H': 1.45,
        'G-I': 2.98,
        'G-K': 0.49,
        'G-L': 1.84,
        'G-M': 1.68,
        'G-N': 0.95,
        'G-P': 0.44,
        'G-Q': 0.64,
        'G-R': 0.65,
        'G-S': 0.83,
        'G-T': 1.27,
        'G-V': 3.04,
        'G-W': 2.20,
        'G-Y': 1.95,
        'G-X': 1.414, # mean

        'H-H': 1.01,
        'H-I': 2.95,
        'H-K': 0.94,
        'H-L': 1.66,
        'H-M': 1.57,
        'H-N': 1.66,
        'H-P': 0.86,
        'H-Q': 0.77,
        'H-R': 1.77,
        'H-S': 1.66,
        'H-T': 2.54,
        'H-V': 3.76,
        'H-W': 3.17,
        'H-Y': 2.34,
        'H-X': 1.9465, # mean

        'I-I': 7.28,
        'I-K': 1.80,
        'I-L': 8.03,
        'I-M': 4.71,
        'I-N': 1.32,
        'I-P': 0.90,
        'I-Q': 1.19,
        'I-R': 1.79,
        'I-S': 2.00,
        'I-T': 3.42,
        'I-V': 14.44,
        'I-W': 6.09,
        'I-Y': 5.80,
        'I-X': 4.1985, # mean

        'K-K': 0.34,
        'K-L': 0.71,
        'K-M': 0.54,
        'K-N': 0.33,
        'K-P': 0.11,
        'K-Q': 0.64,
        'K-R': 0.44,
        'K-S': 0.67,
        'K-T': 1.47,
        'K-V': 1.70,
        'K-W': 1.61,
        'K-Y': 1.38,
        'K-X': 0.8825, # mean

        'L-L': 2.15,
        'L-M': 2.72,
        'L-N': 0.48,
        'L-P': 0.76,
        'L-Q': 0.86,
        'L-R': 1.18,
        'L-S': 1.05,
        'L-T': 2.05,
        'L-V': 8.70,
        'L-W': 2.30,
        'L-Y': 3.11,
        'L-X': 2.377, # mean

        'M-M': 1.94,
        'M-N': 0.68,
        'M-P': 0.48,
        'M-Q': 0.66,
        'M-R': 0.90,
        'M-S': 1.17,
        'M-T': 1.53,
        'M-V': 5.78,
        'M-W': 3.17,
        'M-Y': 2.95,
        'M-X': 1.9685, # mean

        'N-N': 1.28,
        'N-P': 0.23,
        'N-Q': 0.94,
        'N-R': 0.35,
        'N-S': 0.92,
        'N-T': 2.05,
        'N-V': 1.75,
        'N-W': 1.51,
        'N-Y': 1.35,
        'N-X': 1.025, # mean

        'P-P': 0.01,
        'P-Q': 0.36,
        'P-R': 0.27,
        'P-S': 0.28,
        'P-T': 0.58,
        'P-V': 1.39,
        'P-W': 0.61,
        'P-Y': 0.80,
        'P-X': 0.5125, # mean

        'Q-Q': 0.20,
        'Q-R': 0.60,
        'Q-S': 0.91,
        'Q-T': 1.45,
        'Q-V': 1.16,
        'Q-W': 1.83,
        'Q-Y': 1.22,
        'Q-X': 0.873, # mean

        'R-R': 0.12,
        'R-S': 0.73,
        'R-T': 1.21,
        'R-V': 1.81,
        'R-W': 1.06,
        'R-Y': 1.91,
        'R-X': 1.0635, # mean

        'S-S': 0.50,
        'S-T': 1.81,
        'S-V': 1.74,
        'S-W': 1.38,
        'S-Y': 0.82,
        'S-X': 1.121, # mean

        'T-T': 1.49,
        'T-V': 3.78,
        'T-W': 1.42,
        'T-Y': 2.12,
        'T-X': 1.820, # mean

        'V-V': 9.31,
        'V-W': 6.57,
        'V-Y': 6.84,
        'V-X': 4.7155,

        'W-W': 2.23,
        'W-Y': 1.99,
        'W-X': 2.498, # mean

        'Y-Y': 1.88,
        'Y-X': 2.4725, # mean

        'X-X': 1.884, # mean
    },
    'anti-parallel': {
        'A-A': 0.79,
        'A-C': 2.04,
        'A-D': 0.77,
        'A-E': 0.78,
        'A-F': 2.98,
        'A-G': 1.33,
        'A-H': 1.36,
        'A-I': 2.93,
        'A-K': 0.80,
        'A-L': 2.10,
        'A-M': 1.49,
        'A-N': 0.83,
        'A-P': 0.42,
        'A-Q': 0.85,
        'A-R': 1.52,
        'A-S': 1.09,
        'A-T': 1.91,
        'A-V': 4.15,
        'A-W': 2.75,
        'A-Y': 3.00,
        'A-X': 1.6945, # mean

        'C-C': 6.45,
        'C-D': 0.41,
        'C-E': 0.77,
        'C-F': 3.52,
        'C-G': 1.40,
        'C-H': 3.00,
        'C-I': 3.39,
        'C-K': 1.39,
        'C-L': 1.87,
        'C-M': 2.09,
        'C-N': 0.63,
        'C-P': 0.72,
        'C-Q': 0.96,
        'C-R': 2.68,
        'C-S': 1.84,
        'C-T': 1.84,
        'C-V': 3.97,
        'C-W': 7.02,
        'C-Y': 4.22,
        'C-X': 2.5104, # mean

        'D-D': 0.27,
        'D-E': 0.79,
        'D-F': 1.17,
        'D-G': 0.94,
        'D-H': 1.86,
        'D-I': 1.24,
        'D-K': 1.60,
        'D-L': 0.69,
        'D-M': 0.82,
        'D-N': 0.90,
        'D-P': 0.33,
        'D-Q': 1.23,
        'D-R': 1.64,
        'D-S': 0.88,
        'D-T': 1.78,
        'D-V': 1.31,
        'D-W': 1.33,
        'D-Y': 1.49,
        'D-X': 1.0725, # mean

        'E-E': 0.55,
        'E-F': 1.39,
        'E-G': 0.77,
        'E-H': 1.30,
        'E-I': 1.96,
        'E-K': 3.02,
        'E-L': 1.11,
        'E-M': 1.04,
        'E-N': 1.31,
        'E-P': 0.54,
        'E-Q': 1.28,
        'E-R': 3.22,
        'E-S': 1.18,
        'E-T': 2.31,
        'E-V': 2.57,
        'E-W': 1.86,
        'E-Y': 2.44,
        'E-X': 1.5095, # mean

        'F-F': 2.60,
        'F-G': 2.35,
        'F-H': 2.58,
        'F-I': 4.92,
        'F-K': 1.76,
        'F-L': 3.72,
        'F-M': 3.66,
        'F-N': 1.26,
        'F-P': 1.37,
        'F-Q': 1.83,
        'F-R': 2.38,
        'F-S': 1.84,
        'F-T': 2.45,
        'F-V': 6.20,
        'F-W': 5.84,
        'F-Y': 5.75,
        'F-X': 2.9785, # mean

        'G-G': 0.61,
        'G-H': 1.59,
        'G-I': 2.05,
        'G-K': 0.64,
        'G-L': 1.41,
        'G-M': 1.62,
        'G-N': 0.99,
        'G-P': 0.54,
        'G-Q': 1.02,
        'G-R': 1.17,
        'G-S': 0.98,
        'G-T': 1.84,
        'G-V': 3.05,
        'G-W': 2.64,
        'G-Y': 2.72,
        'G-X': 1.482, # mean

        'H-H': 1.46,
        'H-I': 1.97,
        'H-K': 1.28,
        'H-L': 1.84,
        'H-M': 1.98,
        'H-N': 1.54,
        'H-P': 0.80,
        'H-Q': 1.22,
        'H-R': 1.80,
        'H-S': 1.68,
        'H-T': 3.20,
        'H-V': 2.84,
        'H-W': 2.46,
        'H-Y': 3.44,
        'H-X': 1.959, # mean

        'I-I': 3.20,
        'I-K': 2.21,
        'I-L': 3.98,
        'I-M': 3.16,
        'I-N': 1.08,
        'I-P': 0.89,
        'I-Q': 1.70,
        'I-R': 2.21,
        'I-S': 1.53,
        'I-T': 2.94,
        'I-V': 7.63,
        'I-W': 4.27,
        'I-Y': 5.36,
        'I-X': 2.931, # mean

        'K-K': 0.79,
        'K-L': 1.08,
        'K-M': 1.02,
        'K-N': 1.07,
        'K-P': 0.32,
        'K-Q': 1.59,
        'K-R': 1.01,
        'K-S': 1.51,
        'K-T': 2.62,
        'K-V': 2.43,
        'K-W': 2.12,
        'K-Y': 3.35,
        'K-X': 1.5805, # mean

        'L-L': 1.53,
        'L-M': 2.48,
        'L-N': 0.80,
        'L-P': 0.80,
        'L-Q': 1.27,
        'L-R': 1.61,
        'L-S': 1.28,
        'L-T': 1.71,
        'L-V': 4.47,
        'L-W': 3.95,
        'L-Y': 3.56,
        'L-X': 2.063, # mean

        'M-M': 1.47,
        'M-N': 0.65,
        'M-P': 0.86,
        'M-Q': 1.59,
        'M-R': 1.40,
        'M-S': 1.10,
        'M-T': 1.83,
        'M-V': 3.50,
        'M-W': 2.56,
        'M-Y': 3.85,
        'M-X': 1.9085, # mean

        'N-N': 0.63,
        'N-P': 0.54,
        'N-Q': 1.57,
        'N-R': 0.92,
        'N-S': 1.28,
        'N-T': 1.83,
        'N-V': 1.57,
        'N-W': 1.59,
        'N-Y': 2.10,
        'N-X': 1.1545, # mean

        'P-P': 0.19,
        'P-Q': 0.34,
        'P-R': 0.66,
        'P-S': 0.54,
        'P-T': 0.95,
        'P-V': 1.11,
        'P-W': 1.30,
        'P-Y': 1.66,
        'P-X': 0.744, # mean

        'Q-Q': 0.76,
        'Q-R': 1.77,
        'Q-S': 1.38,
        'Q-T': 2.67,
        'Q-V': 2.24,
        'Q-W': 2.24,
        'Q-Y': 2.55,
        'Q-X': 1.503, # mean

        'R-R': 0.95,
        'R-S': 1.30,
        'R-T': 2.59,
        'R-V': 3.07,
        'R-W': 3.95,
        'R-Y': 2.92,
        'R-X': 1.9385, # mean

        'S-S': 0.84,
        'S-T': 2.42,
        'S-V': 2.37,
        'S-W': 1.94,
        'S-Y': 2.83,
        'S-X': 1.4905, # mean

        'T-T': 2.39,
        'T-V': 3.83,
        'T-W': 3.51,
        'T-Y': 3.45,
        'T-X': 2.4035, # mean

        'V-V': 4.91,
        'V-W': 5.95,
        'V-Y': 7.39,
        'V-X': 3.728, # mean

        'W-W': 5.17,
        'W-Y': 7.60,
        'W-X': 3.5025, # mean

        'Y-Y': 4.00,
        'Y-X': 3.683, # mean

        'X-X': 2.091845, # mean overall
    }
}


def compute_segmentation_score(group_index, testing_ID, testing_seq, testing_prediction):
    """
    mostly copied from funct compute_segmentation_score in file inter_segment_interaction_20201101.py
    :param records_dir:
    :param small_group_index:
    :param metric:
    :return:
    """
    # NOTE: we assume they are parallel
    segment_info = summarize_segments(group_index, testing_prediction)
    majority_scores = list()
    for one_index, one_segment_info in segment_info.items():
        majority_scores.append(0)

        subsequence_list = dict()
        for segment_name, segment_index in one_segment_info.items():
            if segment_name not in segment_states[group_index]['beta strand']:
                continue
            subsequence = testing_seq[int(one_index)][segment_index[0]:segment_index[1]]
            subsequence_list[segment_name] = subsequence


        for first_segment_name, first_subsequence in subsequence_list.items():
            for second_segment_name, second_subsequence in subsequence_list.items():
                # only consider once
                if first_segment_name >= second_segment_name:
                    continue

                majority_parallel_type, covered_parallel_type = None, None
                if f'{first_segment_name}<=>{second_segment_name}' in segmentation_property_majority[group_index]:
                    majority_parallel_type = segmentation_property_majority[group_index][
                        f'{first_segment_name}<=>{second_segment_name}']

                for first_amino_acid in first_subsequence:
                    for second_amino_acid in second_subsequence:

                        if majority_parallel_type is not None:
                            if '%s-%s' % (first_amino_acid, second_amino_acid) in amino_acid_relative_frequency[majority_parallel_type]:
                                majority_scores[-1] += amino_acid_relative_frequency[majority_parallel_type]['%s-%s' % (first_amino_acid, second_amino_acid)]
                            elif '%s-%s' % (second_amino_acid, first_amino_acid) in amino_acid_relative_frequency[majority_parallel_type]:
                                majority_scores[-1] += amino_acid_relative_frequency[majority_parallel_type]['%s-%s' % (second_amino_acid, first_amino_acid)]
                            else:
                                # X	Any amino acid
                                raise Exception('We do not have %s-%s propensities' % (first_amino_acid, second_amino_acid))

    majority_scores = np.array(majority_scores)

    return majority_scores


def _construct_sequence_segment(seq_info, label_info, ss3_pred, ss8_pred):
    seq_X, seq_Y = list(), list()
    for protein_index, (cur_seq_info, cur_label_info, cur_ss3_pred, cur_ss8_pred) in enumerate(zip(seq_info, label_info, ss3_pred, ss8_pred)):
        """
        partly copied from func feature_representation_hmm from file stacked_HMM_20200610.py
        """
        seq_X.append(list())
        seq_Y.append(list(map(str, map(int, cur_label_info))))

        invalid_symbol = set(cur_seq_info) - set(list(atchley_factors_dict.keys()))
        for one_invalid_symbol in invalid_symbol:
            cur_seq_info = cur_seq_info.replace(one_invalid_symbol, '*')
        seq_atchley_factor = np.array(list(map(atchley_factors_dict.__getitem__, cur_seq_info)))
        seq_one_hot = np.array(list(map(one_hot_dict.__getitem__, cur_seq_info)))

        for index, cur_amino_acid in enumerate(list(cur_seq_info)):
            seq_X[-1].append(dict())
            # seq_X[-1][-1]['index'] = index
            for factor_index in range(5):
                seq_X[-1][-1]['atchley_factor_%d'%factor_index] = seq_atchley_factor[index][factor_index]
            seq_X[-1][-1]['one_hot'] = str(np.argwhere(seq_one_hot[index] == 1).reshape([-1])[0])

            seq_X[-1][-1]['ss3'] = s3_str_mapping[cur_ss3_pred[index]]

            seq_X[-1][-1]['ss8'] = s8_str_mapping[cur_ss8_pred[index]]

    seq_segment = [SequenceStruct(X, Y, seg_other_symbol="0") for X, Y in zip(seq_X, seq_Y)]
    return seq_segment

