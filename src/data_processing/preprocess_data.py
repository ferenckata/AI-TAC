import numpy as np
import json
import os
import _pickle as pickle
import sys

import preprocess_utils

#input files:
data_file = sys.argv[1] #bed file with peak locations
intensity_file = sys.argv[2] #text file with normalized peak heights
reference_genome = sys.argv[3] #directory with reference genome fasta files 
species =   sys.argv[4] # human or mouse

if species == "human":
    output_directory = "../human_data/"
    num_chr = 23
elif species == "mouse":
    output_directory = "../data/"
    num_chr = 20
else:
    raise Exception("Invalid species")

directory = os.path.dirname(output_directory)
if not os.path.exists(directory):
    os.makedirs(directory)

# read bed file with peak positions
# keep only entries with valid activity vectors - where is it in the code?
positions = preprocess_utils.read_bed(data_file)

# read reference genome fasta file into dictionary
if not os.path.exists(os.path.join(output_directory,'chr_dict.pickle')):
    chr_dict = preprocess_utils.read_fasta(reference_genome, num_chr)
    pickle.dump(chr_dict, open(os.path.join(output_directory,'chr_dict.pickle'), "wb"))
else:
    chr_dict = pickle.load(open(os.path.join(output_directory,'chr_dict.pickle'), "rb"))

one_hot_seqs, peak_seqs, invalid_ids, sequence_peak_names = preprocess_utils.get_sequences(positions, 
                                                                                           chr_dict, 
                                                                                           num_chr)

# read in all intensity values and peak names
cell_type_array, intensity_peak_names = preprocess_utils.format_intensities(intensity_file)
cell_type_array = cell_type_array.astype(np.float32)

# take one_hot encoding of valid sequences of only those peaks that 
# have associated intensity values in cell_type_array
valid_peak_ids = np.intersect1d(sequence_peak_names, intensity_peak_names)

sequence_peak_names, one_hot_seqs, peak_seqs = preprocess_utils.filter_matrix(sequence_peak_names,
                                                                              valid_peak_ids,
                                                                              one_hot_seqs,
                                                                              peak_seqs
                                                                              )

intensity_peak_names, cell_type_array = preprocess_utils.filter_matrix(intensity_peak_names,
                                                                       valid_peak_ids,
                                                                       cell_type_array
                                                                       )

# throw error here, add test for it
if np.sum(sequence_peak_names != intensity_peak_names) > 0:
    print("Order of peaks not matching for sequences/intensities!")

# write to file
np.save(os.path.join(output_directory,'one_hot_seqs.npy'), one_hot_seqs)
np.save(os.path.join(output_directory,'peak_names.npy'), sequence_peak_names)
np.save(os.path.join(output_directory,'peak_seqs.npy'), peak_seqs)

with open(os.path.join(output_directory,'invalid_ids.txt'), 'w') as f:
    f.write(json.dumps(invalid_ids))
f.close()

# write fasta file
with open(os.path.join(output_directory,'sequences.fasta'), 'w') as f:
    for i in range(peak_seqs.shape[0]):
        f.write('>' + sequence_peak_names[i] + '\n')
        f.write (peak_seqs[i] + '\n')
f.close()

np.save(os.path.join(output_directory,'cell_type_array.npy'), cell_type_array)