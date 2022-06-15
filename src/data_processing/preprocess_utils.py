from Bio import SeqIO
from collections import defaultdict
import numpy as np

# read names and postions from bed file
def read_bed(filename):
    positions = defaultdict(list)
    with open(filename) as f:
        for line in f:
            chr, start, stop, name = line.split()
            positions[name].append((chr, int(start), int(stop)))

    return positions

# parse fasta file and turn into dictionary
def read_fasta(genome_dir, num_chr):
    chr_dict = dict()
    for chr in range(1, num_chr):
        # what happend when the file does not exist?
        chr_file_path = genome_dir + "chr{}.fa".format(chr)
        # in case memory becomes an issue, use Bio.SeqIO.index() instead
        chr_dict.update(SeqIO.to_dict(SeqIO.parse(open(chr_file_path), 'fasta')))
    
    return chr_dict

# takes DNA sequence, outputs one-hot-encoded matrix with rows A, T, G, C
def one_hot_encoder(sequence):
    l = len(sequence)
    x = np.zeros((4,l),dtype = 'int8')
    for j, i in enumerate(sequence):
        if i == "a":
            x[0][j] = 1
        elif i == "t":
            x[1][j] = 1
        elif i == "g":
            x[2][j] = 1
        elif i == "c":
            x[3][j] = 1
        else:
            return "contains_N"
    
    return x


# get sequences for peaks from reference genome
def get_sequences(positions, chr_dict, num_chr):
    one_hot_seqs = []
    peak_seqs = []
    invalid_ids = []
    peak_names = []

    target_chr = ['chr{}'.format(i) for i in range(1, num_chr)]

    for name in positions:
        for (chr, start, stop) in positions[name]:
            # somewhat unnecessary here as it would probably throw an error when reading the file in
            # todo: check it once above
            if chr in target_chr: 
                chr_seq = chr_dict[chr].seq
                peak_seq = str(chr_seq)[start - 1:stop].lower()
                # already lowered the character, no need to check for uppercase letter in the encoding step
                one_hot_seq = one_hot_encoder(peak_seq)

                if isinstance(one_hot_seq, np.ndarray):  # it is valid sequence
                    peak_names.append(name)
                    peak_seqs.append(peak_seq)
                    one_hot_seqs.append(one_hot_seq)
                else:
                    # first 20 characters "ImmGenATAC1219.peak_" are skipped, only the peak number is kept
                    # somewhat unnecessary it seems
                    invalid_ids.append(name[20:]) 
            else:
                invalid_ids.append(name[20:])

    one_hot_seqs = np.stack(one_hot_seqs)
    peak_seqs = np.stack(peak_seqs)
    peak_names = np.stack(peak_names)

    return one_hot_seqs, peak_seqs, invalid_ids, peak_names


def format_intensities(intensity_file, invalid_ids):
    cell_type_array = []
    peak_names = []
    with open(intensity_file) as f:
        for i, line in enumerate(f):
            # skip first line of IDs
            if i == 0:
                continue
            columns = line.split()
            peak_name = columns[0]
            # read lines until the EOF is read 
            if '\x1a' not in columns:
                # check if the ID is invalid
                cell_id = columns[0][20:]
                if cell_id not in invalid_ids:
                    cell_act = columns[1:] # removes peak ID
                    cell_type_array.append(cell_act)
                    peak_names.append(peak_name)

    cell_type_array = np.stack(cell_type_array)
    peak_names = np.stack(peak_names)

    return cell_type_array, peak_names



