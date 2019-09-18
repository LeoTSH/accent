from utils import process_raw, generate_input, get_max_len

# Load raw data
with open('../data/raw/raw_dev.txt', 'r') as f:
    raw = f.readlines()[:10]

# Generate target data
target = process_raw(raw)

# Generate input data
inputs = [generate_input(seq) for seq in target]

# Check for maximum sequence length ()
seq_length = get_max_len(inputs)