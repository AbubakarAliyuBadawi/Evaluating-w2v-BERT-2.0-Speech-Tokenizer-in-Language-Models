import os

def read_tsv(file_path):
    """Read a .tsv file and return the lines."""
    with open(file_path, 'r') as file:
        lines = file.readlines()
    return lines

def write_to_tsv_chunks(lines, output_dir, base_filename='train', num_chunks=10):
    """Write lines to multiple .tsv files, split into chunks."""
    chunk_size = len(lines) // num_chunks
    for i in range(num_chunks):
        start_index = i * chunk_size
        # For the last chunk, include any remaining lines
        end_index = (i + 1) * chunk_size if i < num_chunks - 1 else len(lines)
        chunk_lines = lines[start_index:end_index]
        chunk_file_path = os.path.join(output_dir, f'{base_filename}{i+1}.tsv')
        with open(chunk_file_path, 'w') as file:
            file.writelines(chunk_lines)
    print(f"Split {len(lines)} lines into {num_chunks} parts, saved in {output_dir}")

def main(input_tsv_path, output_dir):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Read the existing train.tsv
    lines = read_tsv(input_tsv_path)

    # Check if the actual line count matches the expected count
    actual_count = len(lines)
    expected_count = 3292213
    if actual_count != expected_count:
        print(f"Warning: The actual line count ({actual_count}) does not match the expected count ({expected_count}). Proceeding with the actual count.")

    # Split the lines and write to new .tsv files
    write_to_tsv_chunks(lines, output_dir)

if __name__ == "__main__":
    input_tsv_path = '/mundus/abadawi696/slm_project/slm-60k/train.tsv'
    output_dir = '/mundus/abadawi696/slm_project/slm-60k/train-splits'
    main(input_tsv_path, output_dir)
