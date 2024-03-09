import os
import glob
import random
import shutil  # Needed for copying files

def list_audio_files(base_dir):
    """List all .wav files in the directory and subdirectories."""
    return glob.glob(os.path.join(base_dir, '**', '*.wav'), recursive=True)

def copy_files(audio_files, output_dir):
    """Copy files to the specified output directory."""
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Copy each audio file to the output directory
    for file in audio_files:
        shutil.copy(file, output_dir)

def split_data(audio_files, train_ratio=0.8):
    """Split audio files into training and testing sets."""
    random.shuffle(audio_files)
    split_index = int(len(audio_files) * train_ratio)
    return audio_files[:split_index], audio_files[split_index:]

def main(base_dir, output_dir):
    # List all audio files
    audio_files = list_audio_files(base_dir)
    
    # Split into train and test sets
    train_files, test_files = split_data(audio_files)
    
    # Define train and test directories within the output directory with a '.tsv' extension
    train_dir = os.path.join(output_dir, 'train.tsv')  # Note: This is a folder, not a TSV file
    test_dir = os.path.join(output_dir, 'test.tsv')    # Note: This is a folder, not a TSV file
    
    # Copy train and test sets to their respective directories
    copy_files(train_files, train_dir)
    copy_files(test_files, test_dir)
    print(f"Copied {len(train_files)} train files to {train_dir} and {len(test_files)} test files to {test_dir}.")

if __name__ == "__main__":
    base_dir = '/mundus/data_mundus/librispeech/train-clean-100'
    output_dir = '/mundus/abadawi696/slm_project/slm-100h'  # Update this to your desired output directory
    main(base_dir, output_dir)