import os
import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from tqdm import tqdm

# Initialize the processor and model
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-large-960h")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-960h")

def process_audio_file(audio_file_path):
    # Load the audio file
    waveform, sample_rate = torchaudio.load(audio_file_path)
    
    # Process the waveform through the model
    input_values = processor(waveform, return_tensors="pt", sampling_rate=sample_rate).input_values
    with torch.no_grad():
        outputs = model(input_values, output_hidden_states=True)
        # Extract embeddings from the 16th layer (index 15)
        hidden_states = outputs.hidden_states[16].squeeze().numpy()  # Access the 16th layer
    
    return hidden_states

def process_directory_and_save(source_dir, target_file):
    embeddings = {}
    for root, dirs, files in os.walk(source_dir):
        # Wrap the files iterator with tqdm to display a progress bar
        for file in tqdm(files, desc="Processing audio files"):
            if file.endswith('.wav'):
                audio_file_path = os.path.join(root, file)
                embedding = process_audio_file(audio_file_path)
                # Use a relative path as key to save space and maintain uniqueness
                relative_path = os.path.relpath(audio_file_path, source_dir)
                embeddings[relative_path] = embedding
    
    # Save all embeddings in one .npz file
    np.savez_compressed(target_file, **embeddings)

# Base directory containing the audio files
source_base_dir = "/mundus/data_mundus/librispeech/train-clean-100/"
# File path where the embeddings will be saved
target_file_path = "/mundus/abadawi696/slm_project/all_embeddings.npz"

process_directory_and_save(source_base_dir, target_file_path)
