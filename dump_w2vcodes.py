import torch
import torchaudio
import os
import argparse
from shutil import copyfile
import tqdm
import torch.nn as nn
from torch.nn import functional as F
#from model import SemanticEncoder
import sys
import pickle
import numpy as np
import joblib
#from utils import get_shard_range
from transformers import AutoFeatureExtractor, Wav2Vec2BertModel

    
class SemanticEncoder_W2VBERT(nn.Module):
    def __init__(self, layer, km_path):
        super().__init__()
        
        # Feature extractor
        self.processor = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
        self.model = Wav2Vec2BertModel.from_pretrained("facebook/w2v-bert-2.0")
        self.model.to("cuda" if torch.cuda.is_available() else "cpu")  
        self.layer = layer
        #self.max_chunk = max_chunk
        
        # Quantizer
        km_model = joblib.load(km_path)
        self.C_np = km_model.cluster_centers_.transpose()
        self.Cnorm_np = (self.C_np ** 2).sum(0, keepdims=True)
        self.register_buffer("C", torch.from_numpy(self.C_np))
        self.register_buffer("Cnorm", torch.from_numpy(self.Cnorm_np))

    def wav2code(self, x, sr):
        
        inputs = self.processor(x.squeeze(0).cpu().numpy(), sampling_rate=sr, return_tensors="pt").to(self.model.device)

        # Generate embeddings
        with torch.no_grad():
           outputs = self.model(**inputs, output_hidden_states=True)
           
        feat = []

        # Extract embeddings and store them in the dictionary
        embeddings = outputs.hidden_states[self.layer].squeeze().cpu().numpy()
        embeddings = torch.from_numpy(embeddings)
        feat.append(embeddings)
            
        feat = torch.cat(feat, 1).squeeze(0)
        dist = (
            feat.pow(2).sum(1, keepdim=True)
            - 2 * torch.matmul(feat, self.C.to(feat.device))
            + self.Cnorm_np
        )
        return dist.argmin(dim=1).unsqueeze(0)    

def get_parser():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("data", help="Path with .tsv files pointing to the audio data")
    parser.add_argument("--split", help="Which split", required=True)
    #parser.add_argument("--checkpoint", help="Path to the directory containing the encodec checkpoint", required=True)
    parser.add_argument("--layer", type=int, default=16)
    parser.add_argument("--km_path", help="Path to the directory containing the kmeans checkpoint", required=True)
    parser.add_argument("--save-dir", help="Output path to store the features", required=True)
    parser.add_argument("--nshard", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    #parser.add_argument("--max_chunk", type=int, default=1600000)
    return parser


def get_iterator(args, mdl, device, save_path, eos):
    if os.path.exists(os.path.join(args.data, args.split) + ".tsv"):
        split = args.split
        with open(os.path.join(args.data, args.split) + ".tsv", "r") as fp:
            lines = fp.read().split("\n")
        root = lines.pop(0).strip()
        #start, end = get_shard_range(len(lines), args.nshard, args.rank)
        #lines = lines[start:end]
        files = [os.path.join(root, line.split("\t")[0]) for line in lines if len(line) > 0]
        num = len(files)
        def process_file(fname):
            wav, sr = torchaudio.load(fname)
            # Extract speech tokens
            codes = mdl.wav2code(wav.to(device), sr)
            return codes.squeeze(0).cpu()

        def iterate():
            # Determine how many lines have already been processed
            num_existing_lines = 0
            if os.path.exists(save_path + ".bin") and os.path.getsize(save_path + ".bin") > 0:
                num_existing_lines = len(np.memmap(save_path + ".len", dtype=np.uint16, mode='r'))
                print(f"[Rank {args.rank}] Skipping first {num_existing_lines} already encoded files ...")
            # Calculate the total number of lines to process (including existing lines)
            pbar = tqdm.tqdm(total=num)
            pbar.update(num_existing_lines)
            if num_existing_lines == len(files):
                print(f"[Rank {args.rank}] All files already processed!")
                sys.exit(0)
            for file in files[num_existing_lines:]:
                codes = process_file(file)
                yield codes
                pbar.update(1)
    else:
        # assume we're dealing with a HF dataset
        from datasets import load_dataset
        release = None
        if len(args.split.split("/")) == 2:
            release, split = args.split.split("/")
        elif len(args.split.split("/")) == 1:
            split = args.split
        if release is None:
            dataset = load_dataset(args.data)[split]
        else:
            dataset = load_dataset(args.data, release)[split]
        num = len(dataset)
        def process_wav(wav):
            # Extract speech tokens
            codes = mdl.wav2code(wav.to(device))
            return codes.squeeze(0).cpu()
        def iterate():
            for sample in tqdm.tqdm(dataset):
                wav = torch.from_numpy(sample['audio']['array'].astype(np.float32))
                codes = process_wav(wav)
                yield codes
    return iterate, num, save_path.replace(args.split, split)

def main():
    parser = get_parser()
    args = parser.parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    save_path = os.path.join(args.save_dir, args.split)
    if os.path.exists(os.path.join(args.data, args.split) + ".tsv"):
        copyfile(os.path.join(args.data, args.split) + ".tsv", save_path + ".tsv")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    model = SemanticEncoder_W2VBERT(args.layer, args.km_path)
    model.to(device)
    model.eval()
    n_codes = model.C.size(1)
    eos = n_codes
    save_path = save_path + f"_{args.rank}_{args.nshard}"
    generator, num, save_path = get_iterator(args, model, device, save_path, eos)
    iterator = generator()
    with open(save_path + ".bin", "ab") as bin_data_f,\
         open(save_path + ".dur", "ab") as dur_f,\
         open(save_path + ".len", "ab") as len_f:
        for codes in iterator:
            codes = torch.cat((codes, torch.Tensor([eos]))) # Append end of sentence token
            codes, duration = torch.unique_consecutive(codes, return_counts=True) # Remove adjacent duplicates
            codes = codes.numpy().astype(np.uint16)
            duration = duration.numpy().astype(np.uint8)
            length = np.array([len(codes)], dtype=np.uint16)
            bin_data_f.write(codes.tobytes())
            dur_f.write(duration.tobytes())
            len_f.write(length.tobytes())
    meta = {
        "vocab_size": n_codes + 2, # Accounting for eos and pad tokens
        "eos": n_codes,
        "pad": n_codes + 1
    }
    if args.rank == 0:
        with open(os.path.join(args.save_dir, 'meta.pkl'), 'wb') as f:
            pickle.dump(meta, f)
    del model


if __name__ == "__main__":
    # import ptvsd
    # ptvsd.enable_attach(('0.0.0.0', 7310))
    # print("Attach debugger now")
    # ptvsd.wait_for_attach()
    main()
