import argparse
import logging
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.utils.data
import tqdm

import model as mdl
from Utilities import util
import train as test 

N_SAMPLES = 3
SEQ_LEN = 256
MAX_SEQ_LEN = 15000 
MAX_BEAT = 2048
BATCH_SIZE = 4
MODEL_STEPS = None
TEMPERATURE = 1.0
FILTER = "top_a"
FILTER_THRESHOLD = 0.9
DEVICE = torch.device("cpu")

def save_result(filename, data, sample_dir):

    np.save(sample_dir / "numpy" / f"{filename}.npy", data)

    np.savetxt(sample_dir / "numpy_txt" / f"{filename}.txt", data, fmt = "%d", delimiter='\t')


def main():
    test_names = pathlib.Path(f"test_names.txt")

    test_dataset_path = pathlib.Path(f"test_dataset")
    sample_dir = pathlib.Path(f"results")

    util.write_file_names_to_txt('D:\preprocess-2.0/test_dataset', 'test_names.txt')

    model_dir = pathlib.Path(f"models")

    test_dataset = test.MusicDataset(
        test_names,
        test_dataset_path,
        max_seq_len=MAX_SEQ_LEN,
        max_beat=MAX_BEAT,
        device = DEVICE
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        collate_fn=test.MusicDataset.collate,
    )

    logging.info(f"Creating the model...")
    model = mdl.MusicXTransformer(
        dim=512,
        depth=4,
        heads=6,
        max_seq_len=MAX_SEQ_LEN,
        max_beat=MAX_BEAT,
        rotary_pos_emb=False,
        use_abs_pos_emb=True, 
        emb_dropout=0.2,
        attn_dropout=0.2,
        ff_dropout=0.2,
    ).to(DEVICE)

    checkpoint_dir = model_dir / "checkpoints"
    if MODEL_STEPS is None:
        checkpoint_filename = checkpoint_dir / "best_model.pt"
    else:
        checkpoint_filename = checkpoint_dir / f"model_{MODEL_STEPS}.pt"

    model.load_state_dict(torch.load(checkpoint_filename, map_location=DEVICE))
    print(f"Loaded the model weights from: {checkpoint_filename}")
    model.eval()

    sos = 0
    eos = 2
    beat_0 = 1
    beat_4 = 5
    beat_16 = 17

    with torch.no_grad():
        data_iter = iter(test_loader)
        for i in tqdm.tqdm(range(N_SAMPLES), ncols=80):
            batch = next(data_iter)

            truth_np = batch["seq"][0].cpu().numpy()
            save_result(f"{i}_truth", truth_np, sample_dir)

            # Unconditioned generation ->

            # Output start tokens
            tgt_start = torch.zeros((1, 1, 8), dtype=torch.long, device=DEVICE)
            tgt_start[:, 0, 0] = sos

            generated = model.generate(
                tgt_start,
                SEQ_LEN,
                eos_token=eos,
                temperature=TEMPERATURE,
                filter_logits_fn=FILTER,
                filter_thres=FILTER_THRESHOLD,
                monotonicity_dim=("type", "beat"),
            )
            generated_np = torch.cat((tgt_start, generated), 1).cpu().numpy()

            save_result(f"{i}_unconditioned", generated_np[0], sample_dir)

            # Instrument-informed generation ->

            # Output start tokens
            prefix_len = int(np.argmax(batch["seq"][0, :, 1].cpu() >= beat_0))
            tgt_start = batch["seq"][:1, :prefix_len].to(DEVICE)

            generated = model.generate(
                tgt_start,
                SEQ_LEN,
                eos_token=eos,
                temperature=TEMPERATURE,
                filter_logits_fn=FILTER,
                filter_thres=FILTER_THRESHOLD,
                monotonicity_dim=("type", "beat"),
            )
            generated_np = torch.cat((tgt_start, generated), 1).cpu().numpy()

            save_result(f"{i}_instrument-informed", generated_np[0], sample_dir)

            # 4-beat continuation ->

            # Output start tokens
            cond_len = int(np.argmax(batch["seq"][0, :, 1].cpu() >= beat_4))
            tgt_start = batch["seq"][:1, :cond_len].to(DEVICE)

            generated = model.generate(
                tgt_start,
                SEQ_LEN,
                eos_token=eos,
                temperature=TEMPERATURE,
                filter_logits_fn=FILTER,
                filter_thres=FILTER_THRESHOLD,
                monotonicity_dim=("type", "beat"),
            )
            generated_np = torch.cat((tgt_start, generated), 1).cpu().numpy()

            save_result(f"{i}_4-beat-continuation", generated_np[0], sample_dir)

            # 16-beat continuation ->

            # Output start tokens
            cond_len = int(np.argmax(batch["seq"][0, :, 1].cpu() >= beat_16))
            tgt_start = batch["seq"][:1, :cond_len].to(DEVICE)

            generated = model.generate(
                tgt_start,
                SEQ_LEN,
                eos_token=eos,
                temperature=TEMPERATURE,
                filter_logits_fn=FILTER,
                filter_thres=FILTER_THRESHOLD,
                monotonicity_dim=("type", "beat"),
            )
            generated_np = torch.cat((tgt_start, generated), 1).cpu().numpy()

            save_result(f"{i}_16-beat-continuation", generated_np[0], sample_dir)

if __name__ == "__main__":
    main()
