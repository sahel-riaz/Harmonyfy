import pathlib
import numpy as np
import torch
import torch.utils.data
import tqdm
import shutil
import model as mdl
from Utilities import util


LEARNING_RATE = 0.0005
LR_WARMUP_STEPS = 500
LR_DECAY_STEPS = 1000
LR_DECAY_MULTIPLIER = 0.1
STEPS = 2000
EARLY_STOPPING = True
GRAD_NORM_CLIP = 1.0
VALID_STEPS = 1000
MAX_SEQ_LEN = 15000 
MAX_BEAT = 2048
BATCH_SIZE = 4
DEVICE = torch.device("cuda")
EARLY_STOPPING = True
EARLY_STOPPING_TOLERANCE = 20

def get_mask(data):
    max_seq_len = max(len(sample) for sample in data)
    mask = torch.zeros((len(data), max_seq_len), dtype=torch.bool)
    for i, seq in enumerate(data):
        mask[i, : len(seq)] = 1
    mask = mask.to(DEVICE)
    return mask

def pad(data, maxlen=None):
    if maxlen is None:
        max_len = max(len(x) for x in data)
    else:
        for x in data:
            assert len(x) <= max_len
    if data[0].ndim == 1:
        padded = [np.pad(x, (0, max_len - len(x))) for x in data]
    elif data[0].ndim == 2:
        padded = [np.pad(x, ((0, max_len - len(x)), (0, 0))) for x in data]
    else:
        raise ValueError("Got 3D data.")
    return np.stack(padded)

class MusicDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        filename,
        data_dir,
        max_seq_len=None,
        max_beat=None,
        use_csv=False,
        use_augmentation=False,
        device='cuda'
    ):
        super().__init__()
        self.data_dir = pathlib.Path(data_dir)
        with open(filename) as f:
            self.names = [line.strip() for line in f if line]
        self.max_seq_len = max_seq_len
        self.max_beat = max_beat
        self.use_csv = use_csv
        self.use_augmentation = use_augmentation

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        notes = np.load(self.data_dir / f"{name}.npy")

        # Data augmentation
        if self.use_augmentation:
            # Randomly select a starting beat
            n_beats = notes[-1, 0] + 1
            if n_beats > self.max_beat:
                trial = 0
                # Avoid section with too few notes
                while trial < 10:
                    start_beat = np.random.randint(n_beats - self.max_beat)
                    end_beat = start_beat + self.max_beat
                    sliced_notes = notes[
                        (notes[:, 0] >= start_beat) & (notes[:, 0] < end_beat)
                    ]
                    if len(sliced_notes) > 10:
                        break
                    trial += 1
                sliced_notes[:, 0] = sliced_notes[:, 0] - start_beat
                notes = sliced_notes

        # Trim sequence to max_beat
        elif self.max_beat is not None:
            n_beats = notes[-1, 0] + 1
            if n_beats > self.max_beat:
                notes = notes[notes[:, 0] < self.max_beat]

        # Trim sequence to max_seq_len
        if self.max_seq_len is not None and len(notes) > self.max_seq_len:
            notes = np.concatenate((notes[: self.max_seq_len - 1], notes[-1:]))

        return {"name": name, "seq": notes}

    @classmethod
    def collate(cls, data):
        seq = [sample["seq"] for sample in data]
        return {
            "name": [sample["name"] for sample in data],
            "seq": torch.tensor(pad(seq), dtype=torch.long).to(DEVICE),
            "seq_len": torch.tensor([len(s) for s in seq], dtype=torch.long).to(DEVICE),
            "mask": get_mask(seq),
        }
    
def get_lr_multiplier(step, warmup_steps, decay_end_steps, decay_end_multiplier):
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    if step > decay_end_steps:
        return decay_end_multiplier
    position = (step - warmup_steps) / (decay_end_steps - warmup_steps)
    return 1 - (1 - decay_end_multiplier) * position

   
def main():
    train_names = pathlib.Path(f"train_names.txt")
    val_names = pathlib.Path(f"val_names.txt")

    train_dataset_path = pathlib.Path(f"train_dataset")
    val_dataset_path = pathlib.Path(f"validation_dataset")
    models_path = pathlib.Path(f"models")

    util.write_file_names_to_txt('D:\preprocess-2.0//train_dataset', 'train_names.txt')
    util.write_file_names_to_txt('D:\preprocess-2.0//validation_dataset', 'val_names.txt')

    dataset = MusicDataset(
        train_names,
        train_dataset_path,
        max_seq_len=MAX_SEQ_LEN,
        max_beat=MAX_BEAT,
        use_augmentation=True,
        device = DEVICE
    )
        
    data_loader = torch.utils.data.DataLoader(
        dataset, 
        batch_size = BATCH_SIZE, 
        shuffle = True, 
        collate_fn=MusicDataset.collate
    )
    
    valid_dataset = MusicDataset(
        val_names,
        val_dataset_path,
        max_seq_len=MAX_SEQ_LEN,
        max_beat=MAX_BEAT,
        device = DEVICE
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        BATCH_SIZE,
        num_workers=4,
        collate_fn=MusicDataset.collate,
    )

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

    optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: get_lr_multiplier(
            step,
            LR_WARMUP_STEPS,
            LR_DECAY_STEPS,
            LR_DECAY_MULTIPLIER,
        ),
    )

    step = 0
    min_val_loss = float("inf")
    if EARLY_STOPPING:
        count_early_stopping = 0
    
    train_iterator = iter(data_loader)
    while step < STEPS:
        print(f"Training...")
        model.train()
        recent_losses = []
        print("step:", step, STEPS)
        for batch in (pbar := tqdm.tqdm(range(VALID_STEPS), ncols=80)):
            try:
                batch = next(train_iterator)
            except StopIteration:

                train_iterator = iter(data_loader)
                batch = next(train_iterator)

            seq = batch["seq"].to(DEVICE)
            mask = batch["mask"].to(DEVICE)

            optimizer.zero_grad()
            loss = model(seq, mask=mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_NORM_CLIP)
            optimizer.step()
            scheduler.step()

            # Average loss
            recent_losses.append(float(loss))
            if len(recent_losses) > 10:
                del recent_losses[0]
            train_loss = np.mean(recent_losses)
            pbar.set_postfix(loss=f"{train_loss:8.4f}")

            step += 1

        del seq, mask

        # Validation
        print(f"Validating...")
        model.eval()
        with torch.no_grad():
            total_loss = 0
            total_losses = [0] * 8
            count = 0
            for batch in valid_loader:
                # Get input and output pair
                seq = batch["seq"].to(DEVICE)
                mask = batch["mask"].to(DEVICE)

                # Pass through the model
                loss, losses = model(seq, return_list=True, mask=mask)
            
                # Accumulate validation loss
                count += len(batch)
                total_loss += len(batch) * float(loss)

                for idx in range(8):
                    total_losses[idx] += float(losses[idx])
        val_loss = total_loss / count
        individual_losses = [l / count for l in total_losses]

        print(f"Validation loss: {val_loss:.4f}")
        print(f"Individual losses: \ntype={individual_losses[0]:.4f}, "
            f"\n beat: {individual_losses[1]:.4f}, "
            f"\n position: {individual_losses[2]:.4f}, "
            f"\n pitch: {individual_losses[3]:.4f}, "
            f"\n velocity: {individual_losses[4]:.4f}, "
            f"\n duration: {individual_losses[5]:.4f}, "
            f"\n instrument: {individual_losses[6]:.4f}, "
            f"\n section: {individual_losses[7]:.4f}"
        )   

        del seq, mask

        
        # Save the model
        checkpoint_filename = models_path / "checkpoints" / f"model_{step}.pt"
        torch.save(model.state_dict(), checkpoint_filename)
        print(f"Saved the model to: {checkpoint_filename}")

        # Copy the model if it is the best model so far
        if val_loss < min_val_loss:
            min_val_loss = val_loss
            shutil.copyfile(
                checkpoint_filename,
                models_path / "checkpoints" / "best_model.pt",
            )
            # Reset the early stopping counter if we found a better model
            if EARLY_STOPPING:
                count_early_stopping = 0
        elif EARLY_STOPPING:
            # Increment the early stopping counter if no improvement is found
            count_early_stopping += 1

        # Early stopping
        if (
            EARLY_STOPPING
            and count_early_stopping > EARLY_STOPPING_TOLERANCE
        ):
            print(
                "Stopped the training for no improvements in "
                f"{EARLY_STOPPING_TOLERANCE} rounds."
            )
            break

    # Log minimum validation loss
    print(f"Minimum validation loss achieved: {min_val_loss}")

    # Save the optimizer states
    optimizer_filename = models_path / "checkpoints" / f"optimizer_{step}.pt"
    torch.save(optimizer.state_dict(), optimizer_filename)
    print(f"Saved the optimizer state to: {optimizer_filename}")

    # Save the scheduler states
    scheduler_filename = models_path / "checkpoints" / f"scheduler_{step}.pt"
    torch.save(scheduler.state_dict(), scheduler_filename)
    print(f"Saved the scheduler state to: {scheduler_filename}")


if __name__ == "__main__":
    main()
