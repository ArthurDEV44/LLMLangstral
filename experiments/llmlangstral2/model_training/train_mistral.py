# Copyright (c) 2023 Microsoft
# Licensed under The MIT License [see LICENSE for details]

"""
Training script for MistralForTokenClassification on MeetingBank data.
This creates a Mistral-based LLMLingua-2 compression model.

Usage:
    python train_mistral.py --model_name mistralai/Ministral-3-3B-Instruct-2512
"""

import argparse
import os
import random
import time

import torch
from sklearn.metrics import accuracy_score
from torch import cuda
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import AutoTokenizer, MistralForTokenClassification
from utils import TokenClfDataset

MAX_LEN = 512
MAX_GRAD_NORM = 10

parser = argparse.ArgumentParser(
    description="Train Mistral for compression (by token classification)"
)
parser.add_argument(
    "--model_name",
    help="token classification model",
    default="mistralai/Ministral-3-3B-Instruct-2512",
)
parser.add_argument(
    "--data_path",
    help="training and validation data path",
    default="../../../results/meetingbank/gpt-4-32k_comp/annotation_kept_cs512_meetingbank_train_formated.pt",
)
parser.add_argument(
    "--label_type",
    help="word label or token label",
    default="word_label",
    choices=["word_label", "token_label"],
)
parser.add_argument(
    "--save_path",
    help="save path",
    default="../../../results/models/mistral_3b_meetingbank",
)
parser.add_argument("--lr", help="learning rate", default=1e-5, type=float)
parser.add_argument(
    "--num_epoch", help="number of training epoch", default=10, type=int
)
parser.add_argument("--batch_size", type=int, default=4)  # Smaller batch for Mistral
parser.add_argument(
    "--gradient_accumulation_steps", type=int, default=4, help="Gradient accumulation"
)

args = parser.parse_args()
os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
writer = SummaryWriter(log_dir=os.path.dirname(args.save_path).replace("model", "log"))


def train(epoch):
    tr_loss, tr_accuracy = 0, 0
    nb_tr_examples, nb_tr_steps = 0, 0
    tr_preds, tr_labels = [], []
    model.train()

    for idx, batch in enumerate(train_dataloader):
        t = time.time()
        ids = batch["ids"].to(device, dtype=torch.long)
        mask = batch["mask"].to(device, dtype=torch.long)
        targets = batch["targets"].to(device, dtype=torch.long)

        outputs = model(input_ids=ids, attention_mask=mask, labels=targets)
        loss, tr_logits = outputs.loss, outputs.logits

        # Gradient accumulation
        loss = loss / args.gradient_accumulation_steps
        loss.backward()

        tr_loss += loss.item() * args.gradient_accumulation_steps

        nb_tr_steps += 1
        nb_tr_examples += targets.size(0)

        flattened_targets = targets.view(-1)
        active_logits = tr_logits.view(-1, model.num_labels)
        flattened_predictions = torch.argmax(active_logits, axis=1)
        active_accuracy = mask.view(-1) == 1
        targets = torch.masked_select(flattened_targets, active_accuracy)
        predictions = torch.masked_select(flattened_predictions, active_accuracy)

        tr_preds.extend(predictions)
        tr_labels.extend(targets)

        tmp_tr_accuracy = accuracy_score(
            targets.cpu().numpy(), predictions.cpu().numpy()
        )
        tr_accuracy += tmp_tr_accuracy

        if idx % 100 == 0:
            loss_step = tr_loss / nb_tr_steps
            acc_step = tr_accuracy / nb_tr_steps
            writer.add_scalar(
                "Loss/train", loss_step, idx + epoch * len(train_dataloader)
            )
            writer.add_scalar(
                "Acc/train", acc_step, idx + epoch * len(train_dataloader)
            )
            writer.flush()
            print(f"Training loss per 100 training steps: {loss_step}")

        # Gradient accumulation step
        if (idx + 1) % args.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(
                parameters=model.parameters(), max_norm=MAX_GRAD_NORM
            )
            optimizer.step()
            optimizer.zero_grad()

    tr_loss = tr_loss / nb_tr_steps
    tr_accuracy = tr_accuracy / nb_tr_steps
    print(f"Training loss epoch: {tr_loss}")
    print(f"Training accuracy epoch: {tr_accuracy}")


def test(model, eval_dataloader):
    model.eval()

    eval_loss, eval_accuracy = 0, 0
    nb_eval_examples, nb_eval_steps = 0, 0
    eval_preds, eval_labels = [], []

    with torch.no_grad():
        for idx, batch in enumerate(eval_dataloader):
            ids = batch["ids"].to(device, dtype=torch.long)
            mask = batch["mask"].to(device, dtype=torch.long)
            targets = batch["targets"].to(device, dtype=torch.long)

            outputs = model(input_ids=ids, attention_mask=mask, labels=targets)
            loss, eval_logits = outputs.loss, outputs.logits

            eval_loss += loss.item()

            nb_eval_steps += 1
            nb_eval_examples += targets.size(0)

            flattened_targets = targets.view(-1)
            active_logits = eval_logits.view(-1, model.num_labels)
            flattened_predictions = torch.argmax(active_logits, axis=1)
            active_accuracy = mask.view(-1) == 1
            targets = torch.masked_select(flattened_targets, active_accuracy)
            predictions = torch.masked_select(flattened_predictions, active_accuracy)

            eval_labels.extend(targets)
            eval_preds.extend(predictions)

            tmp_eval_accuracy = accuracy_score(
                targets.cpu().numpy(), predictions.cpu().numpy()
            )
            eval_accuracy += tmp_eval_accuracy

    labels = [label.item() for label in eval_labels]
    predictions = [pred.item() for pred in eval_preds]

    eval_loss = eval_loss / nb_eval_steps
    eval_accuracy = eval_accuracy / nb_eval_steps
    print(f"Validation Loss: {eval_loss}")
    print(f"Validation Accuracy: {eval_accuracy}")

    writer.add_scalar("Loss/eval", eval_loss, epoch * len(eval_dataloader))
    writer.add_scalar("Acc/eval", eval_accuracy, epoch * len(eval_dataloader))
    writer.flush()

    return eval_accuracy


device = "cuda" if cuda.is_available() else "cpu"
print(f"Using device: {device}")

data = torch.load(args.data_path)

# Load tokenizer with Mistral-specific configuration
tokenizer = AutoTokenizer.from_pretrained(args.model_name)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

# Load MistralForTokenClassification
print(f"Loading model: {args.model_name}")
model = MistralForTokenClassification.from_pretrained(
    args.model_name,
    num_labels=2,
    torch_dtype=torch.bfloat16,
    ignore_mismatched_sizes=True,
)
model.to(device)
print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters()):,}")

assert len(data["origin"]) == len(data["labels"])
text_label = [(text, label) for text, label in zip(data["origin"], data["labels"])]
random.shuffle(text_label)
train_data = text_label[: int(len(text_label) * 0.8)]
val_data = text_label[int(len(text_label) * 0.8) :]

train_text = [text for text, label in train_data]
train_label = [label for text, label in train_data]
val_text = [text for text, label in val_data]
val_label = [label for text, label in val_data]

train_dataset = TokenClfDataset(
    train_text, train_label, MAX_LEN, tokenizer=tokenizer, model_name=args.model_name
)
val_dataset = TokenClfDataset(
    val_text, val_label, MAX_LEN, tokenizer=tokenizer, model_name=args.model_name
)

print(f"len training set: {len(train_dataset)}, len validation set: {len(val_dataset)}")
print(train_dataset[0])
for token, label in zip(
    tokenizer.convert_ids_to_tokens(train_dataset[0]["ids"][:30]),
    train_dataset[0]["targets"][:30],
):
    print("{0:10}  {1}".format(token, label.item()))
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

ids = train_dataset[0]["ids"].unsqueeze(0)
mask = train_dataset[0]["mask"].unsqueeze(0)
targets = train_dataset[0]["targets"].unsqueeze(0)
ids = ids.to(device)
mask = mask.to(device)
targets = targets.to(device)
outputs = model(input_ids=ids, attention_mask=mask, labels=targets)
initial_loss = outputs[0]
print(f"Initial loss: {initial_loss}")

tr_logits = outputs[1]
print(f"Logits shape: {tr_logits.shape}")

optimizer = torch.optim.AdamW(params=model.parameters(), lr=args.lr, weight_decay=0.01)

best_acc = 0

for epoch in tqdm(range(args.num_epoch)):
    print(f"Training epoch: {epoch + 1}")
    train(epoch)
    acc = test(model, val_dataloader)
    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), f"{args.save_path}/state_dict.pth")
        model.save_pretrained(args.save_path)
        tokenizer.save_pretrained(args.save_path)
        print(f"New best model saved with accuracy: {best_acc:.4f}")

print(f"Training complete. Best validation accuracy: {best_acc:.4f}")
