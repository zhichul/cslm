import os
from collections import OrderedDict, defaultdict
from datetime import timezone, datetime, timedelta

import orjson
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from cslm.training.utils import sample_indefinitely, NumpyWeightedRandomSampler
from transformers.utils import logging
import torch
import tqdm

from cslm.utils import seq_norm

logger = logging.get_logger(__name__)


class Trainer:

    def __init__(self,
                    model=None,
                    args=None,
                    train_dataset=None,
                    data_collator=None,
                    optimizer=None,
                    lr_scheduler=None,
                    train_dataset_weights=None,
                    train_dataset_lengths=None,
                    evaluations=None):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.data_collator = data_collator
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataset_weights = train_dataset_weights
        self.train_dataset_lengths = train_dataset_lengths
        self.evaluations = evaluations if evaluations is not None else dict()
        self.tensorboard = SummaryWriter(os.path.join(self.args.logging_dir,
                                                      datetime.now(timezone(timedelta(hours=-4))).strftime("%Y-%m-%d %H:%M:%S")))

    def get_train_dataloader(self):
        train_sampler = self._get_train_sampler()

        return DataLoader(
            self.train_dataset,
            batch_size=self.args.per_device_train_batch_size,
            sampler=train_sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
        )

    def _get_train_sampler(self):
        if len(self.train_dataset_weights) == 1:
            assert self.train_dataset_weights == [1.0]
            return RandomSampler(self.train_dataset)
        else:
            weights = [weight for length, weight in zip(self.train_dataset_lengths, self.train_dataset_weights)
                              for _ in range(length)]
            return (
                NumpyWeightedRandomSampler(weights, len(weights))
            )

    def training_step(self, model, inputs):
        raise NotImplementedError

    def train(self):
        model = self.model
        model.to(self.args.device)
        train_dataloader = self.get_train_dataloader()

        # log some stats
        total_train_batch_size = self.args.per_device_train_batch_size * self.args.gradient_accumulation_steps
        num_examples = len(train_dataloader)
        max_steps = self.args.max_steps

        logger.info("***** Running training *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_train_batch_size}")
        logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}")
        logger.info(f"  Gradient Accumulation steps = {self.args.gradient_accumulation_steps}")
        logger.info(f"  Total optimization steps = {max_steps}")

        state = {"logs": defaultdict(dict)}
        # train
        tr_loss_scalar = 0.0
        avg_loss_scalar = 0.0
        count = 0
        model.zero_grad()
        for raw_step, inputs in tqdm.tqdm(enumerate(sample_indefinitely(train_dataloader)),total=max_steps):
            inputs = {k:v.to(self.args.device) for k,v in inputs.items()}
            step = raw_step // self.args.gradient_accumulation_steps
            if step >= self.args.max_steps: break
            # after this call gradient should be accumulated in .grad field without needing further backward passes
            tr_loss_scalar += self.training_step(model, inputs)
            if (raw_step + 1) % self.args.gradient_accumulation_steps == 0:
                # Gradient clipping
                grad_norm_scalar = seq_norm(tuple(p.grad for p in model.parameters())).item()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.args.max_grad_norm)
                # Step optimizer
                self.optimizer.step()
                self.lr_scheduler.step()
                # bookkeep
                avg_loss_scalar += tr_loss_scalar
                count += 1
                # reset grads and loss scalars
                model.zero_grad()
                tr_loss_scalar = 0.0
                # log eval save
                if step == 0 or (step + 1) % self.args.logging_steps == 0:
                    avg_loss_scalar /= count
                    summary = {"step": (step + 1), "loss": avg_loss_scalar, "grad_norm": grad_norm_scalar}
                    logger.info(summary)
                    self.log_to_tensorboard(step + 1, {f"train/{k}":v for k, v in summary.items()})
                    state["logs"][str(step + 1)]["train"] = summary
                    # reset avg_loss accumulators
                    avg_loss_scalar = 0.0
                    count = 0
                if step == 0 or (step + 1) % self.args.eval_steps == 0:
                    summary = {"step": (step + 1)}
                    for eval_list in self.evaluations.values():
                        summary |= eval_list.evaluate()
                    logger.info(summary)
                    self.log_to_tensorboard(step + 1, {f"eval/{k}":v for k, v in summary.items()})
                    state["logs"][str(step + 1)]["eval"] = summary
                if step == 0 or (step + 1) % self.args.save_steps == 0:
                    checkpoint_dir = f"checkpoint-{(step + 1)}"
                    output_dir = os.path.join(self.args.output_dir, checkpoint_dir)
                    os.makedirs(output_dir, exist_ok=True)
                    logger.info(f"Saving model checkpoint to {output_dir}")
                    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
                    torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
                    torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                    with open(os.path.join(output_dir, "state.json"), "wb") as f:
                        f.write(orjson.dumps(state, option=orjson.OPT_INDENT_2) + "\n".encode("utf-8"))


        logger.info("\n\nTraining completed.\n\n")

    def log_to_tensorboard(self, step, d, prefix=""):
        for k, v in d.items():
            if isinstance(v, dict):
                self.log_to_tensorboard(step, v, prefix=f"{prefix}.{k}" if prefix else k)
            else:
                self.tensorboard.add_scalar(f"{prefix}.{k}" if prefix else k,v,step)



