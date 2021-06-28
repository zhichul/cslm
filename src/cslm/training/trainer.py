import os

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
                    eval_dataset=None,
                    data_collator=None,
                    optimizer=None,
                    lr_scheduler=None,
                    train_dataset_weights=None,
                    train_dataset_lengths=None):
        self.model = model
        self.args = args
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_dataset_weights = train_dataset_weights
        self.train_dataset_lengths = train_dataset_lengths

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

    def get_eval_dataloader(self, eval_dataset=None):
        if eval_dataset is None:
            eval_dataset = self.eval_dataset
        eval_sampler = self._get_eval_sampler(eval_dataset)

        return DataLoader(
            eval_dataset,
            sampler=eval_sampler,
            batch_size=self.args.per_device_eval_batch_size,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def _get_eval_sampler(self, eval_dataset):
        return SequentialSampler(eval_dataset)

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
                # eval save and log
                if step % self.args.eval_steps == 0:
                    pass
                if step % self.args.save_steps == 0:
                    checkpoint_dir = f"checkpoint-{step}"
                    output_dir = os.path.join(self.args.output_dir, checkpoint_dir)
                    os.makedirs(output_dir, exist_ok=True)
                    logger.info(f"Saving model checkpoint to {output_dir}")
                    torch.save(model.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))
                    torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
                    torch.save(self.optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                if step % self.args.logging_steps == 0:
                    avg_loss_scalar /= count
                    logger.info({"step": step, "loss": avg_loss_scalar, "grad_norm": grad_norm_scalar})
                    # reset avg_loss accumulators
                    avg_loss_scalar = 0.0
                    count = 0

        logger.info("\n\nTraining completed.\n\n")


