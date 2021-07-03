from torch.utils.data import DataLoader, SequentialSampler
from transformers.utils import logging
import tqdm

logger = logging.get_logger(__name__)


class Prediction:

    def __init__(self,
                    model=None,
                    args=None,
                    eval_dataset=None,
                    data_collator=None,
                    output_file=None):
        self.model = model
        self.args = args
        self.eval_dataset = eval_dataset
        self.data_collator = data_collator
        self.output_file = output_file

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
            num_workers=self.args.dataloader_num_workers
        )

    def _get_eval_sampler(self, eval_dataset):
        return SequentialSampler(eval_dataset)

    def predict_step(self, model, inputs):
        raise NotImplementedError

    def log_step(self, step, predict_result):
        raise NotImplementedError

    def predict_and_log(self):
        for step, predict_result in enumerate(self.predict()):
            self.log_step(step, predict_result)

    def predict(self):
        model = self.model
        model.to(self.args.device)
        eval_dataloader = self.get_eval_dataloader()

        # log some stats
        total_eval_batch_size = self.args.per_device_eval_batch_size * self.args.gradient_accumulation_steps
        num_examples = len(eval_dataloader)

        logger.info("***** Running prediction *****")
        logger.info(f"  Num examples = {num_examples}")
        logger.info(f"  Instantaneous batch size per device = {self.args.per_device_eval_batch_size}")
        logger.info(f"  Total eval batch size (w. parallel, distributed & accumulation) = {total_eval_batch_size}")

        # eval
        for step, inputs in tqdm.tqdm(enumerate(eval_dataloader), total=len(eval_dataloader)):
            if step > 10: break
            inputs = {k:v.to(self.args.device) for k,v in inputs.items()}
            for predict_result in self.predict_step(model, inputs):
                yield predict_result
        logger.info("\n\nPrediction completed.\n\n")