import random
from pathlib import Path
from random import shuffle
from random import sample

import PIL
import pandas as pd
import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torchvision.transforms import ToTensor
from tqdm import tqdm

from tts.base import BaseTrainer
from tts.logger.utils import plot_spectrogram_to_buf
from tts.utils import inf_loop, MetricTracker

import wandb

import os

class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(
            self,
            model,
            criterion,
            metrics,
            optimizer,
            config,
            device,
            dataloaders,
            lr_scheduler=None,
            len_epoch=None,
            skip_oom=True,
    ):
        super().__init__(model, criterion, metrics, optimizer, config, device)
        self.skip_oom = skip_oom
        self.config = config
        self.train_data_obj = dataloaders['train'].dataset
        self.train_dataloader = dataloaders["train"]
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.train_dataloader)
        else:
            # iteration-based training
            self.train_dataloader = inf_loop(self.train_dataloader)
            self.len_epoch = len_epoch
        self.evaluation_dataloaders = {k: v for k, v in dataloaders.items() if k != "train"}
        self.lr_scheduler = lr_scheduler
        self.log_step = 50

        self.train_metrics = MetricTracker(
            "loss", "grad norm", *[m.name for m in self.metrics], writer=self.writer
        )
        self.evaluation_metrics = MetricTracker(
            "loss", *[m.name for m in self.metrics], writer=self.writer
        )

    @staticmethod
    def move_batch_to_device(batch, device: torch.device):
        """
        Move all necessary tensors to the HPU
        """
        for tensor_for_gpu in ["text", "duration", "mel_target", "pitch", "energy"]:
            batch[tensor_for_gpu] = batch[tensor_for_gpu].to(device)
        return batch

    def _clip_grad_norm(self):
        if self.config["trainer"].get("grad_norm_clip", None) is not None:
            clip_grad_norm_(
                self.model.parameters(), self.config["trainer"]["grad_norm_clip"]
            )

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        self.writer.add_scalar("epoch", epoch)
        for batch_idx, batch in enumerate(
                tqdm(self.train_dataloader, desc="train", total=self.len_epoch)
        ):
            try:
                batch = self.process_batch(
                    batch,
                    is_train=True,
                    metrics=self.train_metrics,
                )
            except RuntimeError as e:
                if "out of memory" in str(e) and self.skip_oom:
                    self.logger.warning("OOM on batch. Skipping batch.")
                    for p in self.model.parameters():
                        if p.grad is not None:
                            del p.grad  # free some memory
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise e
            self.train_metrics.update("grad norm", self.get_grad_norm())
            if batch_idx % self.log_step == 0:
                self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
                self.logger.debug(
                    "Train Epoch: {} {} Loss: {:.6f}".format(
                        epoch, self._progress(batch_idx), batch["loss"].item()
                    )
                )
                self.writer.add_scalar(
                    "learning rate", self.lr_scheduler.get_last_lr()[0]
                )
                self._log_predictions(self.model, is_train=True, **batch)
                # TODO: Log target and predicted spectrogram
                # self._log_spectrogram(batch["spectrogram"])
                self._log_scalars(self.train_metrics)
                # we don't want to reset train metrics at the start of every epoch
                # because we are interested in recent train metrics
                last_train_metrics = self.train_metrics.result()
                self.train_metrics.reset()
            if batch_idx >= self.len_epoch:
                break
        log = last_train_metrics

        # XXX: Switch off the evaluation logic
        for part, dataloader in self.evaluation_dataloaders.items():
            val_log = self._evaluation_epoch(epoch, part, dataloader)
            log.update(**{f"{part}_{name}": value for name, value in val_log.items()})

        return log

    def process_batch(self, batch, is_train: bool, metrics: MetricTracker):
        batch = self.move_batch_to_device(batch, self.device)
        if is_train:
            self.optimizer.zero_grad()
        outputs = self.model(**batch)
        if type(outputs) is dict:
            batch.update(outputs)
        batch['loss'] = self.criterion(**batch)
        if is_train:
            batch["loss"].backward()
            self._clip_grad_norm()
            self.optimizer.step()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

        metrics.update("loss", batch["loss"].item())
        for met in self.metrics:
            metrics.update(met.name, met(**batch))
        return batch

    def _evaluation_epoch(self, epoch, part, dataloader):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.evaluation_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch in tqdm(
                    enumerate(dataloader),
                    desc=part,
                    total=len(dataloader),
            ):  
                # print('mel_path:', batch['mel_spec_path'])
                # print('alignment_path:', batch['alignment_path'])
                batch = self.process_batch(
                    batch,
                    is_train=False,
                    metrics=self.evaluation_metrics,
                )
            self.writer.set_step(epoch * self.len_epoch, part)
            self._log_scalars(self.evaluation_metrics)
            self._log_predictions(self.model, is_train=False, **batch)

            # XXX: log spectrogram
            # self._log_spectrogram(batch["spectrogram"])

        # add histogram of model parameters to the tensorboard
        # for name, p in self.model.named_parameters():
        #     self.writer.add_histogram(name, p, bins="auto")
        return self.evaluation_metrics.result()

    def _progress(self, batch_idx):
        base = "[{}/{} ({:.0f}%)]"
        if hasattr(self.train_dataloader, "n_samples"):
            current = batch_idx * self.train_dataloader.batch_size
            total = self.train_dataloader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _log_predictions(
            self,
            model,
            is_train: bool,
            raw_text,
            loss,
            examples_to_log=7,
            *args,
            **kwargs,
    ):
        if self.writer is None:
            return

        if not is_train:
            text_to_voice = [
                'A defibrillator is a device that gives a high energy electric shock to the heart of someone who is in cardiac arrest',
                'Massachusetts Institute of Technology may be best known for its math, science and engineering education',
                'Wasserstein distance or Kantorovich Rubinstein metric is a distance function defined between probability distributions on a given metric space'
            ]
        else:
            examples_to_log = min(examples_to_log, len(raw_text))
            text_to_voice = sample(raw_text, examples_to_log)
        rows = {}
        if is_train:
            coeffs = [1]
        else:
            coeffs = [1]
        for i, text in enumerate(text_to_voice):
            for d in coeffs:
                for p in coeffs:
                    for e in coeffs:
                        output = self.model.text2voice(
                            text, dataset=self.train_data_obj,
                            duration_coeff=d, pitch_coeff=p, energy_coeff=e
                        )
                        audio = output['audio']
                        melspec = output['mel_spec']
                        audio = wandb.Audio(audio, sample_rate=self.train_data_obj.sample_rate)
                        rows[f'{i}.d={d}.p={p}.e={e}'] = {
                            "duration_coeff": d,
                            "pithc_coeff": p,
                            "energy_coeff": e,
                            "text": text,
                            "speech": audio
                        }

                        if not is_train:
                            wandb.log({
                                "config": f'd={d}. p={p}. e={e}',
                                "text": text,
                                "speech": audio
                            })

        output = self.model.text2voice("I am sorry, Dave, I am afraid I can't do that.", dataset=self.train_data_obj, pitch_coeff=0.5)
        audio = output['audio']
        audio = wandb.Audio(audio, sample_rate=self.train_data_obj.sample_rate)
        rows['bruh'] = {
            "text": "I am sorry, Dave, I am afraid I can't do that.",
            "audio": audio
        }

        self.writer.add_table("predictions", pd.DataFrame.from_dict(rows, orient="index"))

    def _log_spectrogram(self, spectrogram_batch):
        spectrogram = random.choice(spectrogram_batch.cpu())
        image = PIL.Image.open(plot_spectrogram_to_buf(spectrogram))
        self.writer.add_image("spectrogram", ToTensor()(image))

    @torch.no_grad()
    def get_grad_norm(self, norm_type=2):
        parameters = self.model.parameters()
        if isinstance(parameters, torch.Tensor):
            parameters = [parameters]
        parameters = [p for p in parameters if p.grad is not None]
        total_norm = torch.norm(
            torch.stack(
                [torch.norm(p.grad.detach(), norm_type).cpu() for p in parameters]
            ),
            norm_type,
        )
        return total_norm.item()

    def _log_scalars(self, metric_tracker: MetricTracker):
        if self.writer is None:
            return
        for metric_name in metric_tracker.keys():
            self.writer.add_scalar(f"{metric_name}", metric_tracker.avg(metric_name))
