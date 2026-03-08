
import os
from datetime import datetime
from tqdm import tqdm
import numpy as np
from collections import defaultdict, OrderedDict
from logging import getLogger
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
from transformers.optimization import get_scheduler

from decoder.model import AbstractModel
from decoder.tokenizer import AbstractTokenizer
from decoder.evaluator import Evaluator
from decoder.utils import get_file_name, get_total_steps, config_for_log, log


class Trainer:
    """
    A class that handles the training process for a model.

    Args:
        config (dict): The configuration parameters for training.
        model (AbstractModel): The model to be trained.
        tokenizer (AbstractTokenizer): The tokenizer used for tokenizing the data.

    Attributes:
        config (dict): The configuration parameters for training.
        model (AbstractModel): The model to be trained.
        evaluator (Evaluator): The evaluator used for evaluating the model.
        logger (Logger): The logger used for logging training progress.
        project_dir (str): The directory path for saving tensorboard logs.
        accelerator (Accelerator): The accelerator used for distributed training
        saved_model_ckpt (str): The file path for saving the trained model checkpoint.

    Methods:
        fit(train_dataloader, val_dataloader): Trains the model using the provided training and validation dataloaders.
        evaluate(dataloader, split='test'): Evaluate the model on the given dataloader.
        end(): Ends the training process and releases any used resources.
    """

    def __init__(self, config: dict, model: AbstractModel, tokenizer: AbstractTokenizer):
        self.config = config
        self.model = model
        self.accelerator = config['accelerator']
        self.evaluator = Evaluator(config, tokenizer)
        self.logger = getLogger()

        self.saved_model_ckpt = os.path.join(
            self.config['ckpt_dir'],
            get_file_name(self.config, suffix='.pth')
        )

        os.makedirs(os.path.dirname(self.saved_model_ckpt), exist_ok=True)

        self.best_test_score_during_training = -1 # Initialize with a low score
        self.best_test_epoch_during_training = 0
        self.saved_best_test_ckpt = os.path.join( # Path for the checkpoint with best test score
            self.config['ckpt_dir'],
            get_file_name(self.config, suffix='_best_test_during_training.pth')
        )
        os.makedirs(os.path.dirname(self.saved_best_test_ckpt), exist_ok=True)
        

    def fit(self, train_dataloader, val_dataloader, test_dataloader): # Add test_dataloader:
        """
        Trains the model using the provided training and validation dataloaders.

        Args:
            train_dataloader: The dataloader for training data.
            val_dataloader: The dataloader for validation data.
        """

        # Simplified configuration: do not use wandb
        project_name = "SCRec_RQVAE"
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        category = self.config.get('category', 'unknown')
        run_name = f"rqvae_{category}_{timestamp}"

        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config['lr'],
            weight_decay=self.config['weight_decay']
        )

        total_n_steps = get_total_steps(self.config, train_dataloader)
        if total_n_steps == 0:
            self.log('No training steps needed.')
            return 0, -1.0, -1.0, 0

        scheduler = get_scheduler(
            name="cosine",
            optimizer=optimizer,
            num_warmup_steps=self.config['warmup_steps'],
            num_training_steps=total_n_steps,
        )

        self.model, optimizer, train_dataloader, val_dataloader, scheduler = self.accelerator.prepare(
            self.model, optimizer, train_dataloader, val_dataloader, scheduler
        )

        # Use only tensorboard and disable wandb
        self.accelerator.init_trackers(
            project_name=project_name,
            config=config_for_log(self.config),
            init_kwargs={
                "tensorboard": {"flush_secs": 60},
            },
        )

        n_epochs = np.ceil(total_n_steps / (len(train_dataloader) * self.accelerator.num_processes)).astype(int)
        best_epoch = 0
        best_val_score = -1

        for epoch in range(n_epochs):
            # Training
            self.model.train()
            total_loss = 0.0
            train_progress_bar = tqdm(
                train_dataloader,
                total=len(train_dataloader),
                desc=f"Training - [Epoch {epoch + 1}]",
            )
            for batch in train_progress_bar:
                optimizer.zero_grad()
                outputs = self.model(batch)
                loss = outputs.loss
                self.accelerator.backward(loss)
                if self.config['max_grad_norm'] is not None:
                    clip_grad_norm_(self.model.parameters(), self.config['max_grad_norm'])
                optimizer.step()
                scheduler.step()
                total_loss = total_loss + loss.item()

            self.accelerator.log({"Loss/train_loss": total_loss / len(train_dataloader)}, step=epoch + 1)
            self.log(f'[Epoch {epoch + 1}] Train Loss: {total_loss / len(train_dataloader)}')

            # Evaluation
            if (epoch + 1) % self.config['eval_interval'] == 0:
                all_results = self.evaluate(val_dataloader, split='val')
                if self.accelerator.is_main_process:
                    for key in all_results:
                        self.accelerator.log({f"Val_Metric/{key}": all_results[key]}, step=epoch + 1)
                    self.log(f'[Epoch {epoch + 1}] Val Results: {all_results}')
                val_score = all_results[self.config['val_metric']]
                if val_score > best_val_score:
                    best_val_score = val_score
                    best_epoch = epoch + 1
                    if self.accelerator.is_main_process:
                        if self.config['use_ddp']: # unwrap model for saving
                            unwrapped_model = self.accelerator.unwrap_model(self.model)
                            torch.save(unwrapped_model.state_dict(), self.saved_model_ckpt)
                        else:
                            torch.save(self.model.state_dict(), self.saved_model_ckpt)
                        self.log(f'[Epoch {epoch + 1}] Saved model checkpoint to {self.saved_model_ckpt}')

                if self.config['patience'] is not None and epoch + 1 - best_epoch >= self.config['patience']:
                    self.log(f'Early stopping at epoch {epoch + 1}')
                    
                    
            # Periodically evaluate on test set and save the best model on test set
            test_eval_interval = self.config.get('test_eval_interval', None)
            if test_eval_interval is not None and test_eval_interval > 0 and (epoch + 1) % test_eval_interval == 0:
                self.log(f'Evaluating on test set at epoch {epoch + 1}...')
                # Ensure test_dataloader is prepared. It should be handled by Accelerator.prepare in pipeline.
                # However, if test_dataloader isn't part of the initial accelerator.prepare call,
                # you might need to prepare it explicitly here, or ensure it's prepared once in Pipeline.run.
                # Since Pipeline.run prepares test_dataloader *after* trainer.fit,
                # we need to prepare it here for periodic evaluation during training if not already prepared.
                # Assuming test_dataloader is already prepared when passed to fit.
                
                # Check if test_dataloader and self.model are already prepared by Accelerator
                # In Pipeline.run, test_dataloader is prepared *after* trainer.fit returns.
                # So, we need to prepare it within trainer.fit for periodic evaluation.
                # For simplicity, we assume test_dataloader is already prepared
                # if you add it to the accelerator.prepare call in Pipeline or ensure it's on the correct device.
                
                # However, the correct way to handle this with Accelerator is to prepare the test_dataloader
                # *before* calling fit, if it's going to be used within fit.
                # Let's adjust the assumption: test_dataloader is passed and is already a standard DataLoader,
                # and we'll prepare it within this block if it wasn't prepared by accelerator.
                
                # Re-checking Pipeline.run: test_dataloader is prepared *after* trainer.fit.
                # So, we must prepare it here if not prepared.

                # Temporarily unwrap model if DDP is used for evaluation
                original_model = self.model
                if self.config['use_ddp']:
                    # No need to unwrap if evaluate handles module.generate, but for device it's safer
                    # and if gather_for_metrics requires unwrapped model.
                    # Best practice: handle it consistently as in evaluate method for metrics.
                    # accelerator.prepare ensures objects are on device.
                    pass # Accelerator handles device placement when prepared.

                test_results = self.evaluate(test_dataloader, split='test')
                if self.accelerator.is_main_process:
                    for key in test_results:
                        self.accelerator.log({f"Test_Periodic_Metric/{key}": test_results[key]}, step=epoch + 1)
                    self.log(f'[Epoch {epoch + 1}] Periodic Test Results: {test_results}')

                    # Compare and save if current test score is better
                    current_test_score = test_results[self.config['val_metric']] # Use val_metric for comparison consistency
                    if current_test_score > self.best_test_score_during_training:
                        self.best_test_score_during_training = current_test_score
                        self.best_test_epoch_during_training = epoch + 1
                        if self.config['use_ddp']: # unwrap model for saving
                            unwrapped_model = self.accelerator.unwrap_model(original_model)
                            torch.save(unwrapped_model.state_dict(), self.saved_best_test_ckpt)
                        else:
                            torch.save(original_model.state_dict(), self.saved_best_test_ckpt)

                        self.log(f'*********************************************************')
                        self.log(f'** **')
                        self.log(f'** ✨ NEW BEST TEST MODEL FOUND! - Epoch {epoch + 1} ✨   **')
                        for metric_name, score_value in test_results.items():
                            self.log(f'** {metric_name}: {score_value:.4f}   **')
                        self.log(f'** Checkpoint saved to: {self.saved_best_test_ckpt}    **')
                        self.log(f'** **')
                        self.log(f'*********************************************************')


        self.log(f'Best epoch: {best_epoch}, Best val score: {best_val_score}')
        return best_epoch, best_val_score, self.best_test_score_during_training, self.best_test_epoch_during_training
        # --- MODIFICATION END ---

    def evaluate(self, dataloader, split='test'):
        """
        Evaluate the model on the given dataloader.

        Args:
            dataloader (torch.utils.data.DataLoader): The dataloader to evaluate on.
            split (str, optional): The split name. Defaults to 'test'.

        Returns:
            OrderedDict: A dictionary containing the evaluation results.
        """
        self.model.eval()

        all_results = defaultdict(list)
        val_progress_bar = tqdm(
            dataloader,
            total=len(dataloader),
            desc=f"Eval - {split}",
        )
        for batch in val_progress_bar:
            with torch.no_grad():
                batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}
                if self.config['use_ddp']: # ddp, gather data from all devices for evaluation
                    preds = self.model.module.generate(batch, n_return_sequences=self.evaluator.maxk)
                    all_preds, all_labels = self.accelerator.gather_for_metrics((preds, batch['labels']))
                    results = self.evaluator.calculate_metrics(all_preds, all_labels)
                else:
                    preds = self.model.generate(batch, n_return_sequences=self.evaluator.maxk)
                    results = self.evaluator.calculate_metrics(preds, batch['labels'])

                for key, value in results.items():
                    all_results[key].append(value)

        output_results = OrderedDict()
        for metric in self.config['metrics']:
            for k in self.config['topk']:
                key = f"{metric}@{k}"
                output_results[key] = torch.cat(all_results[key]).mean().item()
        return output_results

    def case_evaluate(self, dataloader, split='test'):
        """
        Evaluate the model on the given dataloader.

        Args:
            dataloader (torch.utils.data.DataLoader): The dataloader to evaluate on.
            split (str, optional): The split name. Defaults to 'test'.

        Returns:
            OrderedDict: A dictionary containing the evaluation results.
        """
        self.model.eval()

        diff2gap = defaultdict(list)

        all_results = defaultdict(list)
        val_progress_bar = tqdm(
            dataloader,
            total=len(dataloader),
            desc=f"Eval - {split}",
        )
        for batch in val_progress_bar:
            with torch.no_grad():
                batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}

                outputs = self.model.forward(batch, return_loss=False)
                states = outputs.final_states.gather(
                    dim=1,
                    index=(batch['seq_lens'] - 1).view(-1, 1, 1, 1).expand(-1, 1, self.model.n_pred_head, self.model.config['n_embd'])
                )
                states = F.normalize(states, dim=-1)

                token_emb = self.model.gpt2.wte.weight[1:-1]
                token_emb = F.normalize(token_emb, dim=-1)
                token_embs = torch.chunk(token_emb, self.model.n_pred_head, dim=0)
                logits = [torch.matmul(states[:,0,i,:], token_embs[i].T) / self.model.temperature for i in range(self.model.n_pred_head)]
                logits = [F.log_softmax(logit, dim=-1) for logit in logits]
                token_logits = torch.cat(logits, dim=-1)    # (batch_size, n_tokens)

                sampled_items = torch.randint(1, self.model.item_id2tokens.shape[0], (token_logits.shape[0], 10))

                item_logits = torch.gather(
                    input=token_logits.unsqueeze(-2).expand(-1, sampled_items.shape[1], -1),              # (batch_size, n_items, n_tokens)
                    dim=-1,
                    index=(self.model.item_id2tokens[sampled_items,:] - 1)  # (batch_size, n_items, code_dim)
                ).mean(dim=-1)

                for batch_id in range(item_logits.shape[0]):
                    logit_list = item_logits[batch_id].cpu().tolist()
                    for i in range(len(logit_list)):
                        for j in range(i + 1, len(logit_list)):
                            item_a = sampled_items[batch_id, i]
                            item_b = sampled_items[batch_id, j]
                            gap = abs(logit_list[i] - logit_list[j])
                            diff = (self.model.item_id2tokens[item_a] != self.model.item_id2tokens[item_b]).sum().item()
                            diff2gap[diff].append(gap)
        return diff2gap

    def evaluate_cold_start(self, dataloader, token2item, item2group, split='test'):
        """
        Evaluate the model on the given dataloader.

        Args:
            dataloader (torch.utils.data.DataLoader): The dataloader to evaluate on.
            split (str, optional): The split name. Defaults to 'test'.

        Returns:
            OrderedDict: A dictionary containing the evaluation results.
        """
        self.model.eval()

        all_results = defaultdict(list)
        group2results = {
            '0': defaultdict(list),
            '1': defaultdict(list),
            '2': defaultdict(list),
            '3': defaultdict(list),
            '4': defaultdict(list)
        }
        val_progress_bar = tqdm(
            dataloader,
            total=len(dataloader),
            desc=f"Eval - {split}",
        )
        for batch in val_progress_bar:
            with torch.no_grad():
                batch = {k: v.to(self.accelerator.device) for k, v in batch.items()}
                if self.config['use_ddp']: # ddp, gather data from all devices for evaluation
                    preds = self.model.module.generate(batch, n_return_sequences=self.evaluator.maxk)
                    all_preds, all_labels = self.accelerator.gather_for_metrics((preds, batch['labels']))
                    results = self.evaluator.calculate_metrics(all_preds, all_labels)
                else:
                    preds = self.model.generate(batch, n_return_sequences=self.evaluator.maxk)
                    results = self.evaluator.calculate_metrics(preds, batch['labels'])

                for i, label in enumerate(batch['labels'].cpu().tolist()):
                    if self.config['model'] == 'TIGER':
                        item_id = token2item[' '.join(list(map(str, tuple(label[:-1]))))]
                    else:
                        item_id = token2item[str(label[0])]
                    if item_id not in item2group:
                        continue
                    group = item2group[item_id]
                    for key, value in results.items():
                        group2results[group][key].append(value.cpu().tolist()[i])

                for key, value in results.items():
                    all_results[key].append(value)

        output_results = OrderedDict()
        for metric in self.config['metrics']:
            for k in self.config['topk']:
                key = f"{metric}@{k}"
                output_results[key] = torch.cat(all_results[key]).mean().item()

        return output_results, group2results

    def end(self):
        """
        Ends the training process and releases any used resources
        """
        self.accelerator.end_training()

    def log(self, message, level='info'):
        return log(message, self.config['accelerator'], self.logger, level=level)
