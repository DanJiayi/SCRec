"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.

This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from tqdm import tqdm
from transformers import LogitsProcessor, LogitsProcessorList


def model_forward(model, batch, device, n_codebook, method_config, skip_forward=False):
    if method_config["use_id"] == "sid":
        input_sids = batch["input_sids"].to(device)
        attention_mask_sids = batch["attention_mask_sids"].to(device)
        labels_sids = batch["labels_sids"].to(device)
        if method_config["flag_add_input_embedding"]:
            input_text_embeddings = batch["input_embeddings"].to(device).detach()
            max_length = input_text_embeddings.shape[1]

        item_idx_start = 1 if method_config["include_user_id"] else 0
        # Model forwarding
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            if method_config["flag_add_input_embedding"]:
                inputs_embeds = model.shared(input_sids)
                # process the input text embeddings
                input_text_embeddings_shape = (
                    input_text_embeddings.shape
                )  # this is the original embedding
                input_text_embeddings_repeat = (
                    input_text_embeddings[:, :, None, :]
                    .repeat(1, 1, n_codebook, 1)
                    .reshape(-1, input_text_embeddings_shape[-1])
                )
                proj_embd = model.emb_proj(input_text_embeddings_repeat)
                proj_embd = proj_embd.reshape(
                    input_text_embeddings_shape[0], -1, proj_embd.shape[-1]
                )
                # add positional embedding
                pos_id = torch.arange(max_length, dtype=torch.long, device=device)
                pos_id = pos_id[:, None].repeat(1, n_codebook).reshape(-1)
                pos_embd = model.pos_embedding(pos_id)  # [n_seq, n_embd]
                proj_embd += pos_embd[None, :]
                append_embedding = torch.zeros_like(inputs_embeds)
                append_embedding[
                    :, item_idx_start : item_idx_start + n_codebook * max_length
                ] = proj_embd
                # add the text embedding to the inputs embeds
                inputs_embeds += append_embedding
                # add the semantic positional embedding
                seq_len = inputs_embeds.shape[1]
                pattern = torch.arange(n_codebook)
                semantic_pos = pattern.repeat(seq_len // n_codebook + 1)[:seq_len]
                pos_embedding = model.semantic_pos(
                    semantic_pos.to(device)
                )  # [n_seq, n_embd]
                inputs_embeds += pos_embedding[None, :, :]
                # process the inputs embeds
                inputs_embeds = model.input_embed_layernorm(inputs_embeds)
                inputs_embeds = model.input_embed_dropout(inputs_embeds)

                if skip_forward:
                    outputs = None
                else:
                    outputs = model(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask_sids,
                        labels=labels_sids,
                    )

                input_kwargs = {
                    "inputs_embeds": inputs_embeds,
                    "attention_mask": attention_mask_sids,
                }
            else:
                if skip_forward:
                    outputs = None
                else:
                    outputs = model(
                        input_ids=input_sids,
                        attention_mask=attention_mask_sids,
                        labels=labels_sids,
                    )

                input_kwargs = {
                    "input_ids": input_sids,
                    "attention_mask": attention_mask_sids,
                }
    else:
        input_ids = batch["input_ids"].to(device)
        attention_mask_ids = batch["attention_mask_ids"].to(device)
        labels_ids = batch["labels_ids"].to(device)
        if method_config["flag_add_input_embedding"]:
            input_embeddings = batch["input_embeddings"].to(device).detach()
            max_length = input_embeddings.shape[1]

        item_idx_start = 0
        # Model forwarding
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            if method_config["flag_add_input_embedding"]:
                inputs_embeds = model.shared(input_ids)
                # process the input text embeddings
                input_embeddings_shape = input_embeddings.shape
                proj_embd = model.emb_proj(
                    input_embeddings.reshape(
                        input_embeddings_shape[0] * input_embeddings_shape[1],
                        input_embeddings_shape[2],
                    )
                )
                proj_embd = proj_embd.reshape(
                    input_embeddings_shape[0], input_embeddings_shape[1], -1
                )
                # add positional embedding
                pos_id = torch.arange(max_length, dtype=torch.long, device=device)
                pos_embd = model.pos_embedding(pos_id)  # [n_seq, n_embd]
                proj_embd += pos_embd[None, :]
                append_embedding = torch.zeros_like(inputs_embeds)
                append_embedding[:, item_idx_start : item_idx_start + max_length] = (
                    proj_embd
                )
                inputs_embeds += append_embedding

                inputs_embeds = model.input_embed_layernorm(inputs_embeds)
                inputs_embeds = model.input_embed_dropout(inputs_embeds)

                if skip_forward:
                    outputs = None
                else:
                    outputs = model(
                        inputs_embeds=inputs_embeds,
                        attention_mask=attention_mask_ids,
                        labels=labels_ids,
                    )

                input_kwargs = {
                    "inputs_embeds": inputs_embeds,
                    "attention_mask": attention_mask_ids,
                }
            else:
                if skip_forward:
                    outputs = None
                else:
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask_ids,
                        labels=labels_ids,
                    )

                input_kwargs = {
                    "input_ids": input_ids,
                    "attention_mask": attention_mask_ids,
                }
    return outputs, input_kwargs


def dcg_torch(scores: torch.Tensor):
    """Compute DCG using gain = 2^rel - 1."""
    scores = scores.float()
    device = scores.device
    gains = torch.pow(2.0, scores) - 1
    discounts = torch.log2(torch.arange(2, scores.size(0) + 2, device=device).float())
    return torch.sum(gains / discounts)


def ndcg_at_k_torch(r: torch.Tensor, k: int):
    """Compute NDCG at rank k from relevance vector (1D torch.Tensor)"""
    r = r[:k].float()
    dcg_val = dcg_torch(r)
    ideal_r, _ = torch.sort(r, descending=True)
    dcg_max = dcg_torch(ideal_r)

    if dcg_max == 0:
        return torch.tensor(0.0, device=r.device)
    return dcg_val / dcg_max


def calculate_metrics(outputs, labels, KEYS, codebook_level=4, lift_constraint=False):
    """
    n_codebook: the number of semantic id for each item
    :param outputs: shape [batch_size, num_return_seq, n_codebook]
    :param labels: shape [batch_size, n_codebook]
    """
    batch_size = len(outputs)

    ndcg_at_i, recall_at_i = dict({}), dict({})
    for key in KEYS:
        ndcg_at_i[key] = []
        recall_at_i[key] = []

    if not lift_constraint:
        for i in range(batch_size):
            assert (
                torch.unique(outputs[i], dim=0).shape[0] == outputs.shape[1]
            ), "Unless something is wrong, beam search should not return non-unique outputs."

    matches = (outputs == labels[:, None, :codebook_level]).all(axis=-1)
    for key in KEYS:
        recall_at_i[key] = matches[:, :key].any(-1).float().tolist()  # [batch_size]

    for i in range(batch_size):
        for key in KEYS:
            ndcg_at_i[key].append(ndcg_at_k_torch(matches[i], key))

    metrics = (
        recall_at_i,
        ndcg_at_i,
    )

    return metrics


def calculate_metrics_id(outputs, labels, KEYS):
    """
    n_codebook: the number of semantic id for each item
    :param outputs: shape [batch_size, num_return_seq]
    :param labels: shape [batch_size,]
    """
    batch_size = len(outputs)

    ndcg_at_i, recall_at_i = dict({}), dict({})
    for key in KEYS:
        ndcg_at_i[key] = []
        recall_at_i[key] = []

    matches = outputs == labels
    for key in KEYS:
        recall_at_i[key] = matches[:, :key].any(-1).float().tolist()  # [batch_size]

    for i in range(batch_size):
        for key in KEYS:
            ndcg_at_i[key].append(ndcg_at_k_torch(matches[i], key))

    # Calculate mean metrics
    metrics = (
        recall_at_i,
        ndcg_at_i,
    )

    return metrics


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    all_semantic_ids,
    device,
    method_config,
    KEYS,
    RETRIEVE_KEY,
):

    model.eval()
    recall_dict, ndcg_dict = dict({}), dict({})
    returned_cand = []
    returned_predicted_embedding = []

    for batch in tqdm(dataloader):
        labels = batch["labels_sids"].to(device)

        with torch.no_grad():
            _, input_kwargs = model_forward(
                model,
                batch,
                device,
                all_semantic_ids.shape[-1],
                method_config,
                skip_forward=True,
            )

        batch_size, n_codebook = (
            labels.shape[0],
            labels.shape[1],
        )  # this batch_size is before ddp

        num_return_sequences = max(RETRIEVE_KEY)
        num_beams = max(RETRIEVE_KEY)
        assert (
            max(KEYS) <= num_return_sequences
        ), "The number of return sequences should be greater than or equal to the number of keys."
        gen_kwargs = {
            "num_beams": num_beams,
            "max_new_tokens": n_codebook,
            "num_return_sequences": num_return_sequences,
        }
        gen_kwargs["use_cache"] = True

        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model.generate(**input_kwargs, **gen_kwargs)
        predicted_embedding = model.predicted_embedding

        outputs = outputs[:, 1 : 1 + n_codebook].reshape(
            batch_size, num_return_sequences, -1
        )  # [B, n_return_seq, n_codebook]
        if predicted_embedding is not None:
            returned_predicted_embedding.extend(
                predicted_embedding.reshape(batch_size, num_return_sequences, -1)[:, 0]
            )  # [B, n_embd]
        # along the num_return_sequences, all the outputs are the same

        if outputs.shape[-1] < n_codebook:
            # if output shape is smaller than label shape, pad with zeros to label shape
            # can happen when the LM predicts eos early
            # if padded with zero, the remaining prediction should measure the recall / ndcg as 0.
            to_pad = n_codebook - outputs.shape[-1]
            pad_tensor = torch.zeros(
                (batch_size, gen_kwargs["num_return_sequences"], to_pad),
                device=outputs.device,
            )
            outputs = torch.cat([outputs, pad_tensor], dim=-1)

        outputs = outputs
        returned_cand.extend(outputs)

        _recall_at_i, _ndcg_at_i = calculate_metrics(
            outputs, labels, codebook_level=n_codebook, KEYS=KEYS
        )

        for key in _recall_at_i.keys():
            if key not in recall_dict.keys():
                recall_dict[key] = []
                ndcg_dict[key] = []

        for key in _recall_at_i.keys():
            recall_dict[key].extend(_recall_at_i[key])
            ndcg_dict[key].extend(_ndcg_at_i[key])

    return recall_dict, ndcg_dict, returned_cand, returned_predicted_embedding


def get_target_embed(predicted_embedding, model, method_config, item_embedding):
    if len(item_embedding.shape) == 3:  # this is used in generate_then_dense
        assert (
            method_config["embedding_head_dict"]["embed_target"]
            != "ground_truth+item_id"
        ), "We don't expect the use of dense retrieval with item id in liger"
        item_embedding_shape = (
            item_embedding.shape
        )  # [batch_size, num_candidate, n_embd]
        item_embedding = item_embedding.reshape(
            item_embedding_shape[0] * item_embedding_shape[1], item_embedding_shape[2]
        )
        _item_embedding = model.emb_proj(item_embedding)
        _item_embedding = _item_embedding.reshape(
            item_embedding_shape[0], item_embedding_shape[1], -1
        )
    else:
        _item_embedding = model.emb_proj(item_embedding)
        if (
            method_config["embedding_head_dict"]["embed_target"]
            == "ground_truth+item_id"
        ):
            learned_embedding = model.shared.weight[1:-1]
            _item_embedding += learned_embedding

    logits = None
    if predicted_embedding is not None:
        temperature = 1.0
        if method_config["embedding_head_dict"]["normalize_logits"]:
            temperature = method_config["embedding_head_dict"]["logits_temperature"]

        predicted_embedding = F.normalize(predicted_embedding, dim=1)
        _item_embedding = F.normalize(_item_embedding, dim=-1)
        logits = (
            predicted_embedding[:, None, :]
            * _item_embedding.type(predicted_embedding.dtype)
        ).sum(-1) / temperature

    return _item_embedding, logits


@torch.no_grad()
def evaluate_dense_sids(
    model,
    dataloader,
    device,
    item2sid,
    item_embedding,
    method_config,
    KEYS=[
        10,
    ],
):

    model.eval()
    recall_dict, ndcg_dict = dict({}), dict({})
    item2sid_tensor = torch.from_numpy(item2sid).to(device)

    if len(dataloader) == 0:
        return recall_dict, ndcg_dict

    num_candidates = max(KEYS)

    for batch in tqdm(dataloader, desc="Dense Retrieval"):
        labels = batch["labels_sids"].to(device)
        n_codebook = labels.shape[1]

        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                _, _ = model_forward(
                    model,
                    batch,
                    device,
                    n_codebook,
                    method_config,
                )
                predicted_embedding = model.predicted_embedding

        _, logits = get_target_embed(
            predicted_embedding, model, method_config, item_embedding
        )
        # [batch_size, n_items]

        candidate_idx = logits.topk(num_candidates, dim=1, largest=True)[
            1
        ]  # [batch_size, num_candidates]
        candidate_sid = item2sid_tensor[
            candidate_idx
        ]  # [batch_size, num_candidates, n_codebook]
        constrained_outputs = candidate_sid

        _recall_at_i, _ndcg_at_i = calculate_metrics(
            constrained_outputs,
            labels,
            codebook_level=n_codebook,
            KEYS=KEYS,
        )
        for key in _recall_at_i.keys():
            if key not in recall_dict.keys():
                recall_dict[key] = []
                ndcg_dict[key] = []

        for key in _recall_at_i.keys():
            recall_dict[key].extend(_recall_at_i[key])
            ndcg_dict[key].extend(_ndcg_at_i[key])

    return recall_dict, ndcg_dict


@torch.no_grad()
def evaluate_dense_ids(
    model,
    dataloader,
    device,
    item2sid,
    item_embedding,
    method_config,
    KEYS=[
        10,
    ],
):

    model.eval()
    recall_dict, ndcg_dict = dict({}), dict({})

    if len(dataloader) == 0:
        return recall_dict, ndcg_dict

    num_candidates = max(KEYS)

    for batch in dataloader:
        labels = batch["labels_ids"].to(device)
        n_codebook = labels.shape[1]

        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
                outputs, _ = model_forward(
                    model,
                    batch,
                    device,
                    n_codebook,
                    method_config,
                )
                predicted_embedding = model.predicted_embedding

        _, logits = get_target_embed(
            predicted_embedding, model, method_config, item_embedding
        )
        # [batch_size, n_items]

        candidate_idx = logits.topk(num_candidates, dim=1, largest=True)[
            1
        ]  # [batch_size, num_candidates]

        labels = labels.cpu()
        constrained_outputs = candidate_idx + 1

        _recall_at_i, _ndcg_at_i = calculate_metrics_id(
            constrained_outputs.cpu(), labels, KEYS=KEYS
        )
        for key in _recall_at_i.keys():
            if key not in recall_dict.keys():
                recall_dict[key] = []
                ndcg_dict[key] = []

        for key in _recall_at_i.keys():
            recall_dict[key].extend(_recall_at_i[key])
            ndcg_dict[key].extend(_ndcg_at_i[key])

    return recall_dict, ndcg_dict


@torch.no_grad()
def generate_then_dense(
    model,
    dataloader,
    unseen_semantic_ids,
    device,
    method_config,
    returned_cand,
    returned_embd,
    item2sid,
    item_embedding,
    KEYS=[
        10,
    ],
    RETRIEVE_KEY=[20, 40, 60, 80, 100],
):

    model.eval()
    recall_dict_total, ndcg_dict_total = dict({}), dict({})
    if len(returned_cand) == 0:
        return recall_dict_total, ndcg_dict_total

    returned_cand = torch.stack(returned_cand)  # [num_items, num_candidate, n_code]
    item2sid_tensor = torch.from_numpy(item2sid).to(device)
    unseen_semantic_ids = unseen_semantic_ids.to(device)
    returned_embd = torch.stack(returned_embd, dim=0)  # [num_items, n_embd]
    num_candidates = max(KEYS)

    idx_start = 0
    for batch in tqdm(dataloader, desc="Generating and Dense Retrieval"):
        labels = batch["labels_sids"].to(device)
        batch_size, n_codebook = (
            labels.shape[0],
            labels.shape[1],
        )  # this batch_size is before ddp

        predicted_embedding = returned_embd[idx_start : idx_start + batch_size]
        this_batch_cand = returned_cand[
            idx_start : idx_start + batch_size,
        ]  # [batch_size, num_candidate, n_code]
        idx_start += batch_size
        cold_cand = torch.stack([unseen_semantic_ids] * this_batch_cand.shape[0], dim=0)

        for _retrieve_key in RETRIEVE_KEY:
            if _retrieve_key not in recall_dict_total.keys():
                recall_dict_total[_retrieve_key] = dict({})
                ndcg_dict_total[_retrieve_key] = dict({})

            _this_batch_cand = this_batch_cand[:, :_retrieve_key]
            _this_batch_cand = torch.cat([_this_batch_cand, cold_cand], dim=1)
            matches = torch.all(
                _this_batch_cand[:, :, None] == item2sid_tensor[None, None, :, :],
                dim=-1,
            )
            # [batch_size, num_candidate, num_items]
            indices = torch.argmax(matches.int(), dim=2)  # [batch_size, num_candidate]

            cand_item_embedding = item_embedding[
                indices
            ]  # [batch_size, num_candidate_item_embedding]

            _, logits = get_target_embed(
                predicted_embedding, model, method_config, cand_item_embedding
            )
            # [batch_size, n_items]

            _topk = min(num_candidates, logits.shape[1])
            candidate_idx = logits.topk(_topk, dim=1, largest=True)[
                1
            ]  # [batch_size, num_candidates]
            candidate_sid = _this_batch_cand[
                torch.arange(batch_size)[:, None], candidate_idx
            ]  # [batch_size, num_candidates, n_codebook] -> [batch_size, topk, n_codebook]

            _recall_at_i, _ndcg_at_i = calculate_metrics(
                candidate_sid,
                labels,
                codebook_level=n_codebook,
                KEYS=KEYS,
                lift_constraint=True,
            )
            # lift the constraint to have unique candidate, since we append the output from generative retrieval with the cold start candidates
            for key in _recall_at_i.keys():
                if key not in recall_dict_total[_retrieve_key].keys():
                    recall_dict_total[_retrieve_key][key] = []
                    ndcg_dict_total[_retrieve_key][key] = []

            for key in _recall_at_i.keys():
                recall_dict_total[_retrieve_key][key].extend(_recall_at_i[key])
                ndcg_dict_total[_retrieve_key][key].extend(_ndcg_at_i[key])

    return recall_dict_total, ndcg_dict_total
