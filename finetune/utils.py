import torch
import sklearn
import numpy as np
import transformers
from scipy.special import softmax

def safe_save_model_for_hf_trainer(trainer: transformers.Trainer, output_dir: str):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa

"""
Get the reversed complement of the original DNA sequence.
"""
def get_alter_of_dna_sequence(sequence: str):
    MAP = {"A": "T", "T": "A", "C": "G", "G": "C"}
    # return "".join([MAP[c] for c in reversed(sequence)])
    return "".join([MAP[c] for c in sequence])


"""
Compute metrics used for huggingface trainer.
""" 
def compute_metrics(eval_pred):
    logits = eval_pred.predictions[0] # predictions[1] is label pass through preprocess_logits_for_metrics()
    labels = eval_pred.label_ids

    if logits.ndim == 3:
        # Reshape logits to 2D if needed
        logits = logits.reshape(-1, logits.shape[-1])

    predictions = np.argmax(logits, axis=-1)
    valid_mask = labels != -100  # Exclude padding tokens (assuming -100 is the padding token ID)
    valid_predictions = predictions[valid_mask]
    valid_labels = labels[valid_mask]

    pred_prob = softmax(logits, axis=1)
    pred_prob = pred_prob[:,[1]].flatten()

    return {
        "accuracy": sklearn.metrics.accuracy_score(valid_labels, valid_predictions),
        "f1": sklearn.metrics.f1_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "matthews_correlation": sklearn.metrics.matthews_corrcoef(
            valid_labels, valid_predictions
        ),
        "precision": sklearn.metrics.precision_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "recall": sklearn.metrics.recall_score(
            valid_labels, valid_predictions, average="macro", zero_division=0
        ),
        "auPRCs": sklearn.metrics.average_precision_score(
            y_true=valid_labels, y_score=pred_prob
        ),
        "auROC": sklearn.metrics.roc_auc_score(
            y_true=valid_labels, y_score=pred_prob
        )
    }


def preprocess_logits_for_metrics(logits, labels):
    """
    Original Trainer may have a memory leak. 
    This is a workaround to avoid storing too many tensors that are not needed.
    """
    # logits = (model output logits, dnabert output)
    # model output logits : (bs, 2) 
    # dnabert output : (bs, 452, 768)

    return logits[0], labels