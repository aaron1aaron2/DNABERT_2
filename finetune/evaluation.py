import os

import json
from dataclasses import dataclass, field
from typing import Optional

import torch
import transformers

from peft import LoraConfig, get_peft_model

from utils import compute_metrics, preprocess_logits_for_metrics
from dataset import SupervisedDataset, DataCollatorForSupervisedDataset

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    use_lora: bool = field(default=False, metadata={"help": "whether to use LoRA"})
    lora_r: int = field(default=8, metadata={"help": "hidden dimension for LoRA"})
    lora_alpha: int = field(default=32, metadata={"help": "alpha for LoRA"})
    lora_dropout: float = field(default=0.05, metadata={"help": "dropout rate for LoRA"})
    lora_target_modules: str = field(default="query,value", metadata={"help": "where to perform LoRA"})


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    kmer: int = field(default=-1, metadata={"help": "k-mer for input sequence. -1 means not using k-mer."})
    bp_size: int = field(default=-1, metadata={"help": "Only take the bp_size/2 range near the anchor center point. -1 means using original input."})

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    run_name: str = field(default="run")
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length."})
    gradient_accumulation_steps: int = field(default=1)
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    num_train_epochs: int = field(default=1)
    fp16: bool = field(default=False)
    logging_steps: int = field(default=100)
    save_steps: int = field(default=100)
    eval_steps: int = field(default=100)
    evaluation_strategy: str = field(default="steps"),
    warmup_steps: int = field(default=50)
    weight_decay: float = field(default=0.01)
    learning_rate: float = field(default=1e-4)
    save_total_limit: int = field(default=3)
    load_best_model_at_end: bool = field(default=True)
    output_dir: str = field(default="output")
    find_unused_parameters: bool = field(default=False)
    checkpointing: bool = field(default=False)
    dataloader_pin_memory: bool = field(default=False)
    eval_and_save_results: bool = field(default=True)
    save_model: bool = field(default=False)
    seed: int = field(default=42)



def eval():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    os.makedirs(training_args.output_dir, exist_ok=True)

    # load tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=True,
        trust_remote_code=True
    )

    if "InstaDeepAI" in model_args.model_name_or_path:
        tokenizer.eos_token = tokenizer.pad_token

    # define datasets and data collator
#     train_dataset = SupervisedDataset(tokenizer=tokenizer, 
#                                       data_path=os.path.join(data_args.data_path, "train.csv"), 
#                                       kmer=data_args.kmer)
#     val_dataset = SupervisedDataset(tokenizer=tokenizer, 
#                                      data_path=os.path.join(data_args.data_path, "dev.csv"), 
#                                      kmer=data_args.kmer)
    test_dataset = SupervisedDataset(tokenizer=tokenizer, 
                                     data_path=os.path.join(data_args.data_path, "test.csv"), 
                                     bp_size=data_args.bp_size,
                                     kmer=data_args.kmer)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)


    with torch.no_grad():
        # load model
        model = transformers.AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,
            num_labels=test_dataset.num_labels,
            trust_remote_code=True
        )
        model.load_state_dict(
            torch.load(training_args.output_dir + '/pytorch_model.bin')
        )

        # configure LoRA
        if model_args.use_lora:
            lora_config = LoraConfig(
                r=model_args.lora_r,
                lora_alpha=model_args.lora_alpha,
                target_modules=list(model_args.lora_target_modules.split(",")),
                lora_dropout=model_args.lora_dropout,
                bias="none",
                task_type="SEQ_CLS",
                inference_mode=False,
            )
            model = get_peft_model(model, lora_config)
            model.print_trainable_parameters()


        trainer = transformers.Trainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics
            )

        # get the evaluation results from trainer
        if training_args.eval_and_save_results:
            results_path = os.path.join(training_args.output_dir, "results")
            results = trainer.evaluate(eval_dataset=test_dataset)
            os.makedirs(results_path, exist_ok=True)
            with open(os.path.join(results_path, "eval_results.json"), "w") as f:
                json.dump(results, f, indent=2)




if __name__ == "__main__":
    print(torch.__version__)
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    eval()