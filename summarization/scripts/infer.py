import torch 
import pytorch_lightning as pl
import hydra
import glob
import os
from omegaconf import DictConfig
from transformers import AutoTokenizer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from datamodule import NewsSummaryDataset
from model import NewsSummaryModel


@hydra.main(version_base=None, config_path="../../configs", config_name="test")
def infer(cfg: DictConfig):
    pl.seed_everything(cfg.model.seed)
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.data.tokenizer_name)
    val_data = NewsSummaryDataset(json_path=cfg.data.val,
                                  tokenizer=tokenizer,
                                  text_max_token_len=cfg.data.text_max_token_len,
                                  summary_max_token_len=cfg.data.text_max_token_len) 
    
    list_of_files = glob.glob('checkpoints/*') # * means all if need specific format then *.csv
    latest_file = max(list_of_files, key=os.path.getctime)
    trained_model = NewsSummaryModel.load_from_checkpoint(latest_file).model.to(cfg.model.device)

    def summarizeText(text, tokenizer, model):
        text_encoding = tokenizer(
            text,
            max_length=cfg.tokenizer.max_length,
            padding=cfg.tokenizer.padding,
            truncation=cfg.tokenizer.truncation,
            return_attention_mask=cfg.tokenizer.return_attention_mask,
            add_special_tokens=cfg.tokenizer.add_special_tokens,
            return_tensors=cfg.tokenizer.return_tensors
        )
        generated_ids = model.generate(
            input_ids=text_encoding['input_ids'],
            attention_mask=text_encoding['attention_mask'],
            max_length=cfg.model.max_length,
            num_beams=cfg.model.num_beams,
            repetition_penalty=cfg.model.repetition_penalty,
            length_penalty=cfg.model.length_penalty,
            early_stopping=cfg.model.early_stopping
        )

        preds = [
                tokenizer.decode(gen_id, 
                                 skip_special_tokens=cfg.tokenizer.skip_special_tokens,
                                 clean_up_tokenization_spaces=cfg.tokenizer.clean_up_tokenization_spaces)
                for gen_id in generated_ids
        ]
        return "".join(preds)


    text = next(iter(val_data))
    print(summarizeText(text=text['text'],
                        tokenizer=tokenizer,
                        model=trained_model))
    print(text['text'])


if __name__ == "__main__":
    infer()