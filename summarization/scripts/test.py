import glob
import os
from transformers import AutoTokenizer
from .datamodule import NewsSummaryDataset
from .model import NewsSummaryModel

def get_text(
            json_path: str,
            tokenizer: AutoTokenizer,
            text_max_token_len: str,
            summary_max_token_len: str,
            text: str = None
            ):
    if not text: 
       
        val_data = NewsSummaryDataset(json_path=json_path,
                                  tokenizer=tokenizer,
                                  text_max_token_len=text_max_token_len,
                                  summary_max_token_len=summary_max_token_len) 
        text = next(iter(val_data))['text']
    return text




    

def infer(text: str|None = None):
    
    tokenizer = AutoTokenizer.from_pretrained('IlyaGusev/rut5_base_sum_gazeta')
    
    text = get_text(json_path='data/gazeta_test.jsonl',
                        tokenizer=tokenizer,
                        text_max_token_len=512,
                        summary_max_token_len=128,
                        text=text)
    
    list_of_files = glob.glob('checkpoints/*') 
    latest_file = max(list_of_files, key=os.path.getctime)
    trained_model = NewsSummaryModel.load_from_checkpoint(latest_file).model.to('cpu')

    
    def summarizeText(text, tokenizer, model):
        text_encoding = tokenizer(
            text,
            max_length=512,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors='pt'
        )
        generated_ids = model.generate(
            input_ids=text_encoding['input_ids'],
            attention_mask=text_encoding['attention_mask'],
            max_length=150,
            num_beams=2,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True
        )

        preds = [
                tokenizer.decode(gen_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for gen_id in generated_ids
        ]
        return "".join(preds)
    
    



    return(summarizeText(text=text,
                        tokenizer=tokenizer,
                        model=trained_model))


if __name__ == "__main__":
    infer()