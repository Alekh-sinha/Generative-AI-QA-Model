from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, GenerationConfig,AutoModelForCausalLM,TrainingArguments, Trainer
from datasets import load_dataset
from peft import PeftModel, PeftConfig, LoraConfig, TaskType, get_peft_model
import time
import evaluate
import torch
from config import cfg
from peft import prepare_model_for_kbit_training
from peft import PeftModel, PeftConfig
import pandas as pd
import subprocess    
########################################
CFG=cfg()
#####################
cmd='gsutil cp -r gs://{}/training/* .'.format(CFG.bucket)
subprocess.run(cmd.split(' '))

##################
def tokenize_function(example,CFG=CFG):
  start_prompt = 'Answer the question based on the context below. Check if context is relevent or related to the question. If it is not relevant return answer as "not sure about the answer".\n question:'
  context_prompt = '\n\n Context: '
  end_prompt = '\n\n Answer: '
  l=len(example['answer'])
  prompt = [start_prompt + example['question'][i] + context_prompt+example['context'][i]+end_prompt for i in range(l)]
  prompt_answer=[start_prompt + example['question'][i] + context_prompt+example['context'][i]+end_prompt+example['answer] for i in range(l)]
  example['input_ids'] = CFG.tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt").input_ids
  #print(example['input_ids'])
  example['labels'] = CFG.tokenizer(prompt_answer, padding="max_length", truncation=True, return_tensors="pt").input_ids
  return inp
#################################################
def generate_text(input_ids,labels,model):
  input_ids=torch.IntTensor(input_ids)
  input_ids=torch.reshape(input_ids,(1,512))
  inputs={'input_ids':input_ids}
  prediction=tokenizer.decode(outputs[0], skip_special_tokens=True)
  labels=tokenizer.decode(labels, skip_special_tokens=True)
  return prediction,labels

def load_data(cfg):
    dataset = load_dataset('json', data_files=cfg.data)
    tokenized_datasets=dataset.map(tokenize_function, batched=True,batch_size=2)
    tokenized_datasets = tokenized_datasets.remove_columns(['question','context','answer'])
    tokenized_datasets=tokenized_datasets['train'].train_test_split(test_size=0.2)
    return tokenized_datasets
def evaluate_text(cfg,tokenized_datasets):
    res=[]
    for i in range(len(tokenized_datasets['test'])):
        pred,label=generate_text(tokenized_datasets['test'][i]['input_ids'],tokenized_datasets['test'][i]['labels'],cfg.peft_model)
        res.append([pred,label])
        df=pd.DataFrame(res,columns=['prediction','label'])
        rouge = evaluate.load('rouge')
        ##############################################
    peft_model_results = rouge.compute(
        predictions=list(df['prediction']),
        references=list(df['label']),
        use_aggregator=True,
        use_stemmer=True,
    )
    return peft_model_results

def main():
    ###Setting up the tokenizer#############################
    tokenizer = AutoTokenizer.from_pretrained(
    CFG.model,
    model_max_length=CFG.max_length,
    padding_side="left",
    add_eos_token=True)
    tokenizer.pad_token = tokenizer.eos_token
    CFG.tokenizer=tokenizer
    
    
    ####################################setting up model#####################
    model = AutoModelForCausalLM.from_pretrained(CFG.model,low_cpu_mem_usage=True)
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    ###########################################loading data#########################
    tokenized_datasets=load_data(CFG)
    
    #####################################setting up LORA model#######################
    peft_model = get_peft_model(model, lora_config)
    output_dir = f'./peft-upsc-2024-{str(int(time.time()))}'
    ###########################################################
    batch_size=CFG.batch_size
    epochs=CFG.epochs
    max_steps=int(len(tokenized_datasets["train"])/batch_size)*epochs
    ################################################################
    peft_training_args = TrainingArguments(
    output_dir=output_dir,
    auto_find_batch_size=True,
    learning_rate=CFG.lr,
    num_train_epochs=epochs,
    logging_strategy="epoch",
    max_steps=max_steps)

    peft_trainer = Trainer(
        model=peft_model,
        args=peft_training_args,
        train_dataset=tokenized_datasets["train"],
    )
    peft_trainer.train()

    peft_model_path=f"./peft-upsc-2024-checkpoint-local-{str(int(time.time()))"
    peft_trainer.model.save_pretrained(peft_model_path)
    tokenizer.save_pretrained(peft_model_path)
    ###################################################
    peft_model = PeftModel.from_pretrained(model,
                                       peft_model_path,
                                       torch_dtype=torch.bfloat16,
                                       is_trainable=False)  
    #####################################################
    CFG.peft_model=peft_model
    score=evaluate_text(CFG,tokenized_datasets)
    cmd='gsutil cp -r {} gs://{}/training'.format(peft_model_path,CFG.bucket)
    subprocess.run(cmd.split(' '))
    score['model']=peft_model_path
    return str(score)
 ####################################################
if __name__ == '__main__':
  main()