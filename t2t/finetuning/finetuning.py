

from transformers import AutoTokenizer, AutoModel
import pandas
from os import path
from torch.utils.data import DataLoader, Dataset
import torch
import torch
from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig
from torch.optim import Adam
from torch.nn import functional as F
from tqdm import tqdm
import gc
import argparse

gc.collect()
torch.cuda.empty_cache()


class MyDataset(Dataset):
    def __init__(self, input_ids, input_attention_mask, output_ids, output_attention_mask):
        self.input_ids = input_ids
        self.input_attention_mask = input_attention_mask
        self.output_ids = output_ids
        self.output_attention_mask = output_attention_mask
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        input_id = self.input_ids[idx]
        input_mask = self.input_attention_mask[idx]
        output_id = self.output_ids[idx]
        output_mask = self.output_attention_mask[idx]
        
        return {
            'input_ids': input_id,
            'attention_mask': input_mask,
            'labels': output_id,
            'decoder_attention_mask': output_mask
        }
    
def finetuning(data_dir, dataset_name, model_name):

    #Hyperparameters
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_epochs = 20
    batch_size = 16

    #Data Preperation

    with open(path.join(data_dir,dataset_name),'r') as f:
        dataframe = pandas.read_csv(f,sep='\t')

    print(dataframe.keys())
    reports = list(dataframe['table'])
    sentences = list(dataframe['text'])


    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")

    X = tokenizer(reports, padding="max_length", truncation=True, max_length=300, return_tensors="pt")
    Y = tokenizer(sentences, padding="max_length", truncation=True, max_length=300, return_tensors="pt")

    X_ids = X["input_ids"]
    X_masks = X["attention_mask"]

    Y_ids = Y["input_ids"]
    Y_masks = Y["attention_mask"]

    dataset = MyDataset(X_ids, X_masks, Y_ids, Y_masks)

    train_size = int(len(dataset)*0.75)
    val_size = len(dataset)-train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    #Fine Tuning the Model

    for param in model.get_encoder().parameters():
        param.requires_grad = False

    model.to(device)

    # Set up optimizer
    to_optimise = filter(lambda x: x.requires_grad, model.parameters())
    optimizer = Adam(to_optimise, lr=1e-5)

    # Set up training loop
    train_losses = []
    val_losses = []
    for epoch in tqdm(range(num_epochs)):
        model.train()
        total_train_loss = 0
        for batch in train_dataloader:
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}

            # Clear gradients
            optimizer.zero_grad()

            outputs = model(**batch)

        
            # Compute loss and perform backpropagation
            loss = outputs.loss

            total_train_loss += loss.item()
            loss.backward()
            optimizer.step()
        
        #Next use validation data to compute validation losses after the epoch
        total_val_loss = 0
        model.eval()
        with torch.no_grad():
            for batch in val_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                val_outputs = model(**batch)
                val_loss = val_outputs.loss
                total_val_loss+=loss.item()

        # Calculate average training and validation losses for the epoch
        average_train_loss = total_train_loss / len(train_dataloader)
        average_val_loss = total_val_loss/len(val_dataloader)


        train_losses.append(average_train_loss)
        val_losses.append(average_val_loss)
        
        print(f'Epoch {epoch+1}/{num_epochs}, Training Loss: {average_train_loss:.4f}')
        print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {average_val_loss:.4f}')
        
    # Save the fine-tuned model
    model.save_pretrained(path.join(model_name))
    tokenizer.save_pretrained(path.join(model_name))

    with open(path.join(model_name+'.log'),'w+') as f:
        f.write("Training Loss\tValidation Loss")
        for train_loss,val_loss in zip(train_losses,val_losses):
            f.write(str(train_loss)+"\t"+str(val_loss)+"\n")
        
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset-dir', type=str)
    parser.add_argument('--dataset-name', type=str)
    parser.add_argument('--model-name', type=str)

    args = parser.parse_args()

    finetuning(args.dataset_dir,args.dataset_name,args.model_name)

if __name__ == "__main__":
    main()