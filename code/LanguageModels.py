import torch
import os
from transformers import TextDataset, GPT2Tokenizer, GPT2Config, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling

# ======================================================================================================================
def calculate_perplexity(text, model, tokenizer, device):
    encodings = tokenizer(text, return_tensors='pt')
    max_length = model.config.n_positions
    stride = 256

    lls = []
    for i in range(0, encodings.input_ids.size(1), stride):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = i + stride
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-stride] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)
            log_likelihood = outputs[0] * stride

        lls.append(log_likelihood)

    ppl = torch.exp(torch.stack(lls).sum() / i)
    return ppl.item()


def load_lm(foldername, device):
    config = GPT2Config.from_pretrained(foldername)
    model = GPT2LMHeadModel.from_pretrained(foldername, config=config).to(device)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    return model, tokenizer

# ======================================================================================================================
if __name__ == '__main__':
    dataset_folder = "./datasets/dataset_bmj/forLM/"
    ouput_folder = "./Language_Models/"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not os.path.isdir(ouput_folder):
        os.mkdir(ouput_folder)
    files = os.listdir(dataset_folder)

    # Initialize models:
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    config = GPT2Config.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2', config=config)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Train models:
    for filename in files:
        author = filename.split(".")[0]

        train_dataset = TextDataset(
              tokenizer=tokenizer,
              file_path=dataset_folder + filename,
              block_size=128)

        training_args = TrainingArguments(
            output_dir=ouput_folder + author,  # output directory
            num_train_epochs=20,              # total # of training epochs
            per_device_train_batch_size=16,  # batch size per device during training
            warmup_steps=500,                # number of warmup steps for learning rate scheduler
            weight_decay=0.01,               # strength of weight decay
        )

        trainer = Trainer(
            model=model,                         # the instantiated 🤗 Transformers model to be trained
            args=training_args,                  # training arguments, defined above
            data_collator=data_collator,
            train_dataset=train_dataset,         # training dataset
        )

        trainer.train()
        trainer.save_model()

        text = open(dataset_folder + filename, "r").read()
        print(calculate_perplexity(text, model, tokenizer, device))


# config = GPT2Config.from_pretrained(ouput_folder + 'angela wade')
# model = GPT2LMHeadModel.from_pretrained(ouput_folder + 'angela wade', config=config).to(device)

# def calculate_perplexity(text, model, tokenizer):
#     tokenize_input = tokenizer.tokenize(text)
#     tensor_input = torch.tensor([tokenizer.convert_tokens_to_ids(tokenize_input)])
#     with torch.no_grad():
#         loss = model(tensor_input, labels=tensor_input)[0]
#     return math.exp(loss.item())
#