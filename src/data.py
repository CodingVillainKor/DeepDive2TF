import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast

ds = load_dataset("Pradeep016/career-guidance-qa-dataset")

tokenizer = PreTrainedTokenizerFast(tokenizer_file="tokenizer1.json", pad_token="[PAD]")

class Dataset:
    def __init__(self, dataset=ds, tokenizer=tokenizer):
        # dataset["train"] but just [:100]
        self.dataset = dataset["train"].select(range(32))
        
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        text = item["question"]
        label = item["answer"]
        tokenized_text = self.tokenizer.encode(text)
        tokenized_label = self.tokenizer.encode(label)
        return {
            "text": tokenized_text,
            "label": tokenized_label,
        }


def collate_fn(batch):
    questions = [item["text"] for item in batch]
    answers = [item["label"] for item in batch]
    # pad zero to each answer's beginning in answers
    for answer in answers:
        answer.insert(0, 0)

    max_length_question = max(len(text) for text in questions)
    max_length_answer = max(len(text) for text in answers)
    padded_questions = [text + [0] * (max_length_question - len(text)) for text in questions]
    padded_answers = [text + [0] * (max_length_answer - len(text)) for text in answers]
    return torch.tensor(padded_questions), torch.tensor(padded_answers)


dl = DataLoader(Dataset(), batch_size=32, shuffle=True, collate_fn=collate_fn)

if __name__ == "__main__":
    for batch in dl:
        print(batch["question"].shape, batch["answer"].shape)
        breakpoint()