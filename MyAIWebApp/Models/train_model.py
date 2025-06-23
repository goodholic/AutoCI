# train_model.py
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

def fine_tune_model():
    # 모델과 토크나이저 로드
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # 데이터셋 준비
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path="train.txt",
        block_size=128
    )
    
    # 학습 설정
    training_args = TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=500,
        save_total_limit=2,
    )
    
    # 트레이너 생성
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
        ),
        train_dataset=train_dataset,
    )
    
    # 학습 시작
    trainer.train()
    
    # 모델 저장
    model.save_pretrained("./fine_tuned_model")
    tokenizer.save_pretrained("./fine_tuned_model")