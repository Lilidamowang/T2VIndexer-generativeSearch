from transformers import (
    AdamW,
    T5ForConditionalGeneration,
    T5Tokenizer,
    T5Config,
    get_linear_schedule_with_warmup
)
pretrain_model = T5ForConditionalGeneration.from_pretrained('t5-base')