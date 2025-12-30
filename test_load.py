from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(
    r"C:\DESKTOP\5th SEM EL\company document classify\bert_company_model"
)
print("✅ Tokenizer loaded successfully")
