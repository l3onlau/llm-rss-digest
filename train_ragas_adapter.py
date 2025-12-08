import os

import torch
from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from config import settings

OUTPUT_DIR = settings.RAGAS_ADAPTER_PATH


def format_instruction(sample):
    """
    Formats SQuAD v2 data into a RAGAS-style judge prompt.

    ARCHITECTURAL NOTE:
    -------------------
    This training function focuses specifically on 'Context Relevance' (Query <-> Document mapping).
    Prioritize this over 'Faithfulness' because:
    1. Relevance is often the harder task for smaller models (requires semantic understanding of the query).
    2. Irrelevant contexts are the root cause of most hallucinations.

    While this adapter is tuned for Relevance, it helps the model understand the "Judge" persona,
    which can still have positive transfer effects for other metrics, but it is not
    explicitly trained on NLI (Entailment) tasks used for Faithfulness.

    TRAINING NOTE:
    --------------
    **potential overfitting is being ignored** for this phase.
    """
    query = sample["question"]
    context = sample["context"]
    is_relevant = len(sample["answers"]["text"]) > 0
    label = "Relevant" if is_relevant else "Irrelevant"

    prompt = (
        f"<|system|>You are an expert judge evaluating the relevance of a document to a user query. "
        f"Output 'Relevant' or 'Irrelevant'.<|end|>\n"
        f"<|user|>Query: {query}\nDocument: {context}<|end|>\n"
        f"<|assistant|>{label}<|end|>"
    )
    return {"text": prompt}


def train():
    print(f"🚀 Starting Adapter Training for {settings.LLM_MODEL_ID}")
    print(f"📂 Output directory: {OUTPUT_DIR}")

    try:
        print("📚 Loading dataset: squad_v2...")
        dataset = load_dataset("squad_v2", split="train[:1000]")
    except Exception as e:
        print(f"❌ Critical Error loading dataset: {e}")
        return

    dataset = dataset.map(format_instruction)

    # Configs
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    print("🧠 Loading Base Model...")
    model = AutoModelForCausalLM.from_pretrained(
        settings.LLM_MODEL_ID,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model.config.use_cache = False

    # Enable gradients for inputs
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )

    # ------------------------------------------------------------------
    # AUTO-DETECT: Resume vs Fresh Start
    # ------------------------------------------------------------------
    adapter_config_path = os.path.join(OUTPUT_DIR, "adapter_config.json")
    is_resuming = os.path.exists(adapter_config_path)

    if is_resuming:
        print(
            f"🔄 Found existing adapter at {OUTPUT_DIR}. Loading to continue training..."
        )
        model = PeftModel.from_pretrained(model, OUTPUT_DIR, is_trainable=True)
    else:
        print("🆕 No existing adapter found. Initializing new LoRA...")
        peft_config = LoraConfig(
            lora_alpha=32,
            lora_dropout=0.05,
            r=16,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        )
        model = get_peft_model(model, peft_config)

    dynamic_lr = 5e-5 if is_resuming else 2e-4
    print(f"⚙️  Auto-setting Learning Rate to: {dynamic_lr}")

    tokenizer = AutoTokenizer.from_pretrained(settings.LLM_MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    training_args = SFTConfig(
        dataset_text_field="text",
        max_length=settings.CHUNK_SIZE,
        output_dir=OUTPUT_DIR,
        max_steps=30,
        eval_steps=15,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=dynamic_lr,
        fp16=True,
        group_by_length=True,
        logging_steps=5,
        overwrite_output_dir=True,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        processing_class=tokenizer,
        args=training_args,
    )

    print("🏋️‍♂️ Training started...")
    trainer.train()

    print(f"💾 Saving adapter to {OUTPUT_DIR}...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("✅ Done. Adapter is ready for inference.")


if __name__ == "__main__":
    train()
