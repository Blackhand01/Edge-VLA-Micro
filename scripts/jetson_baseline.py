import torch
import time
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from PIL import Image
import urllib.request

print("🚀 Inizializzazione Benchmark Jetson VLM...")
print(f"Dispositivo: {torch.cuda.get_device_name(0)}")

# 1. Carica il Modello (Qwen2-VL 2B) in float16 per risparmiare memoria
model_id = "Qwen/Qwen2-VL-2B-Instruct"
print(f"📥 Scaricamento/Caricamento modello {model_id} (potrebbe volerci un po')...")

processor = AutoProcessor.from_pretrained(model_id)
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    device_map="cuda"
)

# 2. Crea un'immagine dummy (o scaricane una di test)
image_path = "test_drone.jpg"
urllib.request.urlretrieve("https://raw.githubusercontent.com/PX4/PX4-Autopilot/main/boards/px4/fmu-v6x/fmu-v6x.jpg", image_path)
image = Image.open(image_path)

# 3. Prepara il prompt (esattamente come nel tuo drone_controller)
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Describe this image in detail and tell me if you see a drone flight controller."}
        ]
    }
]

text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = processor(text=[text], images=[image], padding=True, return_tensors="pt").to("cuda")

print("\n🧠 Inizio Inferenza...")
torch.cuda.reset_peak_memory_stats()
torch.cuda.synchronize() # Sincronizza la GPU per misurare il tempo esatto

# -- MISURAZIONE TIME-TO-FIRST-TOKEN (TTFT) --
t0 = time.perf_counter()
generated_ids = model.generate(**inputs, max_new_tokens=50)
torch.cuda.synchronize()
t1 = time.perf_counter()

# Estrazione risposta
generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

# -- METRICHE --
total_time = t1 - t0
tokens_generated = len(generated_ids_trimmed[0])
tps = tokens_generated / total_time
peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 3) # in GB

print("\n" + "="*40)
print("🎯 RISULTATO VLM:")
print(output_text)
print("="*40)
print(f"⏱️  Tempo Totale (TTFT + Decode): {total_time:.2f} secondi")
print(f"⚡  Velocità: {tps:.2f} Tokens Per Second (TPS)")
print(f"💾  Picco Memoria GPU (VRAM): {peak_mem:.2f} GB / 8.00 GB")
print("="*40)