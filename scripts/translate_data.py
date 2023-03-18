import os
import openai
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Replace 'your_api_key' with your actual API key
openai.api_key = os.environ.get("OPENAI_API_KEY")

TARGET_LANGUAGE = "zh-Hant"

def download_alpaca_data():
    import requests
    # Source from Standford ALPACA dataset: https://github.com/tatsu-lab/stanford_alpaca
    url = "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json"
    response = requests.get(url)
    if response.status_code == 200:
        with open("alpaca_data.json", "wb") as file:
            file.write(response.content)
        print("File downloaded successfully!")
    else:
            print("Failed to download file.")

def translate_text(value, target_language):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": f"I want you to act as an {target_language} translator,spelling corrector and improver.And you are also proficient in professional vocabulary such as software programs/mathematics/physics/chemistry/literature."},
                {"role": "user", "content": f"I will talk to you in any language and you will detect the language, translate it and answer with a corrected and improved version of my text, in {target_language}.If you encounter any software programming language(Python/Javascript/C++/Swift...etc)/mathematical calculations, please don't translate them, please maintain them. I want you to replace my simplified A0-level words and sentences with more beautiful and elegant high-level {target_language} words and sentences. Keep the same meaning, but make them more literal. I want you to reply only with corrections and improvements, nothing else, no explanation,and don't add any your comment or notes. The text is:\n\n{value}"},
            ],
        max_tokens=1024,
        temperature=0,
        )
    return response.choices[0]["message"]["content"].strip()

def translate_item(item):
    translated_item = {}
    for key, value in item.items():
        if value:
            translated_value = translate_text(value, target_language=TARGET_LANGUAGE)
            translated_item[key] = translated_value
        else:
            translated_item[key] = ''
    return translated_item

# Maximum number of parallel requests
MAX_PARALLEL_REQUESTS = 100

# Check if the ALPACA dataset is downloaded
if not os.path.exists("alpaca_data.json"):
    print("Downloading ALPACA dataset...")
    download_alpaca_data()

# Assuming the input JSON is in a file named 'input.json'
with open('alpaca_data.json', 'r') as f:
    data = json.load(f)

start = 40000
end = 55000
translated_data = []

if start is None:
    start = 0
if end is None:
    end = len(data)
if end > len(data):
    end = len(data)
data = data[start:end]

with ThreadPoolExecutor(max_workers=MAX_PARALLEL_REQUESTS) as executor:
    futures = {executor.submit(translate_item, item): item for item in data}
    
    for future in tqdm(as_completed(futures), total=len(futures), desc="Translating"):
        translated_data.append(future.result())

# Save the translated data to a new JSON file named 'translated_data.json'
with open(f'translated_data_up_to_{start}_to_{end}.json', 'w') as f:
    json.dump(translated_data, f, ensure_ascii=False, indent=4)

print(f"Translation complete. The translated data is saved in 'translated_data_from_{start}_to_{end}.json'")
