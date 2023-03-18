import os
import time
import openai
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from ratelimiter import RateLimiter
from retrying import retry

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

@retry(stop_max_attempt_number=10)
@RateLimiter(max_calls=58, period=60)
def translate_text(value, target_language):
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                {"role": "system", "content": f"I want you to act as an {target_language} translator,spelling corrector and improver.And you are also proficient in professional vocabulary such as software programs/mathematics/physics/chemistry/literature."},
                {"role": "user", "content": f"I will talk to you in any language and you will detect the language, translate it and answer with a corrected and improved version of my text, in {target_language}. Youd could detect which text is programing language or mathmatics. If you encounter any programming languages(Python/Javascript/C++/Swift...etc)/mathematical calculations,ex: `def func: `. Please keep them and don't translate. I want you to replace my simplified A0-level words and sentences with more beautiful and elegant high-level {target_language} words and sentences. Keep the same meaning, but make them more literal. I want you to reply only with corrections and improvements, nothing else, no explanation,and don't add any your comment or notes. The text is:\n\n{value}"},
            ],
        max_tokens=1024,
        temperature=0,
        )
    return response.choices[0]["message"]["content"].strip()

def translate_item(item, target_language):
    translated_item = {}
    for key, value in item.items():
        if value:
            translated_value = translate_text(value, target_language=target_language)
            translated_item[key] = translated_value
        else:
            translated_item[key] = ''
    return translated_item

def load_data():
    # Check if the ALPACA dataset is downloaded
    if not os.path.exists("alpaca_data.json"):
        print("Downloading ALPACA dataset...")
        download_alpaca_data()

    # Assuming the input JSON is in a file named 'input.json'
    with open('alpaca_data.json', 'r') as f:
        data = json.load(f) 
    return data       

def chunk_data(restart_index, data, chunk_size=1000):
    """Chunk the data into smaller chunks to avoid hitting the API rate limit.

    yields:
        start: The starting index of the chunk
        end: The ending index of the chunk
    """
    start = restart_index
    end = restart_index + chunk_size
    while start < len(data):
        if end > len(data):
            end = len(data)
        yield start, end
        start = end
        end += chunk_size
    
def process_data(start, end, data, target_language, max_parallel_requests=100):
    """Translate the data from start to end."""
    translated_data = []

    if start is None:
        start = 0
    if end is None:
        end = len(data)
    if end > len(data):
        end = len(data)
    data = data[start:end]

    with ThreadPoolExecutor(max_workers=max_parallel_requests) as executor:
        futures = {executor.submit(translate_item, item, target_language): item for item in data}

        for future in tqdm(as_completed(futures), total=len(futures), desc="Translating"):
            translated_data.append(future.result())

    # Save the translated data to a new JSON file named 'translated_data.json'
    with open(f'translated_data_up_to_{start}_to_{end}.json', 'w') as f:
        json.dump(translated_data, f, ensure_ascii=False, indent=4)

    print(f"Translation complete. The translated data is saved in 'translated_data_from_{start}_to_{end}.json'")

def combine_data():
    """Combine all the translated data into one JSON file."""
    data = []
    for file in os.listdir():
        if file.startswith("translated_data_up_to_"):
            with open(file, 'r') as f:
                data.extend(json.load(f))
    with open('translated_data.json', 'w') as f:
        json.dump(data, f, ensure_ascii=False, indent=4)
    print("All translated data combined into 'translated_data.json'")

if __name__ == "__main__":
    """
    1. Load the ALPACA dataset
    2. Chunk the dataset into smaller chunks to avoid hitting the API rate limit
    3. Translate each chunk
    """
    # Replace 'your_api_key' with your actual API key
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    TARGET_LANGUAGE = "zh-Hant"        
    # Maximum number of parallel requests
    MAX_PARALLEL_REQUESTS = 3
    CHUNK_SIZE = 20
    data = load_data()
    restart_index = 280
    for start, end in chunk_data(restart_index, data, chunk_size=CHUNK_SIZE):
        process_data(start, end, data, target_language=TARGET_LANGUAGE, max_parallel_requests=MAX_PARALLEL_REQUESTS)
    # Combine all the translated data into one JSON file
    combine_data()
