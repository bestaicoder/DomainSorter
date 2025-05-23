import openai, os, json, dotenv
from tqdm import tqdm
import concurrent.futures, threading, logging
import re

# --- FASTEST SETTINGS: aggressive parallelism, minimal I/O, batch all at once, skip prompt saving ---

logging.basicConfig(level=logging.WARNING, format='[%(asctime)s] %(levelname)s:%(name)s: %(message)s')
logger = logging.getLogger("openai_category_verification")
dotenv.load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# Load all data up front
with open('Results/ratings.json') as f:
    top5_ratings = json.load(f)
with open('Input/categories.txt') as f:
    categories = [c.strip() for c in f.read().split(';') if c.strip()]
with open('Results/ResultsData.json') as f:
    jina_data = json.load(f)
jina_lookup = {
    e["url"]: {
        "title": e.get("title", ""),
        "desc": e.get("desc", ""),
        "content": e.get("content", "")
    }
    for e in jina_data if e.get("url")
}

# Filter out 'Other' category from processing
categories_to_process = [cat for cat in categories if cat.lower() != 'other']

category_to_domains = {cat: [] for cat in categories_to_process}
for entry in top5_ratings:
    url = entry.get("url")
    for cat, rating in entry.get("top5_category_ratings", {}).items():
        if cat in category_to_domains:
            category_to_domains[cat].append((url, rating))
for cat in category_to_domains:
    seen = set()
    sorted_urls = []
    for url, rating in sorted(category_to_domains[cat], key=lambda x: x[1], reverse=True):
        if url not in seen:
            sorted_urls.append((url, rating))
            seen.add(url)
    category_to_domains[cat] = sorted_urls

def count_tokens(text):
    try:
        import tiktoken
        enc = tiktoken.encoding_for_model("o4-mini")
        return len(enc.encode(text))
    except Exception:
        return max(1, len(text) // 4)

def batch_by_token_limit(urls, max_tokens=190_000):
    # Try to maximize batch size for speed (fewer API calls)
    batches, batch, tokens = [], [], 0
    for url, rating in urls:
        info = jina_lookup.get(url)
        if not info:
            continue
        t = count_tokens(f"{url}{rating}{info['title']}{info['desc']}{info['content'][:1000]}")
        if batch and tokens + t > max_tokens:
            batches.append(batch)
            batch, tokens = [], 0
        batch.append((url, rating))
        tokens += t
    if batch:
        batches.append(batch)
    return batches

def build_prompt(batch, cat, all_possible_urls):
    # Enhanced prompt engineering for better categorization accuracy
    p = (
        f"You are an expert website categorization specialist with deep knowledge of business domains and technology sectors.\n\n"
        f"TASK: Categorize websites for the category '{cat}'\n\n"
        f"CATEGORY DEFINITION: '{cat}' refers to websites that are primarily focused on, offer services for, or are core examples of this specific business/technology domain.\n\n"
        f"EVALUATION CRITERIA:\n"
        f"- The website's PRIMARY business model or core offering must align with '{cat}'\n"
        f"- Look for explicit mentions of relevant services, products, or industry focus\n"
        f"- Consider the company's main value proposition and target market\n"
        f"- Exclude websites that only tangentially relate to the category\n"
        f"- Exclude generic platforms unless they specialize in this category\n\n"
        f"REFERENCE CONTEXT: Here are all domains previously associated with '{cat}': {json.dumps(all_possible_urls[:20])}{'...' if len(all_possible_urls) > 20 else ''}\n\n"
        f"CANDIDATE WEBSITES (ranked by AI relevance score):\n"
    )
    
    for url, rating in batch:
        e = jina_lookup.get(url, {})
        title = e.get('title', '').strip()
        desc = e.get('desc', '').strip()
        content = e.get('content', '')[:800].strip()  # Slightly reduced for focus
        
        p += (
            f"\n{'='*50}\n"
            f"URL: {url}\n"
            f"AI Relevance Score: {rating:.3f}\n"
            f"Title: {title or 'N/A'}\n"
            f"Description: {desc or 'N/A'}\n"
            f"Content Preview: {content or 'N/A'}\n"
        )
    
    p += (
        f"\n{'='*50}\n"
        f"INSTRUCTIONS:\n"
        f"1. Analyze each website's primary business focus\n"
        f"2. Determine if it's a clear, unambiguous fit for '{cat}'\n"
        f"3. Be selective - only include websites where '{cat}' is the main business category\n"
        f"4. When in doubt, exclude rather than include\n\n"
        f"OUTPUT FORMAT: Return ONLY a valid JSON array of URLs that clearly belong to '{cat}'.\n"
        f"Example: [\"https://example1.com\", \"https://example2.com\"]\n"
        f"If no websites qualify, return: []\n\n"
        f"JSON Response:"
    )
    return p

def extract_json_list(text):
    try:
        match = re.search(r'\[\s*(".*?"\s*(,\s*".*?"\s*)*)?\]', text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return json.loads(text)
    except Exception:
        return []

file_write_lock = threading.Lock()

def process_batch(args):
    cat, url_batch, results, output_file, batch_idx, all_possible_urls = args
    batch = [(u, r) for u, r in url_batch if jina_lookup.get(u)]
    if not batch:
        return []
    prompt = build_prompt(batch, cat, all_possible_urls)
    try:
        resp = openai.chat.completions.create(model="o4-mini",reasoning_effort="high", messages=[{"role":"user","content":prompt}])
        ans = resp.choices[0].message.content.strip()
        relevant = extract_json_list(ans)
        relevant = [u for u in relevant if isinstance(u, str)]
    except Exception as e:
        logger.error(f"{cat}: {e}")
        relevant = []
    with file_write_lock:
        results.setdefault(cat, []).extend(relevant)
        results[cat] = list(dict.fromkeys(results[cat]))
    return relevant

results, output_file = {}, "Results/ResultsVerif.json"

# MAXIMUM parallelism for speed (use all logical CPUs * 4, but not more than 64)
max_workers = 64
batch_args_all = []
for cat, urls in category_to_domains.items():
    url_batches = batch_by_token_limit(urls, 5000)
    all_possible_urls = [u for u, _ in urls]
    for batch_idx, batch in enumerate(url_batches):
        batch_args_all.append((cat, batch, results, output_file, batch_idx, all_possible_urls))

# Output progress for all iterations in tqdm
with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as ex:
    for i, _ in enumerate(
        tqdm(
            ex.map(process_batch, batch_args_all, chunksize=1),
            total=len(batch_args_all),
            desc="All category batches (max speed)",
            leave=True
        )
    ):
        tqdm.write(f"Processed batch {i+1}/{len(batch_args_all)}")

# Collect all domains that were assigned to any category
assigned_domains = set()
for cat_domains in results.values():
    assigned_domains.update(cat_domains)

# Get all domains from the original data and put unassigned ones in 'Other'
all_domains = set()
for entry in top5_ratings:
    if entry.get("url"):
        all_domains.add(entry["url"])

unassigned_domains = all_domains - assigned_domains
results["Other"] = list(unassigned_domains)

# Write results ONCE at the end for speed
with open(output_file, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

# Print summary
for cat in results:
    if cat == "Other":
        print(f"Category '{cat}': {len(results[cat])} unassigned domains")
    else:
        total_possible = len([u for u, _ in category_to_domains.get(cat, [])])
        verified_count = len(results[cat])
        percent = (verified_count / total_possible * 100) if total_possible else 0
        print(f"Category '{cat}': {verified_count}/{total_possible} ({percent:.1f}%) verified")
