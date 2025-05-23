import asyncio
import aiohttp
import json
import os
import random
import re
import orjson
import aiofiles
import pandas as pd
from pathlib import Path
import time
from tqdm.asyncio import tqdm
import ssl
import certifi
import os

# --- Configuration ---

# Token for API requests
if os.environ.get("JINA_API_TOKEN"):
    token = os.environ.get("JINA_API_TOKEN")
else:
    print("No JINA_API_TOKEN found in environment variables. Please set it.")
    exit(1)

RATE_LIMIT = 500/60
SOCKET_TIMEOUT = 1.5
DNS_TIMEOUT = 1.0
DOMAIN_PROCESSING_TIMEOUT = 90
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.1.1 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1"
]

# --- Helper functions ---

def create_ssl_context():
    """Create SSL context with proper certificate verification for macOS"""
    ssl_context = ssl.create_default_context(cafile=certifi.where())
    ssl_context.check_hostname = True
    ssl_context.verify_mode = ssl.CERT_REQUIRED
    return ssl_context

async def check_api_connectivity():
    """Check if we can connect to the Jina API before processing"""
    ssl_context = create_ssl_context()
    connector = aiohttp.TCPConnector(ssl=ssl_context)
    timeout = aiohttp.ClientTimeout(total=30)
    
    try:
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            headers = {
                'Accept': 'application/json',
                'Authorization': f'Bearer {token}',
                'Content-Type': 'application/json',
                'User-Agent': random.choice(USER_AGENTS)
            }
            
            # Test with a simple request
            test_data = {'url': 'https://httpbin.org/status/200'}
            
            async with session.post('https://r.jina.ai/', headers=headers, json=test_data) as response:
                if response.status == 200:
                    print("✅ API connectivity check passed")
                    return True
                else:
                    print(f"❌ API connectivity check failed with status: {response.status}")
                    return False
                    
    except Exception as e:
        print(f"❌ API connectivity check failed: {str(e)}")
        return False

def calculate_optimal_concurrency(rate_limit: int, total: int) -> int:
    return int(max(2, min(500, total, rate_limit * 5)))

async def check_domain_exists_async(domain, resolver):
    from urllib.parse import urlparse
    import aiodns
    try:
        parsed = urlparse(domain if '://' in domain else f'http://{domain}')
        hostname = parsed.netloc if parsed.netloc else parsed.path
        try:
            await resolver.query(hostname, 'A')
            return domain, True
        except aiodns.error.DNSError:
            return domain, False
    except Exception:
        return domain, False

async def check_domains_existence_async(domains):
    import aiodns
    existing = {}
    non_existing = {}
    resolver = aiodns.DNSResolver(timeout=DNS_TIMEOUT)
    sem = asyncio.Semaphore(calculate_optimal_concurrency(RATE_LIMIT, len(domains)))
    
    async def check_with_semaphore(domain):
        async with sem:
            return await check_domain_exists_async(domain, resolver)
    
    tasks = [check_with_semaphore(domain) for domain in domains]
    
    # Use tqdm for domain existence checking
    with tqdm(total=len(domains), desc="Checking domain existence", unit="domains") as pbar:
        for coro in asyncio.as_completed(tasks):
            domain, exists = await coro
            if exists:
                existing[domain] = {"exists": True}
            else:
                non_existing[domain] = {"exists": False}
            pbar.update(1)
    
    return existing, non_existing

class ImmediateSaver:
    def __init__(self, filename: str, flush_every: int = 20):
        self.filename    = filename
        self.flush_every = max(1, flush_every)
        self.data        = {}
        self._pending    = 0
        self._lock       = asyncio.Lock()
        if os.path.exists(filename):
            try:
                with open(filename, "rb") as f:
                    self.data = orjson.loads(f.read())
            except orjson.JSONDecodeError:
                print(f"[ImmediateSaver] Could not parse {filename}. Starting fresh.")
    async def add(self, new_data: dict):
        async with self._lock:
            self.data.update(new_data)
            self._pending += 1
            if self._pending >= self.flush_every:
                await self._flush()
    def get_data(self) -> dict:
        return self.data
    async def _flush(self):
        async with aiofiles.open(self.filename, "wb") as f:
            await f.write(orjson.dumps(self.data, option=orjson.OPT_INDENT_2))
        self._pending = 0

class RateLimiter:
    def __init__(self, rate_limit):
        self.rate_limit = rate_limit
        self.tokens = rate_limit
        self.updated_at = time.time()
        self.lock = asyncio.Lock()
    async def acquire(self):
        while True:
            async with self.lock:
                now       = time.time()
                elapsed   = now - self.updated_at
                self.tokens = min(self.rate_limit,
                                   self.tokens + elapsed * self.rate_limit)
                self.updated_at = now
                if self.tokens >= 1:
                    self.tokens -= 1
                    return
                sleep_for = (1 - self.tokens) / self.rate_limit
            await asyncio.sleep(sleep_for)

def get_domain_variants(domain):
    from urllib.parse import urlparse
    domain = domain.strip()
    variants = []
    if '://' in domain:
        parsed = urlparse(domain)
        hostname = parsed.netloc
    else:
        hostname = domain
    if hostname.startswith('www.'):
        hostname = hostname[4:]
    variants.append(f"https://{hostname}")
    variants.append(f"https://www.{hostname}")
    variants.append(f"http://{hostname}")
    variants.append(f"http://www.{hostname}")
    return variants

async def fetch_domain_info(session, url):
    headers = {
        'Accept': 'application/json',
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json',
        'User-Agent': random.choice(USER_AGENTS),
        'X-Engine': 'browser',
        'X-Retain-Images': 'none',
        'X-Return-Format': 'markdown',
        'Referer': 'https://www.google.com/'
    }
    data = {
        'url': url.strip()
    }
    
    try:
        async with session.post('https://r.jina.ai/', headers=headers, json=data, timeout=45) as response:
            response.raise_for_status()
            return await response.json()
    except aiohttp.ClientError as e:
        raise Exception(f"Client error: {str(e)}")
    except asyncio.TimeoutError:
        raise Exception("Request timeout")
    except Exception as e:
        raise Exception(f"Request failed: {str(e)}")

async def process_domain(session, domain, rate_limiter, results_saver, error_saver):
    domain_variants = get_domain_variants(domain)
    for url_to_try in domain_variants:
        await rate_limiter.acquire()
        try:
            result = await fetch_domain_info(session, url_to_try)
            domain_result = {
                domain: {
                    "title":       result.get("data", {}).get("title", ""),
                    "description": result.get("data", {}).get("description", ""),
                    "content":     result.get("data", {}).get("content", ""),
                }
            }
            await results_saver.add(domain_result)
            return domain_result
        except Exception as e:
            error_msg = f"Request failed: {str(e)}"
            if url_to_try == domain_variants[-1]:
                await error_saver.add({domain: {"error": error_msg}})
    return {domain: {"error": "All variants failed"}}

# --- Main processing function ---

async def process_domains():
    # Create directories
    os.makedirs("Results", exist_ok=True)
    
    # Check API connectivity before proceeding
    print("Checking API connectivity...")
    if not await check_api_connectivity():
        print("❌ Cannot connect to Jina API. Please check your internet connection and certificates.")
        print("Try running: pip3 install --upgrade certifi")
        return
    
    # Read domains from Input/domains.txt
    domains_file = "Input/domains.txt"
    if not os.path.exists(domains_file):
        print(f"Error: {domains_file} not found!")
        return
    
    with open(domains_file, 'r') as f:
        domains = [line.strip() for line in f if line.strip()]
    
    if not domains:
        print("No domains found in the input file.")
        return
    
    print(f"Found {len(domains)} domains in the input file.")
    
    # Set up savers
    results_saver = ImmediateSaver("Results/ResultsData.json")
    error_saver = ImmediateSaver("Results/ErrorResults.json")
    
    print("Checking domain existence...")
    existing_domains, non_existing_domains = await check_domains_existence_async(domains)
    
    print(f"Found {len(existing_domains)} existing domains and {len(non_existing_domains)} non-existing domains.")
    
    domains_to_process = list(existing_domains.keys())
    if not domains_to_process:
        print("No domains to process.")
        return
    
    concurrency = calculate_optimal_concurrency(RATE_LIMIT, len(domains_to_process))
    print(f"Processing {len(domains_to_process)} domains with concurrency={concurrency}, rate={RATE_LIMIT} req/s")
    
    rate_limiter = RateLimiter(RATE_LIMIT)
    
    # Create SSL context and connector with proper certificate handling
    ssl_context = create_ssl_context()
    connector = aiohttp.TCPConnector(limit=concurrency, ttl_dns_cache=300, ssl=ssl_context)
    timeout = aiohttp.ClientTimeout(total=60)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        queue = asyncio.Queue()
        for d in domains_to_process:
            queue.put_nowait(d)
        
        total = len(domains_to_process)
        
        # Create progress bar for domain processing
        pbar = tqdm(total=total, desc="Processing domains", unit="domains", 
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        async def worker():
            while not queue.empty():
                try:
                    domain = queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                
                try:
                    await asyncio.wait_for(
                        process_domain(session, domain, rate_limiter, results_saver, error_saver),
                        timeout=DOMAIN_PROCESSING_TIMEOUT
                    )
                except asyncio.TimeoutError:
                    await error_saver.add({domain: {"error": f"Processing timed out after {DOMAIN_PROCESSING_TIMEOUT}s"}})
                    tqdm.write(f"Processing domain {domain} timed out after {DOMAIN_PROCESSING_TIMEOUT}s. Skipping.")
                except Exception as e:
                    await error_saver.add({domain: {"error": f"Unexpected error: {str(e)}"}})
                    tqdm.write(f"Unexpected error processing {domain}: {str(e)}")
                
                pbar.update(1)
                queue.task_done()
        
        workers = [asyncio.create_task(worker()) for _ in range(int(concurrency))]
        await asyncio.gather(*workers)
        pbar.close()
    
    print("Processing complete.")
    print(f"Successful: {len(results_saver.get_data())}, Errors: {len(error_saver.get_data())}")
    
    # Convert results to the expected format
    results_list = []
    if os.path.exists("Results/ResultsData.json"):
        with open("Results/ResultsData.json", 'r') as f:
            results_data = json.load(f)
            for domain, data in results_data.items():
                results_list.append({
                    "url": domain,
                    "title": data.get("title", ""),
                    "desc": data.get("description", ""),
                    "content": data.get("content", "")
                })
        
        # Save in the expected format
        with open("Results/ResultsData.json", 'w') as f:
            json.dump(results_list, f, indent=2)
    
    print("All processing complete. Results saved to Results/ directory.")

if __name__ == "__main__":
    asyncio.run(process_domains())