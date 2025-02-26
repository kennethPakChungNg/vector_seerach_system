import torch
from transformers import CLIPProcessor, CLIPModel
import requests
from PIL import Image
from io import BytesIO
import concurrent.futures
import pandas as pd
from bs4 import BeautifulSoup
import time
import random
import json
from tenacity import retry, stop_after_attempt, wait_exponential
import re

# Session setup
session = requests.Session()

# Pool of user agents to rotate
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Safari/605.1.15",
    "Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:89.0) Gecko/20100101 Firefox/89.0",
    "Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/14.0 Mobile/15E148 Safari/604.1"
]

# Establish initial session by visiting the homepage
session.get("https://www.amazon.in", headers={"User-Agent": random.choice(USER_AGENTS)})

# Lazy loading for CLIP model and processor
_clip_model = None
_clip_processor = None

def get_clip_model(device="cpu"):
    """
    Load CLIP model and processor on the specified device if not already loaded.
    
    Args:
        device (str): Device to load the model on (default "cpu").
    
    Returns:
        tuple: CLIP model and processor.
    """
    global _clip_model, _clip_processor
    if _clip_model is None:
        _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        _clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return _clip_model, _clip_processor

def clean_amazon_url(url):
    """
    Clean and standardize Amazon URLs to the format https://www.amazon.in/dp/ASIN
    """
    match = re.search(r'/dp/(\w{10})', url)
    if match:
        asin = match.group(1)
        return f"https://www.amazon.in/dp/{asin}"
    return url

def is_image_url(url):
    """
    Check if the provided URL is likely a direct image URL.
    """
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp')
    return url.lower().endswith(image_extensions)

def download_image(img_url):
    headers = {"User-Agent": random.choice(USER_AGENTS)}
    try:
        response = requests.get(img_url, headers=headers, timeout=5)
        response.raise_for_status()
        return Image.open(BytesIO(response.content)).convert("RGB")
    except requests.exceptions.Timeout:
        print(f"Timeout error for {img_url}")
        return None
    except Exception as e:
        print(f"Error downloading image {img_url}: {e}")
        return None

@retry(stop=stop_after_attempt(2), wait=wait_exponential(multiplier=2, min=5, max=30))
def get_image_url_from_product_link(product_url):
    """
    Fetch the image URL from a product page using a persistent session and rotating user agents.
    
    Args:
        product_url (str): URL of the product page.
    
    Returns:
        str or None: Image URL if found, None if failed.
    """
    time.sleep(random.uniform(10, 15))  # Increased delay to 10-15 seconds
    product_url = clean_amazon_url(product_url)  # Clean URL before fetching
    headers = {
        "User-Agent": random.choice(USER_AGENTS),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "DNT": "1",
        "Connection": "keep-alive",
        "Upgrade-Insecure-Requests": "1",
        "Referer": "https://www.amazon.in/s?k=electronics"  # Simulate coming from search page
    }
    try:
        response = session.get(product_url, headers=headers, timeout=15)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        img_tag = (
            soup.find('img', {'id': 'landingImage'}) or
            soup.find('img', {'data-a-dynamic-image': True}) or
            soup.find('img', {'class': 'a-dynamic-image'}) or
            soup.find('img', {'class': 'imgTagWrapper'}) or
            soup.find('img', {'class': 'image-item'}) or
            soup.find('img', {'alt': lambda x: x and 'product' in x.lower()}) or
            soup.select_one('div#imgTagWrapperId img') or
            soup.find('img', {'data-old-hires': True})
        )
        if img_tag:
            if 'data-a-dynamic-image' in img_tag.attrs:
                try:
                    image_data = json.loads(img_tag['data-a-dynamic-image'].replace('"', '"'))
                    image_url = list(image_data.keys())[0]  # Get the first URL
                    return image_url
                except json.JSONDecodeError:
                    print(f"Failed to parse data-a-dynamic-image for {product_url}")
            elif 'src' in img_tag.attrs:
                return img_tag['src']
        print(f"No image found on product page: {product_url}")
        return None
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error fetching product page {product_url}: {e}")
        with open('failed_urls.txt', 'a') as f:
            f.write(f"{product_url}\n")
        return None
    except Exception as e:
        print(f"Error fetching product page {product_url}: {e}")
        return None
    finally:
        if 'response' in locals():
            response.close()  # Close response to free up connection

def batch_get_image_embeddings(product_urls, batch_size=32, max_workers=2, device="cpu"):
    """
    Fetch image embeddings for multiple product URLs in batches.
    
    Args:
        product_urls (list): List of product URLs.
        batch_size (int): Number of URLs per batch.
        max_workers (int): Number of concurrent workers (reduced to 2).
        device (str): Device to use for model computations (default "cpu").
    
    Returns:
        list: List of image embeddings (or None for failed URLs).
    """
    model, processor = get_clip_model(device)
    image_urls = [None] * len(product_urls)
    images = [None] * len(product_urls)
    embeddings = [None] * len(product_urls)

    # Step 1: Get image URLs from product URLs
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(get_image_url_from_product_link, url): i for i, url in enumerate(product_urls) if pd.notna(url)}
        for future in concurrent.futures.as_completed(futures):
            i = futures[future]
            try:
                image_urls[i] = future.result()
            except Exception as e:
                print(f"Failed to get image URL for {product_urls[i]}: {e}")
                image_urls[i] = None

    # Step 2: Download images from image URLs
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_image, url): i for i, url in enumerate(image_urls) if url is not None}
        for future in concurrent.futures.as_completed(futures):
            i = futures[future]
            try:
                images[i] = future.result()
            except Exception as e:
                print(f"Failed to download image for {image_urls[i]}: {e}")
                images[i] = None

    # Step 3: Compute embeddings for downloaded images in batches
    for start in range(0, len(product_urls), batch_size):
        end = min(start + batch_size, len(product_urls))
        batch_images = [img for img in images[start:end] if img is not None]
        if batch_images:
            inputs = processor(images=batch_images, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                batch_embeddings = model.get_image_features(**inputs).cpu().numpy()
            j = 0
            for i in range(start, end):
                if images[i] is not None:
                    embeddings[i] = batch_embeddings[j]
                    j += 1
            time.sleep(1)  # Avoid rate limiting between batches

    return embeddings