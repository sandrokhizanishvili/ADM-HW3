import pandas as pd
from bs4 import BeautifulSoup
import requests

from tqdm.notebook import tqdm

import time

import asyncio
import aiohttp
import random
import nest_asyncio
import aiofiles

import os


def michelin_restaurant_urls(page_urls):
    """
    Scrapes Michelin restaurant URLs from a list of Michelin Guide page URLs.

    Args:
        page_urls (list of str): List of URLs corresponding to Michelin Guide pages to scrape.

    Returns:
        list of str: A list of URLs for individual restaurant pages.

    This function performs the following steps:
    1. Iterates over the given list of Michelin Guide page URLs.
    2. Sends an HTTP GET request to each page and retrieves the HTML content.
    3. Parses the HTML content using BeautifulSoup and extracts restaurant URLs by locating 
       specific `<div>` elements with a class name that identifies restaurant cards.
    4. Constructs the full restaurant URL by appending the relative path to the base URL.
    5. Appends each restaurant URL to a list and saves this list to a text file named `restaurant_urls.txt`.
    6. Includes a delay (`time.sleep(1)`) between requests to prevent overwhelming the server.
    
    Prints the number of restaurants found on each page for monitoring progress.

    Example:
        page_urls = ["https://guide.michelin.com/en/restaurants/page1", 
                     "https://guide.michelin.com/en/restaurants/page2"]
        restaurant_urls = michelin_restaurant_urls(page_urls)
    """
    restaurant_urls = []

    page = 1
    for url in tqdm(page_urls):
        
        # Get connection
        cnt = requests.get(url)

        # Get content
        soup = BeautifulSoup(cnt.content, features="lxml")
        
        # Find all div tags where class matches the restaurant card
        div = soup.find_all('div', {'class': 'card__menu selection-card box-placeholder js-restaurant__list_item js-match-height js-map'})
        print(f"# of restaurants on page {page} is {len(div)}")

        for element in div:
            restaurant_urls.append('https://guide.michelin.com' + element.select_one('a')['href'])

        page += 1

        # Delay between requests
        time.sleep(1)

    # Save restaurant URLs
    with open("restaurant_urls.txt", "w") as file:
        for url in restaurant_urls:
            file.write(url + "\n")

    return restaurant_urls



def get_michelin_htmls(restaurant_urls):
    """
    Fetches and saves the HTML content of Michelin restaurant pages asynchronously.

    Args:
        restaurant_urls (list of str): List of URLs for individual restaurant pages.

    This function performs the following steps:
    1. Applies `nest_asyncio` to allow asynchronous calls in environments like Jupyter Notebooks.
    2. Defines an asynchronous function `fetch_html` that:
        - Sends an HTTP GET request to a restaurant URL with randomized user-agent headers.
        - Introduces a random delay to mimic human-like behavior and avoid server throttling.
        - Beautifies and saves the HTML content to a structured directory, grouping by page (20 restaurants per page).
    3. Defines an asynchronous function `fetch_all_html` that:
        - Creates tasks for fetching all restaurant URLs concurrently.
        - Executes the tasks using `asyncio.gather` for efficient concurrent processing.
    4. Executes the asynchronous fetching process using `asyncio.run`.

    Example:
        restaurant_urls = ["https://guide.michelin.com/en/restaurant1", 
                           "https://guide.michelin.com/en/restaurant2"]
        get_michelin_htmls(restaurant_urls)
    """
    # Apply nest_asyncio to allow async calls in Jupyter
    nest_asyncio.apply()

    # Function to fetch HTML content from a single URL
    async def fetch_html(url: str, n_rest: int):
        try:
            headers = {
                "User-Agent": random.choice([
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3", 
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/89.0.4389.82 Safari/537.36",
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
                ])
            }
            
            # Create an asynchronous session to send the HTTP GET request
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers) as response:
                    if response.status == 200:
                        # Delay to mimic human behavior
                        await asyncio.sleep(random.uniform(1, 2))

                        # Read the response body as text
                        html = await response.text()

                        # Beautify the HTML content
                        text = BeautifulSoup(html, features='lxml').prettify()

                        # Group by pages (20 restaurants per page)
                        page = n_rest // 20 + 1
                        os.makedirs(f"page {page}", exist_ok=True)

                        # Save HTML to file
                        async with aiofiles.open(f"page {page}/restaurant_{n_rest+1}.html", "w") as f:
                            await f.write(text)
                    else:
                        print(f"Failed to retrieve {url} with status {response.status}")
        except Exception as e:
            print(f'For {url} error: {e}')
            return url

    # Function to fetch HTML from a list of URLs asynchronously
    async def fetch_all_html(urls: list):
        tasks = [fetch_html(url, index) for index, url in enumerate(urls)]
        await asyncio.gather(*tasks)

    # Run the asynchronous HTML fetching function
    asyncio.run(fetch_all_html(restaurant_urls))
