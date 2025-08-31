import csv
import json
import time
import asyncio
import aiohttp
from openai import OpenAI
import os
from urllib.parse import quote_plus
import re

# Set your OpenAI API key
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

async def get_wikipedia_page_length(session, wikipedia_url):
    """Get Wikipedia page length using the API"""
    if not wikipedia_url:
        return 0
    
    # Extract page title from URL
    page_title = wikipedia_url.split('/')[-1]
    
    endpoint = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "titles": page_title,
        "prop": "revisions",
        "rvslots": "main",
        "rvprop": "content",
        "format": "json"
    }
    
    headers = {
        'User-Agent': 'CodyRankingBot/1.0 (enhanced ranking system)'
    }
    
    try:
        async with session.get(endpoint, params=params, headers=headers, timeout=10) as response:
            if response.status == 200:
                data = await response.json()
                pages_data = data["query"]["pages"]
                for _, pdata in pages_data.items():
                    if "revisions" in pdata:
                        content = pdata["revisions"][0]["slots"]["main"]["*"]
                        return len(content)
            return 0
    except Exception as e:
        print(f"Error getting Wikipedia length for {page_title}: {e}")
        return 0

async def get_google_results_count(session, person_name):
    """Get approximate Google results count using Google Custom Search API"""
    # Note: This is a placeholder - you'll need to set up Google Custom Search API
    # For now, we'll simulate with a random-ish number based on name
    # Replace this with actual Google Custom Search API call
    
    # Simulated results for testing - remove this when implementing real API
    import hashlib
    hash_obj = hashlib.md5(person_name.encode())
    hash_hex = hash_obj.hexdigest()
    simulated_count = int(hash_hex[:6], 16) * 1000
    
    print(f"Simulated Google results for {person_name}: {simulated_count}")
    return simulated_count

def get_llm_score(wikipedia_url, metric_name, metric_description):
    """Get LLM score for subjective metrics (0-5 scale)"""
    try:
        prompt = f"""Rate the person on the Wikipedia page {wikipedia_url} on a scale of 0-5 for {metric_name}.

{metric_description}

Respond with only a single number from 0-5, where:
0 = None/Minimal
1 = Very Low  
2 = Low
3 = Moderate
4 = High
5 = Very High/Maximum

Person: {wikipedia_url}
Score:"""

        response = client.chat.completions.create(
            model="gpt-5-nano",
            messages=[{"role": "user", "content": prompt}],
            #max_completion_tokens=5,
            #temperature=0.1
        )
        
        score_text = response.choices[0].message.content.strip()
        
        # Extract number from response
        print(score_text)
        score_match = re.search(r'[0-5]', score_text)
        if score_match:
            score = int(score_match.group())
            return score
        else:
            print(f"Could not parse score for {wikipedia_url} - {metric_name}: {score_text}")
            return 0
            
    except Exception as e:
        print(f"Error getting LLM score for {wikipedia_url} - {metric_name}: {e}")
        return 0

async def process_person(session, person_data, openai_semaphore):
    """Process a single person and get all their scores"""
    name = person_data['name']
    wikipedia_url = person_data['wikipedia']
    
    print(f"Processing: {name}")
    
    # Get Wikipedia page length
    page_length = await get_wikipedia_page_length(session, wikipedia_url)
    await asyncio.sleep(0.1)  # Be respectful to Wikipedia
    
    # Get Google results count
    google_results = await get_google_results_count(session, name)
    # await asyncio.sleep(0.5)  # Rate limiting
    
    # Get LLM scores for subjective metrics
    metrics = {
        'political_influence': 'Political influence and power in government, policy-making, or political movements',
        'cultural_position': 'Cultural significance, impact on arts, entertainment, or social movements',
        'scientific_pioneer': 'Contributions to scientific discovery, research, or technological advancement',
        'cultural_pioneer': 'Pioneering work in culture, arts, philosophy, or social change'
    }
    
    llm_scores = {}
    for metric, description in metrics.items():
        async with openai_semaphore:
            score = get_llm_score(wikipedia_url, metric, description)
            llm_scores[metric] = score
    
#     # Get number of books (using LLM for now)
#     books_prompt = f"""How many books has {name} written or been the primary subject of major biographies? 
    
# Respond with only a number (estimate if exact count unknown).
# If it's someone who has written books, count their authored works.
# If it's someone who hasn't written books, count major biographies about them.

# Person: {name}
# Number of books:"""

#     try:
#         books_response = client.chat.completions.create(
#             model="gpt-5-nano",
#             messages=[{"role": "user", "content": books_prompt}],
#             max_tokens=10,
#             temperature=0.1
#         )
#         books_text = books_response.choices[0].message.content.strip()
#         books_match = re.search(r'\d+', books_text)
#         number_of_books = int(books_match.group()) if books_match else 0
#         time.sleep(1)
        
#     except Exception as e:
#         print(f"Error getting book count for {name}: {e}")
#         number_of_books = 0

    number_of_books = 0
    
    result = {
        'name': name,
        'page_length': page_length,
        'google_results': google_results,
        'political_influence': llm_scores['political_influence'],
        'cultural_position': llm_scores['cultural_position'],
        'number_of_books': number_of_books,
        'scientific_pioneer': llm_scores['scientific_pioneer'],
        'cultural_pioneer': llm_scores['cultural_pioneer'],
        'sitelinks': person_data['sitelinks'],
        'wikipedia_url': wikipedia_url,
        'wikidata_uri': person_data['wikidata_uri']
    }
    
    print(f"Completed: {name} - Page: {page_length}, Google: {google_results}, Books: {number_of_books}")
    return result

def load_existing_enhanced_people():
    """Load existing enhanced people to avoid reprocessing"""
    existing_people = set()
    if os.path.exists('enhanced_people.csv'):
        try:
            with open('enhanced_people.csv', 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['name']:  # Skip empty rows
                        existing_people.add(row['name'].strip())
            print(f"Found {len(existing_people)} existing people in enhanced_people.csv")
        except Exception as e:
            print(f"Error reading existing enhanced_people.csv: {e}")
    return existing_people

async def main():
    start_time = time.time()
    TARGET_COUNT = 100  # Change this number to process more people
    
    # Load existing enhanced people
    existing_people = load_existing_enhanced_people()
    
    # Read people from wikidata_people.csv until we have TARGET_COUNT total
    people_data = []
    with open('wikidata_people.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if len(existing_people) + len(people_data) >= TARGET_COUNT:
                break
            
            # Skip if already processed
            if row['name'].strip() not in existing_people:
                people_data.append(row)
    
    if not people_data:
        elapsed_time = time.time() - start_time
        print(f"All {len(existing_people)} people already processed! Target: {TARGET_COUNT}")
        print(f"Total time: {elapsed_time:.2f} seconds")
        return
        
    print(f"Will process {len(people_data)} new people (existing: {len(existing_people)}, target: {TARGET_COUNT})")
    
    # Create semaphores for rate limiting
    http_semaphore = asyncio.Semaphore(50)  # Max 10 concurrent HTTP requests
    openai_semaphore = asyncio.Semaphore(50)  # Max 5 concurrent OpenAI API calls
    
    async def process_person_with_semaphore(session, person_data):
        async with http_semaphore:
            return await process_person(session, person_data, openai_semaphore)
    
    # Process people with async HTTP session
    connector = aiohttp.TCPConnector(limit=20)
    timeout = aiohttp.ClientTimeout(total=60)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        # Process all people concurrently with semaphore limits
        tasks = [process_person_with_semaphore(session, person_data) for person_data in people_data]
        
        results = []
        completed = 0
        
        # Load existing data first
        existing_data = []
        if os.path.exists('enhanced_people.csv'):
            try:
                with open('enhanced_people.csv', 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    existing_data = [row for row in reader if row['name']]  # Skip empty rows
            except Exception as e:
                print(f"Error reading existing data: {e}")
        
        # Process in batches and save incrementally
        for coro in asyncio.as_completed(tasks):
            try:
                result = await coro
                results.append(result)
                completed += 1
                
                print(f"Progress: {completed}/{len(people_data)} completed")
                
                # Save progress every 10 completions
                if completed % 10 == 0:
                    all_data = existing_data + results
                    with open('enhanced_people.csv', 'w', newline='', encoding='utf-8') as f:
                        fieldnames = ['name', 'page_length', 'google_results', 'political_influence', 
                                    'cultural_position', 'number_of_books', 'scientific_pioneer', 
                                    'cultural_pioneer', 'sitelinks', 'wikipedia_url', 'wikidata_uri']
                        writer = csv.DictWriter(f, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerows(all_data)
                
            except Exception as e:
                print(f"Error processing person: {e}")
                completed += 1
                continue
        
        # Final save with all data
        all_data = existing_data + results
        with open('enhanced_people.csv', 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['name', 'page_length', 'google_results', 'political_influence', 
                        'cultural_position', 'number_of_books', 'scientific_pioneer', 
                        'cultural_pioneer', 'sitelinks', 'wikipedia_url', 'wikidata_uri']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_data)
    
    elapsed_time = time.time() - start_time
    total_people = len(existing_people) + len(results)
    
    print(f"Completed! Processed {len(results)} new people. Total people: {total_people}")
    print(f"Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.1f} minutes)")
    if len(results) > 0:
        avg_time_per_person = elapsed_time / len(results)
        print(f"Average time per person: {avg_time_per_person:.2f} seconds")
    print(f"Results saved to enhanced_people.csv")

if __name__ == "__main__":
    asyncio.run(main())