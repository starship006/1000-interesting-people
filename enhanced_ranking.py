import csv
import json
import time
import asyncio
import aiohttp
import os
from urllib.parse import quote_plus
import re
from inspect_ai import Task, eval, task
from inspect_ai.dataset import Sample
from inspect_ai.solver import generate, system_message

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

def generate_all_samples(people_data):
    """Generate all LLM samples for all people and metrics"""
    samples = []
    sample_metadata = []
    
    metrics = {
        'political_influence': 'Political influence and power in government, policy-making, or political movements',
        'cultural_position': 'Cultural significance, impact on arts, entertainment, or social movements',
        'scientific_pioneer': 'Contributions to scientific discovery, research, or technological advancement',
        'cultural_pioneer': 'Pioneering work in culture, arts, philosophy, or social change'
    }
    
    for person_data in people_data:
        name = person_data['name']
        wikipedia_url = person_data['wikipedia']
        
        # Create samples for each metric
        for metric, description in metrics.items():
            prompt = f"""Rate the person on the Wikipedia page {wikipedia_url} on a scale of 0-5 for {metric}.

{description}

Respond with only a single number from 0-5, where:
0 = None/Minimal
1 = Very Low  
2 = Low
3 = Moderate
4 = High
5 = Very High/Maximum

Person: {wikipedia_url}
Score:"""
            
            samples.append(Sample(input=prompt, target=""))
            sample_metadata.append({
                'person_name': name,
                'metric': metric,
                'wikipedia_url': wikipedia_url,
                'person_data': person_data
            })
        
        # Skip book count for now - just return 0
    
    return samples, sample_metadata

@task
def score_all_people(samples) -> Task:
    """Create task to score all people on all metrics"""
    return Task(
        dataset=samples,
        solver=[
            system_message("You are an expert evaluator. Respond with only the requested number."),
            generate()
        ]
    )

async def get_wikipedia_and_google_data(session, people_data):
    """Get Wikipedia page lengths and Google results for all people concurrently"""
    print(f"Fetching Wikipedia and Google data for {len(people_data)} people...")
    
    tasks = []
    for person_data in people_data:
        name = person_data['name']
        wikipedia_url = person_data['wikipedia']
        
        # Create tasks for each person
        page_length_task = get_wikipedia_page_length(session, wikipedia_url)
        google_results_task = get_google_results_count(session, name)
        
        tasks.append((name, page_length_task, google_results_task, person_data))
    
    # Execute all tasks concurrently
    results = {}
    for name, page_task, google_task, person_data in tasks:
        try:
            page_length, google_results = await asyncio.gather(
                page_task, google_task, return_exceptions=True
            )
            
            # Handle exceptions
            if isinstance(page_length, Exception):
                page_length = 0
            if isinstance(google_results, Exception):
                google_results = 0
                
            results[name] = {
                'page_length': page_length,
                'google_results': google_results,
                'person_data': person_data
            }
            print(f"Fetched data for: {name}")
            
        except Exception as e:
            print(f"Error fetching data for {name}: {e}")
            results[name] = {
                'page_length': 0,
                'google_results': 0,
                'person_data': person_data
            }
    
    return results

def extract_number_from_response(response_text: str, range_0_5: bool = True) -> int:
    """Extract number from LLM response"""
    if range_0_5:
        match = re.search(r'[0-5]', response_text)
        return int(match.group()) if match else 0
    else:
        match = re.search(r'\d+', response_text)
        return int(match.group()) if match else 0

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

def main():
    start_time = time.time()
    TARGET_COUNT = 1000  # Change this number to process more people
    
    # Load existing enhanced people
    existing_people = load_existing_enhanced_people()
    
    # Read people from wikidata_people.csv until we have TARGET_COUNT total
    new_people = []
    with open('wikidata_people.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if len(existing_people) + len(new_people) >= TARGET_COUNT:
                break
            
            # Skip if already processed
            if row['name'].strip() not in existing_people:
                new_people.append(row)
    
    if not new_people:
        elapsed_time = time.time() - start_time
        print(f"All {len(existing_people)} people already processed! Target: {TARGET_COUNT}")
        print(f"Total time: {elapsed_time:.2f} seconds")
        return
        
    print(f"Will process {len(new_people)} new people (existing: {len(existing_people)}, target: {TARGET_COUNT})")
    
    # Step 1: Get Wikipedia and Google data concurrently
    async def fetch_web_data():
        connector = aiohttp.TCPConnector(limit=30)
        timeout = aiohttp.ClientTimeout(total=60)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            return await get_wikipedia_and_google_data(session, new_people)
    
    web_data = asyncio.run(fetch_web_data())
    
    print(f"Completed web data fetching. Now generating LLM samples...")
    
    # Step 2: Generate all LLM samples
    samples, sample_metadata = generate_all_samples(new_people)
    print(f"Generated {len(samples)} LLM samples for {len(new_people)} people")
    
    # Step 3: Create and run the Inspect AI task
    task = score_all_people(samples)
    
    print("Running Inspect AI evaluation...")
    eval_result = eval(
        task,
        model="openai/gpt-5-nano",  # Use your preferred model
        log_dir="./inspect_logs",
        max_connections=20  # Built-in concurrency control
    )

    if isinstance(eval_result, list):
        print(f"Got list of {len(eval_result)} results, taking first one")
        eval_result = eval_result[0]
    

    print("LLM evaluation completed. Processing results...")
    
    # Step 4: Process results and combine with web data
    results = []
    person_scores = {}  # Track scores by person
    
    # Parse LLM results
    for i, sample_result in enumerate(eval_result.samples):
        metadata = sample_metadata[i]
        person_name = metadata['person_name']
        metric = metadata['metric']
        
        if person_name not in person_scores:
            person_scores[person_name] = {}
        
        # Extract score from LLM response
        response_text = sample_result.output.completion if sample_result.output else ""
        score = extract_number_from_response(response_text, range_0_5=True)
        
        person_scores[person_name][metric] = score
        print(f"Parsed {person_name} - {metric}: {score}")
    
    # Step 5: Combine all data
    print(f"Step 5: Combining all data")
    for person_name, scores in person_scores.items():
        if person_name in web_data:
            web_info = web_data[person_name]
            person_data = web_info['person_data']
            
            result = {
                'name': person_name,
                'page_length': web_info['page_length'],
                'google_results': web_info['google_results'],
                'political_influence': scores.get('political_influence', 0),
                'cultural_position': scores.get('cultural_position', 0),
                'number_of_books': 0,  # Skip for now
                'scientific_pioneer': scores.get('scientific_pioneer', 0),
                'cultural_pioneer': scores.get('cultural_pioneer', 0),
                'sitelinks': person_data['sitelinks'],
                'wikipedia_url': person_data['wikipedia'],
                'wikidata_uri': person_data['wikidata_uri']
            }
            results.append(result)
            print(f"Final result for {person_name}: {result}")
    
    # Step 6: Save results
    existing_data = []
    if os.path.exists('enhanced_people.csv'):
        try:
            with open('enhanced_people.csv', 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                existing_data = [row for row in reader if row['name']]  # Skip empty rows
        except Exception as e:
            print(f"Error reading existing data: {e}")
    
    all_data = existing_data + results
    with open('enhanced_people.csv', 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['name', 'page_length', 'google_results', 'political_influence', 
                    'cultural_position', 'number_of_books', 'scientific_pioneer', 
                    'cultural_pioneer', 'sitelinks', 'wikipedia_url', 'wikidata_uri']
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(all_data)
    
    # Done! Wrap up
    elapsed_time = time.time() - start_time
    total_people = len(existing_people) + len(results)
    
    print(f"Completed! Processed {len(results)} new people. Total people: {total_people}")
    print(f"Total time: {elapsed_time:.2f} seconds ({elapsed_time/60:.1f} minutes)")
    if len(results) > 0:
        avg_time_per_person = elapsed_time / len(results)
        print(f"Average time per person: {avg_time_per_person:.2f} seconds")
    print(f"Results saved to enhanced_people.csv")

def patch_zeros():
    """Patch people with zero page length by refetching Wikipedia data"""
    print("Starting patch_zeros to fix zero page lengths...")
    
    # Load current data
    people_to_patch = []
    all_people = []
    
    with open('enhanced_people.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_people.append(row)
            if int(row['page_length']) == 0:
                people_to_patch.append(row)
    
    if not people_to_patch:
        print("No people with zero page length found!")
        return
    
    print(f"Found {len(people_to_patch)} people with zero page length")
    
    # Fetch Wikipedia data for these people using existing function
    async def fetch_wikipedia_data():
        connector = aiohttp.TCPConnector(limit=10)
        timeout = aiohttp.ClientTimeout(total=30)
        
        async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
            # Reuse existing get_wikipedia_and_google_data but only extract page lengths
            patch_data = [{'name': p['name'], 'wikipedia': p['wikipedia_url']} for p in people_to_patch]
            web_results = await get_wikipedia_and_google_data(session, patch_data)
            
            # Extract just the page lengths
            results = {}
            for name, data in web_results.items():
                results[name] = data['page_length']
                print(f"Fetched {name}: {data['page_length']} characters")
            
            return results
    
    # Get the new page lengths
    new_page_lengths = asyncio.run(fetch_wikipedia_data())
    
    # Update the data
    updated_count = 0
    for person in all_people:
        name = person['name']
        if name in new_page_lengths and new_page_lengths[name] > 0:
            old_length = person['page_length']
            person['page_length'] = str(new_page_lengths[name])
            print(f"Updated {name}: {old_length} -> {new_page_lengths[name]}")
            updated_count += 1
    
    # Save the updated data with proper CSV quoting
    with open('enhanced_people.csv', 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['name', 'page_length', 'google_results', 'political_influence', 
                    'cultural_position', 'number_of_books', 'scientific_pioneer', 
                    'cultural_pioneer', 'sitelinks', 'wikipedia_url', 'wikidata_uri']
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        writer.writeheader()
        writer.writerows(all_people)
    
    print(f"Patch complete! Updated {updated_count} people with new page lengths.")
    print(f"Remaining zeros: {len(people_to_patch) - updated_count}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "patch":
        patch_zeros()
    else:
        main()