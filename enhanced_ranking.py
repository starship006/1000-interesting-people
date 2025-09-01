import csv
import json
import time
import asyncio
import aiohttp
import os
import urllib.parse
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
        'political_influence_great_man': {
            'description': """Political influence and power in government, policy-making, or political movements. Someone who directly influenced the political decisions of many individuals and shaped the structure of current day institutions. 
        

        You should rank individuals on a scale of 0-5. It's extremely important we reserve the higher numbers for the tail ends of people - we want to capture the tails of the distribution. Here is a ranking to follow:
        A 0 is someone who was only loosely related to politics, or someone who wasn't particularly influential.
        A 1 should be reserved for *most politicians*, who passed policies that might have slightly influenced individuals. They could have been replaced by another constituent and things would have broadly been the same for most people in the world.
        A 2 should have the bar of passing policies that directly impacted the lives of millions at the time or in the future in a noticable, but potentially medium, way. Their actions are slightly overdetermined, and there were many people pushing for this who would have done these actions too, but they were the one who got it done.
        A 3 should have the bar of either directly impacting billions, or strongly impacting millions of people. Their actions set people's lives down an extremely different path. Their mark was left on the world at the time, and for the future at hand. Not many people could have done what they did. Their influence was dramatic and superb.
        A 4 is reserved for someone who is in the top 1 percentile of people ranked by this category. You should make sure that if you place someone here, they are truly 1%. (I will be checking your callibration!).
        A 5 should be reserved for the most extreme cases, where removing the individual drastically changes the future of the world dramatically. It is likely that my life as a computer scientist in the bay area, or the life of my family or friends, would be dramatically different if they didn't exist. There are only a handful of people like this ever. At the time, they were uniquely qualified to accomplish what they did.        
        """,
            'system_message': """You are a distinguished historian who firmly believes in the Great Man theory of history. You believe that exceptional individuals with unique vision, charisma, and capabilities are the primary drivers of historical change. In your view, key figures possess rare combinations of talent, timing, and determination that allow them to transcend structural limitations and reshape the world through force of personality and genius. You look for evidence of individuals who seized critical moments and bent history to their will through exceptional leadership, innovation, or vision that ordinary people could never have achieved. You are inclined to recognize the transformative power of singular great figures."""
        },
        'scientific_influence': {
            'description': """Scientific influence and impact through research, discovery, or technological advancement. Someone who directly advanced human knowledge, created new scientific paradigms, or developed technologies that shaped how we understand or interact with the world.

        You should rank individuals on a scale of 0-5. It's extremely important we reserve the higher numbers for the tail ends of people - we want to capture the tails of the distribution. In order to make this ranking, it's important to characterize *what are the concrete scientific outcomes* that this person *counterfactually* helped achieve.Here is a ranking to follow:

        A 0 is someone who was only loosely related to science, or someone who wasn't particularly influential in advancing scientific knowledge. They may have been educated in science but didn't make notable contributions to human understanding.

        A 1 should be reserved for *most working scientists*, who made incremental contributions that slightly advanced their field. They published papers that were cited by others, but their work could have been done by many other competent researchers. Their absence wouldn't significantly alter the trajectory of scientific progress. Potentially, scientific educators and teachers can also be in this region.

        A 2 should have the bar of making discoveries or developing technologies that directly impacted how millions of people understand the world or live their lives, either at the time or in the future. You should be able to list the field of science that this person helped push, and how this advancement changed millions of people's lives. This contribution can be slightly overdetermined if it was obvious; potentially others were working on similar problems and would likely have made comparable breakthroughs within a few years.

        A 3 should have the bar of either *fundamentally reshaping entire scientific disciplines* or creating technologies/discoveries that affected billions of people. You should be able to state the concrete advancement they made, and how this advancement changed billions of people's lives. It's important that you consider how the advancement changes not just our understanding of the world, but also the way people live and interact with the world. Few people had the combination of insight, timing, and capability to make their specific contributions.

        A 4 is reserved for someone who is in the top 1 percentile of people ranked by scientific influence. You should make sure that if you place someone here, they are truly in the top 1% of all scientists who ever lived, measured by how much the world is influenced as a result of their work. Their work created foundational shifts in multiple fields or fundamentally altered humanity's relationship with the natural world.

        A 5 should be reserved for the most extreme cases, where removing the individual drastically changes the entire trajectory of scientific progress. It is likely that our current technological civilization, including computers, medicine, and our basic understanding of reality, would be dramatically different or delayed by decades if they didn't exist. There are only a handful of people like this in all of human history. Their contributions were so fundamental and unique that no contemporaries could have replicated them.
        """,
            'system_message': """You are a distinguished historian of science with expertise in the development of scientific knowledge and technological advancement. You understand how scientific breakthroughs build upon previous work and how discoveries can reshape entire fields of human understanding. You evaluate scientific contributions based on their originality, impact on human knowledge, and influence on subsequent research and technology. You consider both the immediate effects of discoveries and their long-term implications for scientific progress and human civilization. Science for the sake of 'understanding the world' isnt enough, it should ideally also change peoples lives for the better."""
        },
        'political_influence_structuralist': {
            'description': """Political influence and power in government, policy-making, or political movements. Someone who directly influenced the political decisions of many individuals and shaped the structure of current day institutions. 
        

        You should rank individuals on a scale of 0-5. It's extremely important we reserve the higher numbers for the tail ends of people - we want to capture the tails of the distribution. Here is a ranking to follow:
        A 0 is someone who was only loosely related to politics, or someone who wasn't particularly influential. They weren't influential in shaping the structures that lead to political decisions, nor did they make political decisions themselves.
        A 1 should be reserved for *most politicians*, who passed policies that might have slightly influenced individuals. They could have been replaced by another constituent and things would have broadly been the same for most people in the world.
        A 2 should have the bar of passing policies that directly impacted the lives of millions at the time or in the future in a noticable, but potentially medium, way. Their actions are slightly overdetermined, and there were many people pushing for this who would have done these actions too, but they were the one who got it done. Alternatively, if someone wasn't a politician but had strong indirect effects on politics and discussions (for example, by making a certain style of thought popular), they might be in this region.
        A 3 should have the bar of either *directly impacting billions*, or strongly impacting millions of people. Their actions set people's lives down an extremely different path. Their mark was left on the world at the time, and for the future at hand. Not many people could have done what they did. Their influence was dramatic and superb. If this person wouldn't describe themself as working in the realm of politics, the case for putting them here must be extraordinarily strong.
        A 4 is reserved for someone who is in the top 1 percentile of people ranked by this category. You should make sure that if you place someone here, they are truly 1%. (I will be checking your callibration!).
        A 5 should be reserved for the most extreme cases, where removing the individual drastically changes the future of the world dramatically. It is likely that my life as a computer scientist in the bay area, or the life of my family or friends, would be dramatically different if they didn't exist. There are only a handful of people like this ever. At the time, they were uniquely qualified to accomplish what they did.        
        """,
            'system_message': """You are a distinguished historian who subscribes to structuralist theories of history. You believe that historical change is primarily driven by deep structural forces - economic systems, social conditions, technological developments, and institutional frameworks - rather than individual agency. In your view, most historical figures are products of their circumstances, elevated by structural conditions that were already pushing society toward certain outcomes. You are deeply skeptical of Great Man narratives and look for evidence that supposed 'great leaders' were simply in the right place at the right time, carried forward by forces beyond their control. You tend to see historical change as overdetermined by structural factors, and believe that if one individual hadn't acted, another would have filled the same role with similar results. You should be extremely critical about evaluating if a persons outcomes were overdetermined or not - and if so, lower their eventual score appropriately."""
        },
        'cultural_influence': {
            'description': """Cultural influence and impact through arts, entertainment, philosophy, social movements, or shaping how people think about society, values, and human experience. Someone who directly influenced how millions of people see themselves, their relationships, or their place in the world through cultural expression or social change.

        You should rank individuals on a scale of 0-5. It's extremely important we reserve the higher numbers for the tail ends of people - we want to capture the tails of the distribution. Here is a ranking to follow:

        A 0 is someone who was only loosely related to culture, or someone who wasn't particularly influential in shaping cultural expression or social thought. They may have been artists, writers, or cultural figures but didn't notably influence how people think or behave.

        A 1 should be reserved for *most cultural figures*, who created works that entertained or inspired some audiences but could have been replaced by other talented contemporaries. Their cultural contributions were appreciated but didn't fundamentally shift how people understand themselves or society. Many other artists were creating similar work.

        A 2 should have the bar of creating cultural works or movements that directly shaped how millions of people think about identity, relationships, or social issues, either at the time or in the future. For example, most famous pop artists might fit in this category: while their work is appreciated by many, people don't take drastically different actions. Additionally, maybe their contributions are somewhat overdetermined - similar cultural shifts were emerging from multiple sources, and other figures were pushing in comparable directions.

        A 3 should have the bar of either *transforming entire cultural domains* or creating works/movements that reshaped how hundreds of millions of people understand human experience. They pioneered entirely new forms of expression, sparked major social movements, or created cultural frameworks that persist across generations. Few people had the combination of talent, insight, and cultural positioning to achieve their specific impact. People may frequently cite this person as 'changing their life' in a way they can gesture to. This person inspired ways of living that dramatically affected large swaths of people.

        A 4 is reserved for someone who is in the top 1 percentile of people ranked by cultural influence. You should make sure that if you place someone here, they are truly in the top 1% of all cultural figures who ever lived. Their work created foundational changes in how humans express themselves, relate to each other, or organize society.

        A 5 should be reserved for the most extreme cases, where removing the individual drastically changes the entire trajectory of human cultural development. It is likely that how we think about art, relationships, social organization, or fundamental human values would be dramatically different if they didn't exist. There are only a handful of people like this in all of human history. Their cultural innovations were so foundational and unique that no contemporaries could have produced equivalent transformations.
        """,
            'system_message': """You are a distinguished cultural historian and critic with expertise in the development of human artistic expression, social movements, and cultural transformation. You understand how cultural innovations spread through societies and how artistic, philosophical, and social movements shape human consciousness and behavior. You evaluate cultural contributions based on their originality, their influence on subsequent cultural development, and their impact on how people understand themselves and their relationships. You consider both immediate cultural effects and long-term influence on human values, social organization, and artistic expression."""
        }
        # 'scientific_pioneer': 'Contributions to scientific discovery, research, or technological advancement',
        #'cultural_pioneer': 'Pioneering work in culture, arts, philosophy, or social change'
    }
    
    for person_data in people_data:
        name = person_data['name']
        wikipedia_url = person_data['wikipedia']
        
        # Create samples for each metric
        for metric, metric_info in metrics.items():
            description = metric_info['description']
            system_msg = metric_info['system_message']
            
            prompt = f"""{system_msg}

Time has a list of the most influential 100 people each year, but it doesn't seem very principled and grounded. You are going to assist our efforts to make a more principled and grounded list. People can achieve extrodinary outcomes, and we want to emphasize and compare people who reached extraodinary outcomes. 

Of course, defining 'influential' is quite subjective and extremely tricky. So we are going to define it relative to *me*. I'm going to suggest some metrics to rate people, and your job is to rate them on the metric. Our goal is to filter people out on the tails. Your job is to consider the following person at Wikipedia URL {wikipedia_url}, and then rate them on a scale of 0-5 for {metric}.

You should do this by:
- First, describe what the person did that matches this metric.
- Then, give the bear case for this person. Was their outcome overdetermined? Were the one of many people?
- Finally, give a rating of 0-5 for the metric.

In particular, here is a more detailed description of the metric:
{description}

Respond with your reasoning, and then a single number from 0-5 inside <Score></Score> brackets.

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
            generate(cache=True)
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

def extract_score_and_reasoning(response_text: str, range_0_5: bool = True) -> tuple:
    """Extract score and full response from LLM response"""
    print(f"Extracting score and reasoning from response: {response_text}")
    reasoning = response_text  # Just use the full response
    print("resultant reasoning: ", reasoning)
    score = 0
    
    if range_0_5:
        # Look for score in <Score></Score> brackets first
        score_match = re.search(r'<Score>([0-5])</Score>', response_text, re.IGNORECASE)
        if score_match:
            score = int(score_match.group(1))
        else:
            # Fallback to any digit 0-5 if no brackets found
            match = re.search(r'[0-5]', response_text)
            score = int(match.group()) if match else 0
    else:
        match = re.search(r'\d+', response_text)
        score = int(match.group()) if match else 0
    
    return score, reasoning

def extract_number_from_response(response_text: str, range_0_5: bool = True) -> int:
    """Backward compatibility wrapper - extract just the score"""
    score, _ = extract_score_and_reasoning(response_text, range_0_5)
    return score

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
    models = [
        "openai/gpt-5-nano",
        "openai/gpt-5-mini",
        "anthropic/claude-3-5-haiku-latest"
    ]
    
    eval_result = eval(
        task,
        model=models,  # Use multiple models
        log_dir="./inspect_logs",
        max_connections=40  # Built-in concurrency control
    )

    # Handle multiple model results
    if not isinstance(eval_result, list):
        eval_result = [eval_result]
    
    print(f"LLM evaluation completed. Processing results from {len(eval_result)} model(s)...")
    
    # Step 4: Process results and combine with web data
    results = []
    person_scores = {}  # Track averaged scores by person
    person_reasoning = {}  # Track reasoning by person (use first model's reasoning)
    person_all_scores = {}  # Track all scores for averaging
    
    # Parse LLM results from all models
    for model_idx, model_result in enumerate(eval_result):
        print(f"Processing results from model {model_idx + 1}/{len(eval_result)}")
        
        for i, sample_result in enumerate(model_result.samples):
            metadata = sample_metadata[i]
            person_name = metadata['person_name']
            metric = metadata['metric']
            
            if person_name not in person_all_scores:
                person_all_scores[person_name] = {}
                person_reasoning[person_name] = {}
            
            if metric not in person_all_scores[person_name]:
                person_all_scores[person_name][metric] = []
            
            # Extract score and reasoning from LLM response
            response_text = sample_result.output.completion if sample_result.output else ""
            score, reasoning = extract_score_and_reasoning(response_text, range_0_5=True)
            
            person_all_scores[person_name][metric].append(score)
            
            # Use reasoning from first model only to avoid overwhelming CSV
            if model_idx == 0:
                person_reasoning[person_name][metric] = reasoning
            
            print(f"Model {model_idx + 1} - {person_name} - {metric}: {score}")
    
    # Calculate averaged scores
    for person_name, metrics in person_all_scores.items():
        if person_name not in person_scores:
            person_scores[person_name] = {}
        
        for metric, scores in metrics.items():
            avg_score = sum(scores) / len(scores)
            person_scores[person_name][metric] = round(avg_score, 1)  # Round to 1 decimal place
            print(f"Averaged {person_name} - {metric}: {scores} -> {avg_score:.1f}")
    
    # Step 5: Combine all data - ensure one result per person
    print(f"Step 5: Combining all data")
    processed_people = set()  # Track processed people to avoid duplicates
    
    for person_name, scores in person_scores.items():
        if person_name in processed_people:
            print(f"Skipping duplicate person: {person_name}") # would happen if Q transform makes multiple copioes
            continue
            
        if person_name in web_data:
            web_info = web_data[person_name]
            person_data = web_info['person_data']
            reasoning = person_reasoning.get(person_name, {})
            
            # Fix names that are just Q-IDs by extracting from Wikipedia URL
            display_name = person_name
            if person_name.startswith('Q') and person_name[1:].isdigit():
                wikipedia_url = person_data['wikipedia']
                if wikipedia_url and '/wiki/' in wikipedia_url:
                    # Extract name from URL and decode URL encoding
                    url_name = wikipedia_url.split('/wiki/')[-1]
                    # Replace underscores with spaces and decode URL encoding
                    display_name = urllib.parse.unquote(url_name).replace('_', ' ')
                    print(f"Fixed name: {person_name} -> {display_name}")
            
            result = {
                'name': display_name,
                'page_length': web_info['page_length'],
                'google_results': web_info['google_results'],
                'political_influence_great_man': scores.get('political_influence_great_man', 0),
                'political_influence_structuralist': scores.get('political_influence_structuralist', 0),
                'political_influence_great_man_reasoning': reasoning.get('political_influence_great_man', ''),
                'political_influence_structuralist_reasoning': reasoning.get('political_influence_structuralist', ''),
                'scientific_influence': scores.get('scientific_influence', 0),
                'scientific_influence_reasoning': reasoning.get('scientific_influence', ''),
                'cultural_influence': scores.get('cultural_influence', 0),
                'cultural_influence_reasoning': reasoning.get('cultural_influence', ''),
                'cultural_position': scores.get('cultural_position', 0),
                'cultural_position_reasoning': reasoning.get('cultural_position', ''),
                'number_of_books': 0,  # Skip for now
                'scientific_pioneer': scores.get('scientific_pioneer', 0),
                'scientific_pioneer_reasoning': reasoning.get('scientific_pioneer', ''),
                'cultural_pioneer': scores.get('cultural_pioneer', 0),
                'cultural_pioneer_reasoning': reasoning.get('cultural_pioneer', ''),
                'sitelinks': person_data['sitelinks'],
                'wikipedia_url': person_data['wikipedia'],
                'wikidata_uri': person_data['wikidata_uri']
            }
            results.append(result)
            processed_people.add(person_name)
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
        fieldnames = ['name', 'page_length', 'google_results', 
                    'political_influence_great_man', 'political_influence_structuralist',
                    'political_influence_great_man_reasoning', 'political_influence_structuralist_reasoning',
                    'scientific_influence', 'scientific_influence_reasoning',
                    'cultural_influence', 'cultural_influence_reasoning',
                    'cultural_position', 'cultural_position_reasoning', 
                    'number_of_books', 
                    'scientific_pioneer', 'scientific_pioneer_reasoning',
                    'cultural_pioneer', 'cultural_pioneer_reasoning',
                    'sitelinks', 'wikipedia_url', 'wikidata_uri']
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
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

def remove_duplicates():
    """Remove duplicate people from enhanced_people.csv based on name"""
    print("Starting remove_duplicates to clean up duplicate entries...")
    
    if not os.path.exists('enhanced_people.csv'):
        print("enhanced_people.csv not found!")
        return
    
    # Load all data
    all_people = []
    with open('enhanced_people.csv', 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            all_people.append(row)
    
    print(f"Found {len(all_people)} total entries")
    
    # Remove duplicates, keeping the first occurrence
    seen_names = set()
    unique_people = []
    duplicates_removed = 0
    
    for person in all_people:
        name = person['name'].strip()
        if name not in seen_names:
            unique_people.append(person)
            seen_names.add(name)
        else:
            print(f"Removing duplicate: {name}")
            duplicates_removed += 1
    
    print(f"Removed {duplicates_removed} duplicates, keeping {len(unique_people)} unique entries")
    
    # Save the deduplicated data
    with open('enhanced_people.csv', 'w', newline='', encoding='utf-8') as f:
        fieldnames = ['name', 'page_length', 'google_results', 
                    'political_influence_great_man', 'political_influence_structuralist',
                    'political_influence_great_man_reasoning', 'political_influence_structuralist_reasoning',
                    'scientific_influence', 'scientific_influence_reasoning',
                    'cultural_influence', 'cultural_influence_reasoning',
                    'cultural_position', 'cultural_position_reasoning', 
                    'number_of_books', 
                    'scientific_pioneer', 'scientific_pioneer_reasoning',
                    'cultural_pioneer', 'cultural_pioneer_reasoning',
                    'sitelinks', 'wikipedia_url', 'wikidata_uri']
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
        writer.writeheader()
        writer.writerows(unique_people)
    
    print(f"Deduplication complete! Saved {len(unique_people)} unique people to enhanced_people.csv")

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
        fieldnames = ['name', 'page_length', 'google_results', 
                    'political_influence_great_man', 'political_influence_structuralist',
                    'political_influence_great_man_reasoning', 'political_influence_structuralist_reasoning',
                    'scientific_influence', 'scientific_influence_reasoning',
                    'cultural_influence', 'cultural_influence_reasoning',
                    'cultural_position', 'cultural_position_reasoning', 
                    'number_of_books', 
                    'scientific_pioneer', 'scientific_pioneer_reasoning',
                    'cultural_pioneer', 'cultural_pioneer_reasoning',
                    'sitelinks', 'wikipedia_url', 'wikidata_uri']
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_NONNUMERIC)
        writer.writeheader()
        writer.writerows(all_people)
    
    print(f"Patch complete! Updated {updated_count} people with new page lengths.")
    print(f"Remaining zeros: {len(people_to_patch) - updated_count}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        if sys.argv[1] == "patch":
            patch_zeros()
        elif sys.argv[1] == "remove_duplicates":
            remove_duplicates()
        else:
            print("Usage: python enhanced_ranking.py [patch|remove_duplicates]")
    else:
        main()