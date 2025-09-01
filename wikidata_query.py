import json
import csv
import time
import os
import argparse
from datetime import datetime
import asyncio
import aiohttp

async def query_wikidata_by_year(session, year):
    start_date = f"{year}-01-01"
    end_date = f"{year+1}-01-01"
    
    sparql_query = f"""
    SELECT ?person ?personLabel ?sitelinks ?birth ?wikipedia WHERE {{
      ?person wdt:P31 wd:Q5;
        wdt:P569 ?birth;
        wikibase:sitelinks ?sitelinks.
      OPTIONAL {{
        ?wikipedia schema:about ?person;
          schema:isPartOf <https://en.wikipedia.org/>.
      }}
      FILTER((?birth >= "{start_date}"^^xsd:dateTime) && (?birth < "{end_date}"^^xsd:dateTime))
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    ORDER BY DESC (?sitelinks)
    """
    
    url = "https://query.wikidata.org/sparql"
    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": "Python SPARQL Client"
    }
    
    try:
        params = {"query": sparql_query}
        async with session.get(url, params=params, headers=headers, timeout=30) as response:
            if response.status == 200:
                data = await response.json()
                return year, data["results"]["bindings"]
            else:
                print(f"Error for {year}: HTTP {response.status}")
                return year, None
    except Exception as e:
        print(f"Request error for {year}: {e}")
        return year, None

async def query_wikidata_by_quarter(session, year, quarter):
    # Define quarter date ranges
    quarters = {
        1: (f"{year}-01-01", f"{year}-04-01"),
        2: (f"{year}-04-01", f"{year}-07-01"),
        3: (f"{year}-07-01", f"{year}-10-01"),
        4: (f"{year}-10-01", f"{year+1}-01-01")
    }
    
    start_date, end_date = quarters[quarter]
    
    sparql_query = f"""
    SELECT ?person ?personLabel ?sitelinks ?birth ?wikipedia WHERE {{
      ?person wdt:P31 wd:Q5;
        wdt:P569 ?birth;
        wikibase:sitelinks ?sitelinks.
      OPTIONAL {{
        ?wikipedia schema:about ?person;
          schema:isPartOf <https://en.wikipedia.org/>.
      }}
      FILTER((?birth >= "{start_date}"^^xsd:dateTime) && (?birth < "{end_date}"^^xsd:dateTime))
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    ORDER BY DESC (?sitelinks)
    """
    
    url = "https://query.wikidata.org/sparql"
    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": "Python SPARQL Client"
    }
    
    try:
        params = {"query": sparql_query}
        async with session.get(url, params=params, headers=headers, timeout=30) as response:
            if response.status == 200:
                data = await response.json()
                return (year, quarter), data["results"]["bindings"]
            else:
                print(f"Error for {year} Q{quarter}: HTTP {response.status}")
                return (year, quarter), None
    except Exception as e:
        print(f"Request error for {year} Q{quarter}: {e}")
        return (year, quarter), None

async def query_years(years_to_process, max_concurrent=6):
    connector = aiohttp.TCPConnector(limit=max_concurrent)
    timeout = aiohttp.ClientTimeout(total=60)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def query_with_semaphore(year):
            async with semaphore:
                result = await query_wikidata_by_year(session, year)
                await asyncio.sleep(0.2)  # Slightly longer delay for full year queries
                return result
        
        tasks = [query_with_semaphore(year) for year in years_to_process]
        
        print(f"Starting async queries for {len(years_to_process)} years with max {max_concurrent} concurrent requests...")
        
        results = await asyncio.gather(*tasks)
        
        return results

async def query_quarters(quarters_to_process, max_concurrent=6):
    connector = aiohttp.TCPConnector(limit=max_concurrent)
    timeout = aiohttp.ClientTimeout(total=60)
    
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def query_with_semaphore(year, quarter):
            async with semaphore:
                result = await query_wikidata_by_quarter(session, year, quarter)
                await asyncio.sleep(0.1)  # Small delay to be respectful
                return result
        
        # Create tasks for specified quarters only
        tasks = [query_with_semaphore(year, quarter) for year, quarter in quarters_to_process]
        
        print(f"Starting async queries for {len(quarters_to_process)} quarters with max {max_concurrent} concurrent requests...")
        
        results = await asyncio.gather(*tasks)
        
        return results

def save_to_csv(all_people, filename="wikidata_people.csv"):
    if not all_people:
        return
    
    with open(filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['name', 'birth_date', 'birth_year', 'birth_quarter', 'sitelinks', 'wikipedia', 'wikidata_uri'])
        writer.writeheader()
        writer.writerows(all_people)
    print(f"Saved {len(all_people)} people to {filename}")

def process_results_year(results, year):
    processed = []
    for result in results:
        birth_date = result.get("birth", {}).get("value", "")
        
        # Calculate quarter from birth date
        quarter = 1  # Default
        if birth_date:
            try:
                month = int(birth_date.split('-')[1])
                quarter = (month - 1) // 3 + 1
            except:
                pass
        
        person_data = {
            'name': result.get("personLabel", {}).get("value", "Unknown"),
            'birth_date': birth_date,
            'birth_year': year,
            'birth_quarter': quarter,
            'sitelinks': int(result.get("sitelinks", {}).get("value", "0")),
            'wikipedia': result.get("wikipedia", {}).get("value", ""),
            'wikidata_uri': result.get("person", {}).get("value", "")
        }
        processed.append(person_data)
    return processed

def process_results(results, year, quarter):
    processed = []
    for result in results:
        birth_date = result.get("birth", {}).get("value", "")
        
        # Calculate actual quarter from birth date
        actual_quarter = quarter  # Default to the queried quarter
        if birth_date:
            try:
                month = int(birth_date.split('-')[1])
                actual_quarter = (month - 1) // 3 + 1
            except:
                pass  # Keep the queried quarter as fallback
        
        person_data = {
            'name': result.get("personLabel", {}).get("value", "Unknown"),
            'birth_date': birth_date,
            'birth_year': year,
            'birth_quarter': actual_quarter,
            'sitelinks': int(result.get("sitelinks", {}).get("value", "0")),
            'wikipedia': result.get("wikipedia", {}).get("value", ""),
            'wikidata_uri': result.get("person", {}).get("value", "")
        }
        processed.append(person_data)
    return processed

def load_existing_data(csv_file):
    """Load existing data from CSV and determine which quarters are already processed."""
    all_people = []
    
    if not os.path.exists(csv_file):
        return all_people, []
    
    print(f"Found existing {csv_file}. Loading previous data...")
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            all_people = list(reader)
            # Convert string fields back to proper types and calculate quarter from birth_date
            for person in all_people:
                person['birth_year'] = int(person['birth_year'])
                person['sitelinks'] = int(person['sitelinks'])
                
                # Calculate quarter from birth_date if not already present
                if 'birth_quarter' not in person or not person['birth_quarter']:
                    birth_date = person.get('birth_date', '')
                    if birth_date:
                        try:
                            # Parse ISO date format (e.g., "1990-03-15T00:00:00Z")
                            month = int(birth_date.split('-')[1])
                            quarter = (month - 1) // 3 + 1
                            person['birth_quarter'] = quarter
                        except:
                            person['birth_quarter'] = 1  # Default fallback
                    else:
                        person['birth_quarter'] = 1
                else:
                    person['birth_quarter'] = int(person['birth_quarter'])
        
        print(f"Loaded {len(all_people)} existing records")
        existing_quarters = set((person['birth_year'], person['birth_quarter']) for person in all_people)
        
    except Exception as e:
        print(f"Error loading existing data: {e}")
        raise e
        # all_people = []
        # existing_quarters = set()
    
    return all_people, existing_quarters

def get_quarters_to_process(start_year, end_year, existing_quarters):
    """Generate list of quarters that need to be processed."""
    all_quarters = []
    for year in range(start_year, end_year + 1):
        for quarter in [1, 2, 3, 4]:
            all_quarters.append((year, quarter))
    
    quarters_to_process = [quarter for quarter in all_quarters if quarter not in existing_quarters]
    return quarters_to_process

async def process_years_with_retry(years_to_process, max_concurrent, max_retries=4):
    """Process years with retry logic for failed years."""
    all_new_people = []
    failed_years = years_to_process.copy()
    retry_count = 0
    
    while failed_years and retry_count < max_retries:
        if retry_count > 0:
            print(f"\nRetry {retry_count}/{max_retries} for {len(failed_years)} failed years...")
        
        results = await query_years(failed_years, max_concurrent)
        
        # Process results and collect failures
        current_failures = []
        year_totals = {}
        
        for year, year_results in results:
            if year_results:
                processed = process_results_year(year_results, year)
                all_new_people.extend(processed)
                year_totals[year] = len(processed)
                print(f"Year {year}: {len(processed)} people")
            else:
                current_failures.append(year)
                print(f"Year {year}: Failed")
        
        # Print year totals for this batch
        if year_totals:
            print(f"\nBatch totals:")
            for year in sorted(year_totals.keys()):
                print(f"  {year}: {year_totals[year]} people")
        
        failed_years = current_failures
        retry_count += 1
        
        if failed_years and retry_count < max_retries:
            print(f"Waiting 5 seconds before retry...")
            time.sleep(5)
    
    return all_new_people, failed_years

async def process_quarters_with_retry(quarters_to_process, max_concurrent, max_retries=4):
    """Process quarters with retry logic for failed quarters."""
    all_new_people = []
    failed_quarters = quarters_to_process.copy()
    retry_count = 0
    
    while failed_quarters and retry_count < max_retries:
        if retry_count > 0:
            print(f"\nRetry {retry_count}/{max_retries} for {len(failed_quarters)} failed quarters...")
        
        results = await query_quarters(failed_quarters, max_concurrent)
        
        # Process results and collect failures
        current_failures = []
        year_totals = {}
        
        for (year, quarter), quarter_results in results:
            if quarter_results:
                processed = process_results(quarter_results, year, quarter)
                all_new_people.extend(processed)
                
                if year not in year_totals:
                    year_totals[year] = 0
                year_totals[year] += len(processed)
                
                print(f"Year {year} Q{quarter}: {len(processed)} people")
            else:
                current_failures.append((year, quarter))
                print(f"Year {year} Q{quarter}: Failed")
        
        # Print year totals for this batch
        if year_totals:
            print(f"\nBatch totals:")
            for year in sorted(year_totals.keys()):
                print(f"  {year}: {year_totals[year]} people")
        
        failed_quarters = current_failures
        retry_count += 1
        
        if failed_quarters and retry_count < max_retries:
            print(f"Waiting 5 seconds before retry...")
            time.sleep(5)
    
    return all_new_people, failed_quarters

def deduplicate_and_sort(all_people):
    """Remove duplicates and sort by sitelinks."""
    seen_uris = set()
    unique_people = []
    for person in all_people:
        uri = person['wikidata_uri']
        if uri not in seen_uris:
            seen_uris.add(uri)
            unique_people.append(person)
    
    unique_people.sort(key=lambda x: x['sitelinks'], reverse=True)
    return unique_people

async def main():
    parser = argparse.ArgumentParser(description='Query Wikidata for people by birth dates')
    parser.add_argument('--year', action='store_true', 
                        help='Query by full years instead of quarters (fewer API calls)')
    parser.add_argument('--start-year', type=int, default=1050,
                        help='Start year (default: 1050)')
    parser.add_argument('--end-year', type=int, default=2024,
                        help='End year (default: 2024)')
    parser.add_argument('--max-concurrent', type=int, default=10,
                        help='Maximum concurrent requests (default: 10)')
    parser.add_argument('--csv-file', default="wikidata_people.csv",
                        help='CSV output file (default: wikidata_people.csv)')
    
    args = parser.parse_args()
    
    start_year = args.start_year
    end_year = args.end_year
    max_concurrent = args.max_concurrent
    csv_file = args.csv_file
    use_year_mode = args.year
    
    # Load existing data
    all_people, existing_quarters = load_existing_data(csv_file)
    
    if use_year_mode:
        # Year mode: get existing years from quarters data
        existing_years = set(year for year, quarter in existing_quarters)
        years_to_process = [year for year in range(start_year, end_year + 1) if year not in existing_years]
        
        if not years_to_process:
            print("All years already processed!")
            return
        
        print(f"Year mode: Will process {len(years_to_process)} remaining years")
        print(f"Covering years: {min(years_to_process)}-{max(years_to_process)}")
        
        # Process years with retry logic
        start_time = time.time()
        new_people, final_failures = await process_years_with_retry(years_to_process, max_concurrent)
    else:
        # Quarter mode (original behavior)
        quarters_to_process = get_quarters_to_process(start_year, end_year, existing_quarters)
        
        if not quarters_to_process:
            print("All quarters already processed!")
            return
        
        print(f"Quarter mode: Will process {len(quarters_to_process)} remaining quarters")
        years_to_process = list(set(year for year, quarter in quarters_to_process))
        print(f"Covering years: {min(years_to_process)}-{max(years_to_process)}")
        
        # Process quarters with retry logic
        start_time = time.time()
        new_people, final_failures = await process_quarters_with_retry(quarters_to_process, max_concurrent)
    
    elapsed = time.time() - start_time
    print(f"\nCompleted in {elapsed:.1f} seconds")
    
    if new_people:
        all_people.extend(new_people)
        all_people = deduplicate_and_sort(all_people)
        
        print(f"Total collected: {len(all_people)} people (after deduplication)")
        print(f"Top 10 by sitelinks:")
        for i, person in enumerate(all_people[:10], 1):
            print(f"{i}. {person['name']} ({person['birth_year']}) - {person['sitelinks']} sitelinks")
        
        save_to_csv(all_people, csv_file)
    
    if final_failures:
        failure_type = "years" if use_year_mode else "quarters"
        print(f"\nFinal failed {failure_type} after retries ({len(final_failures)}): {final_failures}")
    else:
        success_type = "years" if use_year_mode else "quarters"
        print(f"\nAll {success_type} processed successfully!")

if __name__ == "__main__":
    asyncio.run(main())