#!/usr/bin/env python3
import csv
import json

def convert_csv_to_js():
    people = []
    headers = []
    
    with open('enhanced_people.csv', 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        headers = next(reader)
        
        for row in reader:
            person = {}
            for i, value in enumerate(row):
                if i >= len(headers):
                    break
                    
                header = headers[i]
                
                # Keep string fields as strings
                if header in ['name', 'wikipedia_url', 'wikidata_uri'] or header.endswith('_reasoning'):
                    person[header] = value
                else:
                    # Convert numeric fields
                    try:
                        person[header] = float(value) if value else 0
                    except ValueError:
                        person[header] = 0
            
            people.append(person)
    
    # Generate JavaScript data file
    js_content = f"""// Auto-generated from enhanced_people.csv
const csvHeaders = {json.dumps(headers, indent=2)};

const peopleData = {json.dumps(people, indent=2)};
"""
    
    with open('csv-data.js', 'w', encoding='utf-8') as f:
        f.write(js_content)
    
    print(f"Converted {len(people)} people to JavaScript format")
    print(f"Headers: {len(headers)} columns")

if __name__ == "__main__":
    convert_csv_to_js()