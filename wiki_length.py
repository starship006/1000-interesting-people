import requests
import time

pages = [
    "Xi Jinping",
    "Mao Zedong", 
    "Cai Qi"
]

endpoint = "https://en.wikipedia.org/w/api.php"

# Add headers to identify your request
headers = {
    'User-Agent': 'WikipediaLengthChecker/1.0 (thisiscodyr@gmail.com)'
}

for page in pages:
    params = {
        "action": "query",
        "titles": page,
        "prop": "revisions",
        "rvslots": "main",
        "rvprop": "content",
        "format": "json"
    }
    response = requests.get(endpoint, params=params, headers=headers)

    print(f"Status Code: {response.status_code}")
    if response.status_code == 403:
        print(f"403 Forbidden error for {page}")
        print(f"Response headers: {dict(response.headers)}")
        print(f"Response text: {response.text[:500]}...")
        continue
    elif response.status_code != 200:
        print(f"Error {response.status_code} for {page}: {response.text[:200]}...")
        continue
    
    response = response.json()
    pages_data = response["query"]["pages"]
    for _, pdata in pages_data.items():
        if "revisions" in pdata:
            content = pdata["revisions"][0]["slots"]["main"]["*"]
            print(f"{page}: {len(content)} characters")
        else:
            print(f"{page}: not found")
    
    # Add a small delay between requests to be respectful to the API
    time.sleep(0.5)
