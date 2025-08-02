import requests
import json

def test_api():
    """Test the HackRx API with sample data."""
    
    # Test URL (replace with your deployed URL)
    base_url = "http://localhost:8000"  # Change this to your deployed URL
    
    # Sample test data
    test_data = {
        "documents": "https://www.who.int/docs/default-source/coronaviruse/situation-reports/20200121-sitrep-1-2019-ncov.pdf",
        "questions": [
            "What is the main topic of this document?",
            "What are the key recommendations mentioned?"
        ]
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": "Bearer test-api-key"
    }
    
    try:
        print("Testing API...")
        print(f"URL: {base_url}/hackrx/run")
        print(f"Request: {json.dumps(test_data, indent=2)}")
        
        response = requests.post(
            f"{base_url}/hackrx/run",
            json=test_data,
            headers=headers,
            timeout=60
        )
        
        print(f"\nStatus Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        
        if response.status_code == 200:
            print("\n✅ API test successful!")
            return True
        else:
            print(f"\n❌ API test failed with status {response.status_code}")
            return False
            
    except Exception as e:
        print(f"\n❌ Error testing API: {e}")
        return False

if __name__ == "__main__":
    test_api() 