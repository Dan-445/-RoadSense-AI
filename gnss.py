import pandas as pd
import json

# Load GNSS data from JSON file
with open('gnns/GNSS_41.json', 'r') as f:
    gnss_data = json.load(f)['GNSS']  # Access the 'GNSS' key from the JSON

# Extract relevant fields (timestamp, latitude, and longitude) into a DataFrame
data = {
    'timestamp': [entry['ts'] for entry in gnss_data],
    'latitude': [entry['coordinates']['latitude'] for entry in gnss_data],
    'longitude': [entry['coordinates']['longitude'] for entry in gnss_data]
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('gnss_coordinates.csv', index=False)

print("GNSS data saved to gnss_coordinates.csv")
