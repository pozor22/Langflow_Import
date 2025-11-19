import requests
import uuid

api_key = ""

flow_test1 = ""
flow_test2 = ""

flow_id = flow_test2

url = f"http://localhost:8082/api/v1/run/{flow_id}"


payload = {
    "output_type": "chat",
    "input_type": "chat",
    "input_value": "сколько будет 2 + 2"
}
payload["session_id"] = str(uuid.uuid4())

headers = {
    "x-api-key": api_key,
}

try:
    # Send API request
    response = requests.request("POST", url, json=payload, headers=headers)
    response.raise_for_status()  # Raise exception for bad status codes

    # Print response
    print(response.text)

except requests.exceptions.RequestException as e:
    print(f"Error making API request: {e}")
except ValueError as e:
    print(f"Error parsing response: {e}")