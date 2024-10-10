import requests
import json

text = '''
Today, Harry experiences a sudden exacerbation of his COPD symptoms, leading to severe shortness of breath and confusion. David notices that Harry is struggling to catch his breath while trying to get out of bed for breakfast. Despite Harry's protests and stubbornness, David immediately calls for an ambulance, knowing that this is a critical situation. 
'''

# Define the headers
headers = {
    'Content-Type': 'application/json'
}

# Define the data to be sent in the POST request
data = {
    'inputText': text
}

# Send the POST request with JSON payload
response = requests.post('http://4.227.146.175:3000/journal/text/', headers=headers, data=json.dumps(data))

print(response.text)