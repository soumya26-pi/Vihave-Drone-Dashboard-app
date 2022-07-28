import requests
import json
api_url = "https://deploytestt.herokuapp.com/frame"
todo ={"frame":"[[[2,17]]]"} 
response = requests.post(api_url,data=todo)

data = response.json()
print(data)