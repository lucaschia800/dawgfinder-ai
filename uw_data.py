import requests
import pandas as pd
import json

cookies = {
    "_ga": "GA1.1.1169709652.1732947157",
    "_ga_0V5LFWD2KQ": "GS1.1.1732947156.1.1.1732947225.57.0.0",
    "_hp2_id.3001039959": "%7B%22userId%22%3A%223766466782283369%22%2C%22pageviewId%22%3A%22836061124374915%22%2C%22sessionId%22%3A%224498099947331590%22%2C%22identity%22%3A%22uu-2-6523be9528e79cfc59958d23c6f56a841a282a84817f28115c28924b5dc768ad-hMdW75SAnZRIzmj4yn9Zl1cW9eAMfEUMSztnrw0F%22%2C%22trackerVersion%22%3A%224.0%22%2C%22identityField%22%3Anull%2C%22isIdentified%22%3A1%7D",
    "_hp2_props.3001039959": "%7B%22Base.appName%22%3A%22Canvas%22%7D",
    "csrftoken": "q26ZTcxt7iiGfFpcv5D0BrJGJocKLhbb",
    "sessionid": "tiu0wipzn0zfodr1t5shg4kbto05txed"
}

classes = pd.read_csv("ANTH.csv")

classes_list = classes['Course Number'].tolist()

catalog = {}
for class_num in classes_list:
    curr = f"https://dawgpath.uw.edu/api/v1/courses/details/ANTH%20{class_num}"



    response = requests.get(curr, cookies=cookies)

    if response.status_code == 200:
        # If the API returns JSON, try:
        try:
            data = response.json()
            print(data)
        except ValueError:
            # If it's not JSON, print text
            print(response.text)
    else:
        print(f"Error: {response.status_code}, {class_num}")
        print(response.text)

    catalog[class_num] = data
    

with open("data.json", "w", encoding="utf-8") as json_file:
    json.dump(catalog, json_file) 