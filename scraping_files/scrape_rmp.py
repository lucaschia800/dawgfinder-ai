import ratemyprofessor as rmp
import pandas as pd
import json

with open('instructors.json', 'r') as f:
    instructors = json.load(f)  # Load as a dictionar


teachers_df = pd.DataFrame(list(instructors.items()), columns=['index', 'instructor'])
teachers = set(teachers_df['instructor'])


import ratemyprofessor as rmp
import pandas as pd
import concurrent.futures


# Initialize empty DataFrame
columns = ["Name", "Rating", "Difficulty", "Link"]
df = pd.DataFrame(columns=columns)

# Set to track unique names
unique_names = set()

# University of Washington
SCHOOL = rmp.School(school_id=1530)

def get_professors_by_name(name):
    global df  # Reference the global DataFrame
    try:
        professors = rmp.get_professors_by_school_and_name(SCHOOL, name)
        if not professors:
            print(f"No matches found for {name}")
        else:
            for prof in professors:
                if prof.name not in unique_names:  # Avoid duplicates
                    unique_names.add(prof.name)
                    new_data = {
                        "Name": prof.name,
                        "Rating": prof.rating,
                        "Difficulty": prof.difficulty,
                        "Link": f"https://www.ratemyprofessors.com/professor/{prof.id}",
                        "Department": prof.department
                    }
                    df = pd.concat([df, pd.DataFrame([new_data])], ignore_index=True)

                    print(f"Added: {prof.name}, Rating: {prof.rating}, Difficulty: {prof.difficulty}")
    except Exception as e:
        print(f"Error retrieving {name}: {e}")



with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    executor.map(get_professors_by_name, teachers)


df.to_json("rmp_info.json")