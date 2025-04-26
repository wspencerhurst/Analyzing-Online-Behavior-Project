import json
from datetime import date, timedelta

# ==========================
# CONFIGURATION
# ==========================

person_name = "Spencer"  # Change to match the person this file is for
output_path = f"data/timelines/{person_name.lower()}_timeline.json"
start = date(2025, 1, 20)
end = date(2025, 4, 20)

# TOGGLE THIS:
# Set to True to fill each day with one dummy activity per category.
# Set to False to generate a clean template with empty arrays.
USE_DUMMY_ENTRIES = True
# Toggle this to include dummy work entries when USE_DUMMY_ENTRIES is True.
USE_DUMMY_WORK_ENTRIES = False

# ==========================
# GENERATOR FUNCTION
# ==========================

def generate_timeline(person, start_date, end_date, use_dummy=False):
    current = start_date
    output = []

    while current <= end_date:
        entry = {
            "date": current.isoformat(),
            "day_of_week": current.strftime("%A"),
            "person": person,
            "events": {
                "academic": [],
                "work": [],
                "exercise": [],
                "activities": [],
                "notes": ""
            }
        }

        # Scheduled classes
        dow = current.strftime("%A")
        if dow in ["Monday", "Wednesday"]:
            entry["events"]["academic"].append({
                "type": "class",
                "course": "Reinforcement Learning",
                "start_time": "9:30",
                "end_time": "10:45",
                "location": "Rice Hall 340"
            })
            entry["events"]["academic"].append({
                "type": "class",
                "course": "Network Security",
                "start_time": "14:00",
                "end_time": "15:15",
                "location": "MEC 341"
            })
            entry["events"]["academic"].append({
                "type": "class",
                "course": "Analyzing Online Behavior",
                "start_time": "15:30",
                "end_time": "16:45",
                "location": "MEC 341"
            })
        elif dow in ["Tuesday", "Thursday"]:
            entry["events"]["academic"].append({
                "type": "class",
                "course": "Machine Learning",
                "start_time": "9:30",
                "end_time": "10:45",
                "location": "Rice Hall 340"
            })
            entry["events"]["academic"].append({
                "type": "class",
                "course": "Software Analysis",
                "start_time": "12:30",
                "end_time": "13:45",
                "location": "Rice Hall 340"
            })

        if use_dummy:
            # Dummy academic entry
            entry["events"]["academic"].append({
                "type": "assignment_due",
                "course": "CS6501",
                "description": "Dummy assignment due",
                "assignment_type": "report",
                "difficulty": 5
            })
            # Dummy work entry (toggleable)
            if USE_DUMMY_WORK_ENTRIES:
                entry["events"]["work"].append({
                    "job_title": "Research Assistant",
                    "hours": 2,
                    "tasks": ["Dummy task"],
                    "exertion": 4
                })
            # Dummy exercise entry
            entry["events"]["exercise"].append({
                "type": "Lifting",
                "duration_minutes": 120,
                "notes": "Chadwell Fall 2022 Lifting Program - Mem Gym",
                "exertion": 7
            })
            # Dummy activity entry
            entry["events"]["activities"].append({
                "type": "journaling",
                "description": "Dummy journal entry about the day",
                "duration_minutes": 15
            })
            entry["events"]["notes"] = ""

        output.append(entry)
        current += timedelta(days=1)

    return output

# ==========================
# MAIN EXECUTION
# ==========================

timeline_data = generate_timeline(person_name, start, end, USE_DUMMY_ENTRIES)

# Save to file
with open(output_path, "w") as f:
    json.dump(timeline_data, f, indent=2)

mode = "dummy-filled" if USE_DUMMY_ENTRIES else "clean"
print(f"✅ Generated {mode} timeline for {person_name} at: {output_path}")