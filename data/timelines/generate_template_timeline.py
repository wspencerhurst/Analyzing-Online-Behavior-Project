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

# ==========================
# GENERATOR FUNCTION
# ==========================

def generate_timeline(person, start_date, end_date, use_dummy=False):
    current = start_date
    output = []

    while current <= end_date:
        entry = {
            "date": current.isoformat(),
            "person": person,
            "events": {
                "academic": [],
                "work": [],
                "exercise": [],
                "activities": [],
                "notes": ""
            }
        }

        if use_dummy:
            entry["events"]["academic"].append({
                "type": "assignment_due",
                "course": "CS6501",
                "description": "Dummy assignment due",
                "assignment_type": "report",
                "difficulty": 5
            })
            entry["events"]["work"].append({
                "job_title": "Research Assistant",
                "hours": 2,
                "tasks": ["Dummy task"],
                "exertion": 4
            })
            entry["events"]["exercise"].append({
                "type": "walking",
                "duration_minutes": 30,
                "notes": "Dummy walk around campus",
                "exertion": 3
            })
            entry["events"]["activities"].append({
                "type": "journaling",
                "description": "Dummy journal entry about the day",
                "duration_minutes": 15
            })
            entry["events"]["notes"] = "This is a dummy entry to show the schema format."

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
