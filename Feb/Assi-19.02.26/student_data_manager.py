students = []

print("Enter data for 5 students")

for i in range(1, 6):
    name = input(f"Student {i} name: ")
    marks = float(input(f"{name}'s marks (out of 100): "))

    student = {
        "name": name,
        "marks": marks,
    }
    students.append(student)

# Topper
topper = max(students, key=lambda s: s["marks"])

# Class average
class_average = sum(s["marks"] for s in students) / len(students)

# Grade function
def assign_grade(marks):
    if marks >= 90:
        return "A+"
    if marks >= 80:
        return "A"
    if marks >= 70:
        return "B"
    if marks >= 60:
        return "C"
    if marks >= 50:
        return "D"
    return "F"

print("\n--- Student Report ---")
for student in students:
    grade = assign_grade(student["marks"])
    print(f"{student['name']}: Marks = {student['marks']}, Grade = {grade}")

print(f"\nTopper: {topper['name']} ({topper['marks']})")
print(f"Class Average: {class_average:.2f}")
