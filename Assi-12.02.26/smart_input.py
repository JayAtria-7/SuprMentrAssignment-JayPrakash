name = input("Enter your name: ").strip()

while True:
    age_input = input("Enter your age: ").strip()
    if age_input.isdigit():
        age = int(age_input)
        break
    print("Please enter a valid whole number for age.")

hobby = input("Enter your hobby: ").strip()

if age <= 12:
    category = "child"
elif age <= 19:
    category = "teenager"
elif age <= 59:
    category = "adult"
else:
    category = "senior"

print(f"\nHi {name}! You are {age} years old, which means you are an {category}.")
print(f"It's great that you enjoy {hobby}!")