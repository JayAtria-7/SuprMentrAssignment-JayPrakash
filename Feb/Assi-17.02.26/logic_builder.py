def fizz_buzz_value(number):
    if number % 15 == 0:
        return "FizzBuzz"
    if number % 3 == 0:
        return "Fizz"
    if number % 5 == 0:
        return "Buzz"
    return str(number)


def print_fizz_buzz_and_count(start, end):
    counts = {"Fizz": 0, "Buzz": 0, "FizzBuzz": 0, "Number": 0}

    for number in range(start, end + 1):
        value = fizz_buzz_value(number)
        print(value)

        if value in counts:
            counts[value] += 1
        else:
            counts["Number"] += 1

    return counts


def main():
    counts = print_fizz_buzz_and_count(1, 50)

    print("\nCounts:")
    print(f"Fizz: {counts['Fizz']}")
    print(f"Buzz: {counts['Buzz']}")
    print(f"FizzBuzz: {counts['FizzBuzz']}")
    print(f"Number: {counts['Number']}")


if __name__ == "__main__":
    main()
