from collections import Counter

# Function to count occurrences of each character
def count_characters(input_string):
    # Use Counter from collections to count the occurrences of each character
    char_count = Counter(input_string)
    
    # Print each character and its count
    for char, count in char_count.items():
        print(f"'{char}': {count}")

# Example usage
input_string = input("Enter a string: ")
count_characters(input_string)