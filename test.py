def compute_sum_from_file(file_path):
    """
    Compute the sum of floats from a file where each line contains a float.

    Args:
        file_path (str): The path to the file.

    Returns:
        float: The sum of all floats in the file.
    """
    total_sum = 0.0
    with open(file_path, 'r') as file:
        for line in file:
            try:
                total_sum += float(line)
            except ValueError:
                print(f"Skipping line '{line.strip()}' as it doesn't contain a valid float.")

    return total_sum

# Example usage:
file_path = 'your_file.txt'  # Replace 'your_file.txt' with the path to your file
total_sum = compute_sum_from_file('./query3.bin')
print("Sum of floats in the file:", total_sum)