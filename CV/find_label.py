def find_unique_labels(label_file):
    labels = set()  # Use a set to store unique labels

    with open(label_file, 'r') as file:
        for line in file:
            parts = line.strip().split()
            if len(parts) >= 2:
                label = parts[-1]  # The label is the last part of the line
                labels.add(label)  # Add label to the set

    return labels

if __name__ == '__main__':
    label_file_path = 'yolo_data/label/train.txt'
    unique_labels = find_unique_labels(label_file_path)
    print("Unique labels found:", unique_labels)