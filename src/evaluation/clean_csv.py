# clean_csv.py

def clean_csv(file_path, output_path):
    with open(file_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            # Split by commas and ensure each line has exactly 11 fields
            if len(line.split(',')) == 11:
                outfile.write(line)
            else:
                print(f"Skipping line due to incorrect field count: {
                      line.strip()}")


clean_csv('data/reviews.csv', 'data/cleaned_feedback_data.csv')
