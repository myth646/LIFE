import pandas as pd
import matplotlib.pyplot as plt

def process_and_plot_specific_csv(files, labels, output_pdf):
    # Process each CSV file
    for file, label in zip(files, labels):
        # Read CSV file
        try:
            data = pd.read_csv(file, header=None)
        except FileNotFoundError:
            print(f"File '{file}' not found.")
            continue

        # Calculate average of each row and take every fifth row
        averages = data.mean(axis=1)[::5]

        # Plot the data
        plt.plot(averages, label=label)

    # Set plot labels and title
    plt.xlabel('Row (every fifth)')
    plt.ylabel('Average Value')
    plt.title('Average Row Values of Specific CSV Files')
    plt.legend()

    # Save plot to PDF
    plt.savefig(output_pdf)
    plt.close()

    print(f"Plot saved as '{output_pdf}'.")

# Example usage
files = ['', 'path/to/your/file2.csv']  # Replace with your file paths
labels = ['Label 1', 'Label 2']  # Replace with your desired labels
output_pdf = 'output_plot.pdf'  # Output PDF file name
process_and_plot_specific_csv(files, labels, output_pdf)

