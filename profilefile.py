import argparse
from helpers.ProfileMultiple import ProfileMultiple

def main():
    parser = argparse.ArgumentParser(description='Python CLI program to profile a file')
    parser.add_argument('file_path', type=str, help='Path to the file to be profiled')
    parser.add_argument('output_filepath', type=str, help='Path to the file to be profiled')
    parser.add_argument('type', type=str, help='Type of file to be profiled (csv or xlsx)')

    args = parser.parse_args()

    _ = ProfileMultiple(args.file_path, args.output_filepath, args.type)

if __name__ == '__main__':
    main()