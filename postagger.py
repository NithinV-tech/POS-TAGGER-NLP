import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Run")
    parser.add_argument("-f", "--file", action="store_true", help="Run ffnn.py")
    parser.add_argument("-r", "--rnn", action="store_true", help="Run lstm.py")
    
    args = parser.parse_args()
    
    if args.file:
        os.system("python ffnn.py")
    elif args.rnn:
        os.system("python lstm.py")
    else:
        print("Please specify either -f or -r option")

if __name__ == "__main__":
    main()
