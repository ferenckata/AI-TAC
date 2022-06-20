"""Main entry to the code"""
import UI

def main():
    """
    Main steps of the pipeline
    """
    ui = UI.UserInterface()
    ui.run_preprocessing()

if __name__ == "__main__":
    main()