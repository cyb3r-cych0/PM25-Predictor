from meteo_config_scripts.make_monthly_alignment import MakeMonthlyAlignment
from meteo_config_scripts.debug_alignment import DebugAlignment
import time

class Run:
    def __init__(self):
        self.align_monthly = MakeMonthlyAlignment()
        self.align_debug = DebugAlignment()
        self.user_input = input("Type 1 for [debug=True] or 2 for [debug=False]: ")

    def make_alignment(self):
        print("\nMaking Monthly Alignment...\n")
        return self.align_monthly.main()

    def debug_alignment(self):
        print("\nDebbuging Alignment...\n")
        return self.align_debug.main()

    def main(self, user_input=None):
        user_input = self.user_input
        if user_input == "1":
            print("Debug True")
            time.sleep(1)
            self.make_alignment()
            print("====" * 30)
            time.sleep(1)
            self.debug_alignment()
            print("====" * 30)
        elif user_input == "2":
            print("Debug False")
            time.sleep(1)
            run.make_alignment()
        else:
            print("You did not type 1 or 2, exiting...\n")
            exit()
            
    
if __name__ == "__main__":
    run = Run()
    run.main()
    