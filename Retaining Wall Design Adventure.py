import time

def pause():
    time.sleep(1)

def intro():
    print("🏗️ Welcome to the RETAINING WALL DESIGN ADVENTURE! 🧱")
    print("You're the site engineer tasked with selecting the most suitable wall design.")
    print("Make decisions wisely and try to balance cost, safety, and site conditions.\n")
    pause()

def site_understanding():
    print("📍 STEP 1: Understanding the Site")
    input("Have you established the ground model? (Press Enter when done)")
    input("Checked the groundwater regime? (Press Enter when done)")
    input("Do you know the maximum height the wall needs to retain? (Press Enter)")
    input("Is the wall temporary or permanent? (Press Enter)")
    print("✅ Good! You’re ready for environmental checks.")
    pause()

def environmental_considerations():
    print("\n🌿 STEP 2: Environmental Considerations")
    input("Access, noise, vibration, headroom... Have you checked these? (Press Enter)")
    print("🧐 Constraints may limit your options, keep them in mind!")
    pause()

def water_retention():
    print("\n💧 STEP 3: Does the wall need to retain water?")
    while True:
        answer = input("Type 'yes' or 'no': ").strip().lower()
        if answer == "no":
            return False
        elif answer == "yes":
            return True
        else:
            print("❗ Please type 'yes' or 'no'.")

def no_water_path():
    print("\n💡 You don't need to retain water. Choose an economic solution.")
    print("You can use: Contiguous Piles, King Post Piles, or Sheet Piles.")
    print("Just make sure they’re suitable for your retained height!")
    pause()

def water_path():
    print("\n🚰 Water retention detected! Time for some decisions.")
    depth = input("Do you know the depth of required water cut-off? (yes/no): ").strip().lower()
    if depth == "no":
        print("⚠️ You need to determine this before selecting your method.")
        return

    interlock = input("Can you achieve water cut-off with secant or sheet piles? (yes/no): ").strip().lower()
    if interlock == "yes":
        print("✅ Great! Go ahead with interlocked sheet or secant piles.")
    else:
        print("🛠️ You'll need a diaphragm wall – it's more expensive but effective.")
        panel_width = input("Do you know how to size the panels based on height and moment capacity? (yes/no): ").strip().lower()
        if panel_width == "yes":
            print("📏 Good work! Time to move on to detailed design.")
        else:
            print("📚 Review design tables to guide your panel sizing!")

def final_design():
    print("\n🎯 FINAL STEP: Wall Design")
    print("You now determine pile spacing, diameter, and check capacity.")
    print("Make sure it works for both temporary and permanent conditions.")
    print("💪 Don’t forget bending moments and soil pressure!")
    pause()
    print("\n🏁 Congratulations, you've completed the retaining wall design journey! 🎉")

def play_game():
    intro()
    site_understanding()
    environmental_considerations()
    if water_retention():
        water_path()
    else:
        no_water_path()
    final_design()

if __name__ == "__main__":
    play_game()
