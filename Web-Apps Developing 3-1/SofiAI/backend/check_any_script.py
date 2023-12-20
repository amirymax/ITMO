import os

def get_common_programs():
    common_programs = [
        "chrome.exe",    # Google Chrome
        "firefox.exe",   # Mozilla Firefox
        "iexplore.exe",  # Internet Explorer
        "MicrosoftEdge.exe",  # Microsoft Edge
        "notepad.exe",   # Notepad
        "vscode.exe",    # Visual Studio Code
        "sublime_text.exe",   # Sublime Text
        "atom.exe",      # Atom
        "explorer.exe",  # File Explorer
        "winword.exe",   # Microsoft Word
        "excel.exe",     # Microsoft Excel
        "powerpnt.exe",  # Microsoft PowerPoint
        "outlook.exe",   # Microsoft Outlook
        "Skype.exe",     # Skype
        # Add more executables as needed
    ]

    found_programs = []
    for program in common_programs:
        if any((os.path.isfile(os.path.join(os.environ["ProgramFiles(x86)"], program)),
               os.path.isfile(os.path.join(os.environ["ProgramFiles"], program)),
               os.path.isfile(os.path.join(os.environ["LOCALAPPDATA"], "Programs", program)))):
            found_programs.append(program)

    return found_programs

# Get the list of common programs
common_programs = get_common_programs()

# Print the list of common programs
if common_programs:
    print("List of Common Programs:")
    for program in common_programs:
        print(program)
else:
    print("No common programs found.")
