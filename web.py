import eel

eel.init(".\\web")  
  
# Exposing the random_python function to javascript
@eel.expose    
def translate(string:str) -> str:
    outcome = []
    for line in string.splitlines():
        outcome.append('')

    return '\n'.join(outcome)


# Start the index.html file
eel.start("index.html", size=(1200, 800))