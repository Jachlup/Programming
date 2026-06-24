import yaml

def load_yaml():
    with open('C:\\ESA Spaceship Poland\\Programming\\Vision System\\Test\\data.yaml', 'r') as file:
        return yaml.safe_load(file)
    
def save_yaml(data):
    with open('C:\\ESA Spaceship Poland\\Programming\\Vision System\\Test\\data.yaml', 'w') as file:
        yaml.dump(data, file)

print(f"{load_yaml()}")

data = {'hej':'działczy'}

save_yaml(data)

print(f"{load_yaml()}")