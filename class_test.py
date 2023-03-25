class father:
    def __init__(self) -> None:
        self.call()
        
    def call(self):
        print(1)
class sun(father):
    def call(self):
        print(2)

n=father()
p=sun()
