
class Parser():

    def __init__(self) -> None:
        self.book = ""

    def loadBook(self, bookPath: str) -> None:
        with open(bookPath, 'r', encoding='utf-8') as file:
            self.book = file.read()