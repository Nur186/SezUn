import logging

import pyttsx3


class Speaker:
    def __init__(self) -> None:
        self.engine = pyttsx3.init()
        self.engine.setProperty("rate", 165)

    def say(self, text: str) -> None:
        logging.info("Дыбыстық ескерту: %s", text)
        self.engine.say(text)
        self.engine.runAndWait()

    def close(self) -> None:
        self.engine.stop()
